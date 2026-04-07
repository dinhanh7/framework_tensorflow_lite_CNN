import os
import ctypes
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import multiprocessing as mp
from tqdm import tqdm # Thêm thư viện tqdm

# =========================================================================
# CẤU HÌNH TOÀN CỤC
# =========================================================================
NUM_IMAGES = 3000
TFLITE_MODEL_PATH = r"efficientnetv2_b0_bnless_int8_new_weight.tflite"
DLL_PATH = os.path.abspath("efficientnet_c_lib.dll")

LAYER_5_INPUT_INDEX = 201 
FC_WEIGHTS_INDEX = 11
FC_BIAS_INDEX = 10
LAYER_278_INDEX = 475

# Các biến toàn cục dành riêng cho từng Worker
worker_interpreter = None
worker_c_lib = None
worker_metadata = {}

# =========================================================================
# HÀM KHỞI TẠO CHO TỪNG WORKER PROCESS
# =========================================================================
def init_worker():
    """Hàm này chạy một lần cho mỗi process được sinh ra."""
    global worker_interpreter, worker_c_lib, worker_metadata
    
    # ---------------------------------------------------------------------
    # MẸO: TẮT TOÀN BỘ PRINTF TỪ THƯ VIỆN C
    # Chuyển hướng stdout (fd 1) và stderr (fd 2) ở mức hệ điều hành vào devnull
    # ---------------------------------------------------------------------
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1) # Tắt stdout từ C (printf)
    os.dup2(devnull_fd, 2) # Tắt stderr từ C (fprintf(stderr, ...))
    
    # 1. Khởi tạo TFLite riêng biệt cho Process này
    worker_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, experimental_preserve_all_tensors=True)
    worker_interpreter.allocate_tensors()
    details = worker_interpreter.get_tensor_details()
    input_details = worker_interpreter.get_input_details()[0]
    
    # Lấy tham số Quantization
    scale_278 = details[LAYER_278_INDEX]['quantization_parameters']['scales'][0]
    zp_278 = details[LAYER_278_INDEX]['quantization_parameters']['zero_points'][0]
    scale_w = np.array(details[FC_WEIGHTS_INDEX]['quantization_parameters']['scales']).reshape(-1, 1)
    zp_w = np.array(details[FC_WEIGHTS_INDEX]['quantization_parameters']['zero_points']).reshape(-1, 1)

    weights = worker_interpreter.get_tensor(FC_WEIGHTS_INDEX)
    bias = worker_interpreter.get_tensor(FC_BIAS_INDEX)

    # Cân bằng Bias
    bias_scale = (scale_278 * scale_w).flatten()
    bias_float = bias.astype(np.float32) * bias_scale

    worker_metadata = {
        'input_details': input_details,
        'layer_278_index': LAYER_278_INDEX,
        'scale_278': scale_278,
        'zp_278': zp_278,
        'scale_w': scale_w,
        'zp_w': zp_w,
        'weights': weights,
        'bias_float': bias_float
    }

    # 2. Khởi tạo C Engine riêng biệt cho Process này
    worker_c_lib = ctypes.CDLL(DLL_PATH)
    worker_c_lib.init_model.argtypes = []
    worker_c_lib.free_model.argtypes = []
    worker_c_lib.run_model_inference.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags='C_CONTIGUOUS')
    ]
    worker_c_lib.init_model()

# =========================================================================
# HÀM XỬ LÝ LÕI TRONG WORKER
# =========================================================================
def process_single_image(data_dict):
    """Worker sẽ nhận numpy array, chạy suy luận và trả về kết quả."""
    input_data = data_dict['image']
    true_label = data_dict['label']
    idx = data_dict['index']

    input_details = worker_metadata['input_details']
    
    # 1. TFLite chạy mồi lấy IFM
    worker_interpreter.set_tensor(input_details['index'], input_data)
    worker_interpreter.invoke()
    ifm_layer5 = np.ascontiguousarray(worker_interpreter.get_tensor(LAYER_5_INPUT_INDEX).flatten(), dtype=np.int8)
    
    # 2. C Engine tính toán
    c_output_1d = np.zeros(1280, dtype=np.int8)
    worker_c_lib.run_model_inference(ifm_layer5, c_output_1d)

    # Đóng gói dữ liệu debug nếu là 10 ảnh đầu tiên
    debug_info = None
    if idx <= 10:
        tflite_out = worker_interpreter.get_tensor(worker_metadata['layer_278_index']).flatten()
        debug_info = {
            'c_out': c_output_1d[:15].copy(),
            'tf_out': tflite_out[:15].copy(),
            'max_error': np.max(np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16))),
            'err_1': np.sum(np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16)) == 1),
            'err_2': np.sum(np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16)) == 2),
            'err_gt2': np.sum(np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16)) > 2)
        }

    # 3. Tính FC cuối
    out_float = (c_output_1d.astype(np.float32) - worker_metadata['zp_278']) * worker_metadata['scale_278']
    w_float = (worker_metadata['weights'].astype(np.float32) - worker_metadata['zp_w']) * worker_metadata['scale_w']
    logits = np.dot(out_float, w_float.T) + worker_metadata['bias_float']
    
    pred_label = np.argmax(logits[1:]) if len(logits) == 1001 else np.argmax(logits)
    
    return {
        'index': idx,
        'true_label': true_label,
        'pred_label': pred_label,
        'is_correct': pred_label == true_label,
        'debug_info': debug_info
    }

# =========================================================================
# MAIN PROCESS
# =========================================================================
if __name__ == '__main__':
    print("[*] Đang đọc cấu hình TFLite Model (Tiền xử lý)...")
    temp_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    temp_interpreter.allocate_tensors()
    input_details = temp_interpreter.get_input_details()[0]
    input_scale, input_zp = input_details['quantization']
    input_dtype = input_details['dtype']
    
    print("[*] Chuẩn bị pipeline dữ liệu...")
    
    def preprocess_image(example, index):
        img = tf.image.resize(example['image'], [224, 224])
        if input_dtype == np.int8:
            img = (img / input_scale) + input_zp
            img = tf.cast(tf.clip_by_value(tf.round(img), -128, 127), tf.int8)
        elif input_dtype == np.uint8:
            img = (img / input_scale) + input_zp
            img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        else:
            img = tf.cast(img, tf.float32)
            
        img = tf.expand_dims(img, axis=0)
        return {'image': img, 'label': example['label'], 'index': index}

    ds = tfds.load('imagenet_v2', split='test', shuffle_files=False).take(NUM_IMAGES)
    ds = ds.enumerate() 
    ds = ds.map(lambda idx, ex: preprocess_image(ex, idx), num_parallel_calls=tf.data.AUTOTUNE)
    ds_iterator = ds.as_numpy_iterator()

    print(f"[*] Bắt đầu suy luận song song với {mp.cpu_count()} CPU Cores...")
    correct_count = 0
    start_time = time.time()

    with mp.Pool(processes=mp.cpu_count(), initializer=init_worker) as pool:
        
        # ---------------------------------------------------------------------
        # SỬ DỤNG TQDM ĐỂ TẠO THANH TIẾN TRÌNH
        # ---------------------------------------------------------------------
        for result in tqdm(pool.imap(process_single_image, ds_iterator), total=NUM_IMAGES, desc="Tiến trình suy luận", unit="ảnh"):
            idx = result['index']
            
            if result['debug_info']:
                dbg = result['debug_info']
                # Dùng tqdm.write thay cho print() để không làm giật/lỗi thanh tiến trình
                tqdm.write("--- SO SÁNH TRỰC TIẾP TẠI LAYER 278 ---")
                tqdm.write(f"Sai số lớn nhất: {dbg['max_error']}")
                tqdm.write(f"Phần tử sai số = 1: {dbg['err_1']}")
                tqdm.write(f"Phần tử sai số = 2: {dbg['err_2']}")
                tqdm.write(f"Phần tử sai số > 2: {dbg['err_gt2']}")
                tqdm.write("\n" + "!"*50)
                tqdm.write(f"C Output     : {dbg['c_out']}")
                tqdm.write(f"TFLite Output: {dbg['tf_out']}")
                tqdm.write("!"*50 + "\n")

            if result['is_correct']:
                correct_count += 1

            if idx < 10 or (idx + 1) % 100 == 0:
                mark = '✅' if result['is_correct'] else '❌'
                acc = (correct_count / (idx + 1)) * 100
                # Dùng tqdm.write cho log định kỳ
                tqdm.write(f"Ảnh {idx+1}/{NUM_IMAGES} - True: {result['true_label']}, Pred: {result['pred_label']} - {mark} | Accuracy hiện tại: {acc:.2f}%")

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print(f"⏱  Thời gian hoàn thành: {total_time:.2f} giây")
    print("🏆 KẾT QUẢ CUỐI CÙNG: Top-1 Accuracy : {:.2f}%".format((correct_count/NUM_IMAGES)*100))