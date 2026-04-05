import os
import ctypes
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import time

NUM_IMAGES = 3000
TFLITE_MODEL_PATH = r"cell_hswish_hsigmoid\efficientnetv2_b0_bnless_int8.tflite"

LAYER_5_INPUT_INDEX = 201 
FC_WEIGHTS_INDEX = 11
FC_BIAS_INDEX = 10
LAYER_278_INDEX = 475

print("[*] Đang đọc cấu hình TFLite Model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()
input_details = interpreter.get_input_details()[0]

input_scale, input_zp = input_details['quantization']
input_dtype = input_details['dtype']

# =========================================================================
# KHÔI PHỤC TOÁN HỌC FULLY CONNECTED (Xử lý mảng Per-Channel)
# =========================================================================
scale_278 = details[LAYER_278_INDEX]['quantization_parameters']['scales'][0]
zp_278 = details[LAYER_278_INDEX]['quantization_parameters']['zero_points'][0]

scale_w = np.array(details[FC_WEIGHTS_INDEX]['quantization_parameters']['scales']).reshape(-1, 1)
zp_w = np.array(details[FC_WEIGHTS_INDEX]['quantization_parameters']['zero_points']).reshape(-1, 1)

weights = interpreter.get_tensor(FC_WEIGHTS_INDEX)
bias = interpreter.get_tensor(FC_BIAS_INDEX)

# Cân bằng Bias 
bias_scale = (scale_278 * scale_w).flatten()
bias_float = bias.astype(np.float32) * bias_scale

print("[*] Đang nạp thư viện C Engine...")
dll_path = os.path.abspath("efficientnet_c_lib.dll")
c_lib = ctypes.CDLL(dll_path)

c_lib.init_model.argtypes = []
c_lib.free_model.argtypes = []
c_lib.run_model_inference.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags='C_CONTIGUOUS')
]

ds = tfds.load('imagenet_v2', split='test', shuffle_files=False)

c_lib.init_model()

correct_count = 0
start_time = time.time()

for i, example in enumerate(ds.take(NUM_IMAGES)):
    img_tensor = example['image']
    true_label = example['label'].numpy()

    img_resized = tf.image.resize(img_tensor, [224, 224]).numpy()
    if input_dtype == np.int8:
        input_data = (img_resized / input_scale) + input_zp
        input_data = np.clip(np.round(input_data), -128, 127).astype(np.int8)
    elif input_dtype == np.uint8:
        input_data = (img_resized / input_scale) + input_zp
        input_data = np.clip(np.round(input_data), 0, 255).astype(np.uint8)
    else:
        input_data = img_resized.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    # TFLite chạy mồi lấy IFM
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    ifm_layer5 = np.ascontiguousarray(interpreter.get_tensor(LAYER_5_INPUT_INDEX).flatten(), dtype=np.int8)
    
    # C Engine tính toán
    c_output_1d = np.zeros(1280, dtype=np.int8)
    c_lib.run_model_inference(ifm_layer5, c_output_1d)

    # =========================================================================
    # DEBUG CẮT LỚP 
    # =========================================================================
    if i <= 10:  # Chỉ in chi tiết cho 10 ảnh đầu tiên để tránh quá tải thông tin
        print("--- SO SÁNH TRỰC TIẾP TẠI LAYER 278 ---")
        tflite_out = interpreter.get_tensor(LAYER_278_INDEX).flatten()
        max_error = np.max(np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16)))
        error_count_1 = np.sum(np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16)) == 1)
        error_count_2 = np.sum(np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16)) == 2)
        error_count_gt2 = np.sum(np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16)) > 2)
        print(f"Sai số lớn nhất: {max_error}")
        print(f"Phần tử sai số = 1: {error_count_1}")
        print(f"Phần tử sai số = 2: {error_count_2}")
        print(f"Phần tử sai số > 2: {error_count_gt2}")
        print("\n" + "!"*50)
        print(f"C Output     : {c_output_1d[:15]}")
        print(f"TFLite Output: {tflite_out[:15]}")
        print("!"*50 + "\n")

    # Tính FC cuối
    out_float = (c_output_1d.astype(np.float32) - zp_278) * scale_278
    w_float = (weights.astype(np.float32) - zp_w) * scale_w
    logits = np.dot(out_float, w_float.T) + bias_float
    
    pred_label = np.argmax(logits[1:]) if len(logits) == 1001 else np.argmax(logits)
    if pred_label == true_label:
        correct_count += 1
    # In kết quả từng ảnh và acc hiện tại
    if i < 10 or (i + 1) % 100 == 0:  # In kết quả của 10 ảnh đầu tiên và sau đó mỗi 100 ảnh
        print(f"Ảnh {i+1}/{NUM_IMAGES} - True: {true_label}, Pred: {pred_label} - {'✅' if pred_label == true_label else '❌'}")
        print(f"Accuracy hiện tại: {(correct_count/NUM_IMAGES)*100:.2f}%")

print("\n" + "="*50)
c_lib.free_model()
print("🏆 KẾT QUẢ CUỐI CÙNG: Top-1 Accuracy : {:.2f}%".format((correct_count/NUM_IMAGES)*100))