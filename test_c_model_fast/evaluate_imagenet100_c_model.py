import os
import sys
import json
import ctypes
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import time
from PIL import Image
from tqdm import tqdm

# =========================================================================
# CẤU HÌNH TOÀN CỤC
# =========================================================================
TFLITE_MODEL_PATH = os.path.abspath("tf_lite_dropout_imagenet100/efficientnetv2_b0_bnless_int8.tflite")
DLL_PATH          = os.path.abspath("test_c_model_fast/efficientnet_c_lib.dll")

# Đường dẫn dataset val.X (ImageNet-100)
VAL_DIR     = os.path.abspath("val.X")
LABELS_JSON = os.path.join(VAL_DIR, "Labels.json")

# Chỉ số tensor trong TFLite (giữ nguyên như evaluate_3000_images_optimine.py)
LAYER_5_INPUT_INDEX = 201   # IFM đầu vào C Engine (sau block đầu của TFLite)
FC_WEIGHTS_INDEX    = 11    # Trọng số lớp FC cuối (per-channel quantized)
FC_BIAS_INDEX       = 10    # Bias lớp FC cuối
LAYER_278_INDEX     = 475   # Output của C Engine để so sánh debug

# =========================================================================
# SỐ WORKER:
# Trên Windows, mỗi worker spawn một process mới và import lại TensorFlow.
# Quá nhiều worker (> 2-3) sẽ gây OOM hoặc deadlock khi khởi động.
# =========================================================================
NUM_WORKERS = 12   # An toàn trên Windows; tăng nếu máy đủ RAM (> 32GB)

# =========================================================================
# BIẾN TOÀN CỤC CHO TỪNG WORKER PROCESS
# =========================================================================
worker_interpreter = None
worker_c_lib       = None
worker_metadata    = {}

# =========================================================================
# HÀM KHỞI TẠO WORKER (chạy 1 lần mỗi process)
# =========================================================================
def init_worker():
    """Khởi tạo TFLite Interpreter + C Engine riêng cho từng worker."""
    global worker_interpreter, worker_c_lib, worker_metadata

    # Tắt toàn bộ printf từ thư viện C bằng cách redirect fd cấp OS
    if hasattr(os, 'O_WRONLY'):
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 1)  # Tắt stdout từ C (printf)
            os.dup2(devnull_fd, 2)  # Tắt stderr từ C (fprintf)
            os.close(devnull_fd)
        except Exception:
            pass  # Nếu không được thì thôi, không crash

    # --- 1. Khởi tạo TFLite ---
    worker_interpreter = tf.lite.Interpreter(
        model_path=TFLITE_MODEL_PATH,
        num_threads=1,
        experimental_preserve_all_tensors=True
    )
    worker_interpreter.allocate_tensors()
    details       = worker_interpreter.get_tensor_details()
    input_details = worker_interpreter.get_input_details()[0]

    # Thông số quantization lớp 278 (output của C Engine)
    scale_278 = details[LAYER_278_INDEX]['quantization_parameters']['scales'][0]
    zp_278    = details[LAYER_278_INDEX]['quantization_parameters']['zero_points'][0]

    # Thông số quantization lớp FC (per-channel)
    scale_w = np.array(details[FC_WEIGHTS_INDEX]['quantization_parameters']['scales']).reshape(-1, 1)
    zp_w    = np.array(details[FC_WEIGHTS_INDEX]['quantization_parameters']['zero_points']).reshape(-1, 1)

    weights   = worker_interpreter.get_tensor(FC_WEIGHTS_INDEX)  # shape: [num_classes, 1280]
    bias      = worker_interpreter.get_tensor(FC_BIAS_INDEX)     # shape: [num_classes]

    # Dequantize bias một lần duy nhất để tránh tính lặp
    bias_scale = (scale_278 * scale_w).flatten()
    bias_float = bias.astype(np.float32) * bias_scale

    worker_metadata = {
        'input_details' : input_details,
        'input_dtype'   : input_details['dtype'],
        'input_quant'   : input_details['quantization'],    # (scale, zero_point)
        'input_shape'   : input_details['shape'],           # [1, H, W, C]
        'scale_278'     : scale_278,
        'zp_278'        : zp_278,
        'scale_w'       : scale_w,
        'zp_w'          : zp_w,
        'weights'       : weights,
        'bias_float'    : bias_float,
    }

    # --- 2. Khởi tạo C Engine ---
    worker_c_lib = ctypes.CDLL(DLL_PATH)
    worker_c_lib.init_model.argtypes = []
    worker_c_lib.free_model.argtypes = []
    worker_c_lib.run_model_inference.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int8, ndim=1, flags='C_CONTIGUOUS'),
    ]
    worker_c_lib.init_model()


# =========================================================================
# HÀM TIỀN XỬ LÝ ẢNH (PIL, giống eva.py)
# =========================================================================
def preprocess_image(img_path: str, input_shape, input_dtype, input_quant):
    """
    Đọc ảnh từ đường dẫn, resize về kích thước model, quantize nếu cần.
    Trả về numpy array shape [1, H, W, C].
    """
    h, w = int(input_shape[1]), int(input_shape[2])
    img = Image.open(img_path).convert('RGB')
    img = img.resize((w, h), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)  # [0, 255] float

    if input_dtype in (np.int8, np.uint8):
        scale, zero_point = input_quant   # Tuple (scale, zero_point)
        img_array = np.round((img_array / scale) + zero_point)
        lo = np.iinfo(input_dtype).min
        hi = np.iinfo(input_dtype).max
        img_array = np.clip(img_array, lo, hi).astype(input_dtype)
    else:
        img_array = img_array.astype(np.float32)

    return np.expand_dims(img_array, axis=0)  # [1, H, W, C]


# =========================================================================
# HÀM XỬ LÝ MỘT ẢNH (chạy trong Worker)
# =========================================================================
def process_single_image(data_dict):
    """
    Worker nhận dict {'img_path', 'true_label', 'index'},
    trả về dict {'index', 'true_label', 'pred_label', 'pred_top5', 'is_top1', 'is_top5', 'debug_info'}.
    """
    img_path   = data_dict['img_path']
    true_label = data_dict['true_label']
    idx        = data_dict['index']

    meta          = worker_metadata
    input_details = meta['input_details']

    # --- Tiền xử lý ảnh ---
    try:
        input_data = preprocess_image(
            img_path,
            meta['input_shape'],
            meta['input_dtype'],
            meta['input_quant']
        )
    except Exception as e:
        # Nếu ảnh lỗi thì bỏ qua, coi như đoán sai
        return {
            'index'      : idx,
            'true_label' : true_label,
            'pred_label' : -1,
            'pred_top5'  : [],
            'is_top1'    : False,
            'is_top5'    : False,
            'debug_info' : None,
            'error'      : str(e)
        }

    # --- TFLite chạy để lấy IFM cho C Engine ---
    worker_interpreter.set_tensor(input_details['index'], input_data)
    worker_interpreter.invoke()
    ifm_layer5 = np.ascontiguousarray(
        worker_interpreter.get_tensor(LAYER_5_INPUT_INDEX).flatten(),
        dtype=np.int8
    )

    # --- C Engine chạy inference ---
    c_output_1d = np.zeros(1280, dtype=np.int8)
    worker_c_lib.run_model_inference(ifm_layer5, c_output_1d)

    # --- Debug: so sánh output C vs TFLite (10 ảnh đầu) ---
    debug_info = None
    if idx < 10:
        tflite_out = worker_interpreter.get_tensor(LAYER_278_INDEX).flatten()
        diff = np.abs(c_output_1d.astype(np.int16) - tflite_out.astype(np.int16))
        debug_info = {
            'c_out'    : c_output_1d[:15].copy(),
            'tf_out'   : tflite_out[:15].copy(),
            'max_error': int(np.max(diff)),
            'err_1'    : int(np.sum(diff == 1)),
            'err_2'    : int(np.sum(diff == 2)),
            'err_gt2'  : int(np.sum(diff > 2)),
        }

    # --- Dequantize output C Engine → float ---
    out_float = (c_output_1d.astype(np.float32) - meta['zp_278']) * meta['scale_278']

    # --- Dequantize weights FC (per-channel) ---
    w_float = (meta['weights'].astype(np.float32) - meta['zp_w']) * meta['scale_w']

    # --- Tính logits = out @ W^T + bias ---
    logits = np.dot(out_float, w_float.T) + meta['bias_float']

    # Xử lý trường hợp logits có 1001 class (ImageNet-1k với background class)
    if len(logits) == 1001:
        logits = logits[1:]

    # --- Top-1 và Top-5 ---
    top5_idx   = np.argpartition(logits, -5)[-5:]
    pred_label = top5_idx[np.argmax(logits[top5_idx])]

    is_top1 = (pred_label == true_label)
    is_top5 = (true_label in top5_idx)

    return {
        'index'      : idx,
        'true_label' : true_label,
        'pred_label' : int(pred_label),
        'pred_top5'  : top5_idx.tolist(),
        'is_top1'    : is_top1,
        'is_top5'    : is_top5,
        'debug_info' : debug_info,
        'error'      : None,
    }


# =========================================================================
# MAIN
# =========================================================================
def main():
    # ------------------------------------------------------------------
    # 1. Đọc Labels.json và xây dựng mapping folder_name → class_id
    # ------------------------------------------------------------------
    print(f"[*] Đang đọc nhãn từ: {LABELS_JSON}")
    if not os.path.exists(LABELS_JSON):
        print(f"[LỖI] Không tìm thấy file {LABELS_JSON}")
        return

    with open(LABELS_JSON, 'r') as f:
        labels_data = json.load(f)

    # Keras sắp xếp class theo thứ tự bảng chữ cái → dùng sorted()
    sorted_folders = sorted(labels_data.keys())
    class_to_id    = {folder: idx for idx, folder in enumerate(sorted_folders)}
    num_classes    = len(sorted_folders)
    print(f"[*] Số lớp trong dataset: {num_classes}")

    # ------------------------------------------------------------------
    # 2. Quét toàn bộ ảnh trong val.X
    # ------------------------------------------------------------------
    if not os.path.exists(VAL_DIR):
        print(f"[LỖI] Không tìm thấy thư mục {VAL_DIR}")
        return

    all_tasks = []
    print("[*] Đang quét danh sách ảnh...")
    for folder_name, label_id in class_to_id.items():
        folder_path = os.path.join(VAL_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                all_tasks.append({
                    'img_path'  : os.path.join(folder_path, fname),
                    'true_label': label_id,
                    'index'     : len(all_tasks),
                })

    total = len(all_tasks)
    if total == 0:
        print("[LỖI] Không tìm thấy ảnh nào trong val.X")
        return
    print(f"[*] Tổng số ảnh: {total}")

    # ------------------------------------------------------------------
    # 3. Lấy thông tin model (chỉ để log, không dùng trong worker)
    # ------------------------------------------------------------------
    print("[*] Đang đọc cấu hình TFLite (tiền xử lý chính)...")
    temp_interp = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    temp_interp.allocate_tensors()
    inp = temp_interp.get_input_details()[0]
    print(f"    Input shape : {inp['shape']}")
    print(f"    Input dtype : {inp['dtype']}")
    print(f"    Quantization: scale={inp['quantization'][0]:.6f}, zp={inp['quantization'][1]}")
    del temp_interp

    # ------------------------------------------------------------------
    # 4. Chạy song song
    # ------------------------------------------------------------------
    # Trên Windows dùng spawn → mỗi worker re-import TF → giới hạn ở NUM_WORKERS
    num_workers = min(NUM_WORKERS, mp.cpu_count())
    print(f"[*] Bắt đầu suy luận song song với {num_workers} workers (Windows-safe)...\n")

    correct_top1 = 0
    correct_top5 = 0
    error_count  = 0
    start_time   = time.time()

    with mp.Pool(processes=num_workers, initializer=init_worker) as pool:
        for result in tqdm(
            pool.imap(process_single_image, all_tasks, chunksize=16),
            total=total,
            desc="Đánh giá C Model",
            unit="ảnh"
        ):
            idx = result['index']

            # In debug 10 ảnh đầu
            if result['debug_info']:
                dbg = result['debug_info']
                tqdm.write("─" * 55)
                tqdm.write(f"[DEBUG ảnh {idx}] So sánh output C Engine vs TFLite tại layer 278:")
                tqdm.write(f"  Max error  : {dbg['max_error']}")
                tqdm.write(f"  |diff| = 1 : {dbg['err_1']} phần tử")
                tqdm.write(f"  |diff| = 2 : {dbg['err_2']} phần tử")
                tqdm.write(f"  |diff| > 2 : {dbg['err_gt2']} phần tử")
                tqdm.write(f"  C Output   : {dbg['c_out']}")
                tqdm.write(f"  TFLite Out : {dbg['tf_out']}")
                tqdm.write("─" * 55)

            if result['error']:
                error_count += 1
                tqdm.write(f"[WARN] Ảnh {idx} lỗi: {result['error']}")
                continue

            if result['is_top1']:
                correct_top1 += 1
            if result['is_top5']:
                correct_top5 += 1

            # Log định kỳ mỗi 200 ảnh và 10 ảnh đầu
            processed = idx + 1
            if idx < 10 or processed % 200 == 0:
                mark = '✅' if result['is_top1'] else '❌'
                acc1 = (correct_top1 / processed) * 100
                acc5 = (correct_top5 / processed) * 100
                tqdm.write(
                    f"Ảnh {processed}/{total} | True: {result['true_label']:>3} "
                    f"Pred: {result['pred_label']:>3} {mark} | "
                    f"Top-1: {acc1:.2f}%  Top-5: {acc5:.2f}%"
                )

    # ------------------------------------------------------------------
    # 5. In kết quả tổng kết
    # ------------------------------------------------------------------
    total_time    = time.time() - start_time
    valid_total   = total - error_count
    top1_acc      = (correct_top1 / valid_total * 100) if valid_total > 0 else 0.0
    top5_acc      = (correct_top5 / valid_total * 100) if valid_total > 0 else 0.0

    print("\n" + "=" * 55)
    print("                KẾT QUẢ ĐÁNH GIÁ C MODEL")
    print("=" * 55)
    print(f"  Dataset       : val.X (ImageNet-100)")
    print(f"  Tổng ảnh      : {total}")
    print(f"  Ảnh lỗi       : {error_count}")
    print(f"  Ảnh hợp lệ   : {valid_total}")
    print(f"  Top-1 Correct : {correct_top1}")
    print(f"  Top-5 Correct : {correct_top5}")
    print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
    print(f"  Thời gian     : {total_time:.2f}s  ({total_time/valid_total*1000:.1f} ms/ảnh)")
    print("=" * 55)


if __name__ == '__main__':
    # Windows: bắt buộc dùng 'spawn' và freeze_support để tránh spawning vô hạn
    mp.freeze_support()

    # Kiểm tra các file cần thiết trước khi chạy
    missing = []
    for label, path in [
        ("TFLite model", TFLITE_MODEL_PATH),
        ("C DLL",        DLL_PATH),
        ("val.X dir",    VAL_DIR),
        ("Labels.json",  LABELS_JSON),
    ]:
        if not os.path.exists(path):
            missing.append(f"  ✗ {label}: {path}")

    if missing:
        print("[LỖI] Không tìm thấy các file/thư mục sau (CWD={}):".format(os.getcwd()))
        print("\n".join(missing))
        print("\nHãy chạy script từ thư mục gốc của project:")
        print("  cd c:\\Users\\84333\\projects\\framework_tensorflow_lite_CNN")
        print("  python test_c_model_fast/evaluate_imagenet100_c_model.py")
        sys.exit(1)

    print("[*] Đường dẫn đã xác nhận:")
    print(f"    TFLite : {TFLITE_MODEL_PATH}")
    print(f"    DLL    : {DLL_PATH}")
    print(f"    Val dir: {VAL_DIR}")
    main()
