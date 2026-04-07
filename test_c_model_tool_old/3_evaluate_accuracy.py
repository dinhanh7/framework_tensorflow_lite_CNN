import os
import json
import numpy as np
import tensorflow as tf

# --- CẤU HÌNH ---
TFLITE_MODEL_PATH = r"cell_hswish_hsigmoid\efficientnetv2_b0_bnless_int8.tflite"
C_OUTPUT_DIR = "c_outputs/"
LABEL_FILE = "ground_truth_labels.json"

# CHỈ SỐ LAYER (Đảm bảo bạn đã điền đúng)
FC_WEIGHTS_INDEX = 11
FC_BIAS_INDEX = 10   
LAYER_278_INDEX = 475

print("[*] Load nhãn Ground Truth...")
if not os.path.exists(LABEL_FILE):
    raise FileNotFoundError(f"Không tìm thấy {LABEL_FILE}. Hãy chạy Script 1 trước.")

with open(LABEL_FILE, 'r') as f:
    ground_truth_dict = json.load(f)

print("[*] Đang đọc cấu hình Classifier Layer từ TFLite...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
details = interpreter.get_tensor_details()

# 1. Lấy Quantization của Input (Layer 278)
scale_278 = details[LAYER_278_INDEX]['quantization_parameters']['scales'][0]
zp_278 = details[LAYER_278_INDEX]['quantization_parameters']['zero_points'][0]

# 2. Lấy Quantization của Weights (Thường là mảng nhiều giá trị - Per-channel)
scale_w = np.array(details[FC_WEIGHTS_INDEX]['quantization_parameters']['scales'])
zp_w = np.array(details[FC_WEIGHTS_INDEX]['quantization_parameters']['zero_points'])

if len(scale_w) > 1:
    scale_w = scale_w.reshape(-1, 1) # Reshape để nhân ma trận không bị lỗi Broadcast
    zp_w = zp_w.reshape(-1, 1)

# 3. Load Weights và Bias
weights = interpreter.get_tensor(FC_WEIGHTS_INDEX)
bias = interpreter.get_tensor(FC_BIAS_INDEX)

def run_fc_and_get_pred(c_out_int8):
    # A. Lượng tử hóa ngược Input -> Float
    out_float = (c_out_int8.astype(np.float32) - zp_278) * scale_278
    
    # B. Lượng tử hóa ngược Weights -> Float
    w_float = (weights.astype(np.float32) - zp_w) * scale_w
    
    # C. Lượng tử hóa ngược Bias (INT32 -> Float)
    # Công thức: scale_bias = scale_input * scale_weights
    bias_scale = (scale_278 * scale_w).flatten()
    bias_float = bias.astype(np.float32) * bias_scale
    
    # D. Nhân ma trận: Y = X * W^T + Bias
    logits = np.dot(out_float, w_float.T) + bias_float
    
    # E. Trả về Class có xác suất cao nhất
    return np.argmax(logits)

correct_count = 0
total_files = 0

print("\n[*] Đánh giá độ chính xác (Accuracy)...")
for txt_file in os.listdir(C_OUTPUT_DIR):
    if not txt_file.endswith(".txt"): continue
    
    img_name = txt_file.replace("out278_", "").replace(".txt", "")
    
    if img_name not in ground_truth_dict:
        continue
        
    true_label = ground_truth_dict[img_name]
    total_files += 1
    
    with open(os.path.join(C_OUTPUT_DIR, txt_file), "r") as f:
        c_out = np.array([int(line.strip()) for line in f.readlines()])
    
    pred = run_fc_and_get_pred(c_out)
    
    if pred == true_label:
        correct_count += 1

print("========================================")
if total_files > 0:
    acc = (correct_count / total_files) * 100
    print(f"Kết quả Top-1 Accuracy: {correct_count}/{total_files} ({acc:.2f}%)")
else:
    print("Không có kết quả nào được đánh giá.")