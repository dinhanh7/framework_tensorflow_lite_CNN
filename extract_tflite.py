import os

import tensorflow as tf

import numpy as np



# Import hàm build mô hình của bạn

from model_efficientnet_non_dropout import build_efficientnetv2_b0_full_bnless



# ==========================================

# 0) CẤU HÌNH ĐƯỜNG DẪN

# ==========================================

WEIGHTS_PATH    = "./output_non_dropout_image100/best_model.h5"

VAL_DIR         = "/root/dinhanh/kaggle_data/val.X"

OUTPUT_DIR      = "./output_non_dropout_image100"

IMG_SIZE        = 224



# ==========================================

# 1) KHÔI PHỤC ĐỒ THỊ VÀ NẠP TRỌNG SỐ

# ==========================================

print(f"[INFO] Khởi tạo đồ thị Keras và nạp weights từ: {WEIGHTS_PATH}")

base_model = build_efficientnetv2_b0_full_bnless()



# Phục dựng lại node Lambda để topology khớp 100% với file .h5 đã lưu

float32_output = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32), name='cast_to_float32')(base_model.output)

model = tf.keras.Model(inputs=base_model.input, outputs=float32_output, name=base_model.name)



# Nạp ma trận trọng số

model.load_weights(WEIGHTS_PATH)

print("[INFO] Đã nạp trạng thái mô hình thành công.")



# ==========================================

# 2) XUẤT TFLITE: ĐỊNH DẠNG FP32 (Tiêu chuẩn)

# ==========================================

print("\n[INFO] Bắt đầu biên dịch sang TFLite FP32...")

converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(model)



# Cho phép TFLite sử dụng các custom ops của TF nếu tập lệnh chuẩn bị thiếu

# (hỗ trợ quá trình compile các hàm kích hoạt tự định nghĩa như h_swish)

converter_fp32.target_spec.supported_ops = [

    tf.lite.OpsSet.TFLITE_BUILTINS,

    tf.lite.OpsSet.SELECT_TF_OPS

]



tflite_fp32_model = converter_fp32.convert()

fp32_path = os.path.join(OUTPUT_DIR, "efficientnet_bnless_fp32.tflite")

with open(fp32_path, "wb") as f:

    f.write(tflite_fp32_model)

print(f"[SUCCESS] Đã lưu mô hình FP32 tại: {fp32_path}")



# ==========================================

# 3) XUẤT TFLITE: ĐỊNH DẠNG INT8 (Full Integer Quantization)

# ==========================================

print("\n[INFO] Bắt đầu biên dịch sang TFLite INT8 (Post-Training Quantization)...")



# Khởi tạo pipeline nạp dữ liệu hiệu chuẩn (Calibration Dataset)

# Quá trình này cần khoảng 100-500 ảnh để đo đạc dải phân bố động (dynamic range) của các activation

def representative_data_gen():

    dataset = tf.keras.utils.image_dataset_from_directory(

        VAL_DIR,