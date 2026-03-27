import tensorflow as tf
import numpy as np
import os

print(f"TensorFlow Version: {tf.__version__}")

# ============================================================================
# 1. THIẾT LẬP THƯ MỤC VÀ DỮ LIỆU
# ============================================================================

output_dir = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\hardsigmoid_test_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Thư mục dữ liệu test: {output_dir}")

# Dữ liệu float đầu vào để test
ifm_float = np.array([[-10.0, -5.0, -3.0, 0.0, 3.0, 5.0, 10.0]], dtype=np.float32)

# ============================================================================
# 2. TẠO VÀ LƯỢNG TỬ HÓA MODEL HARDSIGMOID
# ============================================================================

# --- Xây dựng model Keras chỉ có Hardsigmoid ---
inputs = tf.keras.Input(shape=(ifm_float.shape[1],), name="input")
outputs = tf.keras.layers.Activation('hard_sigmoid')(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# --- Tạo dữ liệu đại diện cho quá trình lượng tử hóa ---
# TFLite cần dữ liệu này để xác định scale/zp cho các tensor
def representative_dataset():
    for i in range(100):
        yield [np.random.uniform(low=-12.0, high=12.0, size=(1, ifm_float.shape[1])).astype(np.float32)]

# --- Chuyển đổi model sang TFLite INT8 ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
print("\nĐã tạo và lượng tử hóa model TFLite thành công.")

# ============================================================================
# 3. CHẠY TFLITE INTERPRETER ĐỂ LẤY KẾT QUẢ "VÀNG"
# ============================================================================

interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

# --- Lấy thông tin chi tiết của tensor input/output ---
input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

# --- Trích xuất các tham số lượng tử hóa đã được TFLite tính toán ---
input_scale = input_details['quantization_parameters']['scales'][0]
input_zp = input_details['quantization_parameters']['zero_points'][0]
output_scale = output_details['quantization_parameters']['scales'][0]
output_zp = output_details['quantization_parameters']['zero_points'][0]

# --- Lượng tử hóa dữ liệu đầu vào theo đúng tham số của TFLite ---
ifm_quantized = np.clip(np.round(ifm_float / input_scale) + input_zp, -128, 127).astype(np.int8)

# --- Chạy suy luận ---
interpreter.set_tensor(input_details['index'], ifm_quantized)
interpreter.invoke()
ofm_golden_quantized = interpreter.get_tensor(output_details['index'])[0] # Bỏ chiều batch

# ============================================================================
# 4. LƯU TẤT CẢ DỮ LIỆU RA FILE
# ============================================================================

# --- Lưu các file ---
np.savetxt(os.path.join(output_dir, "ifm.txt"), ifm_quantized.flatten(), fmt='%d')
np.savetxt(os.path.join(output_dir, "ofm_golden.txt"), ofm_golden_quantized, fmt='%d')
with open(os.path.join(output_dir, "input_scale.txt"), "w") as f: f.write(str(input_scale))
with open(os.path.join(output_dir, "input_zp.txt"), "w") as f: f.write(str(input_zp))
with open(os.path.join(output_dir, "output_scale.txt"), "w") as f: f.write(str(output_scale))
with open(os.path.join(output_dir, "output_zp.txt"), "w") as f: f.write(str(output_zp))

print("\n--- CÁC THAM SỐ LƯỢNG TỬ HÓA TỪ TFLITE ---")
print(f"Input  Scale: {input_scale}")
print(f"Input  ZP   : {input_zp}")
print(f"Output Scale: {output_scale}")
print(f"Output ZP   : {output_zp}")

print("\n--- KẾT QUẢ ---")
print(f"IFM (Quantized) : {ifm_quantized.flatten()}")
print(f"OFM (Golden)    : {ofm_golden_quantized}")

print(f"\nĐã lưu tất cả dữ liệu vào thư mục: {output_dir}")
print("Bây giờ bạn có thể dùng các file này để kiểm tra lại file C của mình.")