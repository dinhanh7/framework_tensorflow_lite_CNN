import tensorflow as tf
import numpy as np
import sys # Thêm thư viện sys để điều khiển stdout

# --- ĐỊNH NGHĨA FILE OUTPUT ---
output_log_file = 'model_profile.txt'

# --- ĐƯỜNG DẪN ĐẾN FILE MODEL CỦA BẠN ---
tflite_model_path = 'c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\efficientnetv2_b0_bnless_int8.tflite'

# Lưu lại stdout gốc
original_stdout = sys.stdout 

try:
    # Mở file log và chuyển hướng stdout
    print(f"Bắt đầu phân tích model, kết quả sẽ được ghi vào file: {output_log_file}")
    log_file = open(output_log_file, 'w', encoding='utf-8')
    sys.stdout = log_file

    # Tải model TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    
    # Lấy thông tin chi tiết về tất cả các tensor trong model
    tensor_details = interpreter.get_tensor_details()
    
    # Lấy thông tin chi tiết về các operation (lớp)
    ops_details = interpreter._get_ops_details()

    print(f"--- Phân tích Model: {tflite_model_path} ---\n")

    # In thông tin đầu vào và đầu ra tổng thể
    print("--- Model Inputs ---")
    for detail in interpreter.get_input_details():
        print(f"  Name: {detail['name']}, Shape: {detail['shape']}, DType: {detail['dtype']}, Quantization: {detail['quantization']}")
    
    print("\n--- Model Outputs ---")
    for detail in interpreter.get_output_details():
        print(f"  Name: {detail['name']}, Shape: {detail['shape']}, DType: {detail['dtype']}, Quantization: {detail['quantization']}")
    
    print("\n" + "="*50)
    print("--- CHI TIẾT KIẾN TRÚC TỪNG LỚP ---")
    print("="*50)

    # Hàm helper để in thông tin tensor cho dễ đọc
    def print_tensor_info(tensor_index):
        detail = tensor_details[tensor_index]
        # Tensor hằng số (trọng số, bias) thường không có tên
        name = detail['name'] if detail['name'] else f"CONSTANT (Weights/Bias)"
        quant_info = f"Quant(scale={detail['quantization'][0]:.6f}, zp={detail['quantization'][1]})"
        print(f"    - Tensor[{tensor_index}] '{name}': Shape={detail['shape']}, DType={detail['dtype']}, {quant_info}")

    # Lặp qua từng operation và in thông tin
    for i, op in enumerate(ops_details):
        op_name = op['op_name']
        print(f"\n[LỚP {i}]: {op_name}")
        
        print("  Inputs:")
        for tensor_idx in op['inputs']:
            print_tensor_info(tensor_idx)

        print("  Outputs:")
        for tensor_idx in op['outputs']:
            print_tensor_info(tensor_idx)
        print("-" * 40)

except Exception as e:
    # Vẫn in lỗi ra màn hình console gốc nếu có lỗi
    sys.stdout = original_stdout
    print(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")
    print("Vui lòng kiểm tra lại đường dẫn file TFLite và đảm bảo TensorFlow đã được cài đặt.")

finally:
    # Đảm bảo stdout luôn được khôi phục và file được đóng, dù có lỗi hay không
    if 'log_file' in locals() and not log_file.closed:
        print("--- KẾT THÚC PHÂN TÍCH ---")
        log_file.close()
    sys.stdout = original_stdout
    print(f"Phân tích hoàn tất. Kết quả đã được lưu vào file '{output_log_file}'.")