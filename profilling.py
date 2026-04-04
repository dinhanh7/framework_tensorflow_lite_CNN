import tensorflow as tf
import numpy as np
import sys

# --- ĐỊNH NGHĨA FILE OUTPUT ---
output_log_file = 'model_profile_old.txt'

# --- ĐƯỜNG DẪN ĐẾN FILE MODEL CỦA BẠN ---
tflite_model_path = 'c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\cell_hswish_hsigmoid\\efficientnetv2_b0_bnless_int8.tflite'

# Lưu lại stdout gốc để có thể khôi phục sau
original_stdout = sys.stdout 

try:
    # Mở file log và chuyển hướng toàn bộ output của print vào file này
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

    # In thông tin đầu vào và đầu ra tổng thể của model
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
        
        quant_params = detail['quantization_parameters']
        scales = quant_params['scales']
        zero_points = quant_params['zero_points']
        
        # Xử lý trường hợp tensor không được lượng tử hóa (ví dụ: float input)
        if len(scales) == 0:
            quant_info = "Quant(Not Quantized - Float Tensor)"
        # Xử lý trường hợp lượng tử hóa trên mỗi kênh (per-channel)
        elif len(scales) > 1:
            quant_info = f"Quant(per-channel, axis={quant_params['quantized_dimension']})"
        # Xử lý trường hợp lượng tử hóa trên mỗi tensor (per-tensor)
        else:
            quant_info = f"Quant(scale={scales[0]:.8f}, zp={zero_points[0]})"
            
        print(f"    - Tensor[{tensor_index}] '{name}': Shape={list(detail['shape'])}, DType={np.dtype(detail['dtype']).name}, {quant_info}")

    # Lặp qua từng operation và in thông tin chi tiết
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
    # Nếu có lỗi, in lỗi ra màn hình console gốc
    sys.stdout = original_stdout
    print(f"Đã xảy ra lỗi trong quá trình phân tích: {e}")
    print("Vui lòng kiểm tra lại đường dẫn file TFLite và đảm bảo TensorFlow đã được cài đặt đúng cách.")

finally:
    # Đảm bảo stdout luôn được khôi phục và file được đóng, dù có lỗi hay không
    if 'log_file' in locals() and not log_file.closed:
        print("\n--- KẾT THÚC PHÂN TÍCH ---")
        log_file.close()
    sys.stdout = original_stdout
    print(f"Phân tích hoàn tất. Kết quả đã được lưu vào file '{output_log_file}'.")