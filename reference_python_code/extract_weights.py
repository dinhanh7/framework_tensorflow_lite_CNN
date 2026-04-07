import tensorflow as tf
import numpy as np
import os

def save_quant_params(tensor_detail, output_dir, name_prefix):
    """Lưu scale và zero point của một tensor vào file"""
    quant_params = tensor_detail.get('quantization_parameters', {})
    scales = quant_params.get('scales', np.array([]))
    zero_points = quant_params.get('zero_points', np.array([]))

    if isinstance(scales, np.ndarray) and scales.size > 0:
        np.savetxt(os.path.join(output_dir, f"{name_prefix}_scale.txt"), scales, fmt='%s')
    elif isinstance(scales, float):
        with open(os.path.join(output_dir, f"{name_prefix}_scale.txt"), "w") as f:
            f.write(str(scales) + "\n")
            
    if isinstance(zero_points, np.ndarray) and zero_points.size > 0:
        np.savetxt(os.path.join(output_dir, f"{name_prefix}_zp.txt"), zero_points, fmt='%s')
    elif isinstance(zero_points, (int, float, np.integer)):
        with open(os.path.join(output_dir, f"{name_prefix}_zp.txt"), "w") as f:
            f.write(str(int(zero_points)) + "\n")

def save_tensor_data(interpreter, tensor_idx, output_dir, file_name):
    """Lưu giá trị của tensor weight/bias vào tên file tùy chỉnh"""
    if tensor_idx == -1: return
    try:
        data = interpreter.get_tensor(tensor_idx)
        if data is not None and data.size > 0:
            np.savetxt(os.path.join(output_dir, file_name), data.flatten(), fmt='%s')
    except ValueError:
        pass

def extract_tflite_params(model_path, output_dir="extracted_params"):
    # Xoá thư mục cũ nếu cần thiết hoặc chỉ overwrite
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Đang tải cấu hình TFLite từ {model_path}...")
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    ops_details = interpreter._get_ops_details()

    print(f"Bắt đầu trích xuất tham số vào thư mục '{output_dir}/'...")
    layer_counter = 1
    
    # Map index sang thông tin details của tensor
    tensor_map = {t['index']: t for t in tensor_details}

    # Xuất cho TẤT CẢ các layer thao tác
    for op in ops_details:
        op_name = op['op_name']
        
        # Tìm tên block để định danh cho layer dựa vào tensor output
        block_name = "unknown_block"
        if len(op['outputs']) > 0 and op['outputs'][0] != -1:
            raw_name = tensor_map[op['outputs'][0]]['name']
            clean_name = raw_name.replace('/', '_').replace(':', '_').replace(';', '_')
            clean_name = clean_name.replace('build_efficientnetv2_b0_full_bnless_', '')
            clean_name = clean_name.replace('build_efficientnetv2_b0_full_bnless', '') # Đề phòng trường hợp không có dấu _
            block_name = clean_name

        # Tạo thư mục con cho mỗi layer 
        folder_base_name = f"layer{layer_counter:03d}_{op_name}_{block_name}"
        
        # Rút ngắn tên thư mục tối đa, tránh lỗi Filename too long trong git và windows
        if len(folder_base_name) > 80:
            folder_base_name = folder_base_name[:80]
            
        layer_dir = os.path.join(output_dir, folder_base_name)
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
            
        inputs = op['inputs']
        outputs = op['outputs']
        
        # --- Định nghĩa lưu các tham số ---
        if op_name in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED']:
            # Lớp Conv truyền thống
            if len(inputs) > 0 and inputs[0] != -1:
                save_quant_params(tensor_map[inputs[0]], layer_dir, "ifm")
                
            if len(inputs) > 1 and inputs[1] != -1:
                save_quant_params(tensor_map[inputs[1]], layer_dir, "weight")
                save_tensor_data(interpreter, inputs[1], layer_dir, "weight_values.txt")
                
            if len(inputs) > 2 and inputs[2] != -1:
                save_quant_params(tensor_map[inputs[2]], layer_dir, "bias")
                save_tensor_data(interpreter, inputs[2], layer_dir, "bias_value.txt")
        else:
            # Các layer khác (ADD, MAX_POOL_2D, MEAN, QUANTIZE, v.v...)
            for i, inp_idx in enumerate(inputs):
                if inp_idx == -1: continue
                prefix = "ifm" if i == 0 else f"ifm_{i}"
                
                # Check nếu tensor này chứa dữ liệu hằng số/parameters
                is_param = False
                try:
                    data = interpreter.get_tensor(inp_idx)
                    if data is not None and data.size > 0:
                        is_param = True
                except ValueError:
                    pass
                
                if is_param:
                    param_name = "weight" if i == 1 else f"param_idx{i}"
                    save_quant_params(tensor_map[inp_idx], layer_dir, param_name)
                    save_tensor_data(interpreter, inp_idx, layer_dir, f"{param_name}_values.txt")
                else:
                    save_quant_params(tensor_map[inp_idx], layer_dir, prefix)

        # Xử lý Output (OFM) chung cho mọi layer
        for i, out_idx in enumerate(outputs):
            if out_idx != -1:
                prefix = "ofm" if i == 0 else f"ofm_{i}"
                save_quant_params(tensor_map[out_idx], layer_dir, prefix)
            
        print(f"[{layer_counter}] Đã xuất {op_name} ({block_name}) vào {layer_dir}")
        layer_counter += 1

    print(f"Hoàn tất! Các file (.txt) đã được lưu tại thư mục: '{output_dir}'.")

if __name__ == "__main__":
    MODEL_PATH = "tf_lite_dropout_imagenet100/efficientnetv2_b0_bnless_int8.tflite"
    extract_tflite_params(MODEL_PATH, output_dir="tf_lite_dropout_imagenet100/extracted_params")

