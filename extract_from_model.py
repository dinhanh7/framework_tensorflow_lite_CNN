import tensorflow as tf
import numpy as np
import os
import math
from pathlib import Path
from typing import Tuple

# ============================================================================
# PHẦN 1: UTILITY FUNCTIONS TÍNH M, N, EFFECTIVE_BIAS (từ unified_gen_m_n.py)
# ============================================================================

def tflite_round(x: float) -> int:
    """C++ std::round equivalent: half away from zero"""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))

def quantize_multiplier(double_multiplier: float) -> Tuple[int, int]:
    """
    Mirror TensorFlow Lite QuantizeMultiplier() behavior.
    Returns:
      quantized_multiplier (int32), shift (int)
    """
    if double_multiplier == 0.0:
        return 0, 0

    q, shift = math.frexp(double_multiplier)  # double_multiplier = q * 2^shift, q in [0.5, 1)
    q_fixed = tflite_round(q * (1 << 31))

    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1

    # Flush tiny multipliers to zero (same as TFLite safeguard)
    if shift < -31:
        return 0, 0

    # int32 range check
    if q_fixed > 0x7FFFFFFF:
        q_fixed = 0x7FFFFFFF

    return int(q_fixed), int(shift)

def read_single_value(file_path: Path, dtype=float):
    """Đọc 1 giá trị duy nhất từ file."""
    if not file_path.exists(): 
        return None
    txt = file_path.read_text(encoding="utf-8").strip()
    if txt:
        return dtype(txt.split()[-1])
    return None

def read_array(file_path: Path, dtype=float) -> np.ndarray:
    """Đọc mảng 1D từ file."""
    if not file_path.exists(): 
        return np.array([], dtype=dtype)
    return np.loadtxt(file_path, dtype=dtype)

def calculate_m_n_effective_bias(layer_dir: Path, layer_type: str = None):
    """
    Tính M, N (multiplier, shift) và effective_bias cho một layer.
    Kết quả được lưu trực tiếp vào layer_dir.
    """
    print(f"  [Tính toán M, N, effective_bias cho {layer_dir.name}]")
    
    # Tự động nhận diện loại Layer nếu không được truyền vào
    if layer_type is None:
        dir_name = layer_dir.name.upper()
        if "DEPTHWISE" in dir_name:
            layer_type = "DEPTHWISE_CONV_2D"
        elif "CONV_2D" in dir_name:
            layer_type = "CONV_2D"
        else:
            # Nếu không phải Conv-like layers, bỏ qua
            return False
    
    # Kiểm tra xem có phải Conv-like layer không
    if layer_type not in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
        return False
            
    # Đọc Scales
    ifm_scale = read_single_value(layer_dir / "ifm_scale.txt", float)
    ofm_scale = read_single_value(layer_dir / "ofm_scale.txt", float)
    weight_scales = read_array(layer_dir / "weight_scale.txt", float)

    if ifm_scale is None or ofm_scale is None or weight_scales.size == 0:
        print(f"    Lỗi: Thiếu file scale. Bỏ qua tính toán M, N.")
        return False

    channels_count = len(weight_scales)
    
    # 1. TÍNH M VÀ N (MULTIPLIER VÀ SHIFT)
    m_list, n_list = [], []
    for ws in weight_scales:
        real_multiplier = (ifm_scale * ws) / ofm_scale
        m, n = quantize_multiplier(real_multiplier)
        m_list.append(m)
        n_list.append(n)

    np.savetxt(layer_dir / "multiplier.txt", m_list, fmt='%d')
    np.savetxt(layer_dir / "shift.txt", n_list, fmt='%d')
    print(f"    -> Tạo multiplier.txt, shift.txt (Channels: {channels_count})")

    # 2. TÍNH EFFECTIVE BIAS (NẾU CÓ DỮ LIỆU)
    ifm_zp = read_single_value(layer_dir / "ifm_zp.txt", float)
    bias_values = read_array(layer_dir / "bias_value.txt", np.int32)
    weight_values = read_array(layer_dir / "weight_values.txt", np.int32)

    if ifm_zp is not None and bias_values.size > 0 and weight_values.size > 0:
        ifm_zp = int(ifm_zp)
        
        # Reshape weight để tính tổng theo cấu trúc bộ nhớ TFLite (TensorFlow layout)
        try:
            if layer_type == "DEPTHWISE_CONV_2D":
                # Depthwise: weight lưu dưới dạng [1, H, W, O]
                weight_reshaped = weight_values.reshape(-1, channels_count)
                sum_w = np.sum(weight_reshaped, axis=0, dtype=np.int32)
                
            else: # CONV_2D
                # Conv2D: weight lưu theo [O, H, W, I]
                weight_reshaped = weight_values.reshape(channels_count, -1)
                sum_w = np.sum(weight_reshaped, axis=1, dtype=np.int32)

            effective_bias = bias_values - (ifm_zp * sum_w)
            effective_bias = np.round(effective_bias).astype(np.int32)
            
            np.savetxt(layer_dir / "effective_bias.txt", effective_bias, fmt='%d')
            print(f"    -> Tạo effective_bias.txt (Loại: {layer_type})")
            
        except ValueError as e:
            print(f"    -> Lỗi khi reshape weight: {e}")
            return False
    else:
        print("    -> Bỏ qua effective_bias (Thiếu thông tin zp, bias, hoặc weight).")
    
    return True

# ============================================================================
# PHẦN 2: EXTRACT TỪ TFLITE (dựa trên extract_weights.py)
# ============================================================================

def save_quant_params(tensor_detail, output_dir, name_prefix):
    """Lưu scale và zero point của một tensor vào file"""
    quant_params = tensor_detail.get('quantization_parameters', {})
    scales = quant_params.get('scales', np.array([]))
    zero_points = quant_params.get('zero_points', np.array([]))

    if isinstance(scales, np.ndarray) and scales.size > 0:
        np.savetxt(os.path.join(output_dir, f"{name_prefix}_scale.txt"), scales, fmt='%.8f')
    elif isinstance(scales, (float, int, np.floating, np.integer)):
        with open(os.path.join(output_dir, f"{name_prefix}_scale.txt"), "w") as f:
            f.write(str(float(scales)) + "\n")
            
    if isinstance(zero_points, np.ndarray) and zero_points.size > 0:
        np.savetxt(os.path.join(output_dir, f"{name_prefix}_zp.txt"), zero_points, fmt='%d')
    elif isinstance(zero_points, (int, float, np.integer)):
        with open(os.path.join(output_dir, f"{name_prefix}_zp.txt"), "w") as f:
            f.write(str(int(zero_points)) + "\n")

def save_tensor_data(interpreter, tensor_idx, output_dir, file_name):
    """Lưu giá trị của tensor weight/bias vào tên file tùy chỉnh"""
    if tensor_idx == -1: 
        return
    try:
        data = interpreter.get_tensor(tensor_idx)
        if data is not None and data.size > 0:
            np.savetxt(os.path.join(output_dir, file_name), data.flatten(), fmt='%d')
    except (ValueError, IndexError) as e:
        pass

def extract_and_compute_tflite_params(model_path, output_dir="extracted_params", verbose=True):
    """
    Extract tất cả parameters từ TFLite model và tính M, N, effective_bias ngay lập tức.
    
    Args:
        model_path: Đường dẫn tới file .tflite
        output_dir: Thư mục output để lưu kết quả
        verbose: In chi tiết quá trình xử lý
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if verbose:
        print(f"[1] Đang tải model TFLite: {model_path}")
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    tensor_details = interpreter.get_tensor_details()
    ops_details = interpreter._get_ops_details()

    if verbose:
        print(f"[2] Bắt đầu trích xuất tham số vào: '{output_dir}/'")
    
    layer_counter = 1
    tensor_map = {t['index']: t for t in tensor_details}

    # Xuất cho TẤT CẢ các layer thao tác
    for op in ops_details:
        op_name = op['op_name']
        
        # Tìm tên block để định danh layer dựa vào tensor output
        block_name = "unknown_block"
        if len(op['outputs']) > 0 and op['outputs'][0] != -1:
            raw_name = tensor_map[op['outputs'][0]]['name']
            # Làm sạch tên
            clean_name = raw_name.replace('/', '_').replace(':', '_').replace(';', '_')
            clean_name = clean_name.replace('build_efficientnetv2_b0_full_bnless_', '')
            clean_name = clean_name.replace('build_efficientnetv2_b0_full_bnless', '')
            # Cắt ngắn tên nếu quá dài
            if len(clean_name) > 50:
                clean_name = clean_name[:50]
            block_name = clean_name

        # Tạo tên folder cho layer
        folder_base_name = f"layer{layer_counter:03d}_{op_name}_{block_name}"
        if len(folder_base_name) > 100:
            folder_base_name = folder_base_name[:100]
            
        layer_dir = os.path.join(output_dir, folder_base_name)
        if not os.path.exists(layer_dir):
            os.makedirs(layer_dir)
        
        layer_path = Path(layer_dir)
        inputs = op['inputs']
        outputs = op['outputs']
        
        if verbose:
            print(f"\n[Layer {layer_counter}] {op_name}")
        
        # --- Lưu các tham số theo loại layer ---
        if op_name in ['CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED']:
            # Layer Conv truyền thống
            if len(inputs) > 0 and inputs[0] != -1:
                save_quant_params(tensor_map[inputs[0]], layer_dir, "ifm")
                
            if len(inputs) > 1 and inputs[1] != -1:
                save_quant_params(tensor_map[inputs[1]], layer_dir, "weight")
                save_tensor_data(interpreter, inputs[1], layer_dir, "weight_values.txt")
                
            if len(inputs) > 2 and inputs[2] != -1:
                save_quant_params(tensor_map[inputs[2]], layer_dir, "bias")
                save_tensor_data(interpreter, inputs[2], layer_dir, "bias_value.txt")
        else:
            # Các layer khác (ADD, MAX_POOL_2D, MEAN, QUANTIZE, ...)
            for i, inp_idx in enumerate(inputs):
                if inp_idx == -1: 
                    continue
                prefix = "ifm" if i == 0 else f"ifm_{i}"
                
                # Check nếu tensor này chứa dữ liệu hằng số/parameters
                is_param = False
                try:
                    data = interpreter.get_tensor(inp_idx)
                    if data is not None and data.size > 0:
                        is_param = True
                except (ValueError, IndexError):
                    pass
                
                if is_param:
                    param_name = "weight" if i == 1 else f"param_idx{i}"
                    save_quant_params(tensor_map[inp_idx], layer_dir, param_name)
                    save_tensor_data(interpreter, inp_idx, layer_dir, f"{param_name}_values.txt")
                else:
                    save_quant_params(tensor_map[inp_idx], layer_dir, prefix)

        # Xử lý Output (OFM)
        for i, out_idx in enumerate(outputs):
            if out_idx != -1:
                prefix = "ofm" if i == 0 else f"ofm_{i}"
                save_quant_params(tensor_map[out_idx], layer_dir, prefix)
        
        # ===== BƯỚC 3: TÍNH M, N, EFFECTIVE_BIAS NGAY LẬP TỨC =====
        if verbose:
            print(f"  -> Trích xuất xong. Tiến hành tính M, N, effective_bias...")
        
        calculate_m_n_effective_bias(layer_path, layer_type=None)
        
        layer_counter += 1

    if verbose:
        print(f"\n✓ Hoàn tất! Các file đã được lưu tại: '{output_dir}'")
        print(f"  Tổng cộng {layer_counter - 1} layers được xử lý.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract & compute M, N, effective_bias từ TFLite model trong 1 lệnh."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cell_hswish_hsigmoid/efficientnetv2_b0_bnless_int8.tflite",
        help="Đường dẫn file .tflite"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="extracted_params_complete",
        help="Thư mục output"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="In chi tiết quá trình xử lý"
    )
    
    args = parser.parse_args()
    
    extract_and_compute_tflite_params(
        model_path=args.model,
        output_dir=args.output,
        verbose=args.verbose
    )
