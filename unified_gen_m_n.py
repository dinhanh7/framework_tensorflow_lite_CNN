import argparse
import math
import os
import numpy as np
from pathlib import Path
from typing import Tuple

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
    if not file_path.exists(): return None
    txt = file_path.read_text(encoding="utf-8").strip()
    return dtype(txt.split()[-1])

def read_array(file_path: Path, dtype=float) -> np.ndarray:
    """Đọc mảng 1D từ file."""
    if not file_path.exists(): return np.array([], dtype=dtype)
    return np.loadtxt(file_path, dtype=dtype)

def process_layer(layer_dir: Path, layer_type: str = None):
    print(f"[{layer_dir.name}] Đang phân tích...")
    
    # Tự động nhận diện loại Layer nếu không được truyền vào
    if layer_type is None:
        dir_name = layer_dir.name.upper()
        if "DEPTHWISE" in dir_name:
            layer_type = "DEPTHWISE_CONV_2D"
        elif "CONV_2D" in dir_name:
            layer_type = "CONV_2D"
        else:
            print(f"Cảnh báo: Không thể nhận diện loại layer từ tên thư mục. Mặc định dùng CONV_2D.")
            layer_type = "CONV_2D"
            
    # Đọc Scales
    ifm_scale = read_single_value(layer_dir / "ifm_scale.txt", float)
    ofm_scale = read_single_value(layer_dir / "ofm_scale.txt", float)
    weight_scales = read_array(layer_dir / "weight_scale.txt", float)

    if ifm_scale is None or ofm_scale is None or weight_scales.size == 0:
        print(f"Lỗi: Thiếu file scale trong {layer_dir}. Bỏ qua.")
        return

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
    print(f"  -> Đã tạo multiplier.txt, shift.txt (Channels: {channels_count})")

    # 2. TÍNH EFFECTIVE BIAS (NẾU CÓ DỮ LIỆU)
    ifm_zp = read_single_value(layer_dir / "ifm_zp.txt", float)
    bias_values = read_array(layer_dir / "bias_value.txt", np.int32)  # Output bias có thể là file bias_values.txt...
    if bias_values.size == 0: 
        # Thử tên file khác nếu không lấy được
        bias_values = read_array(layer_dir / "bias_value.txt", np.int32)
        
    weight_values = read_array(layer_dir / "weight_values.txt", np.int32)

    if ifm_zp is not None and bias_values.size > 0 and weight_values.size > 0:
        ifm_zp = int(ifm_zp)
        
        # Reshape weight để tính tổng theo cấu trúc bộ nhớ TFLite (TensorFlow layout)
        try:
            if layer_type == "DEPTHWISE_CONV_2D":
                # Depthwise: weight lưu dưới dạng [1, H, W, O] -> O chính là số groups/channels filter.
                # Khi làm phẳng, chênh lệch vị trí giữa các phần tử liền kề nhau thuộc về khác filter.
                # => Nhóm thành cột theo số lượng channel (O)
                weight_reshaped = weight_values.reshape(-1, channels_count)
                sum_w = np.sum(weight_reshaped, axis=0, dtype=np.int32) # Sum theo dọc => Output: [channels_count]
                
            else: # CONV_2D
                # Conv2D: weight lưu theo [O, H, W, I] -> O là output channels nằm ở ngoài cùng.
                # Khi làm phẳng, một cụm (H*W*I) phần tử liên tiếp sẽ thuộc về cùng 1 filter O.
                # => Nhóm thành mảng 2D với số dòng = số lượng channel output (O)
                weight_reshaped = weight_values.reshape(channels_count, -1)
                sum_w = np.sum(weight_reshaped, axis=1, dtype=np.int32) # Sum theo ngang => Output: [channels_count]

            effective_bias = bias_values - (ifm_zp * sum_w)
            # Ép kiểu an toàn (int32 output)
            effective_bias = np.round(effective_bias).astype(np.int32)
            
            np.savetxt(layer_dir / "effective_bias.txt", effective_bias, fmt='%d')
            print(f"  -> Đã tạo effective_bias.txt (Loại: {layer_type})")
            
        except ValueError as e:
            print(f"  -> Lỗi khi reshape weight để tính effective bias: {e}")
    else:
        print("  -> Bỏ qua effective_bias (Thiếu thông tin zp, bias, hoặc weight).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified script generate m, n, and effective_bias for TFLite Conv2D & Depthwise layers."
    )
    parser.add_argument(
        "--layer_dir",
        type=str,
        help="Đường dẫn tới duy nhất 1 thư mục layer (ví dụ: extracted_params_hsigmoid/layer001_CONV_2D)",
    )
    parser.add_argument(
        "--all_dir",
        type=str,
        help="Đường dẫn thư mục cha chứa tất cả các output layers cần xử lý. Script sẽ lặp qua và tự tính.",
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["CONV_2D", "DEPTHWISE_CONV_2D"],
        default=None,
        help="Bắt buộc xét theo loại này (Nên để None để script tự động tìm từ tên thư mục)",
    )
    
    args = parser.parse_args()

    if args.all_dir:
        base_path = Path(args.all_dir)
        if not base_path.exists():
            print(f"Thư mục {base_path} không tồn tại!")
        else:
            for subdir in base_path.iterdir():
                if subdir.is_dir() and ("CONV" in subdir.name.upper() or "DEPTHWISE" in subdir.name.upper()):
                    process_layer(subdir, layer_type=args.type)
    elif args.layer_dir:
        process_layer(Path(args.layer_dir), layer_type=args.type)
    else:
        print("Vui lòng cung cấp --layer_dir hoặc --all_dir. Dùng -h để xem cách dùng.")
