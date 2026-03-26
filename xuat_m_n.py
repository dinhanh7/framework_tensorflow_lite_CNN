import numpy as np
import os
import math

# Hàm QuantizeMultiplier này được sao chép y hệt từ file flow_tflite_gen_tf_layer_01.py
# để đảm bảo kết quả tính toán hoàn toàn tương đồng.
# Mô phỏng chính xác hàm QuantizeMultiplier trong genMn.cpp
def QuantizeMultiplier(double_multiplier):
    if double_multiplier == 0.:
        return 0, 0
    
    # math.frexp trả về (mantissa, exponent) tương tự std::frexp
    q, shift = math.frexp(double_multiplier)
    
    # q_fixed = round(q * 2^31)
    q_fixed = int(round(q * (1 << 31)))
    
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
        
    if shift < -31:
        shift = 0
        q_fixed = 0
        
    return q_fixed, shift

# --- Bắt đầu các hàm đọc file giống flow_tflite_gen_tf_layer_01.py ---

# Hàm đọc file chứa 1 số thực (Scale của IFM, OFM)
def read_float_file(filename):
    with open(filename, "r") as file:
        content = file.read().strip()
        # Lấy phần tử cuối cùng nếu file có dạng "0.69..."
        val_str = content.split()[-1]
    return float(val_str)

# Hàm đọc file chứa nhiều số thực (Scale của Weight)
def read_float_array_file(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    # Lọc và lấy số thực từ mỗi dòng
    data = []
    for line in lines:
        val_str = line.strip().split()[-1]
        data.append(float(val_str))
    return np.array(data, dtype=np.float64)

# Hàm đọc ZP từ file
def read_zp_file(filename):
    try:
        with open(filename, "r") as file:
            content = file.read().strip()
            val_str = content.split()[-1]
        val = int(val_str)
        if val > 0x7F:
            val -= 0x100
        return val
    except Exception as e:
        print(f"Cảnh báo: Không đọc được file {filename}, dùng ZP=0. Lỗi: {e}")
        return 0

# Hàm đọc bias từ file
def read_bias_file(filename, length):
    with open(filename, "r") as file:
        lines = file.readlines()
    data = np.array([int(x.strip()) for x in lines], dtype=np.int64)
    for i in range(len(data)):
        if data[i] >= 0x80000000:
            data[i] -= 0x100000000
    data = data.astype(np.int32)
    return data.reshape((length,))

# Hàm đọc weight từ file
def read_hex_file_weight(filename, shape):
    with open(filename, "r") as file:
        lines = file.readlines()
    data = np.array([int(x.strip()) for x in lines], dtype=np.int16)
    for i in range(len(data)):
        if data[i] > 0x7F:
            data[i] -= 0x100
    H, W, C, F = shape
    reshaped_data = np.zeros((H, W, C, F), dtype=np.int16)
    index = 0
    for f in range(F):
        for h in range(H):
            for w in range(W):
                for c in range(C):
                    reshaped_data[h, w, c, f] = data[index]
                    index += 1
    return reshaped_data

# --- Kết thúc các hàm đọc file ---

def main():
    # --- Cấu hình ---
    # Đường dẫn đến thư mục chứa các tham số đã trích xuất của layer
    params_path = r'c:\Code c\Tensorflow\framework_tensorflow_lite_CNN\all_layer_io\layer_34_CONV_2D'
    
    # Đường dẫn đến thư mục để lưu các file output
    # Lưu trực tiếp vào thư mục params_path theo yêu cầu
    output_dir = params_path

    # Đường dẫn file output
    output_m_file_path = os.path.join(output_dir, 'multiplier.txt')
    output_n_file_path = os.path.join(output_dir, 'shift.txt')
    output_eff_bias_path = os.path.join(output_dir, 'effective_bias.txt')

    # --- Đọc dữ liệu ---
    try:
        # Đường dẫn file scale
        scale_ifm_path = os.path.join(params_path, 'ifm_scale.txt')
        scale_ofm_path = os.path.join(params_path, 'ofm_0_scale.txt') # Sửa tên file cho layer 34
        scale_w_path = os.path.join(params_path, 'weight_scale.txt')

        # Đường dẫn file cho effective bias
        bias_values_path = os.path.join(params_path, 'bias.txt') # Sửa tên file cho layer 34
        weight_values_path = os.path.join(params_path, 'weight.txt') # Sửa tên file cho layer 34
        ifm_zp_path = os.path.join(params_path, 'ifm_zp.txt')

        # Đọc scale
        scale_ifm = read_float_file(scale_ifm_path)
        
        print(f"Reading OFM scale from: {scale_ofm_path}")
        scale_ofm = read_float_file(scale_ofm_path)
        
        print(f"Reading Weight scale from: {scale_w_path}")
        scale_w_all = read_float_array_file(scale_w_path)
        num_filters = len(scale_w_all)

        # Đọc dữ liệu để tính effective bias
        print(f"Reading IFM ZP from: {ifm_zp_path}")
        zp_in = read_zp_file(ifm_zp_path)
        
        print(f"Reading Bias from: {bias_values_path}")
        bias_data = read_bias_file(bias_values_path, num_filters)
        
        # Đọc và reshape weight để tính sum_w
        print(f"Reading Weights from: {weight_values_path}")
        with open(weight_values_path, "r") as f:
            lines = f.readlines()
            num_weights = len([l for l in lines if l.strip()]) # Count non-empty lines
        
        # Layer 34 là conv 1x1
        kernel_size_squared = 1
        if (num_weights % (kernel_size_squared * num_filters)) != 0:
            print(f"Lỗi: Số lượng weight ({num_weights}) không chia hết cho ({kernel_size_squared} * {num_filters}).")
            return
        ifm_channel = num_weights // (kernel_size_squared * num_filters)
        weight_shape = (1, 1, ifm_channel, num_filters) # Sửa thành kernel 1x1
        weight_data = read_hex_file_weight(weight_values_path, weight_shape)

    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file cần thiết. {e}")
        print(f"Đảm bảo các file tồn tại trong thư mục '{params_path}'.")
        return

    # --- Tính toán m, n ---
    m_values = []
    n_values = []
    for scale_w in scale_w_all:
        effective_scale = (scale_ifm * scale_w) / scale_ofm
        multiplier_m, shift_n = QuantizeMultiplier(effective_scale)
        m_values.append(multiplier_m)
        n_values.append(shift_n)

    # --- Tính toán effective bias ---
    # sum_w có shape (num_filters,)
    sum_w = np.sum(weight_data, axis=(0, 1, 2))
    # correction = zp_in * sum_w
    correction = zp_in * sum_w
    # effective_bias = bias - correction
    effective_bias = bias_data - correction

    # --- Xuất kết quả ra file ---
    # Ghi m, n
    with open(output_m_file_path, 'w') as f_m:
        for m in m_values:
            f_m.write(f"{m}\n")
    with open(output_n_file_path, 'w') as f_n:
        for n in n_values:
            f_n.write(f"{n}\n")
    print(f"Đã ghi thành công các giá trị m vào file '{output_m_file_path}'")
    print(f"Đã ghi thành công các giá trị n vào file '{output_n_file_path}'")

    # Ghi effective_bias
    with open(output_eff_bias_path, 'w') as f_eff_bias:
        for bias in effective_bias:
            f_eff_bias.write(f"{int(bias)}\n") # Ghi dưới dạng số nguyên
    print(f"Đã ghi thành công effective bias vào file '{output_eff_bias_path}'")

if __name__ == '__main__':
    main()