import numpy as np
import tensorflow as tf
import argparse
import math

# BIAS_FRAC_BIT = 7
# --- BẮT ĐẦU ĐOẠN BỔ SUNG CÁC HÀM XỬ LÝ SCALE ---

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
def QuantizeMultiplier_hs(double_multiplier):
    if double_multiplier == 0.:
        return 0, 0
    
    q, shift = math.frexp(double_multiplier)
    
    # SỬA Ở ĐÂY: Dùng floor(x + 0.5) thay vì round()
    q_fixed = int(math.floor(q * (1 << 31) + 0.5))
    
    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1
        
    if shift < -31:
        shift = 0
        q_fixed = 0
        
    return q_fixed, shift
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



# Tương ứng với phần _mm256_mul_epi32 và _mm256_add_epi64 trong C++
def SaturatingRoundingDoublingHighMul(a, b):
    # Check tràn số đặc biệt: -2^31 * -2^31 (rất hiếm gặp nhưng logic C++ có handle)
    if a == -2147483648 and b == -2147483648:
        return 2147483647
    
    # Thực hiện nhân 64-bit
    a_64 = int(a)
    b_64 = int(b)
    ab_64 = a_64 * b_64
    
    # Cộng offset (1 << 30) - Tương ứng 'offset_vector' trong C++
    nudge = 1 << 30
    
    # Dịch phải 31 bit
    # Python >> là arithmetic shift (giữ dấu), giống hệt logic xử lý int32 của C++
    result = (ab_64 + nudge) >> 31
    
    return result

# [MỚI] Mô phỏng hàm rounding_right_shift trong C++
# Xử lý chính xác logic "nudge" và Overflow Mask
def RoundingRightShift(x, shift):
    if shift <= 0:
        return x
        
    # Tính giá trị làm tròn (nudge)
    nudge = 1 << (shift - 1)
    
    # --- LOGIC QUAN TRỌNG NHẤT GIỐNG C++ ---
    # C++ Code:
    # mask_num_plus_nudge_overflow = _mm256_cmpgt_epi32(results, INT_MAX - nudge)
    # result = overflow ? (1 << (31 - shift)) : ((results + nudge) >> shift)
    
    int32_max = 2147483647
    
    # Kiểm tra overflow nếy cộng nudge
    if x > int32_max - nudge:
        # Nếu tràn, trả về giá trị đặc biệt như logic C++
        return 1 << (31 - shift)
    else:
        # Nếu không tràn, thực hiện cộng và dịch bình thường
        return (x + nudge) >> shift

# Hàm chính kết hợp 2 bước trên
def MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift):
    # Logic tính toán shift giống TFLite:
    # shift từ QuantizeMultiplier thường là số âm (ví dụ -7).
    # Total right shift = 31 - shift_exponent.
    # Tuy nhiên, hàm này nhận đầu vào 'shift' chính là exponent từ math.frexp
    
    # 1. Tính toán lượng dịch phải thực tế
    # shift ở đây là exponent (ví dụ -6).
    # Trong công thức: Real_Mul = Quantized_Mul * 2^(shift - 31)
    # => Ta cần HighMul (nhân 2^-31) và sau đó dịch phải thêm (-shift) nữa nếu shift < 0
    
    left_shift = shift if shift > 0 else 0
    right_shift = -shift if shift < 0 else 0
    
    # Bước 1: Left Shift (thường là 0 với Output Layer)
    x_shifted = x * (1 << left_shift)
    
    # Bước 2: Fixed Point Multiply (High Mul)
    # Tương ứng đoạn _mm256_mul_epi32... đến khi ra results trong C++
    high_mul = SaturatingRoundingDoublingHighMul(x_shifted, quantized_multiplier)
    
    # Bước 3: Rounding Right Shift
    # Tương ứng đoạn lambda rounding_right_shift trong C++
    result = RoundingRightShift(high_mul, right_shift)
    
    return result


# # Mô phỏng chính xác hàm MultiplyByQuantizedMultiplier trong genMn.cpp
# def MultiplyByQuantizedMultiplier(x, quantized_multiplier, shift):
#     # x: int32, quantized_multiplier: int32, shift: int
    
#     total_shift = 31 - shift
    
#     # Tính round: 1 << (total_shift - 1)
#     round_val = 1 << (total_shift - 1)
    
#     # Thực hiện nhân 64-bit: result = x * multiplier + round
#     result = x * quantized_multiplier + round_val
    
#     # Dịch phải
#     result = result >> total_shift
    
#     # Trong Python số nguyên tự mở rộng, nhưng ta cần mô phỏng int32 nếu cần
#     # Tuy nhiên kết quả phép này thường được dùng để cộng ZP rồi clip
#     return result

# --- KẾT THÚC ĐOẠN BỔ SUNG ---

# --- [BỔ SUNG 1] Hàm đọc giá trị ZP từ file ---
def read_zp_file(filename):
    try:
        with open(filename, "r") as file:
            # Đọc toàn bộ nội dung, ví dụ: "F2"
            content = file.read().strip()
            # Lấy chuỗi cuối cùng (F2) để tránh các ký tự thừa
            val_str = content.split()[-1]
            
        val = int(val_str)
        
        # Xử lý số âm 8-bit (nếu > 127 thì trừ 256)
        # Ví dụ: F2 (242) -> -14
        if val > 0x7F:
            val -= 0x100
        return val
    except Exception as e:
        print(f"Cảnh báo: Không đọc được file {filename}, dùng ZP=0. Lỗi: {e}")
        return 0
    
# Hàm đọc dữ liệu từ file HEX với thứ tự hàng → cột → channel → filter
def read_bias_file(filename, length):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # 1. Đọc dữ liệu Hex (Python tự hiểu 0xFF...F là số dương lớn)
    # Chúng ta dùng int(x, 16) sẽ ra số dương unsigned nếu x >= 0x80000000
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int64)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int64)
    # 2. Xử lý số âm (Signed 32-bit conversion)
    # Nếu giá trị >= 2^31 (0x80000000), tức là số âm trong hệ bù 2 32-bit
    for i in range(len(data)):
        if data[i] >= 0x80000000: 
            data[i] -= 0x100000000  # Trừ đi 2^32 để về số âm

    # 3. Ép kiểu về đúng int32 để đưa vào Model
    data = data.astype(np.int32)

    # LƯU Ý QUAN TRỌNG: 
    # Đã XÓA đoạn: data[j] = data[j] * (1 << BIAS_FRAC_BIT)
    # Bias là số nguyên cộng trực tiếp vào accumulator, không cần shift.

    return data.reshape((length,))
# def read_bias_file(filename, length):
#     with open(filename, "r") as file:
#         hex_values = file.readlines()
    
#     # Chuyển đổi từ HEX thành số nguyên 8-bit có dấu
#     data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int16)

#     # Đảm bảo dữ liệu trong phạm vi số nguyên có dấu 8-bit
#     for i in range(len(data)):
#         if data[i] > 0x7F:  # Nếu giá trị > 127, chuyển thành số âm
#             data[i] -= 0x100  # 0x100 là 256, nên ta trừ đi để có giá trị âm
            
#     for j in range(len(data)):
#         data[j] = data[j] * (1 << BIAS_FRAC_BIT)
        
#     return data.reshape((length,))
def read_hex_file_weight(filename, shape):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # Chuyển đổi từ HEX thành số nguyên 8-bit có dấu
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int16)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int16)
    # Đảm bảo dữ liệu trong phạm vi số nguyên có dấu 8-bit
    for i in range(len(data)):
        if data[i] > 0x7F:  # Nếu giá trị > 127, chuyển thành số âm
            data[i] -= 0x100  # 0x100 là 256, nên ta trừ đi để có giá trị âm

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

def read_hex_file(filename, shape):
    with open(filename, "r") as file:
        # hex_values = file.readlines()
        lines = file.readlines()    
    # Chuyển đổi từ HEX thành số nguyên 8-bit có dấu
    # data = np.array([int(x.strip(), 16) for x in hex_values], dtype=np.int32)
    data = np.array([int(x.strip()) for x in lines], dtype=np.int32)
    # Đảm bảo dữ liệu trong phạm vi số nguyên có dấu 8-bit
    # Nếu giá trị lớn hơn 127, chúng ta sẽ chuyển thành số âm
    for i in range(len(data)):
        if data[i] > 0x7F:  # Nếu giá trị > 127, chuyển thành số âm
            data[i] -= 0x100  # 0x100 là 256, nên ta trừ đi để có giá trị âm
    H, W, C = shape
    reshaped_data = np.zeros((H, W, C), dtype=np.int32)
    index = 0
    for h in range(H):
        for w in range(W):
            for c in range(C):
                reshaped_data[h, w, c] = data[index]

                index += 1

    return reshaped_data

# Hàm ghi dữ liệu ra file HEX
# Sửa trong gen_tf_layer_01.py
# Sửa trong gen_tf_layer_01.py
def write_hex_file(filename, data):
    H, W, C = data.shape
    with open(filename, "w") as file:
        for h in range(H):          # Loop Channel trước (để khớp với genhex.py)
            for w in range(W):
                for c in range(C):
                    int_value = int(round(data[h, w, c]))
                    
                    # SỬA LẠI: Mask 32-bit và format 8 ký tự
                    hex_value = int_value & 0xFFFFFFFF 
                    file.write(f"{int_value}\n")

# ======================================================
# 2. XỬ LÝ HARD SWISH (Mô phỏng chính xác TFLite Reference - hard_swish.h)
# ======================================================
print("\n--- Bắt đầu xử lý Hard Swish (Reference Logic) ---")
    
    # --- Định nghĩa các hàm helper mô phỏng int16 hardware ---
    
def clamp_int16(val):
        return max(-32768, min(32767, int(val)))

    # Hàm nhân KHÔNG làm tròn (Truncating) - Quan trọng nhất để sửa lỗi lệch 1 đơn vị
    # Tương ứng: SaturatingDoublingHighMul trong hard_swish.h
def SaturatingDoublingHighMul_Int16(a, b):
        if a == -32768 and b == -32768:
            return 32767
        ab = a * b
        # C++ integer division truncates towards 0 (ví dụ -3.5 -> -3)
        # Python // truncates towards -infinity (ví dụ -3.5 -> -4)
        # Dùng int() trên float để mô phỏng C++ truncate
        return clamp_int16(int(ab / 32768.0))

    # Hàm nhân CÓ làm tròn (Rounding) 
    # Tương ứng: gemmlowp::SaturatingRoundingDoublingHighMul
def SaturatingRoundingDoublingHighMul_Int16(a, b):
        if a == -32768 and b == -32768:
            return 32767
        ab = int(a) * int(b)
        nudge = 16384 # 1 << 14 (cho int16 logic tương đương 1<<30 của int32)
        # Logic: (ab + 16384) / 32768
        return clamp_int16((ab + nudge) >> 15)

def SaturatingLeftShift(val, shift):
        res = val * (1 << shift)
        return clamp_int16(res)

def RoundingDivideByPOT(x, exponent):
        # Chia x cho 2^exponent có làm tròn
        if exponent == 0: return x
        mask = (1 << exponent) - 1
        remainder = x & mask
        threshold = (mask >> 1) + (1 if x < 0 else 0)
        return (x >> exponent) + (1 if remainder > threshold else 0)
    
    # --- A. Khởi tạo tham số Hard Swish ---
    # Đường dẫn đã được cập nhật theo yêu cầu
input_file = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_6_HARD_SWISH\\ifm6.txt"
params_dir = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\extracted_params\\layer006_HARD_SWISH_1_stem_activation_1_truediv__1_stem_activation_1_Relu6_1_stem_activation_1_add_1_stem_activation_1_mul"
hs_output_file = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_6_HARD_SWISH\\ofm6_py_sim.txt"

    # Shape của input feature map (cần được xác nhận)
    # Giả sử shape là (28, 28, 32) dựa trên các layer trước
IFM_SHAPE = (112, 112, 16) 

    # Đọc dữ liệu input trực tiếp từ file
hs_input_data = read_hex_file(input_file, IFM_SHAPE)

    # Đọc các tham số quantization từ thư mục params
    # Input scale/zp
sx = read_float_file(params_dir + "\\param_idx0_scale.txt")
zx = read_zp_file(params_dir + "\\param_idx0_zp.txt")
    # Output scale/zp
sy = read_float_file(params_dir + "\\ofm_scale.txt")
zy = read_zp_file(params_dir + "\\ofm_zp.txt")

print(f"Hard Swish Params: Sx={sx}, Zx={zx} -> Sy={sy}, Zy={zy}")
def Multippliers_for_RefLogic(sx, sy):
        # Tính toán Multipliers cho Reference Logic
        # Reference Logic không dùng scale_0, scale_1 mà dùng output_multiplier và reluish_multiplier
        
        # 1. Output Multiplier (Input -> Output scale)
        # TFLite Ref: input_value_on_preshift = input_hires * output_mul
        # input_hires = input * 128. => Ta cần: (sx / sy) / 128.0
        real_output_mul = (sx / sy) / 128.0
        m_out, n_out = QuantizeMultiplier_hs(real_output_mul)
        
        # 2. Reluish Multiplier (Input -> Gate [0,1] scale)
        # Ta cần map giá trị thực 3.0 về int16 là 32768.
        # input_hires = input * 128.
        # Ta cần: input_hires * M_relu ~= input_real / 3.0 (normalized)
        # => (input * 128) * M_relu = (input * sx) * (32768 / 3.0)
        # => M_relu = sx * (32768 / 3.0) / 128 = sx * 256.0 / 3.0
        real_relu_mul = sx * 256.0 / 3.0
        m_relu, n_relu = QuantizeMultiplier_hs(real_relu_mul)
        
        return m_out, n_out, m_relu, n_relu

m_out, n_out, m_relu, n_relu = Multippliers_for_RefLogic(sx, sy)

    

def simulate_hard_swish(hs_input_data, m_out, m_relu, zx,zy, n_out, n_relu):
        # --- C. Vòng lặp xử lý (Mô phỏng hard_swish.h) ---
        flat_input = hs_input_data.flatten()
        flat_output = []
        
        # Chuyển đổi multipliers về dạng dùng cho hàm int16 (SaturatingRoundingDoublingHighMul)
        # QuantizeMultiplier trả về m (32-bit). Ta cần lấy high part cho phép nhân 16-bit nếu cần.
        # Nhưng hàm SaturatingRoundingDoublingHighMul_Int16 nhận vào int16.
        # Tuy nhiên, trong TFLite Reference, params.output_multiplier_fixedpoint_int16 là int16.
        # Ta cần convert m_out (đang là 31-bit) về int16.
        # Thông thường m_out nằm trong [0.5, 1.0), tức 1<<30 đến 1<<31.
        # Lấy 16 bit cao: m >> 15 (giữ sign bit? No, m is usually positive normalized)
        # Đúng chuẩn: FixedPointInt16 = FixedPointInt32_Rounding >> 16 ??
        # Let's use simple logic: m_out_16 = int(m_out >> 15) if m_out is 31-bit signed logic
        # QuantizeMultiplier_hs trả về q_fixed ở dạng 31-bit signed (max 2^31-1).
        # Để khớp int16, ta lấy 16 bit cao:
        
        m_out_16 = clamp_int16(m_out >> 16)
        m_relu_16 = clamp_int16(m_relu >> 16)
        
        # Lưu ý: n_out và n_relu từ QuantizeMultiplier là shift phải (số âm).
        # hard_swish.h mong đợi shift exponent cho RoundingDivideByPOT hoặc SaturatingLeftShift.
        
        for X in flat_input:
            # 1. Input Offset
            # const int16_t input_value = input_data[i] - params.input_zero_point;
            input_val_16 = clamp_int16(int(X) - zx)
            
            # 2. Hires Input (Shift Left 7)
            # input_value_on_hires_input_scale = input_value * (1 << 7);
            input_hires = clamp_int16(input_val_16 * 128)
            
            # 3. Preshift Output (Rounding Mul)
            # gemmlowp::SaturatingRoundingDoublingHighMul(..., params.output_multiplier_fixedpoint_int16)
            # Lưu ý: C++ code tự động handle shift nếu exponent khác 0? 
            # Trong hard_swish.h chỉ có 1 tham số fixedpoint. Shift được xử lý ở bước cuối cùng.
            # Ở đây ta chỉ nhân với M (Multiplier).
            preshift_out = SaturatingRoundingDoublingHighMul_Int16(input_hires, m_out_16)
            
            # 4. Relu-ish calculation
            reluish = input_hires
            # Shift exponent handling for reluish
            # Trong file .h có đoạn: if (params.reluish_multiplier_exponent > 0) SaturatingLeftShift...
            # n_relu từ hàm Quantize thường <= 0 (right shift).
            # Nếu n_relu > 0 (hiếm): Shift Left (reluish_exponent - 1)
            if n_relu > 0:
                reluish = SaturatingLeftShift(reluish, n_relu - 1)
                
            # Nhân Multiplier
            reluish = SaturatingRoundingDoublingHighMul_Int16(reluish, m_relu_16)
            
            # Shift tiếp
            if n_relu > 0:
                reluish = SaturatingLeftShift(reluish, 1)
            elif n_relu < 0:
                # Shift Right (Rounding)
                reluish = RoundingDivideByPOT(reluish, -n_relu)
                
            # Convert to [0, 1] range in 16-bit
            # reluish_value = (reluish_value + (1 << 15)) >> 1;
            reluish = (reluish + 32768) >> 1
            
            # 5. Final Multiplication (NON-ROUNDING / Truncating) - KEY FIX
            # const int16_t preshift_output_value = SaturatingDoublingHighMul(...)
            preshift_final = SaturatingDoublingHighMul_Int16(reluish, preshift_out)
            
            # 6. Output Rescale (Shift Right)
            # int16_t output_value = gemmlowp::RoundingDivideByPOT(preshift_output_value, -params.output_multiplier_exponent);
            # n_out là exponent. Thường n_out < 0 -> shift right -n_out
            val_final = preshift_final
            if n_out < 0:
                val_final = RoundingDivideByPOT(val_final, -n_out)
            else:
                # Hiếm khi xảy ra
                val_final = SaturatingLeftShift(val_final, n_out)
                
            # 7. Add Output ZP
            val_final += zy
            
            # Clip to 8-bit range
            flat_output.append(max(-128, min(127, val_final)))
        return flat_output

    
flat_output = simulate_hard_swish(hs_input_data, m_out, m_relu, zx, zy, n_out, n_relu)

    # --- Reshape và Xuất file ---
hs_output_arr = np.array(flat_output, dtype=np.int32).reshape(hs_input_data.shape)
write_hex_file(hs_output_file, hs_output_arr)
print(f"Kết quả Hard Swish (Ref) đã được ghi vào {hs_output_file}")