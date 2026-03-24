#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>




// ============================================================================
// 0. CÁC HÀM HELPER MỚI ĐỂ TÍNH TOÁN THAM SỐ LƯỢNG TỬ HÓA
// ============================================================================

// Hàm đọc một giá trị float từ file
float read_float_from_file(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    float val;
    if (fscanf(fp, "%f", &val) != 1) {
        fprintf(stderr, "Error: Failed to read float from %s\n", path);
        exit(1);
    }
    fclose(fp);
    return val;
}

// Hàm tính toán multiplier và shift từ một giá trị scale (float)
// Mô phỏng chính xác logic QuantizeMultiplier của TFLite
void GetQuantizedMultiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift) {
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }
    int s;
    double mantissa = frexp(double_multiplier, &s);
    int64_t q_fixed = (int64_t)round(mantissa * (1LL << 31));

    if (q_fixed == (1LL << 31)) {
        q_fixed /= 2;
        s++;
    }

    *quantized_multiplier = (int32_t)q_fixed;
    *shift = s;
}


// ============================================================================
// 1. ĐỊNH NGHĨA KÍCH THƯỚC TENSOR
// ============================================================================
#define TENSOR_HEIGHT 112
#define TENSOR_WIDTH 112
#define TENSOR_CHANNEL 16
#define TENSOR_SIZE (TENSOR_HEIGHT * TENSOR_WIDTH * TENSOR_CHANNEL)

// ============================================================================
// 2. CÁC HÀM HELPER MÔ PHỎNG LOGIC INT16 TỪ FILE PYTHON
// ============================================================================

// Hàm kẹp giá trị trong khoảng int16_t
int16_t clamp_int16(int32_t val) {
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return (int16_t)val;
}

// Hàm kẹp giá trị trong khoảng int8_t
int8_t clip_int8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// Hàm nhân bão hòa KHÔNG LÀM TRÒN (Truncating) cho int16
// Tương ứng: SaturatingDoublingHighMul_Int16 trong file Python (loại không có Rounding)
int16_t SaturatingDoublingHighMul_Int16_trunc(int16_t a, int16_t b) {
    // Xử lý trường hợp đặc biệt
    if (a == -32768 && b == -32768) {
        return 32767;
    }
    int32_t ab = (int32_t)a * (int32_t)b;
    // Dùng phép chia số nguyên để mô phỏng chính xác hành vi `int(a/b)` của Python
    // (cắt cụt về 0 - truncate towards zero). Phép dịch bit `>>` sẽ làm tròn về âm vô cùng.
    int32_t result = ab / 32768;
    return clamp_int16(result);
}

// Hàm nhân bão hòa CÓ LÀM TRÒN (Rounding) cho int16
// Tương ứng: SaturatingRoundingDoublingHighMul_Int16 trong file Python
int16_t SaturatingRoundingDoublingHighMul_Int16_round(int16_t a, int16_t b) {
    // Xử lý trường hợp đặc biệt
    if (a == -32768 && b == -32768) {
        return 32767;
    }
    int32_t ab = (int32_t)a * (int32_t)b;
    int32_t nudge = 16384; // 1 << 14
    int32_t result = (ab + nudge) >> 15;
    return clamp_int16(result);
}

// [MỚI] Hàm dịch trái có bão hòa để khớp với Python
int16_t SaturatingLeftShift(int16_t val, int shift) {
    int32_t res = (int32_t)val * (1 << shift);
    return clamp_int16(res);
}

// Hàm chia cho lũy thừa của 2 CÓ LÀM TRÒN
// Tương ứng: RoundingDivideByPOT trong TFLite
int16_t RoundingDivideByPOT(int16_t x, int exponent) {
    if (exponent == 0) return x;
    if (exponent < 0) { // Dịch trái nếu exponent âm
        return clamp_int16((int32_t)x << -exponent);
    }
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    int32_t result = (x >> exponent) + (remainder > threshold ? 1 : 0);
    return clamp_int16(result);
}



// [MỚI] Hàm RoundingDivideByPOT phiên bản 32-bit để xử lý multiplier
// Tránh lỗi cắt cụt khi truyền int32_t vào hàm cho int16_t
int32_t RoundingDivideByPOT_32(int32_t x, int exponent) {
    if (exponent == 0) return x;
    if (exponent < 0) {
        // Cần cẩn thận với tràn số khi dịch trái int32
        int64_t val = (int64_t)x << -exponent;
        if (val > 2147483647) val = 2147483647;
        if (val < -2147483648) val = -2147483648;
        return (int32_t)val;
    }
    int64_t mask = (1LL << exponent) - 1;
    int64_t remainder = (int64_t)x & mask;
    int64_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    int32_t result = (int32_t)((x >> exponent) + (remainder > threshold ? 1 : 0));
    return result;
}


// ============================================================================
// 3. CÁC HÀM ĐỌC/GHI FILE (Giữ nguyên)
// ============================================================================
void read_int8_array_from_file(const char* path, int8_t* arr, int size) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    int temp_val;
    for (int i = 0; i < size; ++i) {
        if (fscanf(fp, "%d", &temp_val) != 1) {
            fprintf(stderr, "Error: Failed to read element %d from %s\n", i, path);
            exit(1);
        }
        arr[i] = (int8_t)temp_val;
    }
    fclose(fp);
}

int8_t read_zp_from_file(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    int val;
    fscanf(fp, "%d", &val);
    fclose(fp);
    return (int8_t)val;
}

void write_ofm_to_file(const char* path, int8_t* ofm, int size) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    for (int i = 0; i < size; ++i) {
        fprintf(fp, "%d\n", ofm[i]);
    }
    fclose(fp);
}

// ============================================================================
// 4. HÀM MAIN - VIẾT LẠI HOÀN TOÀN THEO LOGIC INT16
// ============================================================================
int main() {
    printf("--- Hardswish Simulation (C version) ---\n");
    printf("--- LOGIC INT16 - Matching Python Reference ---\n");

    // --- Cấp phát bộ nhớ ---
    int8_t* input_data = (int8_t*)malloc(TENSOR_SIZE * sizeof(int8_t));
    int8_t* output_data = (int8_t*)malloc(TENSOR_SIZE * sizeof(int8_t));

    // --- Đường dẫn file (ĐÃ SỬA LẠI CHO ĐÚNG) ---
    const char* input_file = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_6_HARD_SWISH\\ifm6.txt";
    const char* output_file = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_6_HARD_SWISH\\ofm6_c_sim.txt";
    const char* params_path = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\extracted_params\\layer006_HARD_SWISH_1_stem_activation_1_truediv__1_stem_activation_1_Relu6_1_stem_activation_1_add_1_stem_activation_1_mul\\";

    char input_zp_file[256], ofm_zp_file[256], input_scale_file[256], ofm_scale_file[256];
    sprintf(input_zp_file, "%sparam_idx0_zp.txt", params_path);
    sprintf(ofm_zp_file, "%sofm_zp.txt", params_path);
    sprintf(input_scale_file, "%sparam_idx0_scale.txt", params_path);
    sprintf(ofm_scale_file, "%sofm_scale.txt", params_path);

    // --- Đọc dữ liệu đầu vào và các tham số scale/zp ---
    printf("Reading input file: %s\n", input_file);
    read_int8_array_from_file(input_file, input_data, TENSOR_SIZE);
    
    int8_t input_zp = read_zp_from_file(input_zp_file);
    int8_t output_zp = read_zp_from_file(ofm_zp_file);
    float input_scale_float = read_float_from_file(input_scale_file);
    float output_scale_float = read_float_from_file(ofm_scale_file);

    printf("Input ZP = %d, Input Scale = %f\n", input_zp, input_scale_float);
    printf("Output ZP = %d, Output Scale = %f\n", output_zp, output_scale_float);

    // --- TÍNH TOÁN MULTIPLIER GIỐNG HỆT PYTHON ---
    // 1. Output Multiplier (cho preshift)
    double real_output_mul = (input_scale_float / output_scale_float) / 128.0;
    int32_t m_out;
    int n_out;
    GetQuantizedMultiplier(real_output_mul, &m_out, &n_out);

    // 2. Reluish Multiplier (cho gating)
    double real_relu_mul = input_scale_float * 256.0 / 3.0;
    int32_t m_relu;
    int n_relu;
    GetQuantizedMultiplier(real_relu_mul, &m_relu, &n_relu);

    // Lấy 16 bit cao của multiplier để dùng trong phép nhân int16
    // [FIX 1] Thêm clamp_int16 để tránh tràn số và khớp 100% với Python
    int16_t m_out_16 = clamp_int16(m_out >> 16);
    int16_t m_relu_16 = clamp_int16(m_relu >> 16);

    printf("\nCalculated Params (Python Logic):\n");
    printf("Output Mul: m_out_16=%d, n_out=%d\n", m_out_16, n_out);
    printf("Reluish Mul: m_relu_16=%d, n_relu=%d\n", m_relu_16, n_relu);
    printf("\nPerforming quantized hardswish (Python logic)...\n");

    // --- VÒNG LẶP XỬ LÝ HARDSWISH GIỐNG HỆT PYTHON ---
    for (int i = 0; i < TENSOR_SIZE; ++i) {
        // 1. Input Offset
        const int16_t input_val_16 = clamp_int16((int32_t)input_data[i] - input_zp);

        // 2. Hires Input (Shift Left 7)
        const int16_t input_hires = SaturatingLeftShift(input_val_16, 7);

        // 3. Preshift Output (Rounding Mul)
        const int16_t preshift_out = SaturatingRoundingDoublingHighMul_Int16_round(input_hires, m_out_16);

        // 4. Relu-ish calculation
        int16_t reluish = input_hires;
        if (n_relu > 0) {
            reluish = SaturatingLeftShift(reluish, n_relu - 1);
        }
        reluish = SaturatingRoundingDoublingHighMul_Int16_round(reluish, m_relu_16);
        if (n_relu > 0) {
            reluish = SaturatingLeftShift(reluish, 1);
        } else if (n_relu < 0) {
            reluish = RoundingDivideByPOT(reluish, -n_relu);
        }

        // Convert to [0, 1] range in 16-bit
        reluish = (reluish + 32768) >> 1;

        // 5. Final Multiplication - [FIX 2] Đổi sang hàm CÓ LÀM TRÒN (_round) để khớp với Python
        // Ghi chú: File Python có lỗi định nghĩa lại hàm, khiến nó luôn dùng bản CÓ LÀM TRÒN.
        // Thay đổi này mô phỏng chính xác hành vi thực tế của file Python.
        const int16_t preshift_final = SaturatingRoundingDoublingHighMul_Int16_round(reluish, preshift_out);

        // 6. Output Rescale (Shift Right)
        int16_t val_final = preshift_final;
        if (n_out < 0) {
            val_final = RoundingDivideByPOT(val_final, -n_out);
        } else {
            val_final = SaturatingLeftShift(val_final, n_out);
        }

        // 7. Add Output ZP and Clip
        int32_t final_val_32 = val_final + output_zp;
        output_data[i] = clip_int8(final_val_32);
    }

    // --- Ghi kết quả và giải phóng bộ nhớ ---
    printf("Writing output to %s...\n", output_file);
    write_ofm_to_file(output_file, output_data, TENSOR_SIZE);
    printf("Hardswish simulation finished successfully!\n");

    free(input_data);
    free(output_data);

    return 0;
}