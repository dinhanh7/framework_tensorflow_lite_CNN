#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define IFM_SIZE 1000
#define BETA 1.0 

static const int32_t kInt32Max = 2147483647;
static const int32_t kInt32Min = -2147483648;

// Các hằng số định dạng Fixed-point từ TFLite softmax.h
static const int kScaledDiffIntegerBits = 5;
static const int kAccumulationIntegerBits = 12;

// =================================================================================================
// CÁC HÀM TOÁN HỌC FIXED-POINT (MÔ PHỎNG GEMMLOWP)
// =================================================================================================

// Nhân 2 số 32-bit, lấy 32-bit cao, làm tròn và nhân đôi (SaturatingRoundingDoublingHighMul)
int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
    int64_t a_64 = (int64_t)a;
    int64_t b_64 = (int64_t)b;
    int64_t ab_64 = a_64 * b_64;
    int32_t nudge = (ab_64 >= 0) ? (1 << 30) : (1 - (1 << 30));
    int32_t result = (int32_t)((ab_64 + nudge) >> 31);
    if (ab_64 > 0 && result < 0) result = kInt32Max;
    if (ab_64 < 0 && result > 0) result = kInt32Min;
    return result;
}

// Dịch bit phải có làm tròn (RoundingDivideByPOT)
int32_t RoundingDivideByPOT(int32_t x, int exponent) {
    if (exponent == 0) return x;
    if (exponent >= 31) return x >= 0 ? 0 : -1;
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (1 << (exponent - 1));
    if (x >= 0) {
        return (x >> exponent) + (remainder > threshold ? 1 : 0);
    } else {
        return (x >> exponent) + (remainder >= threshold ? 1 : 0);
    }
}

// Nhân với Multiplier > 1
int32_t MultiplyByQuantizedMultiplierGreaterThanOne(int32_t x, int32_t quantized_multiplier, int left_shift) {
    int32_t x_shifted = x * (1 << left_shift);
    return SaturatingRoundingDoublingHighMul(x_shifted, quantized_multiplier);
}

// Mô phỏng exp(x) cho số âm, trả về Q0.31
int32_t exp_on_negative_values(int32_t a_q5_26) {
    const double kInputScale = 1.0 / (double)(1 << 26);
    double real_a = (double)a_q5_26 * kInputScale;
    double real_exp = exp(real_a);
    return (int32_t)round(real_exp * 2147483648.0); // Chuyển về Q0.31
}

// Tính nghịch đảo 1/x, trả về Q0.31 và số bit dịch (num_bits_over_unit)
int32_t GetReciprocal(int32_t x, int x_integer_bits, int* num_bits_over_unit) {
    if (x <= 0) return kInt32Max;
    
    int headroom = 0;
    uint32_t tmp = (uint32_t)x;
    while ((tmp & 0x40000000) == 0) {
        tmp <<= 1;
        headroom++;
    }
    
    // Q-format logic của TFLite
    *num_bits_over_unit = x_integer_bits - (31 - headroom);
    
    int64_t one_q62 = (int64_t)1 << 62;
    int32_t reciprocal = (int32_t)(one_q62 / (int64_t)(x << headroom));
    return reciprocal;
}

// =================================================================================================
// ĐỌC/GHI FILE
// =================================================================================================

float read_float_from_file(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Error: %s\n", path); exit(1); }
    float val; fscanf(fp, "%f", &val); fclose(fp);
    return val;
}

int8_t read_zp_from_file(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Error: %s\n", path); exit(1); }
    int val; fscanf(fp, "%d", &val); fclose(fp);
    return (int8_t)val;
}

void read_int8_array_from_file(const char* path, int8_t* arr, int size) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "Error: %s\n", path); exit(1); }
    int temp;
    for (int i = 0; i < size; ++i) {
        if (fscanf(fp, "%d", &temp) == 1) arr[i] = (int8_t)temp;
    }
    fclose(fp);
}

void write_ofm_to_file(const char* path, int8_t* ofm, int size) {
    FILE* fp = fopen(path, "w");
    if (!fp) { exit(1); }
    for (int i = 0; i < size; ++i) fprintf(fp, "%d\n", ofm[i]);
    fclose(fp);
}

// =================================================================================================
// HÀM SOFTMAX CHÍNH - BIT EXACT VỚI TFLite
// =================================================================================================

void QuantizedSoftmax(const int8_t* input_data, int input_size,
                      int32_t input_multiplier, int input_left_shift,
                      int32_t diff_min, int8_t ofm_zp, int8_t* output_data) {

    // 1. Tìm Max
    int8_t max_in_row = -128;
    for (int i = 0; i < input_size; ++i) {
        if (input_data[i] > max_in_row) max_in_row = input_data[i];
    }

    // 2. Tính Tổng Exp (Dùng Q12.19 để tích lũy)
    int32_t sum_of_exps_q12_19 = 0;
    for (int i = 0; i < input_size; ++i) {
        int32_t input_diff = (int32_t)input_data[i] - max_in_row;
        if (input_diff >= diff_min) {
            int32_t scaled_diff = MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
            
            int32_t exp_val_q0_31 = exp_on_negative_values(scaled_diff);
            
            // Rescale Q0.31 -> Q12.19 (Dịch phải 12 bit)
            sum_of_exps_q12_19 += RoundingDivideByPOT(exp_val_q0_31, 12);
        }
    }

    // 3. Tính Nghịch đảo của tổng
    int num_bits_over_unit;
    int32_t shifted_scale_q0_31 = GetReciprocal(sum_of_exps_q12_19, kAccumulationIntegerBits, &num_bits_over_unit);

    // 4. Tính toán kết quả cuối cùng
    // Exponent dựa trên: num_bits_over_unit + 31 - (số bit của OutputT)
    const int exponent = num_bits_over_unit + 31 - 8;

    for (int i = 0; i < input_size; ++i) {
        int32_t input_diff = (int32_t)input_data[i] - max_in_row;
        if (input_diff >= diff_min) {
            int32_t scaled_diff = MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_multiplier, input_left_shift);
            
            int32_t exp_in_0_q0_31 = exp_on_negative_values(scaled_diff);

            // Nhân Exp (Q0.31) với Scale (Q0.31) -> Kết quả Q0.31 (sau khi bão hòa doubling mul)
            int32_t product = SaturatingRoundingDoublingHighMul(exp_in_0_q0_31, shifted_scale_q0_31);
            
            // Dịch về định dạng 8-bit
            int32_t unsat_output = RoundingDivideByPOT(product, exponent);

            // Rescale về OFM Zero Point (Thường Softmax int8 có OFM ZP là -128)
            int32_t shifted_output = unsat_output + (int32_t)ofm_zp;

            // Clamp
            if (shifted_output < -128) shifted_output = -128;
            if (shifted_output > 127) shifted_output = 127;

            output_data[i] = (int8_t)shifted_output;
        } else {
            output_data[i] = ofm_zp;
        }
    }
}

// =================================================================================================
// MAIN
// =================================================================================================

int main() {
    printf("--- Accurate Quantized Softmax Simulation ---\n");

    const char* params_path = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\extracted_params\\layer280_SOFTMAX_StatefulPartitionedCall_1_01\\";
    const char* ifm_file = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_280_SOFTMAX\\ifm280.txt";
    const char* ofm_file = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_280_SOFTMAX\\ofm280_c_sim.txt";

    char path[256];
    sprintf(path, "%sifm_scale.txt", params_path); float ifm_scale = read_float_from_file(path);
    sprintf(path, "%sofm_zp.txt", params_path);    int8_t ofm_zp = read_zp_from_file(path);

    // Tính toán Multiplier và Shift theo PopulateSoftmaxParams của TFLite
    double input_scale_beta_rescaled = (double)ifm_scale * BETA / (1.0 / (double)(1 << (31 - kScaledDiffIntegerBits)));
    
    int s;
    double mantissa = frexp(input_scale_beta_rescaled, &s);
    int32_t input_multiplier = (int32_t)round(mantissa * 2147483648.0);
    int input_left_shift = s;

    // Chuẩn hóa nếu cần
    if (input_multiplier == (int32_t)2147483648LL) {
        input_multiplier /= 2;
        input_left_shift++;
    }

    const int32_t diff_min = -255; // Khoảng an toàn cho int8

    int8_t* input_data = (int8_t*)malloc(IFM_SIZE);
    int8_t* output_data = (int8_t*)malloc(IFM_SIZE);

    read_int8_array_from_file(ifm_file, input_data, IFM_SIZE);

    printf("Executing...\n");
    QuantizedSoftmax(input_data, IFM_SIZE, input_multiplier, input_left_shift, diff_min, ofm_zp, output_data);

    write_ofm_to_file(ofm_file, output_data, IFM_SIZE);
    printf("Done. Output saved to %s\n", ofm_file);

    free(input_data); free(output_data);
    return 0;
}