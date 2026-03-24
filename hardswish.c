#include "hardswish.h"
#include <math.h>
#include <stdio.h>
#include <stdint.h>

// ============================================================================
// CÁC HÀM HELPER (Được sao chép từ hardswish_in_tflite.c)
// ============================================================================

// Hàm tính toán multiplier và shift từ một giá trị scale (float)
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

// Hàm nhân bão hòa CÓ LÀM TRÒN (Rounding) cho int16
int16_t SaturatingRoundingDoublingHighMul_Int16_round(int16_t a, int16_t b) {
    if (a == -32768 && b == -32768) {
        return 32767;
    }
    int32_t ab = (int32_t)a * (int32_t)b;
    int32_t nudge = 16384; // 1 << 14
    int32_t result = (ab + nudge) >> 15;
    return clamp_int16(result);
}

// Hàm dịch trái có bão hòa để khớp với Python
int16_t SaturatingLeftShift(int16_t val, int shift) {
    int32_t res = (int32_t)val * (1 << shift);
    return clamp_int16(res);
}

// Hàm chia cho lũy thừa của 2 CÓ LÀM TRÒN
int16_t RoundingDivideByPOT(int16_t x, int exponent) {
    if (exponent == 0) return x;
    if (exponent < 0) {
        return clamp_int16((int32_t)x << -exponent);
    }
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    int32_t result = (x >> exponent) + (remainder > threshold ? 1 : 0);
    return clamp_int16(result);
}


// ============================================================================
// HÀM HARDSWISH CHÍNH
// ============================================================================

void quantized_hardswish(
    const int8_t* input_data,
    int8_t* output_data,
    int tensor_size,
    int8_t input_zp,
    int8_t output_zp,
    float input_scale_float,
    float output_scale_float
) {
    // --- Tính toán các Multiplier ---
    double real_output_mul = (input_scale_float / output_scale_float) / 128.0;
    int32_t m_out;
    int n_out;
    GetQuantizedMultiplier(real_output_mul, &m_out, &n_out);

    double real_relu_mul = input_scale_float * 256.0 / 3.0;
    int32_t m_relu;
    int n_relu;
    GetQuantizedMultiplier(real_relu_mul, &m_relu, &n_relu);

    int16_t m_out_16 = clamp_int16(m_out >> 16);
    int16_t m_relu_16 = clamp_int16(m_relu >> 16);

    printf("\n>> Inside quantized_hardswish - Calculated Params:\n");
    printf("   Output Mul: m_out_16=%d, n_out=%d\n", m_out_16, n_out);
    printf("   Reluish Mul: m_relu_16=%d, n_relu=%d\n", m_relu_16, n_relu);
    printf(">> Performing quantized hardswish logic...\n");

    // --- Vòng lặp xử lý ---
    for (int i = 0; i < tensor_size; ++i) {
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

        // 5. Final Multiplication
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
}