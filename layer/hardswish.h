#ifndef HARDSWISH_H
#define HARDSWISH_H

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "requantize_utils.h"


/**
 * @brief Thực hiện phép toán hardswish lượng tử hóa trên một tensor.
 *
 * Hàm này mô phỏng logic hardswish lượng tử hóa của TFLite, sử dụng
 * các phép toán số học int16 cho các tính toán trung gian để khớp với
 * một file tham chiếu Python.
 *
 * @param input_data Con trỏ đến dữ liệu tensor đầu vào (int8_t).
 * @param output_data Con trỏ đến dữ liệu tensor đầu ra (int8_t).
 * @param tensor_size Tổng số phần tử trong tensor.
 * @param input_zp Điểm zero-point của tensor đầu vào.
 * @param output_zp Điểm zero-point của tensor đầu ra.
 * @param input_scale_float Tham số scale lượng tử hóa của tensor đầu vào.
 * @param output_scale_float Tham số scale lượng tử hóa của tensor đầu ra.
 */
void quantized_hardswish(
    const int8_t* input_data,
    int8_t* output_data,
    int tensor_size,
    int8_t input_zp,
    int8_t output_zp,
    float input_scale_float,
    float output_scale_float
){
    // --- Tính toán các Multiplier ---
    double real_output_mul = (input_scale_float / output_scale_float) / 128.0;
    int32_t m_out;
    int n_out;
    QuantizeMultiplier(real_output_mul, &m_out, &n_out);

    double real_relu_mul = input_scale_float * 256.0 / 3.0;
    int32_t m_relu;
    int n_relu;
    QuantizeMultiplier(real_relu_mul, &m_relu, &n_relu);

    int16_t m_out_16 = clamp_int16(((int64_t)m_out + 32768) >> 16);
    int16_t m_relu_16 = clamp_int16(((int64_t)m_relu + 32768) >> 16);

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
        // Use of SaturatingDoublingHighMul_Int16_trunc here is important to cancel the biases
        // from the above SaturatingRoundingDoublingHighMul_Int16_round.
        const int16_t preshift_final = SaturatingDoublingHighMul_Int16_trunc(reluish, preshift_out);

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
#endif