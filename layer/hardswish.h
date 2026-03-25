#ifndef HARDSWISH_H
#define HARDSWISH_H

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "requantize_utils.h"


/**
 * @brief Thực hiện lớp Hardswish lượng tử hóa (logic int16).
 *
 * Hàm này thực hiện thuật toán hardswish trên mỗi phần tử của tensor đầu vào.
 * Logic được sao chép chính xác từ hardswish_in_tflite.c để đảm bảo kết quả tương đồng,
 * đặc biệt là việc sử dụng logic tính toán 16-bit.
 *
 * @param ofm_data Con trỏ đến buffer để lưu trữ tensor đầu ra.
 * @param ifm_data Con trỏ đến dữ liệu tensor đầu vào.
 * @param tensor_size Tổng số phần tử trong tensor (H * W * C).
 * @param ifm_zp Điểm zero-point của IFM.
 * @param ofm_zp Điểm zero-point của OFM.
 * @param input_scale_multiplier Số nhân để scale đầu vào sang int16.
 * @param reluish_multiplier Số nhân cho phần reluish(x+3).
 * @param output_scale_multiplier Số nhân để scale đầu ra.
 * @param output_scale_shift Giá trị dịch bit để scale đầu ra.
 */
void quantized_hardswish_int16(
    int8_t* ofm_data,
    const int8_t* ifm_data,
    int tensor_size,
    int8_t ifm_zp,
    int8_t ofm_zp,
    int16_t input_scale_multiplier,
    int16_t reluish_multiplier,
    int16_t output_scale_multiplier,
    int output_scale_shift)
{
    for (int i = 0; i < tensor_size; ++i) {
        const int16_t input_val = ifm_data[i];
        const int16_t input_val_minus_zp = input_val - ifm_zp;

        // 1. Đưa input về thang đo 16-bit trung gian
        const int16_t input_val_s16 = SaturatingRoundingDoublingHighMul_Int16_round(
            clamp_int16((int32_t)input_val_minus_zp << 7),
            input_scale_multiplier
        );

        // 2. Tính (x+3) và kẹp trong khoảng [0, 6] (reluish)
        const int16_t reluish_val = SaturatingRoundingDoublingHighMul_Int16_round(
            input_val_s16,
            reluish_multiplier
        );

        // 3. Nhân x * reluish(x) -> DÙNG PHÉP NHÂN KHÔNG LÀM TRÒN (TRUNCATING)
        const int16_t product = SaturatingDoublingHighMul_Int16_trunc(
            reluish_val,
            input_val_s16
        );

        // 4. Đưa kết quả về thang đo output
        int16_t output_s16 = SaturatingRoundingDoublingHighMul_Int16_round(
            product,
            output_scale_multiplier
        );
        output_s16 = RoundingDivideByPOT(output_s16, output_scale_shift);

        // 5. Cộng ZP và kẹp về int8
        int32_t final_val = output_s16 + ofm_zp;
        ofm_data[i] = clip_int8(final_val);
    }
}
#endif