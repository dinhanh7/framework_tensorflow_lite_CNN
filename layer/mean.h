#ifndef MEAN_H
#define MEAN_H

#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "requantize_utils.h"

/**
 * @brief Thực thi phép tính Mean lượng tử hóa cho tensor int8.
 *
 * Hàm này mô phỏng 100% logic từ `tflite::optimized_integer_ops::Mean` (phiên bản không NEON).
 * Nó thực hiện global average pooling trên các chiều height và width của tensor đầu vào.
 *
 * @param input_data Con trỏ tới dữ liệu tensor đầu vào.
 * @param output_data Con trỏ tới buffer dữ liệu tensor đầu ra.
 * @param input_batch Batch size của tensor đầu vào.
 * @param input_height Chiều cao của tensor đầu vào.
 * @param input_width Chiều rộng của tensor đầu vào.
 * @param input_channels Số kênh của tensor đầu vào.
 * @param output_channels Số kênh của tensor đầu ra (phải bằng input_channels).
 * @param input_scale Tham số scale của tensor đầu vào.
 * @param input_zp Tham số zero-point của tensor đầu vào.
 * @param output_scale Tham số scale của tensor đầu ra.
 * @param output_zp Tham số zero-point của tensor đầu ra.
 */
void quantized_mean_int8(
    const int8_t* input_data,
    int8_t* output_data,
    int input_batch,
    int input_height,
    int input_width,
    int input_channels,
    int output_channels,
    float input_scale,
    int32_t input_zp,
    float output_scale,
    int32_t output_zp
) {
    // 1. Tính toán real_scale (giống hệt mean.h)
    const int32_t num_elements_in_axis = input_width * input_height;
    const double real_multiplier =
        (double)input_scale / (double)output_scale;

    // 2. Lượng tử hóa real_scale để lấy multiplier và shift (giống hệt mean.h)
    int32_t output_multiplier;
    int output_shift;
    QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);

    // 3. Tính toán bias (giống hệt mean.h)
    int reduction_shift =
        63 - CountLeadingZeros64((uint64_t)num_elements_in_axis);
    reduction_shift = reduction_shift < 32 ? reduction_shift : 32;
    reduction_shift =
        reduction_shift < (31 + output_shift) ? reduction_shift
                                              : (31 + output_shift);

    output_multiplier =
        (int32_t)(((int64_t)output_multiplier << reduction_shift) /
                  num_elements_in_axis);
    output_shift -= reduction_shift;

    // 4. Thực thi phép tính (mô phỏng MeanImpl từ mean.h)
    for (int b = 0; b < input_batch; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            int32_t temp_sum = 0;
            // Tích lũy giá trị trên height và width
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    int index = b * (input_height * input_width * input_channels) +
                                h * (input_width * input_channels) +
                                w * (input_channels) +
                                oc;
                    temp_sum += input_data[index];
                }
            }

            // Tái lượng tử hóa theo chuẩn TFLite
            int32_t shifted_sum = temp_sum - input_zp * num_elements_in_axis;
            int32_t acc = MultiplyByQuantizedMultiplier(
                shifted_sum, output_multiplier, output_shift);
            acc += output_zp;

            // Kẹp giá trị và lưu kết quả
            int output_index = b * output_channels + oc;
            output_data[output_index] = clip_int8(acc);
        }
    }
}


#endif
