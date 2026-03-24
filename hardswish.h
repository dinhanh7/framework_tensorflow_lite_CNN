#ifndef HARDSWISH_H
#define HARDSWISH_H

#include <stdint.h>

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
);

#endif // HARDSWISH_H
