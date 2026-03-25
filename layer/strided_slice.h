#ifndef STRIDED_SLICE_H
#define STRIDED_SLICE_H
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "requantize_utils.h"

/**
 * @brief Thực hiện phép cắt (slice) trên một tensor 1D int32.
 *
 * Hàm này mô phỏng logic của lớp StridedSlice trong TFLite cho trường hợp
 * cụ thể trong model (lớp 60), hoạt động trên tensor int32. Nó không
 * thực hiện lượng tử hóa.
 *
 * @param output_tensor Con trỏ tới mảng để lưu kết quả đầu ra.
 * @param input_tensor Con trỏ tới mảng dữ liệu đầu vào.
 * @param begin_params Con trỏ tới mảng chứa chỉ số bắt đầu.
 * @param end_params Con trỏ tới mảng chứa chỉ số kết thúc.
 * @param strides_params Con trỏ tới mảng chứa bước nhảy.
 * @param output_size Kích thước của mảng đầu ra để kiểm tra biên.
 */
void strided_slice_int32(
    int32_t* output_tensor,
    const int32_t* input_tensor,
    const int32_t* begin_params,
    const int32_t* end_params,
    const int32_t* strides_params,
    int output_size
) {
    int output_idx = 0;
    // Logic này giả định các tham số chỉ có 1 phần tử, đúng với model_profile.txt
    for (int i = begin_params[0]; i < end_params[0]; i += strides_params[0]) {
        if (output_idx < output_size) {
            output_tensor[output_idx] = input_tensor[i];
            output_idx++;
        }
    }
}

#endif