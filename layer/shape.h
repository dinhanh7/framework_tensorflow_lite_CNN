#ifndef SHAPE_H
#define SHAPE_H
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "requantize_utils.h"

/**
 * @brief Lấy shape của một tensor và lưu vào một mảng int32.
 * 
 * Hàm này mô phỏng logic của lớp SHAPE trong TFLite. Nó sao chép các chiều
 * của một tensor (được cung cấp dưới dạng mảng) vào tensor đầu ra.
 * Hàm này có tính tổng quát, hoạt động với mọi số chiều (rank).
 *
 * @param output_shape_tensor Con trỏ tới mảng int32 để lưu kết quả shape.
 * @param input_dims Một mảng chứa các kích thước của tensor đầu vào.
 * @param rank Số lượng chiều của tensor đầu vào (kích thước của mảng input_dims).
 */
void get_shape(
    int32_t* output_shape_tensor,
    const int32_t* input_dims,
    int rank
) {
    for (int i = 0; i < rank; ++i) {
        output_shape_tensor[i] = input_dims[i];
    }
}
#endif