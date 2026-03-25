#ifndef PACK_H
#define PACK_H
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "requantize_utils.h"

/**
 * @brief Thực hiện phép Pack (stack) trên 2 tensor int32 có shape khác nhau.
 * 
 * Hàm này mô phỏng logic của lớp PACK trong TFLite (ví dụ: Layer 62), 
 * trong đó các tensor đầu vào có thể có độ dài khác nhau. Tensor ngắn hơn
 * sẽ được đệm (pad) bằng số 0 để khớp với chiều dài của tensor dài nhất
 * (hoặc một chiều dài cột xác định).
 *
 * @param output_tensor Con trỏ tới mảng đầu ra (đã được làm phẳng - flattened).
 * @param input0 Con trỏ tới tensor đầu vào thứ nhất.
 * @param input0_size Kích thước của tensor đầu vào thứ nhất.
 * @param input1 Con trỏ tới tensor đầu vào thứ hai.
 * @param input1_size Kích thước của tensor đầu vào thứ hai.
 * @param output_cols Chiều rộng của ma trận đầu ra. Đây là kích thước để các tensor con được đệm tới.
 */
void pack_int32_with_padding(
    int32_t* output_tensor,
    const int32_t* input0,
    int input0_size,
    const int32_t* input1,
    int input1_size,
    int output_cols
) {
    // Sao chép input0 vào dòng đầu tiên của ma trận output
    memcpy(output_tensor, input0, input0_size * sizeof(int32_t));
    // Đệm phần còn lại của dòng đầu tiên (nếu cần)
    for (int i = input0_size; i < output_cols; ++i) {
        output_tensor[i] = 0;
    }

    // Sao chép input1 vào dòng thứ hai
    memcpy(output_tensor + output_cols, input1, input1_size * sizeof(int32_t));
    // Đệm phần còn lại của dòng thứ hai (nếu cần)
    for (int i = input1_size; i < output_cols; ++i) {
        output_tensor[output_cols + i] = 0;
    }
}
#endif