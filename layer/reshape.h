#ifndef RESHAPE_H
#define RESHAPE_H
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "requantize_utils.h"
/**
 * @brief Thực hiện phép Reshape cho tensor lượng tử hóa.
 * 
 * Hàm này mô phỏng logic của lớp RESHAPE trong TFLite. Vì reshape
 * chỉ thay đổi metadata (hình dạng) của tensor mà không thay đổi dữ liệu
 * bên trong, hàm này thực chất chỉ là một phép sao chép bộ nhớ.
 * Các tham số lượng tử hóa của đầu vào và đầu ra là giống hệt nhau.
 *
 * @param output_data Con trỏ tới buffer để lưu trữ tensor đầu ra.
 * @param input_data Con trỏ tới dữ liệu tensor đầu vào.
 * @param num_elements Tổng số phần tử trong tensor (phải bằng nhau cho cả input và output).
 */
void quantized_reshape(
    int8_t* output_data,
    const int8_t* input_data,
    int num_elements
) {
    // Reshape chỉ là thay đổi cách diễn giải dữ liệu, không thay đổi giá trị.
    // Do đó, chúng ta chỉ cần sao chép toàn bộ dữ liệu từ input sang output.
    memcpy(output_data, input_data, num_elements * sizeof(int8_t));
}
#endif