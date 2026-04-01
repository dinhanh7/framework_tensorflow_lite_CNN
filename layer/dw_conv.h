// ===============================================================================================
// File chứa định nghĩa các hàm phục vụ cho tính có khả năng 3 hàm đầu xung đột tên với conv thường
// Cách sử dụng các hàm này trong việc build model inference hoặc test dw_conv.c
// 1. Cấp phát bộ nhớ cho các biến
// 2. Đọc dữ liệu từ file (trong file test dw_conv.c)
// 3. Cấu hình tham số
// 4. Tính toán
// 5. Ghi dữ liệu ra file (trong file test dw_conv.c)
// ===============================================================================================
#ifndef DW_CONV_H
#define DW_CONV_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include "utils.h"
#include "requantize_utils.h"



// Hàm Offset để lấy vị trí từ mảng 1D (trải phẳng Tensor 4D)
int Offset(int batch, int h, int w, int c, int height, int width, int depth) {
    return ((batch * height + h) * width + w) * depth + c;
}

// Cấu trúc chứa các tham số thiết lập Depthwise Convolution
typedef struct {
    int32_t input_zp;
    int32_t output_zp;
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;
    int stride_width;
    int stride_height;
    int pad_width;
    int pad_height;
    int dilation_width_factor;
    int dilation_height_factor;
} DepthwiseParams;

// Lõi Inference của Depthwise Convolution (INT8 Per-Channel)
void DepthwiseConvPerChannel(
    const DepthwiseParams* params,
    const int32_t* output_multiplier, // Mảng multiplier cho từng kênh đã tính ở Prepare
    const int32_t* output_shift,      // Mảng shift cho từng kênh đã tính ở Prepare
    int batches,
    int input_h, int input_w, int input_c, const int8_t* input_data,
    int filter_h, int filter_w, int filter_c, const int8_t* filter_data,
    int bias_size, const int32_t* effective_bias_data,
    int output_h, int output_w, int output_c, int8_t* output_data,
    int32_t* output_acc) {

  // 1. Gán tham số lượng tử hóa (Quantization params) & stride, pad
  const int32_t input_zp = params->input_zp;
  const int32_t output_zp = params->output_zp;
  const int32_t output_activation_min = params->quantized_activation_min;
  const int32_t output_activation_max = params->quantized_activation_max;
  
  const int stride_width = params->stride_width;
  const int stride_height = params->stride_height;
  const int pad_width = params->pad_width;
  const int pad_height = params->pad_height;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;

  // 2. Trích xuất kích thước các chiều của Tensor
  const int input_height = input_h;
  const int input_width = input_w;
  const int input_depth = input_c;
  
  const int filter_height = filter_h;
  const int filter_width = filter_w;
  
  const int output_height = output_h;
  const int output_width = output_w;

  const int depth_multiplier = output_c / input_c;

  // ==========================================================
  // 3. TẠO BUFFER PADDING ẢO (PHYSICAL PADDING TO TENSOR)
  // ==========================================================
  int pad_h_total = (output_height - 1) * stride_height + filter_height - input_height;
  if (pad_h_total < 0) pad_h_total = 0;
  int pad_top = pad_h_total / 2;
  int pad_bottom = pad_h_total - pad_top;

  int pad_w_total = (output_width - 1) * stride_width + filter_width - input_width;
  if (pad_w_total < 0) pad_w_total = 0;
  int pad_left = pad_w_total / 2;
  int pad_right = pad_w_total - pad_left;

  int padded_h = input_height + pad_top + pad_bottom;
  int padded_w = input_width + pad_left + pad_right;
  
  int8_t* padded_input = (int8_t*)malloc(batches * padded_h * padded_w * input_depth * sizeof(int8_t));
  for (int i = 0; i < batches * padded_h * padded_w * input_depth; ++i) {
      padded_input[i] = (int8_t)input_zp;
  }

  for (int b = 0; b < batches; ++b) {
      for (int y = 0; y < input_height; ++y) {
          for (int x = 0; x < input_width; ++x) {
              for (int c = 0; c < input_depth; ++c) {
                  int src_idx = Offset(b, y, x, c, input_height, input_width, input_depth);
                  int dst_idx = Offset(b, y + pad_top, x + pad_left, c, padded_h, padded_w, input_depth);
                  padded_input[dst_idx] = input_data[src_idx];
              }
          }
      }
  }

  // ==========================================================
  // 4. CHẠY LÕI INFERENCE
  // ==========================================================
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        for (int in_channel = 0; in_channel < input_depth; ++in_channel) {
          for (int m = 0; m < depth_multiplier; ++m) {
            
            const int output_channel = m + in_channel * depth_multiplier;
            
            const int in_x_origin = (out_x * stride_width);
            const int in_y_origin = (out_y * stride_height);
            
            // Thanh ghi tích lũy (Accumulator) dùng int32 để tránh tràn số (overflow)
            int32_t acc = 0;

            // --- BẮT ĐẦU TÍNH TÍCH CHẬP 2D CHO 1 KÊNH ---
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int in_x = in_x_origin + dilation_width_factor * filter_x;
                const int in_y = in_y_origin + dilation_height_factor * filter_y;

                // Flatten index (đã dùng Physical Padding nên mảng luôn hợp lệ)
                int input_idx = Offset(batch, in_y, in_x, in_channel, padded_h, padded_w, input_depth);
                int filter_idx = Offset(0, filter_y, filter_x, output_channel, filter_height, filter_width, output_c);
                
                int32_t input_val = padded_input[input_idx];
                int32_t filter_val = filter_data[filter_idx];
                
                // Lõi MAC (input_zp đã nằm trong effective_bias)
                acc += filter_val * input_val;
              }
            }
            // --- KẾT THÚC TÍCH CHẬP 2D ---
            
            // LƯU TRỮ ACCUMULATOR CHƯA BIAS DÙNG ĐỂ DEBUG
            if (output_acc != NULL) {
                int out_idx = Offset(batch, out_y, out_x, output_channel, output_height, output_width, output_c);
                output_acc[out_idx] = acc;
            }

            // ==========================================================
            // 5. RE-QUANTIZATION
            // ==========================================================

            // 6. Cộng bias
            if (effective_bias_data && bias_size > 0 && output_channel < bias_size) {
              acc += effective_bias_data[output_channel];
            }
            
            // 7. Nhân với multiplier và shift
            acc = MultiplyByQuantizedMultiplierDWConv(
                acc, 
                output_multiplier[output_channel], 
                output_shift[output_channel]);
            
            // 8. Khôi phục zero-point của Output
            acc += output_zp;
            
            // 8. Ràng buộc kết quả trong giới hạn (Activation)
            acc = MAX(acc, output_activation_min);
            acc = MIN(acc, output_activation_max);
            
            // 9. GHI KẾT QUẢ RA BỘ NHỚ
            int out_idx = Offset(batch, out_y, out_x, output_channel, output_height, output_width, output_c);
            output_data[out_idx] = (int8_t)(acc);
          }
        }
      }
    }
  }

  // GIẢI PHÓNG BỘ NHỚ PADDED
  free(padded_input);
}

#endif // DW_CONV_H
