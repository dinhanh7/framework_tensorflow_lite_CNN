#ifndef CONV2D_H
#define CONV2D_h
#include <stdint.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "requantize_utils.h"
#include "utils.h"


/**
 * @brief Thực hiện một lớp tích chập 2D lượng tử hóa đầy đủ.
 * 
 * Hàm này bao gồm tất cả các bước: đệm (padding), tích chập, cộng effective bias,
 * tái lượng tử hóa (re-quantization) và kẹp giá trị (clipping).
 * Logic được sao chép chính xác từ conv_in_tflite.c để đảm bảo kết quả tương đồng.
 *
 * @param ofm_data Con trỏ đến buffer để lưu trữ bản đồ đặc trưng đầu ra (OFM). Buffer này phải được cấp phát từ bên ngoài.
 * @param out_ofm_height Con trỏ để trả về chiều cao của OFM đã được tính toán.
 * @param out_ofm_width Con trỏ để trả về chiều rộng của OFM đã được tính toán.
 * @param ifm_data Con trỏ đến dữ liệu bản đồ đặc trưng đầu vào (IFM).
 * @param ifm_height Chiều cao của IFM.
 * @param ifm_width Chiều rộng của IFM.
 * @param ifm_channels Số kênh của IFM.
 * @param weight_data Con trỏ đến dữ liệu trọng số (đã được sắp xếp theo thứ tự OFM_CHANNEL, H, W, IFM_CHANNEL).
 * @param ofm_channels Số kênh của OFM (cũng là số bộ lọc).
 * @param kernel_h Chiều cao của kernel.
 * @param kernel_w Chiều rộng của kernel.
 * @param effective_bias_data Con trỏ đến dữ liệu effective bias (int32).
 * @param output_multiplier Con trỏ đến mảng các giá trị số nhân tái lượng tử hóa (int32).
 * @param output_shift Con trỏ đến mảng các giá trị dịch bit tái lượng tử hóa (int8).
 * @param ifm_zp Điểm zero-point của IFM.
 * @param ofm_zp Điểm zero-point của OFM.
 * @param stride_h Sải bước theo chiều dọc.
 * @param stride_w Sải bước theo chiều ngang.
 * @param padding_type Kiểu đệm. Hiện tại hỗ trợ "SAME".
 */
void quantized_conv2d(
    int8_t* ofm_data, int* out_ofm_height, int* out_ofm_width,
    const int8_t* ifm_data, int ifm_height, int ifm_width, int ifm_channels,
    const int8_t* weight_data, int ofm_channels, int kernel_h, int kernel_w,
    const int32_t* effective_bias_data, const int32_t* output_multiplier, const int8_t* output_shift,
    int8_t ifm_zp, int8_t ofm_zp,
    int stride_h, int stride_w, const char* padding_type)
{
    // --- 1. Tính toán kích thước đầu ra và padding ---
    int ofm_height, ofm_width;
    int pad_top = 0, pad_bottom = 0, pad_left = 0, pad_right = 0;

    if (strcmp(padding_type, "SAME") == 0) {
        ofm_height = (ifm_height + stride_h - 1) / stride_h;
        ofm_width = (ifm_width + stride_w - 1) / stride_w;
        int pad_h_total = (ofm_height - 1) * stride_h + kernel_h - ifm_height;
        int pad_w_total = (ofm_width - 1) * stride_w + kernel_w - ifm_width;
        pad_h_total = (pad_h_total > 0) ? pad_h_total : 0;
        pad_w_total = (pad_w_total > 0) ? pad_w_total : 0;
        pad_top = pad_h_total / 2;
        pad_bottom = pad_h_total - pad_top;
        pad_left = pad_w_total / 2;
        pad_right = pad_w_total - pad_left;
    } else { // "VALID"
        ofm_height = (ifm_height - kernel_h) / stride_h + 1;
        ofm_width = (ifm_width - kernel_w) / stride_w + 1;
    }
    
    *out_ofm_height = ofm_height;
    *out_ofm_width = ofm_width;

    int padded_ifm_height = ifm_height + pad_top + pad_bottom;
    int padded_ifm_width = ifm_width + pad_left + pad_right;

    // --- 2. Cấp phát bộ nhớ tạm ---
    int8_t* padded_ifm = (int8_t*)calloc(padded_ifm_height * padded_ifm_width * ifm_channels, sizeof(int8_t));
    int32_t* accumulator = (int32_t*)malloc(ofm_height * ofm_width * ofm_channels * sizeof(int32_t));
    if (!padded_ifm || !accumulator) {
        fprintf(stderr, "Error: Failed to allocate temporary memory in quantized_conv2d.\\n");
        if(padded_ifm) free(padded_ifm);
        if(accumulator) free(accumulator);
        return;
    }

    // --- 3. Thực hiện Padding ---
    for(int i=0; i < padded_ifm_height * padded_ifm_width * ifm_channels; ++i) {
        padded_ifm[i] = ifm_zp;
    }
    for (int c = 0; c < ifm_channels; ++c) {
        for (int h = 0; h < ifm_height; ++h) {
            for (int w = 0; w < ifm_width; ++w) {
                int padded_idx = (h + pad_top) * padded_ifm_width * ifm_channels + (w + pad_left) * ifm_channels + c;
                int orig_idx = h * ifm_width * ifm_channels + w * ifm_channels + c;
                padded_ifm[padded_idx] = ifm_data[orig_idx];
            }
        }
    }

    // --- 4. Tích chập 2D thủ công ---
    for (int oh = 0; oh < ofm_height; ++oh) {
        for (int ow = 0; ow < ofm_width; ++ow) {
            for (int oc = 0; oc < ofm_channels; ++oc) {
                int32_t acc = 0;
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        for (int ic = 0; ic < ifm_channels; ++ic) {
                            int ih = oh * stride_h + kh;
                            int iw = ow * stride_w + kw;
                            int ifm_idx = ih * padded_ifm_width * ifm_channels + iw * ifm_channels + ic;
                            int weight_idx = oc * kernel_h * kernel_w * ifm_channels + kh * kernel_w * ifm_channels + kw * ifm_channels + ic;
                            acc += (int32_t)padded_ifm[ifm_idx] * (int32_t)weight_data[weight_idx];
                        }
                    }
                }
                int acc_idx = oh * ofm_width * ofm_channels + ow * ofm_channels + oc;
                accumulator[acc_idx] = acc;
            }
        }
    }

    // --- 5. Cộng Effective Bias (giữ nguyên logic gốc) ---
    for (int i = 0; i < ofm_height * ofm_width; ++i) {
        for (int oc = 0; oc < ofm_channels; ++oc) {
            accumulator[i * ofm_channels + oc] += effective_bias_data[oc];
        }
    }

    // --- 6. Tái lượng tử hóa ---
    for (int i = 0; i < ofm_height * ofm_width; ++i) {
        for (int oc = 0; oc < ofm_channels; ++oc) {
            int idx = i * ofm_channels + oc;
            int32_t acc_val = accumulator[idx];
            int32_t m = output_multiplier[oc];
            int8_t n = output_shift[oc];
            int32_t res_scaled = MultiplyByQuantizedMultiplier(acc_val, m, n);
            int32_t res_final = res_scaled + ofm_zp;
            ofm_data[idx] = clip_int8(res_final);
        }
    }

    // --- 7. Giải phóng bộ nhớ tạm ---
    free(padded_ifm);
    free(accumulator);
}
#endif