#include "cnn_layers.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// CÁC HÀM HELPER TĨNH (STATIC) - Chỉ sử dụng trong file này
// ============================================================================

// Tái hiện hàm `SaturatingRoundingDoublingHighMul` (Đã đồng bộ với phiên bản TFLite chính xác)
static int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
    int64_t a_64 = a;
    int64_t b_64 = b;
    int64_t ab_64 = a_64 * b_64;
    int64_t nudge = (1LL << 30);
    int64_t result_64 = (ab_64 + nudge) >> 31;
    if (result_64 > 2147483647LL) return 2147483647;
    if (result_64 < -2147483648LL) return -2147483648LL;
    return (int32_t)result_64;
}

// Tái hiện hàm `RoundingRightShift` (Đã đồng bộ với phiên bản TFLite chính xác)
static int32_t RoundingRightShift(int32_t x, int shift) {
    if (shift <= 0) return x;
    int64_t x_64 = x;
    int64_t nudge = (1LL << (shift - 1));
    int64_t result_64 = (x_64 + nudge) >> shift;
    if (result_64 > 2147483647LL) return 2147483647;
    if (result_64 < -2147483648LL) return -2147483648LL;
    return (int32_t)result_64;
}

// Tái hiện hàm `MultiplyByQuantizedMultiplier` (Đã đồng bộ với phiên bản TFLite chính xác)
static int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    int64_t x_64 = (int64_t)x << left_shift;
    // Giả định rằng x_64 sau khi dịch trái vẫn nằm trong khoảng int32. 
    // Điều này thường đúng với các tham số lượng tử hóa thông thường.
    int32_t x_32 = (int32_t)x_64; 
    int32_t result = SaturatingRoundingDoublingHighMul(x_32, quantized_multiplier);
    result = RoundingRightShift(result, right_shift);
    return result;
}

// Hàm kẹp giá trị trong khoảng int8_t
static int8_t clip_int8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// ============================================================================
// CÁC HÀM HELPER TĨNH (STATIC) - LOGIC INT16 CHO HARDSWISH
// Sao chép y hệt từ hardswish_in_tflite.c
// ============================================================================

// Hàm kẹp giá trị trong khoảng int16_t
static int16_t clamp_int16(int32_t val) {
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return (int16_t)val;
}

// Hàm nhân bão hòa KHÔNG LÀM TRÒN (Truncating) cho int16
static int16_t SaturatingDoublingHighMul_Int16_trunc(int16_t a, int16_t b) {
    if (a == -32768 && b == -32768) return 32767;
    int32_t ab = (int32_t)a * (int32_t)b;
    int32_t result = ab / 32768;
    return clamp_int16(result);
}

// Hàm nhân bão hòa CÓ LÀM TRÒN (Rounding) cho int16
static int16_t SaturatingRoundingDoublingHighMul_Int16_round(int16_t a, int16_t b) {
    if (a == -32768 && b == -32768) return 32767;
    int32_t ab = (int32_t)a * (int32_t)b;
    int32_t nudge = 16384;
    int32_t result = (ab + nudge) >> 15;
    return clamp_int16(result);
}

// Hàm chia cho lũy thừa của 2 CÓ LÀM TRÒN
static int16_t RoundingDivideByPOT(int16_t x, int exponent) {
    if (exponent == 0) return x;
    if (exponent < 0) return clamp_int16((int32_t)x << -exponent);
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    int32_t result = (x >> exponent) + (remainder > threshold ? 1 : 0);
    return clamp_int16(result);
}


// ============================================================================
// THÂN HÀM CONV2D LƯỢNG TỬ HÓA
// ============================================================================

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
        fprintf(stderr, "Error: Failed to allocate temporary memory in quantized_conv2d.\n");
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

// ============================================================================
// THÂN HÀM HARDSWISH LƯỢNG TỬ HÓA (LOGIC INT16)
// ============================================================================

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

// ============================================================================
// CÁC HÀM HELPER TĨNH (STATIC) - LOGIC CHO MEAN
// ============================================================================

// Tái hiện hàm `QuantizeMultiplier` từ TFLite
static void QuantizeMultiplier(double real_multiplier, int32_t* quantized_multiplier, int* shift) {
    if (real_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }
    const double q = frexp(real_multiplier, shift);
    *shift += 31;
    int64_t q_fixed = (int64_t)round(q * (1LL << 31));
    if (q_fixed == (1LL << 31)) {
        q_fixed /= 2;
        ++*shift;
    }
    *quantized_multiplier = (int32_t)q_fixed;
}


// ============================================================================
// THÂN HÀM MEAN LƯỢNG TỬ HÓA
// ============================================================================

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
    const float num_elements_in_axis = (float)(input_width * input_height);
    const double real_scale = (double)input_scale / (num_elements_in_axis * (double)output_scale);

    // 2. Lượng tử hóa real_scale để lấy multiplier và shift (giống hệt mean.h)
    int32_t multiplier;
    int shift;
    QuantizeMultiplier(real_scale, &multiplier, &shift);

    // 3. Tính toán bias (giống hệt mean.h)
    double temp = (double)input_zp * (double)input_scale / (double)output_scale;
    temp = (temp > 0) ? (temp + 0.5f) : (temp - 0.5f);
    int32_t bias = output_zp - (int32_t)temp;

    // 4. Thực thi phép tính (mô phỏng MeanImpl từ mean.h)
    for (int b = 0; b < input_batch; ++b) {
        for (int oc = 0; oc < output_channels; ++oc) {
            int32_t acc = 0;
            // Tích lũy giá trị trên height và width
            for (int h = 0; h < input_height; ++h) {
                for (int w = 0; w < input_width; ++w) {
                    int index = b * (input_height * input_width * input_channels) +
                                h * (input_width * input_channels) +
                                w * (input_channels) +
                                oc;
                    acc += input_data[index];
                }
            }

            // Tái lượng tử hóa theo chuẩn TFLite
            acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
            acc += bias;

            // Kẹp giá trị và lưu kết quả
            int output_index = b * output_channels + oc;
            output_data[output_index] = clip_int8(acc);
        }
    }
}


// ============================================================================
// THÂN HÀM PACK (INT32) VỚI PADDING
// ============================================================================

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


// ============================================================================
// THÂN HÀM STRIDED SLICE (INT32)
// ============================================================================

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


// ============================================================================
// THÂN HÀM GET SHAPE (TỔNG QUÁT)
// ============================================================================

void get_shape(
    int32_t* output_shape_tensor,
    const int32_t* input_dims,
    int rank
) {
    for (int i = 0; i < rank; ++i) {
        output_shape_tensor[i] = input_dims[i];
    }
}


// ============================================================================
// THÂN HÀM RESHAPE (TỔNG QUÁT)
// ============================================================================

void quantized_reshape(
    int8_t* output_data,
    const int8_t* input_data,
    int num_elements
) {
    // Reshape chỉ là thay đổi cách diễn giải dữ liệu, không thay đổi giá trị.
    // Do đó, chúng ta chỉ cần sao chép toàn bộ dữ liệu từ input sang output.
    memcpy(output_data, input_data, num_elements * sizeof(int8_t));
}