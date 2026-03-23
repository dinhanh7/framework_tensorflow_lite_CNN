#ifndef CNN_LAYERS_H
#define CNN_LAYERS_H

#include <stdint.h>

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
    // Output Tensor
    int8_t* ofm_data,
    int* out_ofm_height,
    int* out_ofm_width,

    // Input Tensor
    const int8_t* ifm_data,
    int ifm_height,
    int ifm_width,
    int ifm_channels,

    // Weight Tensor
    const int8_t* weight_data,
    int ofm_channels,
    int kernel_h,
    int kernel_w,

    // Quantization Params
    const int32_t* effective_bias_data,
    const int32_t* output_multiplier,
    const int8_t* output_shift,
    int8_t ifm_zp,
    int8_t ofm_zp,

    // Conv Params
    int stride_h,
    int stride_w,
    const char* padding_type
);


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
    // Output Tensor
    int8_t* ofm_data,

    // Input Tensor
    const int8_t* ifm_data,
    int tensor_size,

    // Quantization Params
    int8_t ifm_zp,
    int8_t ofm_zp,
    int16_t input_scale_multiplier,
    int16_t reluish_multiplier,
    int16_t output_scale_multiplier,
    int output_scale_shift
);


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
);


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
);


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
);


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
);


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
);

#endif // CNN_LAYERS_H