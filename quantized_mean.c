#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>

// =================================================================================================
// 1. CÁC HÀM PHỤ TRỢ LƯỢNG TỬ HÓA (PHIÊN BẢN CHUẨN TFLITE)
// Tái hiện các hàm từ `optimized_ops.h` và `common.h` để khớp 100% với `mean.h`
// =================================================================================================

// Hàm kẹp giá trị trong khoảng int8_t
int8_t clip_int8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// Tái hiện hàm `SaturatingRoundingDoublingHighMul` từ TFLite (common.h)
// Chính xác, xử lý cả số âm.
// Tái hiện hàm `SaturatingRoundingDoublingHighMul` từ TFLite (common.h)
// Hàm này nhân 2 số 32-bit, lấy 32-bit cao của kết quả 64-bit sau khi làm tròn.
int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
    int64_t a_64 = a;
    int64_t b_64 = b;
    int64_t ab_64 = a_64 * b_64;
    int64_t nudge = (ab_64 >= 0) ? (1LL << 30) : (-(1LL << 30));
    int64_t ab_x2_high32 = (ab_64 + nudge) >> 31;
    // Bão hòa kết quả về khoảng int32
    if (ab_x2_high32 > INT_MAX) return INT_MAX;
    if (ab_x2_high32 < INT_MIN) return INT_MIN;
    return (int32_t)ab_x2_high32;
}

// Tái hiện hàm `RoundingRightShift` từ TFLite (common.h)
int32_t RoundingRightShift(int32_t x, int shift) {
    if (shift == 0) return x;
    int64_t x_64 = x;
    int64_t nudge = (shift > 0) ? (1LL << (shift - 1)) : (-(1LL << (shift - 1)));
    int64_t result_64 = (x_64 + nudge) >> shift;
    // Bão hòa kết quả về khoảng int32
    if (result_64 > INT_MAX) return INT_MAX;
    if (result_64 < INT_MIN) return INT_MIN;
    return (int32_t)result_64;
}

// Tái hiện hàm `MultiplyByQuantizedMultiplier` từ TFLite (common.h)
// Phiên bản này đã được sửa lỗi ép kiểu và logic để khớp 100%.
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;

    // Dịch trái x, giữ nguyên trên int64 để tránh tràn số
    int64_t x_64 = (int64_t)x << left_shift;

    // Bão hòa giá trị về int32 TRƯỚC khi gọi SaturatingRoundingDoublingHighMul
    // Đây là bước quan trọng mà TFLite thực hiện
    int32_t x_32_sat;
    if (x_64 > INT_MAX) x_32_sat = INT_MAX;
    else if (x_64 < INT_MIN) x_32_sat = INT_MIN;
    else x_32_sat = (int32_t)x_64;

    int32_t result = SaturatingRoundingDoublingHighMul(x_32_sat, quantized_multiplier);
    result = RoundingRightShift(result, right_shift);
    return result;
}

// Tái hiện hàm `QuantizeMultiplier` từ TFLite (common.h) một cách chính xác tuyệt đối.
// Hàm này nhận đầu vào là double, khớp với mã nguồn gốc của TFLite.
void QuantizeMultiplier(double real_multiplier, int32_t* quantized_multiplier, int* shift) {
    if (real_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }
    int s;
    // Sử dụng frexp cho kiểu double
    double mantissa = frexp(real_multiplier, &s);

    // Phương pháp làm tròn và nhân của TFLite
    int64_t q_fixed = (int64_t)round(mantissa * (1LL << 31));

    if (q_fixed == (1LL << 31)) {
        q_fixed /= 2;
        s++;
    }

    *quantized_multiplier = (int32_t)q_fixed;
    *shift = s;
}

// =================================================================================================
// 2. CÁC HÀM ĐỌC/GHI FILE (CHO MỤC ĐÍCH KIỂM THỬ)
// =================================================================================================

void read_int8_array_from_file(const char* path, int8_t* arr, int size) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    int temp_val;
    for (int i = 0; i < size; ++i) {
        if (fscanf(fp, "%d", &temp_val) != 1) {
            fprintf(stderr, "Error: Failed to read element %d from %s\n", i, path);
            fclose(fp);
            exit(1);
        }
        arr[i] = (int8_t)temp_val;
    }
    fclose(fp);
}

void write_ofm_to_file(const char* path, int8_t* ofm, int size) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    for (int i = 0; i < size; ++i) {
        fprintf(fp, "%d\n", ofm[i]);
    }
    fclose(fp);
}

// Hàm đọc một giá trị float từ file
float read_float_from_file(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    float val;
    if (fscanf(fp, "%f", &val) != 1) {
        fprintf(stderr, "Error: Failed to read float from %s\n", path);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    return val;
}

// Hàm đọc một giá trị int32 từ file
int32_t read_int32_from_file(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    int32_t val;
    if (fscanf(fp, "%d", &val) != 1) {
        fprintf(stderr, "Error: Failed to read int32 from %s\n", path);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    return val;
}

// =================================================================================================
// 3. HÀM LÕI QUANTIZED MEAN (ĐỂ ĐÓNG GÓI VÀO cnn_layers.h)
// =================================================================================================

/**
 * @brief Thực thi phép tính Mean lượng tử hóa cho tensor int8.
 *
 * Hàm này mô phỏng 100% logic từ `tflite::optimized_integer_ops::Mean` (phiên bản không NEON).
 * Nó được thiết kế để có thể đóng gói, nhận tất cả các tham số cần thiết thay vì
 * sử dụng macro, giúp nó linh hoạt và có thể tái sử dụng.
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
    // --- BƯỚC 1: Chuẩn bị các tham số (Mô phỏng 100% tflite::optimized_integer_ops::Mean) ---
    const int num_elements_in_axis = input_width * input_height;

    // [FIX] Tính real_scale bằng float, y hệt TFLite source (mean.h) để đảm bảo bit-exact
    const float real_scale = (float)input_scale / (num_elements_in_axis * (float)output_scale);

    // Tính bias, y hệt TFLite source
    float temp = (float)input_zp * (float)input_scale / (float)output_scale;
    temp = (temp > 0) ? (temp + 0.5f) : (temp - 0.5f); // Tương đương roundf()
    int32_t bias = output_zp - (int32_t)temp;

    // [FIX] Lượng tử hóa real_scale (float) bằng hàm QuantizeMultiplier (double) chuẩn
    // Trình biên dịch sẽ tự động thăng cấp float -> double, mô phỏng chính xác TFLite
    int32_t multiplier;
    int shift;
    QuantizeMultiplier(real_scale, &multiplier, &shift);

    printf("Calculating Mean with OPTIMIZED params: multiplier=%d, shift=%d, bias=%d\n", 
           multiplier, shift, bias);

    // --- BƯỚC 2: Thực thi phép tính (Mô phỏng y hệt MeanImpl, non-NEON path) ---
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

            // Tái lượng tử hóa kết quả
            acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
            
            // Thêm bias
            acc += bias;

            // Kẹp giá trị trong khoảng int8 và lưu kết quả
            int output_index = b * output_channels + oc;
            output_data[output_index] = clip_int8(acc);
        }
    }
}

// =================================================================================================
// 4. HÀM MAIN (TRÌNH CHẠY KIỂM THỬ)
// =================================================================================================
int main() {
    // --- Định nghĩa các hằng số cho bài toán cụ thể (từ model_profile.txt) ---
    const int INPUT_BATCH      = 1;
    const int INPUT_HEIGHT     = 14;
    const int INPUT_WIDTH      = 14;
    const int INPUT_CHANNELS   = 192;
    const int OUTPUT_CHANNELS  = 192;

    // --- Đường dẫn tới các file tham số và I/O ---
    const char* params_path = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\extracted_params\\layer027_MEAN_1_block4a_se_squeeze_1_Mean\\";
    const char* input_filename = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_27_MEAN\\ifm27.txt";
    const char* output_filename = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_27_MEAN\\ofm27_c_sim.txt";

    // --- Đọc các tham số lượng tử hóa từ file ---
    char path_buffer[512];

    snprintf(path_buffer, sizeof(path_buffer), "%sparam_idx0_scale.txt", params_path);
    const float INPUT_SCALE = read_float_from_file(path_buffer);

    snprintf(path_buffer, sizeof(path_buffer), "%sparam_idx0_zp.txt", params_path);
    const int32_t INPUT_ZP = read_int32_from_file(path_buffer);

    snprintf(path_buffer, sizeof(path_buffer), "%sofm_scale.txt", params_path);
    const float OUTPUT_SCALE = read_float_from_file(path_buffer);

    snprintf(path_buffer, sizeof(path_buffer), "%sofm_zp.txt", params_path);
    const int32_t OUTPUT_ZP = read_int32_from_file(path_buffer);

    printf("Loaded params: IN_SCALE=%.8f, IN_ZP=%d, OUT_SCALE=%.8f, OUT_ZP=%d\n", 
           INPUT_SCALE, INPUT_ZP, OUTPUT_SCALE, OUTPUT_ZP);

    const int input_size = INPUT_BATCH * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS;
    const int output_size = INPUT_BATCH * OUTPUT_CHANNELS;

    // --- Cấp phát bộ nhớ ---
    int8_t* input_tensor = (int8_t*)malloc(input_size * sizeof(int8_t));
    int8_t* output_tensor = (int8_t*)malloc(output_size * sizeof(int8_t));

    if (!input_tensor || !output_tensor) {
        fprintf(stderr, "Failed to allocate memory.\n");
        return 1;
    }

    // --- Đọc dữ liệu đầu vào ---
    printf("Reading input tensor from %s...\n", input_filename);
    read_int8_array_from_file(input_filename, input_tensor, input_size);

    // --- Thực thi phép tính Mean bằng cách gọi hàm lõi ---
    printf("Performing Quantized Mean...\n");
    quantized_mean_int8(
        input_tensor,
        output_tensor,
        INPUT_BATCH,
        INPUT_HEIGHT,
        INPUT_WIDTH,
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
        INPUT_SCALE,
        INPUT_ZP,
        OUTPUT_SCALE,
        OUTPUT_ZP
    );

    // --- Ghi kết quả ---
    printf("Writing output tensor to %s...\n", output_filename);
    write_ofm_to_file(output_filename, output_tensor, output_size);

    printf("Simulation finished successfully!\n");

    // --- Giải phóng bộ nhớ ---
    free(input_tensor);
    free(output_tensor);

    return 0;
}