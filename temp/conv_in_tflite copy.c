#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// ============================================================================
// 1. ĐỊNH NGHĨA CÁC KÍCH THƯỚC (Thay đổi nếu cần)
// ============================================================================
// Đây là các tham số từ command line của script python
#define IFM_HEIGHT 224
#define IFM_WIDTH 224
#define IFM_CHANNEL 3
#define WEIGHT_FILTER 32
#define KERNEL_H 3
#define KERNEL_W 3
#define STRIDE 2
#define PADDING 1

// Kích thước OFM được tính toán
#define OFM_HEIGHT ((IFM_HEIGHT - KERNEL_H + 2 * PADDING) / STRIDE + 1)
#define OFM_WIDTH ((IFM_WIDTH - KERNEL_W + 2 * PADDING) / STRIDE + 1)
#define OFM_CHANNEL WEIGHT_FILTER


// ============================================================================
// 2. CÁC HÀM MÔ PHỎNG RE-QUANTIZATION (QUAN TRỌNG NHẤT)
// ============================================================================
// Tái hiện hàm `SaturatingRoundingDoublingHighMul` trong Python
// Hàm này nhân 2 số 32-bit, lấy 32-bit cao của kết quả 64-bit sau khi làm tròn.
int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
    int64_t a_64 = a;
    int64_t b_64 = b;
    int64_t ab_64 = a_64 * b_64;

    int64_t nudge = (int64_t)1 << 30;
    int64_t result_64 = (ab_64 + nudge) >> 31;

    // Ép kiểu về int32_t, C sẽ tự xử lý tràn số nếu cần (nhưng logic trên đã chuẩn)
    return (int32_t)result_64;
}

// Tái hiện hàm `RoundingRightShift` trong Python
int32_t RoundingRightShift(int32_t x, int32_t shift) {
    if (shift <= 0) {
        return x;
    }
    int64_t nudge = (int64_t)1 << (shift - 1);
    int64_t result_64 = (x + nudge) >> shift;
    return (int32_t)result_64;
}


// Tái hiện hàm `MultiplyByQuantizedMultiplier` trong Python
// *** CẢNH BÁO: Logic này mô phỏng theo script Python `gen_tf_layer_01_verilog1.py`
// *** và không khớp với chuẩn TFLite, nhưng cần thiết để đồng bộ kết quả.
int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int32_t shift) {
    // Logic của Python: 
    // left_shift = shift if shift > 0 else 0
    // right_shift = -shift if shift < 0 else 0
    int32_t left_shift = (shift > 0) ? shift : 0;
    int32_t right_shift = (shift < 0) ? -shift : 0;

    // x_shifted = x * (1 << left_shift)
    // Cần cẩn thận với tràn số khi dịch trái x
    int64_t x_64 = (int64_t)x;
    int64_t x_shifted_64 = x_64 << left_shift;

    // Bão hòa giá trị x_shifted về int32
    int32_t x_shifted_32;
    if (x_shifted_64 > 2147483647LL) {
        x_shifted_32 = 2147483647;
    } else if (x_shifted_64 < -2147483648LL) {
        x_shifted_32 = -2147483648LL;
    } else {
        x_shifted_32 = (int32_t)x_shifted_64;
    }

    // high_mul = SaturatingRoundingDoublingHighMul(x_shifted, quantized_multiplier)
    int32_t high_mul = SaturatingRoundingDoublingHighMul(x_shifted_32, quantized_multiplier);

    // result = RoundingRightShift(high_mul, right_shift)
    return RoundingRightShift(high_mul, right_shift);
}

// Hàm clip giá trị trong khoảng int8_t
int8_t clip_int8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// ============================================================================
// 3. CÁC HÀM ĐỌC/GHI FILE
// ============================================================================

// Hàm đọc một mảng số nguyên 32-bit từ file text
void read_int32_array_from_file(const char* path, int32_t* arr, int size) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    for (int i = 0; i < size; ++i) {
        if (fscanf(fp, "%d", &arr[i]) != 1) {
            fprintf(stderr, "Error: Failed to read element %d from %s\n", i, path);
            exit(1);
        }
    }
    fclose(fp);
}

// *** HÀM MỚI ĐỂ SỬA LỖI ***
// Hàm đọc một mảng số nguyên 8-bit từ file text
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
            exit(1);
        }
        arr[i] = (int8_t)temp_val; // Đọc số nguyên rồi ép kiểu về 8-bit
    }
    fclose(fp);
}

// Hàm đọc một giá trị ZP (số nguyên 8-bit)
int8_t read_zp_from_file(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    int val;
    fscanf(fp, "%d", &val);
    fclose(fp);
    return (int8_t)val;
}

// Hàm ghi OFM ra file
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


// ============================================================================
// 4. HÀM MAIN
// ============================================================================
int main() {
    // --- Cấp phát bộ nhớ ---
    int8_t* ifm = (int8_t*)malloc(IFM_HEIGHT * IFM_WIDTH * IFM_CHANNEL * sizeof(int8_t));
    int8_t* weight = (int8_t*)malloc(KERNEL_H * KERNEL_W * IFM_CHANNEL * WEIGHT_FILTER * sizeof(int8_t));
    int32_t* effective_bias = (int32_t*)malloc(OFM_CHANNEL * sizeof(int32_t));
    int32_t* m_values = (int32_t*)malloc(OFM_CHANNEL * sizeof(int32_t));
    int32_t* n_values = (int32_t*)malloc(OFM_CHANNEL * sizeof(int32_t)); // Đọc là int32 để an toàn
    int32_t* accumulator = (int32_t*)malloc(OFM_HEIGHT * OFM_WIDTH * OFM_CHANNEL * sizeof(int32_t));
    int8_t* ofm = (int8_t*)malloc(OFM_HEIGHT * OFM_WIDTH * OFM_CHANNEL * sizeof(int8_t));
    int8_t* padded_ifm = NULL; // Sẽ được cấp phát sau khi tính toán padding

    // --- Đường dẫn file (giống trong script python) ---
    const char* base_path = "golden_verilog/";
    char ifm_file[256], weight_file[256], eff_bias_file[256], m_file[256], n_file[256], ifm_zp_file[256], ofm_zp_file[256], ofm_file[256];

    sprintf(ifm_file, "%sop004_CONV_2D_ifm_values.hex", base_path);
    sprintf(weight_file, "%sop004_CONV_2D_weight_values.hex", base_path);
    sprintf(eff_bias_file, "%sop004_CONV_2D_effective_bias.txt", base_path);
    sprintf(m_file, "%sop004_CONV_2D_multiplier.txt", base_path);
    sprintf(n_file, "%sop004_CONV_2D_shift.txt", base_path);
    sprintf(ifm_zp_file, "%sop004_CONV_2D_ifm_zp.hex", base_path);
    sprintf(ofm_zp_file, "%sop004_CONV_2D_ofm_zp.hex", base_path);
    sprintf(ofm_file, "%sop004_CONV_2D_ofm_c_sim.hex", base_path); // Tên file output mới

    // --- Đọc dữ liệu từ file --- (ĐÃ SỬA LỖI)
    printf("Reading input files...\n");
    read_int8_array_from_file(ifm_file, ifm, IFM_HEIGHT * IFM_WIDTH * IFM_CHANNEL);
    read_int8_array_from_file(weight_file, weight, KERNEL_H * KERNEL_W * IFM_CHANNEL * WEIGHT_FILTER);
    read_int32_array_from_file(eff_bias_file, effective_bias, OFM_CHANNEL);
    read_int32_array_from_file(m_file, m_values, OFM_CHANNEL);
    read_int32_array_from_file(n_file, n_values, OFM_CHANNEL);
    int8_t zp_in = read_zp_from_file(ifm_zp_file);
    int8_t zp_ofm = read_zp_from_file(ofm_zp_file);
    printf("Info: ZP_IN = %d, ZP_OFM = %d\n", zp_in, zp_ofm);

    // --- Đệm cho IFM (Padding) - ĐỒNG BỘ VỚI PYTHON ---
    printf("Calculating asymmetric padding like Python script...\n");
    int pad_h_total = (OFM_HEIGHT - 1) * STRIDE + KERNEL_H - IFM_HEIGHT;
    if (pad_h_total < 0) pad_h_total = 0;
    int pad_w_total = (OFM_WIDTH - 1) * STRIDE + KERNEL_W - IFM_WIDTH;
    if (pad_w_total < 0) pad_w_total = 0;

    int pad_top = pad_h_total / 2;
    int pad_bottom = pad_h_total - pad_top;
    int pad_left = pad_w_total / 2;
    int pad_right = pad_w_total - pad_left;

    int padded_ifm_height = IFM_HEIGHT + pad_top + pad_bottom;
    int padded_ifm_width = IFM_WIDTH + pad_left + pad_right;

    printf("Padding info (C): Top=%d, Bottom=%d, Left=%d, Right=%d\n", pad_top, pad_bottom, pad_left, pad_right);
    printf("Padded dims (C): %d x %d\n", padded_ifm_height, padded_ifm_width);

    padded_ifm = (int8_t*)calloc(padded_ifm_height * padded_ifm_width * IFM_CHANNEL, sizeof(int8_t));
    if (!padded_ifm) { fprintf(stderr, "Failed to allocate memory for padded_ifm\n"); exit(1); }

    printf("Padding IFM with value %d...\n", zp_in);
    for(int i=0; i < padded_ifm_height * padded_ifm_width * IFM_CHANNEL; ++i) {
        padded_ifm[i] = zp_in;
    }

    for (int c = 0; c < IFM_CHANNEL; ++c) {
        for (int h = 0; h < IFM_HEIGHT; ++h) {
            for (int w = 0; w < IFM_WIDTH; ++w) {
                int padded_idx = (h + pad_top) * padded_ifm_width * IFM_CHANNEL + (w + pad_left) * IFM_CHANNEL + c;
                int orig_idx = h * IFM_WIDTH * IFM_CHANNEL + w * IFM_CHANNEL + c;
                padded_ifm[padded_idx] = ifm[orig_idx];
            }
        }
    }

    // --- TÍCH CHẬP 2D THỦ CÔNG (Thay thế cho Conv2D của TensorFlow) ---
    printf("Performing manual 2D convolution...\n");
    for (int oh = 0; oh < OFM_HEIGHT; ++oh) {
        for (int ow = 0; ow < OFM_WIDTH; ++ow) {
            for (int oc = 0; oc < OFM_CHANNEL; ++oc) {
                int32_t acc = effective_bias[oc];
                for (int kh = 0; kh < KERNEL_H; ++kh) {
                    for (int kw = 0; kw < KERNEL_W; ++kw) {
                        for (int ic = 0; ic < IFM_CHANNEL; ++ic) {
                            int ih = oh * STRIDE + kh;
                            int iw = ow * STRIDE + kw;
                            
                            int ifm_idx = ih * padded_ifm_width * IFM_CHANNEL + iw * IFM_CHANNEL + ic;
                            int weight_idx = oc * KERNEL_H * KERNEL_W * IFM_CHANNEL + 
                                             kh * KERNEL_W * IFM_CHANNEL + 
                                             kw * IFM_CHANNEL + 
                                             ic;

                            acc += (int32_t)padded_ifm[ifm_idx] * (int32_t)weight[weight_idx];
                        }
                    }
                }
                int acc_idx = oh * OFM_WIDTH * OFM_CHANNEL + ow * OFM_CHANNEL + oc;
                accumulator[acc_idx] = acc;
            }
        }
    }

    // --- RE-QUANTIZATION (Giống hệt bước 6 trong script Python) ---
    printf("Performing re-quantization...\n");
    for (int oh = 0; oh < OFM_HEIGHT; ++oh) {
        for (int ow = 0; ow < OFM_WIDTH; ++ow) {
            for (int oc = 0; oc < OFM_CHANNEL; ++oc) {
                int idx = oh * OFM_WIDTH * OFM_CHANNEL + ow * OFM_CHANNEL + oc;
                
                int32_t acc_val = accumulator[idx];
                int32_t m = m_values[oc];
                int32_t n = n_values[oc];

                // Áp dụng phép nhân và dịch bit
                int32_t res_scaled = MultiplyByQuantizedMultiplier(acc_val, m, n);
                
                // Cộng ZP của OFM
                int32_t res_final = res_scaled + zp_ofm;
                
                // Clip về khoảng int8
                ofm[idx] = clip_int8(res_final);
            }
        }
    }

    // --- Ghi kết quả và giải phóng bộ nhớ ---
    printf("Writing output to %s...\n", ofm_file);
    write_ofm_to_file(ofm_file, ofm, OFM_HEIGHT * OFM_WIDTH * OFM_CHANNEL);
    printf("Simulation finished successfully!\n");

    free(ifm);
    free(weight);
    free(effective_bias);
    free(m_values);
    free(n_values);
    free(accumulator);
    free(ofm);
    free(padded_ifm);

    return 0;
}