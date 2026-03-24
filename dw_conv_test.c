// ============================================================================
//                                 USER MANUAL
// ============================================================================
// 1. Thay đổi các thông số trong phần 1. ĐỊNH NGHĨA CÁC KÍCH THƯỚC
// 2. Thay đổi các đường dẫn trong phần 3. HÀM MAIN TEST DW_CONV là tên folder chứa các file ifm_values.txt, weight_values.txt, effective_bias.txt, m_values.txt, n_values.txt, ifm_zp.txt, ofm_zp.txt, ofm_c_sim.txt, acc_c_sim.txt
// 3. Output gồm 2 file ofm_c_sim.txt và acc_c_sim.txt được in ra cùng folder
// ============================================================================
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "dw_conv.h"

// ============================================================================
// 1. ĐỊNH NGHĨA CÁC KÍCH THƯỚC (Thay đổi tương ứng với layer025)
// ============================================================================
// Thông số cho xem bằng netron
#define IFM_HEIGHT 28  // Thay đổi kích thước ifm thực tế của layer này
#define IFM_WIDTH 28   // Thay đổi kích thước ifm thực tế của layer này
#define CHANNELS 192   // Thay đổi kích thước ifm thực tế của layer này
#define KERNEL_H 3
#define KERNEL_W 3
#define STRIDE_H 2
#define STRIDE_W 2

// Tinh nham padding
#define PAD_H 0
#define PAD_W 0

#define OFM_HEIGHT 14
#define OFM_WIDTH 14
#define OFM_CHANNEL CHANNELS 

// ============================================================================
// 2. CÁC HÀM ĐỌC/GHI FILE
// ============================================================================

void read_int32_array_from_file(const char* path, int32_t* arr, int size) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Loi: Khong the mo file (hoac file khong ton tai) %s\n", path);
        // Điền 0 nếu không tìm thấy file để test vẫn chạy được
        for(int i=0; i<size; i++) arr[i] = 0;
        return;
    }
    for (int i = 0; i < size; ++i) {
        if (fscanf(fp, "%d", &arr[i]) != 1) {
            fprintf(stderr, "Loi: Doc phan tu %d tu %s that bai\n", i, path);
            break;
        }
    }
    fclose(fp);
}

void read_int8_array_from_file(const char* path, int8_t* arr, int size) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Loi: Khong the mo file %s\n", path);
        for(int i=0; i<size; i++) arr[i] = 0;
        return;
    }
    int temp_val;
    for (int i = 0; i < size; ++i) {
        if (fscanf(fp, "%d", &temp_val) != 1) {
            fprintf(stderr, "Loi: Doc phan tu %d tu %s that bai\n", i, path);
            break;
        }
        arr[i] = (int8_t)temp_val;
    }
    fclose(fp);
}

int32_t read_single_int_from_file(const char* path, int32_t def_val) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Loi: Khong the mo file %s. Dung gia tri mac dinh %d.\n", path, def_val);
        return def_val;
    }
    int val = def_val;
    fscanf(fp, "%d", &val);
    fclose(fp);
    return val;
}

void write_ofm_to_file(const char* path, int8_t* ofm, int size) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Loi: Khong the mo file ghi %s\n", path);
        return;
    }
    for (int i = 0; i < size; ++i) {
        fprintf(fp, "%d\n", ofm[i]);
    }
    fclose(fp);
}

void write_acc_to_file(const char* path, int32_t* acc_arr, int size) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Loi: Khong the mo file ghi acc %s\n", path);
        return;
    }
    for (int i = 0; i < size; ++i) {
        fprintf(fp, "%d\n", acc_arr[i]);
    }
    fclose(fp);
}

void build_path(char* out, size_t out_size, const char* base, const char* file_name) {
    size_t base_len = strlen(base);
    const char* sep = (base_len > 0 && (base[base_len - 1] == '/' || base[base_len - 1] == '\\')) ? "" : "/";
    snprintf(out, out_size, "%s%s%s", base, sep, file_name);
}

// ============================================================================
// 3. HÀM MAIN TEST DW_CONV
// ============================================================================
int main() {
    printf("Bat dau khoi tao du lieu cho test DepthwiseConvPerChannel...\n");

    // --- Cấp phát bộ nhớ ---
    int8_t* ifm = (int8_t*)malloc(1 * IFM_HEIGHT * IFM_WIDTH * CHANNELS * sizeof(int8_t));
    int8_t* weight = (int8_t*)malloc(1 * KERNEL_H * KERNEL_W * CHANNELS * sizeof(int8_t)); 
    
    int32_t* effective_bias = (int32_t*)malloc(CHANNELS * sizeof(int32_t));
    int32_t* m_values = (int32_t*)malloc(CHANNELS * sizeof(int32_t));
    int32_t* n_values = (int32_t*)malloc(CHANNELS * sizeof(int32_t)); 
    
    int8_t* ofm = (int8_t*)malloc(1 * OFM_HEIGHT * OFM_WIDTH * CHANNELS * sizeof(int8_t));
    int32_t* acc_out = (int32_t*)malloc(1 * OFM_HEIGHT * OFM_WIDTH * CHANNELS * sizeof(int32_t));

    // --- Đường dẫn file hay thay đổi theo layer ---
    // --- Thay đổi ở đây ---
    const char* param_base_path = "layer025_DEPTHWISE_CONV_2D_1_block4a_dwconv2_1_BiasAdd_1_block4a_dwconv2_1_depthwise_1_block4a_dwconv2_1_Squeeze";
    // Ở quyen's machine
    const char* io_base_path = "input_output/layer025_DEPTHWISE_CONV_2D_1_block4a_dwconv2_1_BiasAdd_1_block4a_dwconv2_1_depthwise_1_block4a_dwconv2_1_Squeeze";
    char ifm_file[512], weight_file[512], eff_bias_file[512], m_file[512], n_file[512];
    char ifm_zp_file[512], ofm_zp_file[512], ofm_out_file[512], acc_out_file[512];

    build_path(ifm_file, sizeof(ifm_file), io_base_path, "ifm_values.txt");
    build_path(weight_file, sizeof(weight_file), param_base_path, "weight_values.txt");
    build_path(eff_bias_file, sizeof(eff_bias_file), param_base_path, "effective_bias.txt"); 
    build_path(m_file, sizeof(m_file), param_base_path, "multiplier.txt");
    build_path(n_file, sizeof(n_file), param_base_path, "shift.txt");
    build_path(ifm_zp_file, sizeof(ifm_zp_file), param_base_path, "ifm_zp.txt");
    build_path(ofm_zp_file, sizeof(ofm_zp_file), param_base_path, "ofm_zp.txt");
    build_path(ofm_out_file, sizeof(ofm_out_file), param_base_path, "ofm_c_sim.txt");
    build_path(acc_out_file, sizeof(acc_out_file), param_base_path, "acc_c_sim.txt");

    // --- Đọc dữ liệu từ file ---
    printf("Doc file cac mang trong so, multiplier, shift...\n");
    read_int8_array_from_file(ifm_file, ifm, IFM_HEIGHT * IFM_WIDTH * CHANNELS);
    read_int8_array_from_file(weight_file, weight, KERNEL_H * KERNEL_W * CHANNELS);
    read_int32_array_from_file(eff_bias_file, effective_bias, CHANNELS);
    read_int32_array_from_file(m_file, m_values, CHANNELS);
    read_int32_array_from_file(n_file, n_values, CHANNELS); 

    int32_t zp_in = read_single_int_from_file(ifm_zp_file, 0);
    int32_t zp_ofm = read_single_int_from_file(ofm_zp_file, 0);
    printf("Thong tin ZP: ZP_IN = %d, ZP_OFM = %d\n", zp_in, zp_ofm);

    // --- Cấu hình Tham số (DepthwiseParams) ---
    DepthwiseParams params;
    params.input_zp = zp_in;  
    params.output_zp = zp_ofm; 
    params.quantized_activation_min = -128; // Cấu hình giới hạn cho relu/none (-128 giả định)
    params.quantized_activation_max = 127; 
    
    params.stride_width = STRIDE_W;
    params.stride_height = STRIDE_H;
    params.pad_width = PAD_W;
    params.pad_height = PAD_H;
    params.dilation_width_factor = 1;
    params.dilation_height_factor = 1;

    // --- GỌI HÀM INFERENCE ---
    printf("Dang thuc hien tinh toan Depthwise Convolution...\n");
    DepthwiseConvPerChannel(
        &params,
        m_values,      
        n_values,      
        1,             // batch size
        IFM_HEIGHT, IFM_WIDTH, CHANNELS, ifm,
        KERNEL_H, KERNEL_W, CHANNELS, weight, 
        CHANNELS, effective_bias, 
        OFM_HEIGHT, OFM_WIDTH, CHANNELS, ofm,
        acc_out
    );

    // --- Ghi kết quả ---
    printf("Ghi ket qua ra file %s...\n", ofm_out_file);
    write_ofm_to_file(ofm_out_file, ofm, OFM_HEIGHT * OFM_WIDTH * CHANNELS);
    
    printf("Ghi acc ra file %s...\n", acc_out_file);
    write_acc_to_file(acc_out_file, acc_out, OFM_HEIGHT * OFM_WIDTH * CHANNELS);

    printf("Hoan thanh test!\n");

    // Dọn dẹp memory
    free(ifm);
    free(weight);
    free(effective_bias);
    free(m_values);
    free(n_values);
    free(ofm);
    free(acc_out);

    return 0;
}