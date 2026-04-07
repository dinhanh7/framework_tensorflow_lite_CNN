#ifndef INFERENCE_HELPER_H
#define INFERENCE_HELPER_H

#include "../layer/add.h"
#include "../layer/conv2d.h"
#include "../layer/dw_conv.h"
#include "../layer/hardswish.h"
#include "../layer/utils.h"
#include "../layer/mean.h"
#include "../layer/mul.h"
#include "../layer/requantize_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ============================================================================
// 1. CẤU TRÚC CACHE VẠN NĂNG (UNIVERSAL LAYER CACHE)
// ============================================================================
typedef struct {
    // Arrays
    int8_t* weights;
    int32_t* effective_bias; 
    int32_t* multipliers;
    int32_t* shifts;

    // Input 0 / Output Quantization Params
    float    ifm_scale;
    int32_t  ifm_zp;
    float    ofm_scale;
    int32_t  ofm_zp;

    // Input 1 / Weight Quantization Params (Lưu cả 2 để giải quyết động tại runtime)
    float    weight_scale;
    int32_t  weight_zp;
    float    ifm1_scale;
    int32_t  ifm1_zp;
} LayerCache;

#define MAX_LAYERS 300
extern LayerCache g_cache[MAX_LAYERS]; 

// ============================================================================
// CÁC HÀM TIỆN ÍCH ĐỌC FILE
// ============================================================================

int file_exists(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (file) {
        fclose(file);
        return 1;
    }
    return 0;
}

static int8_t* cache_read_int8_array(const char* path, int* out_size) {
    if (!file_exists(path)) return NULL;
    int size = count_elements(path);
    if (size == 0) return NULL;
    int8_t* data = (int8_t*)malloc(size * sizeof(int8_t));
    read_int8_array(path, data, size);
    if (out_size) *out_size = size;
    return data;
}

static int32_t* cache_read_int32_array(const char* path) {
    if (!file_exists(path)) return NULL;
    int size = count_elements(path);
    if (size == 0) return NULL;
    int32_t* data = (int32_t*)malloc(size * sizeof(int32_t));
    read_int_array(path, data, size);
    return data;
}

static void cache_read_scalar_float(const char* path, float* target, float default_val) {
    if (file_exists(path)) read_float_array(path, target, 1);
    else *target = default_val;
}

static void cache_read_scalar_int(const char* path, int32_t* target, int32_t default_val) {
    if (file_exists(path)) read_int_array(path, target, 1);
    else *target = default_val;
}

// ============================================================================
// 2. NẠP DỮ LIỆU VÀO CACHE (CHẠY 1 LẦN)
// ============================================================================
void init_single_layer_cache(const char* params_dir, LayerCache* cache) {
    char path_buf[512];

    memset(cache, 0, sizeof(LayerCache));
    if (!params_dir || strlen(params_dir) == 0) return;

    // Arrays
    sprintf(path_buf, "%s/weight_values.txt", params_dir);
    cache->weights = cache_read_int8_array(path_buf, NULL);

    sprintf(path_buf, "%s/effective_bias.txt", params_dir);
    cache->effective_bias = cache_read_int32_array(path_buf);

    sprintf(path_buf, "%s/multiplier.txt", params_dir);
    cache->multipliers = cache_read_int32_array(path_buf);

    sprintf(path_buf, "%s/shift.txt", params_dir);
    cache->shifts = cache_read_int32_array(path_buf);

    // Params Input 0
    sprintf(path_buf, "%s/ifm_scale.txt", params_dir);
    if (file_exists(path_buf)) {
        cache_read_scalar_float(path_buf, &cache->ifm_scale, 0.0f);
        sprintf(path_buf, "%s/ifm_zp.txt", params_dir);
        cache_read_scalar_int(path_buf, &cache->ifm_zp, 0);
    } else {
        sprintf(path_buf, "%s/param_idx0_scale.txt", params_dir);
        cache_read_scalar_float(path_buf, &cache->ifm_scale, 0.0f);
        sprintf(path_buf, "%s/param_idx0_zp.txt", params_dir);
        cache_read_scalar_int(path_buf, &cache->ifm_zp, 0);
    }

    // Params Output
    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    cache_read_scalar_float(path_buf, &cache->ofm_scale, 0.0f);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    cache_read_scalar_int(path_buf, &cache->ofm_zp, 0);

    // ĐỌC ĐỒNG THỜI CẢ 2 LOẠI SCALE INPUT 1 (Phục vụ giải quyết cờ ở Runtime)
    sprintf(path_buf, "%s/weight_scale.txt", params_dir);
    cache_read_scalar_float(path_buf, &cache->weight_scale, 0.0f);
    sprintf(path_buf, "%s/weight_zp.txt", params_dir);
    cache_read_scalar_int(path_buf, &cache->weight_zp, 0);

    sprintf(path_buf, "%s/ifm_1_scale.txt", params_dir);
    cache_read_scalar_float(path_buf, &cache->ifm1_scale, 0.0f);
    sprintf(path_buf, "%s/ifm_1_zp.txt", params_dir);
    cache_read_scalar_int(path_buf, &cache->ifm1_zp, 0);
}

// ============================================================================
// 3. CÁC HÀM INFERENCE ENGINES
// ============================================================================

int8_t* run_conv2d_layer(LayerCache* cache, const char* input_dir, int8_t* ifm_data, 
                         int ifm_h, int ifm_w, int ifm_c, int ofm_c, int kernel_h, 
                         int kernel_w, int stride_h, int stride_w, const char* padding_type, 
                         int* out_h_check, int* out_w_check) 
{
    int ofm_height, ofm_width;
    if (strcmp(padding_type, "SAME") == 0) {
        ofm_height = (ifm_h + stride_h - 1) / stride_h;
        ofm_width = (ifm_w + stride_w - 1) / stride_w;
    } else { 
        ofm_height = (ifm_h - kernel_h) / stride_h + 1;
        ofm_width = (ifm_w - kernel_w) / stride_w + 1;
    }
    *out_h_check = ofm_height;
    *out_w_check = ofm_width;

    int max_ofm_size = ofm_height * ofm_width * ofm_c; 
    int8_t* ofm_data = (int8_t*)calloc(max_ofm_size, sizeof(int8_t));

    int8_t* output_shifts = (int8_t*)calloc(ofm_c, sizeof(int8_t));
    if (cache->shifts) {
        for(int i = 0; i < ofm_c; i++) output_shifts[i] = (int8_t)cache->shifts[i];
    }

    if(ifm_data == NULL){
        printf("Input data for Conv2d is NULL, read from file instead.\n");
        int ifm_size = ifm_h * ifm_w * ifm_c;
        ifm_data = (int8_t*)malloc(ifm_size * sizeof(int8_t));
        char ifm_path[512];
        sprintf(ifm_path, "%s/ifm.txt", input_dir);
        read_int8_array(ifm_path, ifm_data, ifm_size);
    }

    quantized_conv2d(
        ofm_data,
        out_h_check,
        out_w_check,
        ifm_data,
        ifm_h, ifm_w, ifm_c,
        cache->weights,
        ofm_c, kernel_h, kernel_w,
        cache->effective_bias,
        cache->multipliers,
        output_shifts,
        (int8_t)cache->ifm_zp,
        (int8_t)cache->ofm_zp,
        stride_h, stride_w,
        (char*)padding_type
    );

    free(output_shifts);
    return ofm_data;
}

int8_t* run_hardswish_layer(LayerCache* cache, const char* input_dir, int8_t* input_data, int* size) {
    int8_t* output_data = (int8_t*)calloc(*size, sizeof(int8_t));

    if(input_data == NULL){
        printf("Input data for HardSwish is NULL, read from file instead.\n");
        int ifm_size = *size;
        input_data = (int8_t*)malloc(ifm_size * sizeof(int8_t));
        char ifm_path[512];
        sprintf(ifm_path, "%s/ifm_0.txt", input_dir);
        read_int8_array(ifm_path, input_data, ifm_size);
    }

    quantized_hardswish(
        input_data,
        output_data,
        *size,
        (int8_t)cache->ifm_zp,
        (int8_t)cache->ofm_zp,
        cache->ifm_scale,
        cache->ofm_scale
    );

    return output_data;
}

int8_t* run_dw_conv_layer(LayerCache* cache, const char* input_dir, int8_t* ifm_data, 
                          int ifm_h, int ifm_w, int ifm_c, int ofm_c, int kernel_h, int kernel_w, 
                          int stride_h, int stride_w, const char* padding_type, 
                          int* out_h_check, int* out_w_check) 
{
    int ofm_height, ofm_width;
    if (strcmp(padding_type, "SAME") == 0) {
        ofm_height = (ifm_h + stride_h - 1) / stride_h;
        ofm_width = (ifm_w + stride_w - 1) / stride_w;
    } else { 
        ofm_height = (ifm_h - kernel_h) / stride_h + 1;
        ofm_width = (ifm_w - kernel_w) / stride_w + 1;
    }
    *out_h_check = ofm_height;
    *out_w_check = ofm_width;

    int max_ofm_size = ofm_height * ofm_width * ofm_c; 
    int8_t* ofm_data = (int8_t*)calloc(max_ofm_size, sizeof(int8_t));

    DepthwiseParams dw_params = {
        .input_zp = cache->ifm_zp,
        .output_zp = cache->ofm_zp,
        .quantized_activation_min = -128,
        .quantized_activation_max = 127,
        .stride_width = stride_w,
        .stride_height = stride_h,
        .pad_width = 0, 
        .pad_height = 0,
        .dilation_width_factor = 1,
        .dilation_height_factor = 1
    };

    DepthwiseConvPerChannel(
        &dw_params,
        cache->multipliers,
        cache->shifts, 
        1,
        ifm_h, ifm_w, ifm_c, ifm_data,
        kernel_h, kernel_w, ofm_c, cache->weights,
        ofm_c, cache->effective_bias,
        ofm_height, ofm_width, ofm_c, ofm_data,
        NULL
    );

    return ofm_data;
}

int8_t* run_add_layer(LayerCache* cache, const char* input_dir, 
                      int8_t* input0, int ifm0_size, 
                      int8_t* input1, int ifm1_size) 
{
    int max_size = (ifm0_size > ifm1_size) ? ifm0_size : ifm1_size;
    int8_t* output_data = (int8_t*)calloc(max_size, sizeof(int8_t));

    // Runtime Resolve Params (Giống helper_read_input1_quant_params gốc)
    float actual_ifm1_scale = (cache->ifm1_scale != 0.0f) ? cache->ifm1_scale : cache->weight_scale;
    int32_t actual_ifm1_zp = (cache->ifm1_scale != 0.0f) ? cache->ifm1_zp : cache->weight_zp;

    struct AddArithmeticParams params;
    CalculateAddArithmeticParams(cache->ifm_scale, -cache->ifm_zp, 
                                 actual_ifm1_scale, -actual_ifm1_zp, 
                                 cache->ofm_scale, cache->ofm_zp, &params);
    
    struct RuntimeShape shape = {1, (int32_t[]){max_size}};
    Add(&params, &shape, input0, &shape, input1, &shape, output_data);

    return output_data;
}

int8_t* run_mul_layer(LayerCache* cache, const char* input_dir, 
                      int8_t* input0, int ifm0_size, 
                      int8_t* input1, int ifm1_size, int use_ifm_1_naming) 
{
    // Cờ use_ifm_1_naming điều hướng chính xác như code model.c ban đầu
    float actual_ifm1_scale = use_ifm_1_naming ? cache->ifm1_scale : cache->weight_scale;
    int32_t actual_ifm1_zp = use_ifm_1_naming ? cache->ifm1_zp : cache->weight_zp;

    MulArithmeticParams mul_params;
    int32_t quantized_multiplier = 0;
    int shift = 0;

    // Tính toán lượng tử hóa ngay tại runtime (chiếm vài nano giây)
    if (cache->ifm_scale != 0.0f && actual_ifm1_scale != 0.0f && cache->ofm_scale != 0.0f) {
        float real_multiplier = (cache->ifm_scale * actual_ifm1_scale) / cache->ofm_scale;
        QuantizeMultiplier(real_multiplier, &quantized_multiplier, &shift);
    }

    mul_params.output_multiplier = quantized_multiplier;
    mul_params.output_shift = shift;
    mul_params.input1_offset = -cache->ifm_zp;
    mul_params.input2_offset = -actual_ifm1_zp;
    mul_params.output_offset = cache->ofm_zp;
    mul_params.quantized_activation_min = -128;
    mul_params.quantized_activation_max = 127;

    int8_t* actual_input1 = input1 ? input1 : cache->weights; 

    int8_t *output_data = (int8_t*)calloc(ifm0_size, sizeof(int8_t));
    BroadcastMulInt8(&mul_params, input0, ifm0_size, actual_input1, ifm1_size, output_data);

    return output_data;
}

int8_t* run_mean_layer(LayerCache* cache, const char* input_dir, int8_t* input_data, 
                       int ifm_h, int ifm_w, int ifm_c, int batch_size, int use_input_file) 
{
    int ofm_size = batch_size * ifm_c; 
    int8_t* output_data = (int8_t*)calloc(ofm_size, sizeof(int8_t));
    if(input_data == NULL && use_input_file){
        printf("Input data for Mean is NULL, read from file instead.\n");
        int ifm_size = ifm_h * ifm_w * ifm_c;
        input_data = (int8_t*)malloc(ifm_size * sizeof(int8_t));
        char ifm_path[512];
        sprintf(ifm_path, "%s/ifm_0.txt", input_dir);
        read_int8_array(ifm_path, input_data, ifm_size);
    }
    quantized_mean_int8(
        input_data,
        output_data,
        batch_size,
        ifm_h,
        ifm_w,
        ifm_c,      
        ifm_c,      
        cache->ifm_scale,
        cache->ifm_zp,
        cache->ofm_scale,
        cache->ofm_zp
    );

    return output_data;
}
// Hàm so sánh trực tiếp bộ nhớ RAM với file Golden, không lưu thêm file phụ
void compare_debug_only(const char* layer_name, int8_t* c_output, int size, const char* golden_file_path) {
    if (golden_file_path == NULL || strlen(golden_file_path) == 0) {
        printf("\n[LOI] Chua truyen duong dan file Golden!\n");
        exit(1);
    }

    FILE *f_golden = fopen(golden_file_path, "r");
    if (!f_golden) {
        printf("\n[CANH BAO] Khong tim thay file golden tai: %s\n", golden_file_path);
        exit(1);
    } 
    
    int8_t* golden_data = (int8_t*)malloc(size * sizeof(int8_t));
    int temp_val;
    int elements_read = 0;
    
    // Đọc file text TFLite vào RAM
    while(fscanf(f_golden, "%d", &temp_val) == 1 && elements_read < size) {
        golden_data[elements_read] = (int8_t)temp_val;
        elements_read++;
    }
    fclose(f_golden);

    if (elements_read != size) {
        printf("\n[CANH BAO] File golden có %d phần tử (Của C yêu cầu %d phần tử)\n", elements_read, size);
    } else {
        // So sánh toán học trên RAM
        int max_diff = 0;
        int error_count_all = 0; 
        int error_count_gt2 = 0; 

        for (int i = 0; i < size; ++i) {
            int diff = abs((int)c_output[i] - (int)golden_data[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > 0) error_count_all++;
            if (diff > 2) error_count_gt2++;
        }

        // Báo cáo
        printf("\n==================================================\n");
        printf(" KET QUA SO SANH LAYER [%s]:\n", layer_name);
        printf("--------------------------------------------------\n");
        printf(" - Sai so lon nhat (Max Diff) : %d\n", max_diff);
        printf(" - So phan tu lech (> 0)      : %d / %d (%.2f%%)\n", error_count_all, size, (float)error_count_all/size*100);
        printf(" - So phan tu lech > 2        : %d / %d (%.2f%%)\n", error_count_gt2, size, (float)error_count_gt2/size*100);
        
        if (max_diff <= 1) { 
            printf(" -> CHUAN XAC TUYET DOI! (Sai so %d don vi do lam tron float)\n", max_diff);
        } else {
            printf(" -> CO LOI TOAN HOC / LUONG TU HOA TAI LAYER NAY!\n");
        }
        printf("==================================================\n");
    }
    free(golden_data);
    
    // printf("=== DUNG CHUONG TRINH DE DEBUG (exit 0) ===\n");
    // exit(0); 
}
#endif