#ifndef LAYER_HELPER_H
#define LAYER_HELPER_H
#include "add.h"
#include "conv2d.h"
#include "dw_conv.h"
#include "hardswish.h"
#include "utils.h"
#include "mean.h"
#include "mul.h"
#include "requantize_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int helper_file_exists(const char* path) {
    FILE* file = fopen(path, "r");
    if (!file) {
        return 0;
    }
    fclose(file);
    return 1;
}

static void helper_read_input_quant_params(const char* params_dir, float* ifm_scale, int32_t* ifm_zp) {
    char path_buf[512];

    sprintf(path_buf, "%s/ifm_scale.txt", params_dir);
    if (helper_file_exists(path_buf)) {
        read_float_array(path_buf, ifm_scale, 1);
        sprintf(path_buf, "%s/ifm_zp.txt", params_dir);
        read_int_array(path_buf, ifm_zp, 1);
        return;
    }

    sprintf(path_buf, "%s/param_idx0_scale.txt", params_dir);
    read_float_array(path_buf, ifm_scale, 1);
    sprintf(path_buf, "%s/param_idx0_zp.txt", params_dir);
    read_int_array(path_buf, ifm_zp, 1);
}

static int helper_has_ifm0_quant_params(const char* params_dir) {
    char path_buf[512];

    sprintf(path_buf, "%s/ifm_scale.txt", params_dir);
    return helper_file_exists(path_buf);
}

static int helper_has_ifm1_quant_params(const char* params_dir) {
    char path_buf[512];

    sprintf(path_buf, "%s/ifm_1_scale.txt", params_dir);
    return helper_file_exists(path_buf);
}

static void helper_read_input1_quant_params(const char* params_dir, float* ifm1_scale, int32_t* ifm1_zp) {
    char path_buf[512];

    if (helper_has_ifm1_quant_params(params_dir)) {
        sprintf(path_buf, "%s/ifm_1_scale.txt", params_dir);
        read_float_array(path_buf, ifm1_scale, 1);
        sprintf(path_buf, "%s/ifm_1_zp.txt", params_dir);
        read_int_array(path_buf, ifm1_zp, 1);
        return;
    }

    sprintf(path_buf, "%s/weight_scale.txt", params_dir);
    read_float_array(path_buf, ifm1_scale, 1);
    sprintf(path_buf, "%s/weight_zp.txt", params_dir);
    read_int_array(path_buf, ifm1_zp, 1);
}

int8_t* run_hardswish_layer(const char* params_dir, const char* input_dir, int8_t* input_data_override, int* out_size) {
    char path_buf[512];
    float ifm_scale, ofm_scale;
    int32_t ifm_zp, ofm_zp;

    helper_read_input_quant_params(params_dir, &ifm_scale, &ifm_zp);

    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_float_array(path_buf, &ofm_scale, 1);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    int size;
    int8_t* input_data;
    if (input_data_override) {
        // Size must have been provided or assumed same, reading from ifm_0 just for size info but not data
        sprintf(path_buf, "%s/ifm_0.txt", input_dir);
        size = count_elements(path_buf);
        input_data = input_data_override;
    } else {
        sprintf(path_buf, "%s/ifm_0.txt", input_dir);
        size = count_elements(path_buf);
        input_data = (int8_t*)malloc(size * sizeof(int8_t));
        read_int8_array(path_buf, input_data, size);
    }
    
    if (out_size) *out_size = size;

    int8_t* output_data = (int8_t*)malloc(size * sizeof(int8_t));

    quantized_hardswish(
        input_data,
        output_data,
        size,
        (int8_t)ifm_zp,
        (int8_t)ofm_zp,
        ifm_scale,
        ofm_scale
    );

    if (!input_data_override) free(input_data);
    return output_data;
}

int8_t* run_conv2d_layer(const char* params_dir, const char* input_dir, int8_t* ifm_data, int ifm_h, int ifm_w, int ifm_c, int ofm_c, int kernel_h, int kernel_w, int stride_h, int stride_w, const char* padding_type, int* out_h_check, int* out_w_check) {
    char path_buf[512];
    float ifm_scale, ofm_scale;
    int32_t ifm_zp, ofm_zp;
    float* weight_scales = (float*)malloc(ofm_c * sizeof(float));
    int32_t* biases = (int32_t*)malloc(ofm_c * sizeof(int32_t));
    int32_t* effective_biases = (int32_t*)malloc(ofm_c * sizeof(int32_t));
    int32_t* output_multipliers = (int32_t*)malloc(ofm_c * sizeof(int32_t));
    int8_t* output_shifts = (int8_t*)malloc(ofm_c * sizeof(int8_t));

    sprintf(path_buf, "%s/ifm_scale.txt", params_dir);
    read_float_array(path_buf, &ifm_scale, 1);
    sprintf(path_buf, "%s/ifm_zp.txt", params_dir);
    read_int_array(path_buf, &ifm_zp, 1);
    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_float_array(path_buf, &ofm_scale, 1);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    sprintf(path_buf, "%s/weight_scale.txt", params_dir);
    read_float_array(path_buf, weight_scales, ofm_c);
    sprintf(path_buf, "%s/bias_value.txt", params_dir);
    read_int_array(path_buf, biases, ofm_c);
    sprintf(path_buf, "%s/effective_bias.txt", params_dir);
    read_int_array(path_buf, effective_biases, ofm_c);
    sprintf(path_buf, "%s/multiplier.txt", params_dir);
    read_int_array(path_buf, output_multipliers, ofm_c);
    sprintf(path_buf, "%s/shift.txt", params_dir);
    read_int8_array(path_buf, output_shifts, ofm_c);

    int weight_size = ofm_c * kernel_h * kernel_w * ifm_c;
    int8_t* weight_data = (int8_t*)malloc(weight_size * sizeof(int8_t));
    sprintf(path_buf, "%s/weight_values.txt", params_dir);
    read_int8_array(path_buf, weight_data, weight_size);

    int max_ofm_size = ifm_h * ifm_w * ofm_c; 
    int8_t* ofm_data = (int8_t*)malloc(max_ofm_size * sizeof(int8_t));

    if(ifm_data == NULL) {
        printf("No IFM data provided, attempting to read from file...\n");
        ifm_data = (int8_t*)malloc(ifm_h * ifm_w * ifm_c * sizeof(int8_t));
        sprintf(path_buf, "%s/ifm.txt", input_dir);
        read_int8_array(path_buf, ifm_data, ifm_h * ifm_w * ifm_c);
    }

    quantized_conv2d(
        ofm_data,
        out_h_check,
        out_w_check,
        ifm_data,
        ifm_h, ifm_w, ifm_c,
        weight_data,
        ofm_c, kernel_h, kernel_w,
        effective_biases,
        output_multipliers,
        output_shifts,
        (int8_t)ifm_zp,
        (int8_t)ofm_zp,
        stride_h, stride_w,
        (char*)padding_type
    );

    free(weight_data);
    free(weight_scales);
    free(biases);
    free(effective_biases);
    free(output_multipliers);
    free(output_shifts);

    return ofm_data;
}

int8_t* run_mul_layer(const char* params_dir, const char* input_dir, int8_t* input0_override, int ifm0_size, int8_t* input1_override, int ifm1_size, int use_ifm_1_naming) {
    char path_buf[512];
    float ifm0_scale, ifm1_scale, ofm_scale;
    int32_t ifm0_zp, ifm1_zp, ofm_zp;

    helper_read_input_quant_params(params_dir, &ifm0_scale, &ifm0_zp);

    if (use_ifm_1_naming) {
        sprintf(path_buf, "%s/ifm_1_scale.txt", params_dir);
        read_float_array(path_buf, &ifm1_scale, 1);
        sprintf(path_buf, "%s/ifm_1_zp.txt", params_dir);
        read_int_array(path_buf, &ifm1_zp, 1);
    } else {
        sprintf(path_buf, "%s/weight_scale.txt", params_dir);
        read_float_array(path_buf, &ifm1_scale, 1);
        sprintf(path_buf, "%s/weight_zp.txt", params_dir);
        read_int_array(path_buf, &ifm1_zp, 1);
    }

    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_float_array(path_buf, &ofm_scale, 1);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    int8_t* input0_data;
    if (input0_override) {
        input0_data = input0_override;
    } else {
        input0_data = (int8_t*)malloc(ifm0_size * sizeof(int8_t));
        sprintf(path_buf, "%s/ifm_0.txt", input_dir);
        if (!helper_file_exists(path_buf)) {
            sprintf(path_buf, "%s/param_idx0_values.txt", params_dir);
        }
        read_int8_array(path_buf, input0_data, ifm0_size);
    }

    int8_t* input1_data;
    if (input1_override) {
        input1_data = input1_override;
    } else {
        input1_data = (int8_t*)malloc(ifm1_size * sizeof(int8_t));
        if (use_ifm_1_naming) {
             sprintf(path_buf, "%s/ifm_1.txt", input_dir); // Normally from input
        } else {
             sprintf(path_buf, "%s/weight_values.txt", params_dir);
        }
        read_int8_array(path_buf, input1_data, ifm1_size);
    }

    MulArithmeticParams mul_params;
    float real_multiplier = (ifm0_scale * ifm1_scale) / ofm_scale;
    int32_t quantized_multiplier;
    int shift;
    QuantizeMultiplier(real_multiplier, &quantized_multiplier, &shift);

    mul_params.output_multiplier = quantized_multiplier;
    mul_params.output_shift = shift;
    mul_params.input1_offset = -ifm0_zp;
    mul_params.input2_offset = -ifm1_zp;
    mul_params.output_offset = ofm_zp;
    mul_params.quantized_activation_min = -128;
    mul_params.quantized_activation_max = 127;

    int8_t *output_data = (int8_t*)malloc(ifm0_size * sizeof(int8_t));
    BroadcastMulInt8(&mul_params, input0_data, ifm0_size, input1_data, ifm1_size, output_data);

    if (!input0_override) free(input0_data);
    if (!input1_override) free(input1_data);

    return output_data;
}


int8_t* run_add_layer(const char* params_dir, const char* input_dir, int8_t* input0_override, int ifm0_size, int8_t* input1_override, int ifm1_size) {
    char path_buf[512];
    float ifm0_scale, ifm1_scale, ofm_scale;
    int32_t ifm0_zp, ifm1_zp, ofm_zp;

    helper_read_input_quant_params(params_dir, &ifm0_scale, &ifm0_zp);
    helper_read_input1_quant_params(params_dir, &ifm1_scale, &ifm1_zp);

    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_float_array(path_buf, &ofm_scale, 1);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    int8_t* input0_data;
    if (input0_override) {
        input0_data = input0_override;
    } else {
        input0_data = (int8_t*)malloc(ifm0_size * sizeof(int8_t));
        sprintf(path_buf, "%s/ifm_0.txt", input_dir);
        read_int8_array(path_buf, input0_data, ifm0_size);
    }

    int8_t* input1_data;
    if (input1_override) {
        input1_data = input1_override;
    } else {
        input1_data = (int8_t*)malloc(ifm1_size * sizeof(int8_t));
        if (helper_has_ifm1_quant_params(params_dir)) {
            sprintf(path_buf, "%s/ifm_1.txt", input_dir);
        } else {
            sprintf(path_buf, "%s/weight_values.txt", params_dir);
        }
        read_int8_array(path_buf, input1_data, ifm1_size);
    }

    int max_size = (ifm0_size > ifm1_size) ? ifm0_size : ifm1_size;
    int8_t* output_data = (int8_t*)malloc(max_size * sizeof(int8_t));

    struct AddArithmeticParams params;
    CalculateAddArithmeticParams(ifm0_scale, -ifm0_zp, ifm1_scale, -ifm1_zp, ofm_scale, ofm_zp, &params);
    
    struct RuntimeShape shape = {1, (int32_t[]){max_size}};
    Add(&params, &shape, input0_data, &shape, input1_data, &shape, output_data);

    if (!input0_override) free(input0_data);
    if (!input1_override) free(input1_data);
    return output_data;
}
int8_t* run_dw_conv_layer(const char* params_dir, const char* input_dir, int8_t* ifm_data_override, int ifm_h, int ifm_w, int ifm_c, int ofm_c, int kernel_h, int kernel_w, int stride_h, int stride_w, const char* padding_type, int* out_h_check, int* out_w_check) {
    char path_buf[512];
    float ifm_scale, ofm_scale;
    int32_t ifm_zp, ofm_zp;
    float* weight_scales = (float*)malloc(ofm_c * sizeof(float));
    int32_t* biases = (int32_t*)malloc(ofm_c * sizeof(int32_t));
    int32_t* effective_biases = (int32_t*)malloc(ofm_c * sizeof(int32_t));
    int32_t* output_multipliers = (int32_t*)malloc(ofm_c * sizeof(int32_t));
    int32_t* output_shifts_int32 = (int32_t*)malloc(ofm_c * sizeof(int32_t)); // DW conv use int32 shift
    int8_t* output_shifts = (int8_t*)malloc(ofm_c * sizeof(int8_t));

    sprintf(path_buf, "%s/ifm_scale.txt", params_dir);
    read_float_array(path_buf, &ifm_scale, 1);
    sprintf(path_buf, "%s/ifm_zp.txt", params_dir);
    read_int_array(path_buf, &ifm_zp, 1);
    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_float_array(path_buf, &ofm_scale, 1);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    sprintf(path_buf, "%s/weight_scale.txt", params_dir);
    read_float_array(path_buf, weight_scales, ofm_c);
    sprintf(path_buf, "%s/bias_value.txt", params_dir);
    read_int_array(path_buf, biases, ofm_c);
    sprintf(path_buf, "%s/effective_bias.txt", params_dir);
    read_int_array(path_buf, effective_biases, ofm_c);
    sprintf(path_buf, "%s/multiplier.txt", params_dir);
    read_int_array(path_buf, output_multipliers, ofm_c);
    sprintf(path_buf, "%s/shift.txt", params_dir);
    read_int_array(path_buf, output_shifts_int32, ofm_c);
    for (int i=0; i<ofm_c; i++) output_shifts[i] = (int8_t)output_shifts_int32[i];

    int weight_size = 1 * kernel_h * kernel_w * ofm_c; // depthwise weights shape [1, h, w, c]
    int8_t* weight_data = (int8_t*)malloc(weight_size * sizeof(int8_t));
    sprintf(path_buf, "%s/weight_values.txt", params_dir);
    read_int8_array(path_buf, weight_data, weight_size);

    if(ifm_data_override == NULL) {
        printf("No IFM data provided, attempting to read from file...\n");
        ifm_data_override = (int8_t*)malloc(ifm_h * ifm_w * ifm_c * sizeof(int8_t));
        sprintf(path_buf, "%s/ifm.txt", input_dir);
        read_int8_array(path_buf, ifm_data_override, ifm_h * ifm_w * ifm_c);
    }

    int ofm_height, ofm_width;
    if (strcmp(padding_type, "SAME") == 0) {
        ofm_height = (ifm_h + stride_h - 1) / stride_h;
        ofm_width = (ifm_w + stride_w - 1) / stride_w;
    } else { // "VALID"
        ofm_height = (ifm_h - kernel_h) / stride_h + 1;
        ofm_width = (ifm_w - kernel_w) / stride_w + 1;
    }
    
    *out_h_check = ofm_height;
    *out_w_check = ofm_width;

    int max_ofm_size = ofm_height * ofm_width * ofm_c; 
    int8_t* ofm_data = (int8_t*)malloc(max_ofm_size * sizeof(int8_t));

    DepthwiseParams dw_params = {
        .input_zp = ifm_zp,
        .output_zp = ofm_zp,
        .quantized_activation_min = -128,
        .quantized_activation_max = 127,
        .stride_width = stride_w,
        .stride_height = stride_h,
        .pad_width = 0, // Padding will be computed inside
        .pad_height = 0,
        .dilation_width_factor = 1,
        .dilation_height_factor = 1
    };

    DepthwiseConvPerChannel(
        &dw_params,
        output_multipliers,
        output_shifts_int32,
        1,
        ifm_h, ifm_w, ifm_c, ifm_data_override,
        kernel_h, kernel_w, ofm_c, weight_data, // Depthwise filter channel
        ofm_c, effective_biases,
        ofm_height, ofm_width, ofm_c, ofm_data,
        NULL
    );

    free(weight_data);
    free(weight_scales);
    free(biases);
    free(effective_biases);
    free(output_multipliers);
    free(output_shifts_int32);
    free(output_shifts);

    return ofm_data;
}

int8_t* run_mean_layer(const char* params_dir, const char* input_dir, int8_t* input_data_override, int ifm_h, int ifm_w, int ifm_c, int batch_size, int use_input_file) {
    char path_buf[512];
    float ifm_scale, ofm_scale;
    int32_t ifm_zp, ofm_zp;

    helper_read_input_quant_params(params_dir, &ifm_scale, &ifm_zp);

    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_float_array(path_buf, &ofm_scale, 1);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    int ifm_size = batch_size * ifm_h * ifm_w * ifm_c;
    int8_t* input_data;
    
    if (input_data_override) {
        input_data = input_data_override;
    } else {
        if (!use_input_file) {
            fprintf(stderr, "run_mean_layer: missing input buffer for %s\n", params_dir);
            return NULL;
        }
        input_data = (int8_t*)malloc(ifm_size * sizeof(int8_t));
        if (use_input_file) {
            sprintf(path_buf, "%s/ifm_0.txt", input_dir);
            read_int8_array(path_buf, input_data, ifm_size);
        }
    }

    int ofm_size = batch_size * ifm_c; // output is [batch, channels] due to global average pooling
    int8_t* output_data = (int8_t*)malloc(ofm_size * sizeof(int8_t));

    quantized_mean_int8(
        input_data,
        output_data,
        batch_size,
        ifm_h,
        ifm_w,
        ifm_c,      // input_channels
        ifm_c,      // output_channels
        ifm_scale,
        ifm_zp,
        ofm_scale,
        ofm_zp
    );

    if (!input_data_override) free(input_data);
    return output_data;
}
#endif
