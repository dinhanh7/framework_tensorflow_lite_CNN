#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../layer/conv2d.h"

// Define constants for this specific layer test (Layer 5: Stem Conv)
#define IFM_H 56
#define IFM_W 56
#define IFM_C 128
#define KERNEL_H 1
#define KERNEL_W 1
#define STRIDE_H 1
#define STRIDE_W 1
#define OFM_C 32
#define PADDING_TYPE "SAME"





int main() {
    printf("Starting CONV_2D Test (Layer 5)...\n");

    // 1. Allocate scaling factors and quantization params
    float ifm_scale, ofm_scale;
    int32_t ifm_zp, ofm_zp;
    float* weight_scales = (float*)malloc(OFM_C * sizeof(float));
    int32_t* biases = (int32_t*)malloc(OFM_C * sizeof(int32_t));
    int32_t* effective_biases = (int32_t*)malloc(OFM_C * sizeof(int32_t));
    int32_t* output_multipliers = (int32_t*)malloc(OFM_C * sizeof(int32_t));
    int8_t* output_shifts = (int8_t*)malloc(OFM_C * sizeof(int8_t));

    // 2. Read Quantization Parameters
    // Paths assumed correct relative to execution dir
    char params_dir[] = "extracted_params_hsigmoid/layer014_CONV_2D_1_block2b_project_conv_1_BiasAdd_1_block2b_project_conv_1_convo";
    char input_dir[] = "all_layer_io/layer_14_CONV_2D";
    char path_buf[512];

    // Read scalars
    sprintf(path_buf, "%s/ifm_scale.txt", params_dir);
    read_float_array(path_buf, &ifm_scale, 1);
    
    sprintf(path_buf, "%s/ifm_zp.txt", params_dir);
    read_int_array(path_buf, &ifm_zp, 1);

    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_float_array(path_buf, &ofm_scale, 1);

    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    // Read arrays
    sprintf(path_buf, "%s/weight_scale.txt", params_dir);
    read_float_array(path_buf, weight_scales, OFM_C);

    sprintf(path_buf, "%s/bias_value.txt", params_dir);
    read_int_array(path_buf, biases, OFM_C);

    // 3. Load input Data and Weights
    int ifm_size = IFM_H * IFM_W * IFM_C;
    int8_t* ifm_data = (int8_t*)malloc(ifm_size * sizeof(int8_t));
    sprintf(path_buf, "%s/ifm.txt", input_dir);
    read_int8_array(path_buf, ifm_data, ifm_size);

    int weight_size = OFM_C * KERNEL_H * KERNEL_W * IFM_C;
    int8_t* weight_data = (int8_t*)malloc(weight_size * sizeof(int8_t));
    sprintf(path_buf, "%s/weight_values.txt", params_dir);
    read_int8_array(path_buf, weight_data, weight_size);

    // 4. Compute Derived Parameters (Effective Bias, Multiplier, Shift)
    printf("Computing quantization parameters...\n");
    for (int oc = 0; oc < OFM_C; oc++) {
        // Calculate Effective Bias
        // formula: effective_bias = bias - ifm_zp * sum(weight)
        int32_t weight_sum = 0;
        for (int kh = 0; kh < KERNEL_H; kh++) {
            for (int kw = 0; kw < KERNEL_W; kw++) {
                for (int ic = 0; ic < IFM_C; ic++) {
                    // Weight layout: [OFM, H, W, IFM]
                    // Index = oc * (H*W*IC) + kh * (W*IC) + kw * IC + ic
                    int idx = oc * (KERNEL_H * KERNEL_W * IFM_C) + 
                              kh * (KERNEL_W * IFM_C) + 
                              kw * IFM_C + ic;
                    weight_sum += weight_data[idx];
                }
            }
        }
        effective_biases[oc] = biases[oc] - (int8_t)ifm_zp * weight_sum; // Be careful with sign of ifm_zp. 
        // Note: In TFLite quantization, (input - zp) * weight. 
        // = input * weight - zp * weight. 
        // So we SUBTRACT zp * sum(weight). Correct.

        // Calculate Multiplier and Shift
        double double_multiplier = (double)ifm_scale * (double)weight_scales[oc] / (double)ofm_scale;
        int temp_shift = 0;
        QuantizeMultiplier(double_multiplier, &output_multipliers[oc], &temp_shift);
        output_shifts[oc] = (int8_t)temp_shift;
        if (oc == 0) {
            printf("DEBUG: oc=0, double_multiplier=%f, multiplier=%d, shift=%d\\n", double_multiplier, output_multipliers[0], temp_shift);
        }
    }

    // 5. Prepare Output Buffer
    // Assuming SAME padding and stride 2, output size is ceil(input/stride)
    int out_h_check, out_w_check; // To capture output from function
    int max_ofm_size = IFM_H * IFM_W * OFM_C; // Overallocate to be safe
    int8_t* ofm_data = (int8_t*)malloc(max_ofm_size * sizeof(int8_t));

    // 6. Run Quantized Conv2D
    printf("Running quantized_conv2d...\n");
    quantized_conv2d(
        ofm_data,
        &out_h_check,
        &out_w_check,
        ifm_data,
        IFM_H, IFM_W, IFM_C,
        weight_data,
        OFM_C, KERNEL_H, KERNEL_W,
        effective_biases,
        output_multipliers,
        output_shifts,
        (int8_t)ifm_zp,
        (int8_t)ofm_zp,
        STRIDE_H, STRIDE_W,
        PADDING_TYPE
    );

    printf("Inference done. Output shape: %d x %d x %d\n", out_h_check, out_w_check, OFM_C);

    // 7. Save Result to Params Dir
    sprintf(path_buf, "%s/ofm_c_sim.txt", params_dir);
    FILE* f_out = fopen(path_buf, "w");
    if (f_out) {
        int total_count = out_h_check * out_w_check * OFM_C;
        for (int i = 0; i < total_count; i++) {
            fprintf(f_out, "%d\n", ofm_data[i]);
        }
        fclose(f_out);
        printf("Result saved to: %s\n", path_buf);
    } else {
        printf("Error: Could not open file for writing: %s\n", path_buf);
    }

    // Cleanup
    free(ifm_data);
    free(weight_data);
    free(ofm_data);
    // free(expected_ofm); // Removed
    free(weight_scales);
    free(biases);
    free(effective_biases);
    free(output_multipliers);
    free(output_shifts);

    return 0;
}
