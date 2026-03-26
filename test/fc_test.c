#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../layer/fc.h"  // Assuming internal include structure works
#include "../layer/requantize_utils.h"

#define MAX_PATH 1024

// Paths to folders
const char* PARAMS_DIR = "c:\\Users\\84333\\projects\\framework_tensorflow_lite_CNN\\extracted_params\\layer279_FULLY_CONNECTED_1_predictions_1_MatMul_1_predictions_1_BiasAdd";
const char* IO_DIR = "c:\\Users\\84333\\projects\\framework_tensorflow_lite_CNN\\all_layer_io\\layer_279_FULLY_CONNECTED";

// Helper to count elements in file
int count_elements(const char* filepath) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        printf("Failed to open %s\n", filepath);
        return -1;
    }
    int count = 0;
    float temp; // Use float to generic read, works for int too
    while (fscanf(f, "%f", &temp) == 1) {
        count++;
    }
    fclose(f);
    return count;
}

// Helper to read float data
float* read_floats(const char* filepath, int count) {
    FILE* f = fopen(filepath, "r");
    if (!f) return NULL;
    float* data = (float*)malloc(count * sizeof(float));
    for (int i = 0; i < count; i++) {
        fscanf(f, "%f", &data[i]);
    }
    fclose(f);
    return data;
}

// Helper to read int32 data
int32_t* read_int32(const char* filepath, int count) {
    FILE* f = fopen(filepath, "r");
    if (!f) return NULL;
    int32_t* data = (int32_t*)malloc(count * sizeof(int32_t));
    for (int i = 0; i < count; i++) {
        fscanf(f, "%d", &data[i]);
    }
    fclose(f);
    return data;
}

// Helper to read int8 data
int8_t* read_int8(const char* filepath, int count) {
    FILE* f = fopen(filepath, "r");
    if (!f) return NULL;
    int8_t* data = (int8_t*)malloc(count * sizeof(int8_t));
    for (int i = 0; i < count; i++) {
        int temp;
        fscanf(f, "%d", &temp);
        data[i] = (int8_t)temp;
    }
    fclose(f);
    return data;
}

// Helper to write int8 data
void write_int8(const char* filepath, int8_t* data, int count) {
    FILE* f = fopen(filepath, "w");
    if (!f) {
        printf("Failed to open %s for writing\n", filepath);
        return;
    }
    for (int i = 0; i < count; i++) {
        fprintf(f, "%d\n", data[i]);
    }
    fclose(f);
    printf("Saved output to: %s\n", filepath);
}

void get_file_path(char* buffer, const char* dir, const char* filename) {
    sprintf(buffer, "%s\\%s", dir, filename);
}

int main() {
    char path[MAX_PATH];

    // 1. Determine Dimensions
    get_file_path(path, PARAMS_DIR, "bias_value.txt");
    int output_depth = count_elements(path);
    printf("Output Depth: %d\n", output_depth);

    get_file_path(path, PARAMS_DIR, "weight_values.txt");
    int total_weights = count_elements(path);
    int accum_depth = total_weights / output_depth;
    printf("Accum Depth (Input Channels): %d\n", accum_depth);

    get_file_path(path, IO_DIR, "ifm_0.txt");
    int total_input = count_elements(path);
    int batches = total_input / accum_depth;
    printf("Batches: %d\n", batches);

    // 2. Load Data
    // IFM
    get_file_path(path, IO_DIR, "ifm_0.txt");
    int8_t* input_data = read_int8(path, total_input);
    
    // Weights
    get_file_path(path, PARAMS_DIR, "weight_values.txt");
    int8_t* weight_data = read_int8(path, total_weights);

    // TRY TRANSPOSE: If weights are [In, Out], convert to [Out, In]
    // Uncomment to use transposed weights
    // free(weight_data);
    // weight_data = transposed_weights; 
    // printf("Using Transposed Weights [In, Out] -> [Out, In]\n");
    
    // Or keep original if we think it's [Out, In]
    // free(transposed_weights); 

    // Bias
    get_file_path(path, PARAMS_DIR, "bias_value.txt");
    int32_t* bias_data = read_int32(path, output_depth);

    // Expected OFM
    get_file_path(path, IO_DIR, "ofm_0.txt");
    int total_output = count_elements(path);
    int8_t* expected_output = read_int8(path, total_output);

    // 3. Load Quantization Params
    // Scales
    get_file_path(path, PARAMS_DIR, "ifm_scale.txt");
    float* ifm_scale_arr = read_floats(path, 1);
    float ifm_scale = ifm_scale_arr[0];
    free(ifm_scale_arr);

    get_file_path(path, PARAMS_DIR, "ofm_scale.txt");
    float* ofm_scale_arr = read_floats(path, 1);
    float ofm_scale = ofm_scale_arr[0];
    free(ofm_scale_arr);

    get_file_path(path, PARAMS_DIR, "weight_scale.txt");
    float* weight_scales = read_floats(path, output_depth);

    // ZPs
    get_file_path(path, PARAMS_DIR, "ifm_zp.txt");
    int32_t* ifm_zp_arr = read_int32(path, 1);
    int32_t ifm_zp = ifm_zp_arr[0];
    free(ifm_zp_arr);

    get_file_path(path, PARAMS_DIR, "ofm_zp.txt");
    int32_t* ofm_zp_arr = read_int32(path, 1);
    int32_t ofm_zp = ofm_zp_arr[0];
    free(ofm_zp_arr);

    // Note: weight_zp usually 0 for per-channel symmetric, but let's check
    // get_file_path(path, PARAMS_DIR, "weight_zp.txt");
    // ... not needed for symmetric per-channel standard FC, assumed 0 in code comment, 
    // but usually handled by weight_offset if needed. FC PerChannel spec says weights symmetric => zp=0.

    // 4. Calculate Multipliers and Shifts
    int32_t* output_multiplier = (int32_t*)malloc(output_depth * sizeof(int32_t));
    int* output_shift = (int*)malloc(output_depth * sizeof(int));

    for (int i = 0; i < output_depth; i++) {
        double effective_scale = (double)ifm_scale * weight_scales[i] / (double)ofm_scale;
        QuantizeMultiplier(effective_scale, &output_multiplier[i], &output_shift[i]);
    }

    // 5. Setup Structs
    FullyConnectedParams params;
    params.input_offset = ifm_zp; // ZP is subtracted, so if ZP=-109, offset=109. TFLite does val + offset
    params.output_offset = ofm_zp;
    params.weights_offset = 0; // Per-channel typically has 0 weights_offset
    params.output_multiplier = output_multiplier[0]; // Used only in FullyConnected, but we'll populate it
    params.output_shift = output_shift[0]; // Used only in FullyConnected
    params.quantized_activation_min = -128; // Standard int8
    params.quantized_activation_max = 127;

    int32_t input_dims[] = {batches, accum_depth};
    RuntimeShape input_shape = {2, input_dims};

    int32_t filter_dims[] = {output_depth, accum_depth};
    RuntimeShape filter_shape = {2, filter_dims};

    int32_t bias_dims[] = {output_depth};
    RuntimeShape bias_shape = {1, bias_dims};

    int32_t output_dims[] = {batches, output_depth};
    RuntimeShape output_shape = {2, output_dims};

    // Print Data Samples
    printf("--- Data Samples ---\n");
    printf("Input[0..4]: %d, %d, %d, %d, %d\n", input_data[0], input_data[1], input_data[2], input_data[3], input_data[4]);
    printf("Weight[0..4]: %d, %d, %d, %d, %d\n", weight_data[0], weight_data[1], weight_data[2], weight_data[3], weight_data[4]);
    printf("Bias[0..4]: %d, %d, %d, %d, %d\n", bias_data[0], bias_data[1], bias_data[2], bias_data[3], bias_data[4]);
    printf("Expected OFM[0..4]: %d, %d, %d, %d, %d\n", expected_output[0], expected_output[1], expected_output[2], expected_output[3], expected_output[4]);
    printf("Input Scale: %f, Input ZP: %d\n", ifm_scale, ifm_zp);
    printf("Output Scale: %f, Output ZP: %d\n", ofm_scale, ofm_zp);
    printf("Weight Scale[0]: %f\n", weight_scales[0]);
    printf("Calculated Mult[0]: %d, Shift[0]: %d\n", output_multiplier[0], output_shift[0]);
    printf("Params Input Offset: %d\n", params.input_offset);
    printf("--------------------\n");

    // 6. Run
    int8_t* output_data = (int8_t*)malloc(total_output * sizeof(int8_t));
    memset(output_data, 0, total_output);

    FullyConnectedPerChannel(
        &params,
        output_multiplier,
        output_shift,
        &input_shape,
        input_data,
        &filter_shape,
        weight_data,
        &bias_shape,
        bias_data,
        &output_shape,
        output_data
    );

    // 7. Verify
    int correct = 0;
    int max_diff = 0;
    for (int i = 0; i < total_output; i++) {
        if (output_data[i] == expected_output[i]) {
            correct++;
        } else {
            int diff = abs(output_data[i] - expected_output[i]);
            if (diff > max_diff) max_diff = diff;
            // printf(" mismatch at %d: got %d, expected %d\n", i, output_data[i], expected_output[i]);
        }
    }

    printf("Accuracy: %d/%d (%.2f%%)\n", correct, total_output, 100.0 * correct / total_output);
    printf("Max Diff: %d\n", max_diff);

    if (max_diff == 0) {
        printf("TEST PASSED\n");
    } else {
        printf("TEST FAILED (with strict equality)\n");
    }

    // Save actual output for debugging
    get_file_path(path, PARAMS_DIR, "ofm.txt");
    write_int8(path, output_data, total_output);

    // Cleanup
    free(input_data);
    free(weight_data);
    free(bias_data);
    free(expected_output);
    free(weight_scales);
    free(output_multiplier);
    free(output_shift);
    free(output_data);

    return 0;
}
