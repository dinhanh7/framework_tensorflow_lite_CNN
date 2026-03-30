#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "../layer/add.h"

// Define max buffer size for reading files
#define MAX_DATA_SIZE 200000

// Helper to read a single float from file
float read_float(const char* filepath) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        printf("Error opening file: %s\n", filepath);
        exit(1);
    }
    float val;
    fscanf(f, "%f", &val);
    fclose(f);
    return val;
}

// Helper to read a single int from file
int read_int(const char* filepath) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        printf("Error opening file: %s\n", filepath);
        exit(1);
    }
    int val;
    fscanf(f, "%d", &val);
    fclose(f);
    return val;
}

// Helper to read array of int8_t from file
int read_data(const char* filepath, int8_t* buffer, int max_size) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        printf("Error opening file: %s\n", filepath);
        exit(1);
    }
    int val;
    int count = 0;
    while (fscanf(f, "%d", &val) != EOF && count < max_size) {
        buffer[count++] = (int8_t)val;
    }
    fclose(f);
    return count;
}

// Helper to write array of int8_t to file
void write_data(const char* filepath, int8_t* buffer, int size) {
    FILE* f = fopen(filepath, "w");
    if (!f) {
        printf("Error opening file for writing: %s\n", filepath);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        fprintf(f, "%d\n", buffer[i]);
    }
    fclose(f);
    printf("Output written to: %s\n", filepath);
}

// Helper: Calculate Arithmetic Params for ADD
void CalculateArithmeticParams(float input1_scale, int32_t input1_zp,
                               float input2_scale, int32_t input2_zp,
                               float output_scale, int32_t output_zp,
                               struct AddArithmeticParams* params) {
    params->input1_zp = input1_zp;
    params->input2_zp = input2_zp;
    params->output_zp = output_zp;
    
    // Standard TFLite Add parameters usually set left_shift to 20
    params->left_shift = 20;

    double twice_max_input_scale = 2 * (input1_scale > input2_scale ? input1_scale : input2_scale);
    double real_input1_multiplier = input1_scale / twice_max_input_scale;
    double real_input2_multiplier = input2_scale / twice_max_input_scale;
    double real_output_multiplier = twice_max_input_scale / ((1 << params->left_shift) * output_scale);

    QuantizeMultiplier(real_input1_multiplier, &params->input1_multiplier, &params->input1_shift);
    QuantizeMultiplier(real_input2_multiplier, &params->input2_multiplier, &params->input2_shift);
    QuantizeMultiplier(real_output_multiplier, &params->output_multiplier, &params->output_shift);

    params->quantized_activation_min = -128;
    params->quantized_activation_max = 127;
}

int main() {
    // Paths
    const char* param_dir = "extracted_params/layer015_ADD_1_block2b_add_1_Add/";
    const char* io_dir = "all_layer_io/layer_15_ADD/";
    
    char path_buffer[256];

    // 1. Read Quantization Parameters
    sprintf(path_buffer, "%sifm_scale.txt", param_dir);
    float ifm_scale = read_float(path_buffer);
    
    sprintf(path_buffer, "%sifm_zp.txt", param_dir);
    int ifm_zp = read_int(path_buffer);
    
    sprintf(path_buffer, "%sweight_scale.txt", param_dir);
    float w_scale = read_float(path_buffer);
    
    sprintf(path_buffer, "%sweight_zp.txt", param_dir); // Assuming weight ZP exists, or use 0 if not
    // Wait, let's assume weight_zp.txt exists, if not we might default to 0. 
    // The user context showed weight_zp.txt exists.
    int w_zp = read_int(path_buffer);

    sprintf(path_buffer, "%sofm_scale.txt", param_dir);
    float ofm_scale = read_float(path_buffer);
    
    sprintf(path_buffer, "%sofm_zp.txt", param_dir);
    int ofm_zp = read_int(path_buffer);

    printf("Params Loaded:\n");
    printf("Input1 (Scale, ZP): %f, %d\n", ifm_scale, ifm_zp);
    printf("Input2 (Scale, ZP): %f, %d\n", w_scale, w_zp);
    printf("Output (Scale, ZP): %f, %d\n", ofm_scale, ofm_zp);

    // 2. Prepare Arithmetic Params
    // Important: input_zp are offsets to add to the value.
    // In quantization formula: R = (Q - Z) * S. So offset = -Z.
    struct AddArithmeticParams params;
    CalculateArithmeticParams(ifm_scale, -ifm_zp, w_scale, -w_zp, ofm_scale, ofm_zp, &params);
    
    printf("\nCalculated Add Arithmetic Params:\n");
    printf("input1_mult: %d, shift: %d\n", params.input1_multiplier, params.input1_shift);
    printf("input2_mult: %d, shift: %d\n", params.input2_multiplier, params.input2_shift);
    printf("output_mult: %d, shift: %d\n", params.output_multiplier, params.output_shift);
    printf("left_shift: %d\n", params.left_shift);


    // 3. Read IO Data
    int8_t* input1_data = (int8_t*)malloc(MAX_DATA_SIZE);
    int8_t* input2_data = (int8_t*)malloc(MAX_DATA_SIZE);
    int8_t* output_data = (int8_t*)malloc(MAX_DATA_SIZE);

    sprintf(path_buffer, "%sifm_0.txt", io_dir);
    int count1 = read_data(path_buffer, input1_data, MAX_DATA_SIZE);
    
    sprintf(path_buffer, "%sifm_1.txt", io_dir);
    int count2 = read_data(path_buffer, input2_data, MAX_DATA_SIZE);
    
    if (count1 != count2) {
        printf("\nError: Size mismatch! Input1: %d, Input2: %d\n", count1, count2);
    }
    
    int num_elements = count1;

    // Defines Shapes (1D)
    int32_t dims[] = {num_elements};
    struct RuntimeShape shape = {1, dims};

    // 4. Run Add
    printf("\nRunning Add...\n");
    Add(&params, &shape, input1_data, &shape, input2_data, &shape, output_data);

    // 5. Write Output
    sprintf(path_buffer, "%sofm_c_sim.txt", param_dir);
    write_data(path_buffer, output_data, num_elements);

    free(input1_data);
    free(input2_data);
    free(output_data);

    return 0;
}
