#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "../layer/mean.h"

#define BATCH 1
#define HEIGHT 14
#define WIDTH 14
#define CHANNELS 576
#define OUTPUT_CHANNELS 576

// Helper to read int8 array from file
void read_int8_array(const char* filename, int8_t* buffer, int size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        int val;
        if (fscanf(fp, "%d", &val) != 1) {
            printf("Error reading file at index %d: %s\n", i, filename);
            fclose(fp);
            exit(1);
        }
        buffer[i] = (int8_t)val;
    }
    fclose(fp);
}

// Helper to read single float
float read_float(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    float val;
    fscanf(fp, "%f", &val);
    fclose(fp);
    return val;
}

// Helper to read single int
int32_t read_int(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    int val;
    fscanf(fp, "%d", &val);
    fclose(fp);
    return val;
}

void write_int8_file(const char* filename, int8_t* data, int size) {
    FILE* f = fopen(filename, "w");
    if (!f) { printf("Cannot open %s for write\n", filename); return; }
    for (int i = 0; i < size; ++i) fprintf(f, "%d\n", data[i]);
    fclose(f);
    printf("Wrote %s\n", filename);
}

int main() {
    int input_size = BATCH * HEIGHT * WIDTH * CHANNELS;
    int output_size = BATCH * OUTPUT_CHANNELS;

    int8_t* input_data = (int8_t*)malloc(input_size * sizeof(int8_t));
    int8_t* output_data = (int8_t*)malloc(output_size * sizeof(int8_t));
    
    
    const char* param_dir = "extracted_params_hsigmoid\\layer074_MEAN_1_block5a_se_squeeze_1_Mean";
    const char* io_dir = "all_layer_io\\layer_74_MEAN";
    char param_path[256];

    printf("Reading input data...\n");
    printf("IO Directory: %s\n", io_dir);
    char in_path[256];
    sprintf(in_path, "%s\\ifm_0.txt", io_dir);
    printf("Full input path: %s\n", in_path);
    read_int8_array(in_path, input_data, input_size);


    printf("Reading quantization parameters...\n");
    sprintf(param_path, "%s\\ifm_scale.txt", param_dir);
    float input_scale = read_float(param_path);

    sprintf(param_path, "%s\\ifm_zp.txt", param_dir);
    int32_t input_zp = read_int(param_path);

    sprintf(param_path, "%s\\ofm_scale.txt", param_dir);
    float output_scale = read_float(param_path);

    sprintf(param_path, "%s\\ofm_zp.txt", param_dir);
    int32_t output_zp = read_int(param_path);

    printf("Params: Input Scale=%f, Input ZP=%d, Output Scale=%f, Output ZP=%d\n", 
           input_scale, input_zp, output_scale, output_zp);

    printf("Running quantized_mean_int8...\n");
    quantized_mean_int8(
        input_data,
        output_data,
        BATCH,
        HEIGHT,
        WIDTH,
        CHANNELS,
        OUTPUT_CHANNELS,
        input_scale,
        input_zp,
        output_scale,
        output_zp
    );

    char out_path[256];
    sprintf(out_path, "%s\\ofm_0_c_sim.txt", io_dir);
    write_int8_file(out_path, output_data, output_size);

    free(input_data);
    free(output_data);
    return 0;
}