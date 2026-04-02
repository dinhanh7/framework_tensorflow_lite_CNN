#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "../layer/hardswish.h"
#include "../layer/utils.h"

// ============================================================================
// Helper Functions (File I/O)
// ============================================================================

// int count_elements(const char* filename) {
//     FILE* f = fopen(filename, "r");
//     if (!f) return -1;
//     int count = 0;
//     int temp;
//     while (fscanf(f, "%d", &temp) == 1) count++;
//     fclose(f);
//     return count;
// }
int count_elements(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) return -1;
    int count = 0;
    int temp;
    while (fscanf(f, "%d", &temp) == 1) count++;
    fclose(f);
    return count;
}

void read_int8_array(const char* filename, int8_t* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) { printf("Cannot open %s\n", filename); exit(1); }
    int temp;
    for (int i = 0; i < size; i++) {
        fscanf(f, "%d", &temp);
        buffer[i] = (int8_t)temp;
    }
    fclose(f);
}

void read_float_array(const char* filename, float* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) { printf("Cannot open %s\n", filename); exit(1); }
    for (int i = 0; i < size; i++) fscanf(f, "%f", &buffer[i]);
    fclose(f);
}

void read_int_array(const char* filename, int32_t* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) { printf("Cannot open %s\n", filename); exit(1); }
    for (int i = 0; i < size; i++) fscanf(f, "%d", &buffer[i]);
    fclose(f);
}

void write_int8_file(const char* filename, int8_t* data, int size) {
    FILE* f = fopen(filename, "w");
    if (!f) { printf("Cannot open %s for write\n", filename); return; }
    for (int i = 0; i < size; ++i) fprintf(f, "%d\n", data[i]);
    fclose(f);
    printf("Wrote %s\n", filename);
}



// ============================================================================
// MAIN
// ============================================================================
int main() {
    printf("Starting HardSwish Test...\n");

    // Paths
    const char params_dir[] = "extracted_params/layer020_HARD_SWISH_1_block3b_expand_activation_1_truediv__1_block3b_expand_activation_1_Relu6_1_block3b_expand_activation_1_add_1_block3b_expand_activation_1_mul";
    const char io_dir[] = "all_layer_io/layer_20_HARD_SWISH";
    char path_buf[512];

    // 1. Read Scale/ZP
    float ifm_scale, ofm_scale;
    int32_t ifm_zp, ofm_zp;

    sprintf(path_buf, "%s/param_idx0_scale.txt", params_dir);
    read_float_array(path_buf, &ifm_scale, 1);
    sprintf(path_buf, "%s/param_idx0_zp.txt", params_dir);
    read_int_array(path_buf, &ifm_zp, 1);

    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_float_array(path_buf, &ofm_scale, 1);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    printf("Params:\n");
    printf("  IFM Scale: %f, ZP: %d\n", ifm_scale, ifm_zp);
    printf("  OFM Scale: %f, ZP: %d\n", ofm_scale, ofm_zp);

    // 2. Read Input
    sprintf(path_buf, "%s/ifm_0.txt", io_dir);
    int size = count_elements(path_buf);
    if (size <= 0) {
        printf("Error: Could not read input file or file is empty.\n");
        return 1;
    }
    printf("  Input size: %d\n", size);

    int8_t* input_data = (int8_t*)malloc(size * sizeof(int8_t));
    if (!input_data) return 1;
    read_int8_array(path_buf, input_data, size);

    // 3. Alloc Output
    int8_t* output_data = (int8_t*)malloc(size * sizeof(int8_t));
    if (!output_data) {
        free(input_data);
        return 1;
    }

    // 4. Run HardSwish
    quantized_hardswish(
        input_data,
        output_data,
        size,
        (int8_t)ifm_zp,
        (int8_t)ofm_zp,
        ifm_scale,
        ofm_scale
    );

    // 5. Write Output
    sprintf(path_buf, "%s/ofm.txt", params_dir);
    write_int8_file(path_buf, output_data, size);

    free(input_data);
    free(output_data);

    return 0;
}
