#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "layer/mean.h"

void read_int8_array(const char* filename, int8_t* buffer, int size) {
    FILE* fp = fopen(filename, "r");
    for (int i = 0; i < size; i++) { int val; fscanf(fp, "%d", &val); buffer[i] = (int8_t)val; }
    fclose(fp);
}
float read_float(const char* filename) { FILE* fp = fopen(filename, "r"); float val; fscanf(fp, "%f", &val); fclose(fp); return val; }
int32_t read_int(const char* filename) { FILE* fp = fopen(filename, "r"); int val; fscanf(fp, "%d", &val); fclose(fp); return val; }
int main() {
    int8_t input[75264], output[384];
    read_int8_array("all_layer_io/layer_58_MEAN/ifm_0.txt", input, 75264);
    float input_scale = read_float("extracted_params_hsigmoid/layer058_MEAN_1_block4c_se_squeeze_1_Mean/ifm_scale.txt");
    int32_t input_zp = read_int("extracted_params_hsigmoid/layer058_MEAN_1_block4c_se_squeeze_1_Mean/ifm_zp.txt");
    float output_scale = read_float("extracted_params_hsigmoid/layer058_MEAN_1_block4c_se_squeeze_1_Mean/ofm_scale.txt");
    int32_t output_zp = read_int("extracted_params_hsigmoid/layer058_MEAN_1_block4c_se_squeeze_1_Mean/ofm_zp.txt");
    quantized_mean_int8(input, output, 1, 14, 14, 384, 384, input_scale, input_zp, output_scale, output_zp);
    for (int i=0;i<10;i++) printf("%d\n", output[i]);
    return 0;
}
