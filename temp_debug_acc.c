#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "layer/mean.h"
int main() {
  FILE* f = fopen("all_layer_io/layer_58_MEAN/ifm_0.txt", "r");
  int8_t input[75264];
  for (int i=0;i<75264;i++){ int v; fscanf(f, "%d", &v); input[i]=(int8_t)v; }
  fclose(f);
  float input_scale = 0.21199346f, output_scale = 0.05856107f;
  int32_t input_zp = -126, output_zp = -122;
  const float num_elements_in_axis = 14.0f * 14.0f;
  const float real_scale = input_scale / (num_elements_in_axis * output_scale);
  int32_t qm; int shift; QuantizeMultiplier(real_scale, &qm, &shift);
  float temp = input_zp * input_scale / output_scale; temp = temp > 0 ? temp + 0.5f : temp - 0.5f; int32_t bias = output_zp - (int32_t)temp;
  for (int c=0;c<10;c++) {
    int32_t acc=0; for (int h=0;h<14;h++) for (int w=0;w<14;w++) acc += input[h*14*384 + w*384 + c];
    int32_t scaled = MultiplyByQuantizedMultiplier(acc, qm, shift);
    int32_t out = scaled + bias;
    printf("c=%d acc=%d scaled=%d out=%d\n", c, acc, scaled, out);
  }
  printf("qm=%d shift=%d bias=%d\n", qm, shift, bias);
  return 0;
}
