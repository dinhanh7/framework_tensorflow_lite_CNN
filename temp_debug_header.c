#include <stdio.h>
#include <stdint.h>
#include "layer/mean.h"
int main() {
  FILE* f = fopen("all_layer_io/layer_58_MEAN/ifm_0.txt", "r");
  FILE* g = fopen("all_layer_io/layer_58_MEAN/ofm_0.txt", "r");
  int8_t input[75264]; int gold[384];
  for (int i=0;i<75264;i++){ int v; fscanf(f, "%d", &v); input[i]=(int8_t)v; }
  for (int i=0;i<384;i++){ fscanf(g, "%d", &gold[i]); }
  fclose(f); fclose(g);

  const int32_t num_elements_in_axis = 14 * 14;
  const double real_multiplier = (double)0.21199346f / (double)0.05856107f;
  int32_t output_multiplier; int output_shift;
  QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  int reduction_shift = 63 - CountLeadingZeros64((uint64_t)num_elements_in_axis);
  reduction_shift = reduction_shift < 32 ? reduction_shift : 32;
  reduction_shift = reduction_shift < (31 + output_shift) ? reduction_shift : (31 + output_shift);
  output_multiplier = (int32_t)(((int64_t)output_multiplier << reduction_shift) / num_elements_in_axis);
  output_shift -= reduction_shift;
  printf("output_multiplier=%d output_shift=%d reduction_shift=%d\n", output_multiplier, output_shift, reduction_shift);
  for (int c=0; c<10; c++) {
    int32_t temp_sum=0;
    for (int h=0; h<14; h++) for (int w=0; w<14; w++) temp_sum += input[h*14*384 + w*384 + c];
    int32_t shifted_sum = temp_sum - (-126) * num_elements_in_axis;
    int32_t acc = MultiplyByQuantizedMultiplier(shifted_sum, output_multiplier, output_shift);
    int32_t out = acc + (-122);
    printf("c=%d temp_sum=%d shifted_sum=%d acc=%d out=%d gold=%d\n", c, temp_sum, shifted_sum, acc, out, gold[c]);
  }
  return 0;
}
