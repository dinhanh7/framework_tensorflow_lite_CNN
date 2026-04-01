#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "layer/mean.h"
int main() {
  FILE* f = fopen("all_layer_io/layer_58_MEAN/ifm_0.txt", "r");
  int8_t input[75264];
  int8_t output[384];
  for (int i=0;i<75264;i++){ int v; fscanf(f, "%d", &v); input[i]=(int8_t)v; }
  fclose(f);
  quantized_mean_int8(input, output, 1, 14, 14, 384, 384, 0.21199346f, -126, 0.05856107f, -122);
  for (int i=0;i<10;i++) printf("%d\n", output[i]);
  return 0;
}
