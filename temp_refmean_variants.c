#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

static int CountLeadingZeros64(uint64_t x){ if(!x) return 64; int n=0; while((x & (1ull<<63))==0){ x <<= 1; n++; } return n; }
static void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift) {
    if (double_multiplier == 0.) { *quantized_multiplier = 0; *shift = 0; return; }
    const double q = frexp(double_multiplier, shift);
    long long q_fixed = llround(q * (1ll << 31));
    if (q_fixed == (1ll << 31)) { q_fixed /= 2; ++*shift; }
    if (*shift < -31) { *shift = 0; q_fixed = 0; }
    *quantized_multiplier = (int32_t)q_fixed;
}
static int32_t SRDHM(int32_t a, int32_t b) {
    long long ab = (long long)a * (long long)b;
    int overflow = (a == INT32_MIN && b == INT32_MIN);
    long long nudge = ab >= 0 ? (1ll << 30) : (1ll - (1ll << 30));
    int32_t res = (int32_t)((ab + nudge) >> 31);
    return overflow ? INT32_MAX : res;
}
static int32_t RoundingDivideByPOT_v1(int32_t x, int exponent) {
    if (exponent <= 0) return x;
    const int32_t mask = (1 << exponent) - 1;
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}
static int32_t RoundingDivideByPOT_v2(int32_t x, int exponent) {
    if (exponent <= 0) return x;
    const int32_t mask = (1 << exponent) - 1;
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}
static int32_t RoundingDivideByPOT_v3(int32_t x, int exponent) {
    if (exponent <= 0) return x;
    const int32_t mask = (1 << exponent) - 1;
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >> exponent) + (remainder >= threshold ? 1 : 0);
}
static int32_t MBQM(int32_t x, int32_t qm, int shift, int variant) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    int32_t y = SRDHM(x * (1 << left_shift), qm);
    if (variant == 1) return RoundingDivideByPOT_v1(y, right_shift);
    if (variant == 2) return RoundingDivideByPOT_v2(y, right_shift);
    return RoundingDivideByPOT_v3(y, right_shift);
}
static int32_t clip_int8(int32_t val){ if(val>127) return 127; if(val<-128) return -128; return val; }
int main() {
  FILE* f = fopen("all_layer_io/layer_58_MEAN/ifm_0.txt", "r");
  FILE* g = fopen("all_layer_io/layer_58_MEAN/ofm_0.txt", "r");
  int8_t input[75264]; int gold[384];
  for(int i=0;i<75264;i++){ int v; fscanf(f, "%d", &v); input[i]=(int8_t)v; }
  for(int i=0;i<384;i++){ fscanf(g, "%d", &gold[i]); }
  fclose(f); fclose(g);
  float input_scale=0.21199346f, output_scale=0.05856107f;
  int32_t input_zp=-126, output_zp=-122;
  double real_multiplier = (double)input_scale / (double)output_scale;
  int32_t output_multiplier; int output_shift;
  QuantizeMultiplier(real_multiplier, &output_multiplier, &output_shift);
  int64_t num_elements = 196;
  int shift = 63 - CountLeadingZeros64((uint64_t)num_elements);
  if (shift > 32) shift = 32;
  if (shift > 31 + output_shift) shift = 31 + output_shift;
  output_multiplier = (int32_t)(((int64_t)output_multiplier << shift) / num_elements);
  output_shift = output_shift - shift;
  for(int variant=1; variant<=3; variant++) {
    int mism=0;
    for(int c=0;c<384;c++){
      int32_t temp_sum=0; for(int h=0;h<14;h++) for(int w=0;w<14;w++) temp_sum += input[h*14*384+w*384+c];
      int32_t shifted_sum = temp_sum - input_zp * (int32_t)num_elements;
      int32_t out = clip_int8(MBQM(shifted_sum, output_multiplier, output_shift, variant) + output_zp);
      if(out != gold[c]) mism++;
    }
    printf("variant%d mism=%d\n", variant, mism);
  }
  return 0;
}
