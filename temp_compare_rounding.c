#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static int32_t clip_int8(int32_t val) { if (val > 127) return 127; if (val < -128) return -128; return val; }
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
    int32_t res = (int32_t)((ab + nudge) / (1ll << 31));
    return overflow ? INT32_MAX : res;
}
static int32_t RoundingDivideByPOT(int32_t x, int exponent) {
    if (exponent <= 0) return x;
    const int32_t mask = (1 << exponent) - 1;
    const int32_t remainder = x & mask;
    const int32_t threshold = (mask >> 1) + ((x < 0) ? 1 : 0);
    return (x >> exponent) + (remainder > threshold ? 1 : 0);
}
static int32_t MBQM_double(int32_t x, int32_t qm, int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    return RoundingDivideByPOT(SRDHM(x * (1 << left_shift), qm), right_shift);
}
static int32_t MBQM_single(int32_t x, int32_t qm, int shift) {
    long long total_shift = 31 - shift;
    long long round = 1ll << (total_shift - 1);
    long long result = (long long)x * (long long)qm + round;
    result >>= total_shift;
    return (int32_t)result;
}
int main() {
    const char* in_path = "all_layer_io/layer_58_MEAN/ifm_0.txt";
    const char* gold_path = "all_layer_io/layer_58_MEAN/ofm_0.txt";
    FILE* f = fopen(in_path, "r");
    FILE* g = fopen(gold_path, "r");
    if (!f || !g) return 1;
    int8_t input[75264];
    int gold[384];
    for (int i=0;i<75264;i++){ int v; fscanf(f, "%d", &v); input[i]=(int8_t)v; }
    for (int i=0;i<384;i++){ fscanf(g, "%d", &gold[i]); }
    fclose(f); fclose(g);
    float input_scale = 0.21199346f, output_scale = 0.05856107f;
    int32_t input_zp = -126, output_zp = -122;
    float num = 14.0f * 14.0f;
    float temp = input_zp * input_scale / output_scale;
    temp = temp > 0 ? temp + 0.5f : temp - 0.5f;
    int32_t bias = output_zp - (int32_t)temp;
    float real_scale = input_scale / (num * output_scale);
    int32_t qm; int shift;
    QuantizeMultiplier(real_scale, &qm, &shift);
    int mism_double=0, mism_single=0;
    for (int c=0;c<384;c++) {
      int32_t acc=0;
      for (int h=0;h<14;h++) for(int w=0;w<14;w++) acc += input[h*14*384 + w*384 + c];
      int out_d = clip_int8(MBQM_double(acc, qm, shift) + bias);
      int out_s = clip_int8(MBQM_single(acc, qm, shift) + bias);
      if (out_d != gold[c]) mism_double++;
      if (out_s != gold[c]) mism_single++;
    }
    printf("qm=%d shift=%d bias=%d\n", qm, shift, bias);
    printf("mism_double=%d\n", mism_double);
    printf("mism_single=%d\n", mism_single);
    return 0;
}
