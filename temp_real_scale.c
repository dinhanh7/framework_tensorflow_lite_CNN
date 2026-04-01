#include <stdio.h>
#include <stdint.h>
#include <math.h>
int main() {
    float input_scale = 0.21199346f;
    float output_scale = 0.05856107f;
    float num = 14.0f * 14.0f;
    float real_scale_f = input_scale / (num * output_scale);
    double real_scale_d = (double)input_scale / ((double)num * (double)output_scale);
    printf("real_scale_f=%.20f\n", real_scale_f);
    printf("real_scale_d=%.20f\n", real_scale_d);
    return 0;
}
