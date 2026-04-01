#include <stdio.h>
#include "layer/requantize_utils.h"
int main() {
    float input_scale = 0.21199346f;
    float output_scale = 0.05856107f;
    float num = 14.0f * 14.0f;
    float real_scale_f = input_scale / (num * output_scale);
    double real_scale_d = (double)input_scale / ((double)num * (double)output_scale);
    int32_t qm_f, qm_d; int s_f, s_d;
    QuantizeMultiplier(real_scale_f, &qm_f, &s_f);
    QuantizeMultiplier(real_scale_d, &qm_d, &s_d);
    printf("f: qm=%d shift=%d\n", qm_f, s_f);
    printf("d: qm=%d shift=%d\n", qm_d, s_d);
    return 0;
}
