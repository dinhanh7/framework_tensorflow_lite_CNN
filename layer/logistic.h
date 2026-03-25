#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <stdint.h>
#include <math.h>
#include "requantize_utils.h"

/**
 * @brief Implement function Logistic (Sigmoid) reference quantized
 * Logic follows tflite::reference_ops::Logistic
 * 
 * @param input_data: Input data pointer (int8)
 * @param output_data: Output data pointer (int8)
 * @param flat_size: Total number of elements
 * @param input_scale: Input scale
 * @param input_zp: Input zero point
 * @param output_scale: Output scale
 * @param output_zp: Output zero point
 */
void quantized_logistic_int8(
    const int8_t* input_data,
    int8_t* output_data,
    int flat_size,
    float input_scale,
    int32_t input_zp,
    float output_scale,
    int32_t output_zp
) {
    const float cutoff_upper = 16.619047164916992188f;
    const float cutoff_lower = -9.0f;

    for (int i = 0; i < flat_size; i++) {
        // Dequantize.
        float val = (float)(input_data[i] - input_zp) * input_scale;
        float result;

        if (val > cutoff_upper) {
            result = 1.0f;
        } else if (val < cutoff_lower) {
            result = expf(val);
        } else {
            result = 1.0f / (1.0f + expf(-val));
        }

        // Requantize
        // Note: The provided reference code used static_cast (truncation), 
        // but standard TFLite quantization usually involves rounding to nearest.
        // We use roundf to match the golden data which likely expects rounding.
        float scaled_val = result / output_scale + (float)output_zp;
        int32_t output_val = (int32_t)roundf(scaled_val);
        
        output_data[i] = clip_int8(output_val);
    }
}

#endif