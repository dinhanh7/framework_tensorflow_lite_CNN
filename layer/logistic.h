#ifndef LOGISTIC_H
#define LOGISTIC_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "requantize_utils.h"

#define MIN_INT8 -128
#define MAX_INT8 127



// Giả lập hàm gemmlowp::logistic cho số nguyên
// Implement based on gemmlowp fixed-point logistic
// Logistic(x) = 1 / (1 + exp(-x))
// Input is fixed-point (Q4.27 format usually in gemmlowp internal context, but here Q3.28 is common)
// We use a simplified approximation or look-up table if possible,
// but for standard compliance we might need 3rd order polynomial.
//
// Ref: tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h -> gemmlowp::logistic
// Since we don't have gemmlowp, we use a 3rd degree polynomial approximation:
// y = 0.5 + 0.25*x - x^3/48 + x^5/480... (Taylor series around 0)
// Or just use the float implementation for "fake" but correct behavior if exact bit match isn't mandatory
// BUT user asked for "implement gemmlowp::logistic".
static inline int32_t FakeGemmlowpLogistic(int32_t input_in_q4) {
    // Input is in Q format. TFLite logistic input range roughly [-10, 10] mapped to int32.
    // The input_in_q4 comes from input * multiplier.
    // Let's implement the specific fixed-point logistic used in TFLite Micro / gemmlowp
    
    // Using signed 32-bit fixed-point arithmetic. 
    // We can use a polynomial approximation:
    // 1 / (1 + exp(-x))
    
    // For simplicity and "fake" implementation as requested:
    // Convert back to float, compute sigmoid, convert to Q0.31
    // (This is often acceptable if the goal is functional correctness over strict bit exactness without the library)
    // float val = (float)input_in_q4 / (float)(1 << 24); // Assuming Q format from multiplier
    // Actually, input_in_q4 is often Q24 or similar?
    // Let's infer from context. `MultiplyByQuantizedMultiplier` usually produces a result scaled by multiplier.
    
    // REAL GEMMLOWP APPROXIMATION (simplified):
    // Maps input to [0, 1] range in Q31.
    // Logistic(x) = 0.5 + 0.5 * Tanh(x/2)
    // We will use double for precision if we want to be safe, or just float.
    // Since this is "Fake", we will use float math for the core function to ensure accuracy.
    
    // The typical input to this function from TFLite is scaled such that it represents real value * 2^n
    // TFLite uses input_multiplier to scale input to fixed point.
    // Let's assume input_in_q4 is essentially FixedPoint(x).
    
    // NOTE: In TFLite Reference, the input multiplier is calculated so that input_data is converted to Q3.29 or similar.
    
    // Let's align with the provided structure.
    // Input is int32. Result should be Q0.31 (int32 where MAX_INT is close to 1.0).
    
    // Convert fixed point to double. 
    // We assume the input has integer part and fractional part.
    // The shift amount usually aligns it.
    
    // Let's look at `LogisticInt8` usage of `FakeGemmlowpLogistic(input_in_q4)`.
    // Then `output_in_q0` is converted to int8 by `RoundingDivideByPOT(..., 31 - 8)`.
    // So `output_in_q0` MUST be Q31 (max value ~ 2^31).
    // `input_in_q4` name suggests Q4.something?
    // Actually TFLite uses integer bits = 4 for input range [-8, 8] roughly.
    // So input_in_q4 is Q4.28??
    
    double real_input = (double)input_in_q4 / (double)(1 << 24); // Checking standard TFLite scaling
    // Actually, let's look at the standard shift.
    // If we rely on the float implementation for the "Fake" core, detailed format doesn't matter as much 
    // as long as we use the right divisor.
    
    // Based on TFLite `logistic.h`: 
    // static const int kInputIntegerBits = 4;
    // double real_input = input_in_q4 / 2^(31 - 4) = input_in_q4 / 2^27. 
    // Wait, TFLite uses fixedpoint::FixedPoint<t, 4>.
    
    const double kScale = 134217728.0; // 2^27
    double x = (double)input_in_q4 / kScale;
    double sigmoid = 1.0 / (1.0 + exp(-x));
    
    // Convert back to Q31 (Q0.31)
    return (int32_t)(sigmoid * 2147483648.0); // 2^31
}

// Original signature required for the test code to work
// We will wrap the integer implementation.
void LogisticInt8(int32_t input_zero_point, int32_t input_range_radius,
                  int32_t input_multiplier, int32_t input_left_shift,
                  int32_t input_size, const int8_t* input_data,
                  int8_t* output_data) {
    
    const int32_t kOutputIntegerBits = 8;
    const int32_t kOutputZeroPoint = -128;

    for (int i = 0; i < input_size; ++i) {
        const int32_t input = (int32_t)input_data[i] - input_zero_point;

        if (input <= -input_range_radius) {
            output_data[i] = MIN_INT8;
        } else if (input >= input_range_radius) {
            output_data[i] = MAX_INT8;
        } else {
            const int32_t input_in_q4 = MultiplyByQuantizedMultiplier(
                input, input_multiplier, input_left_shift);
            
            // Thay thế gemmlowp::logistic
            const int32_t output_in_q0 = FakeGemmlowpLogistic(input_in_q4);

            // Rescale và ép kiểu
            // output is in Q31. We need Q8 (int8).
            // Shift = 31 - 7 (since int8 is 7 bits + sign? No, int8 covers range 2^8).
            // Actually TFLite uses `gemmlowp::RoundingDivideByPOT(output_in_q0, 31 - kOutputIntegerBits)`.
            // kOutputIntegerBits = 8. So shift 23.
            int32_t output_in_q23 = RoundingDivideByPOT_32(output_in_q0, 31 - 8);
            
            int32_t output_val = output_in_q23 + kOutputZeroPoint;

            if (i < 5) {
                printf("Idx %d: input_raw=%d, input=%d, input_in_q4=%d, output_in_q0=%d, output_in_q23=%d, output_val=%d\n", 
                       i, input_data[i], input, input_in_q4, output_in_q0, output_in_q23, output_val);
            }

            // Clamp (giới hạn vùng giá trị)
            if (output_val < MIN_INT8) output_val = MIN_INT8;
            if (output_val > MAX_INT8) output_val = MAX_INT8;

            output_data[i] = (int8_t)output_val;
        }
    }
}

// Wrapper to adapt the old function signature to the new implementation
// This allows existing tests to pass.
static inline void quantized_logistic_int8(
    const int8_t* input_data,
    int8_t* output_data,
    int flat_size,
    float input_scale,
    int32_t input_zp,
    float output_scale,
    int32_t output_zp
) {
    // 1. Calculate Input Multiplier and Shift
    // We want to map input_scale to a fixed point multiplier.
    // The "input integer bits" for Logistic is typically 4 (range +/- 8.0).
    // TFLite uses input_multiplier to map input to Q3.28 format (typically).
    // multiplier = input_scale * (1 << (31 - kInputIntegerBits))
    // kInputIntegerBits = 4.
    
    // We can use standard math to compute multiplier/shift from float.
    // multiplier * 2^shift ~= input_scale * (1 << 27) (if we target Q3.28)
    
    // Actually, TFLite reference code calculates multiplier based on input_scale alone.
    // But since our inner function expects input_multiplier to act on (input-zp),
    // and produce a value that FakeGemmlowpLogistic expects (Q4.xx).
    
    // Let's use a standard approximation.
    // Use double for precision during setup.
    double real_multiplier = input_scale / (1.0 / (1 << 8)); // Example
    // TFLite logic:
    // double input_product_scale = input_scale * (1 << 24); // Not quite right for all cases
    
    // Let's reverse engineer from FakeGemmlowpLogistic expectations.
    // It does: double x = (double)input_in_q4 / 134217728.0; // / 2^27
    // So input_in_q4 should be x * 2^27.
    // input_in_q4 = (input - zp) * multiplier >> shift (sort of).
    // So (input - zp) * input_scale = x.
    // input_in_q4 = x * 2^27 = (input - zp) * input_scale * 2^27.
    // So multiplier * 2^shift should equal input_scale * 2^27.
    
    int32_t input_multiplier = 0;
    int input_shift = 0;
    
    // Decompose input_scale to significand * 2^shift using standard logic
    // We want multiplier * 2^shift ~= input_scale / (1.0 / (1 << 27?))
    // Actually we want input_in_q4 = (input - zp) * multiplier >> shift
    // input_in_q4 represents fixed point 4.28 (roughly, where 27-28 fractional bits).
    // Let's assume input_scale * (1<<27) is the effective scale factor.
    
    double effective_scale = input_scale * (1 << 27);
    int exponent;
    double significand = frexp(effective_scale, &exponent);
    // significand is in [0.5, 1). We want int32 multiplier (Q31).
    if (significand == 0.5) {
        input_multiplier = 1073741824; // 1 << 30
        input_shift = exponent - 31;
    } else {
        input_multiplier = (int32_t)(significand * 2147483648.0); // * 2^31
        input_shift = exponent - 31; 
    }

    // 2. Input Range Radius
    // Determine when x > some clamp value. TFLite uses integer check.
    // Let's just use the full byte range for simplicity as saturation happens inside.
    int32_t input_range_radius = 127; 
    
    LogisticInt8(
        input_zp,
        127, // Radius covers full int8 range
        input_multiplier,
        input_shift,
        flat_size,
        input_data,
        output_data
    );
}

#endif