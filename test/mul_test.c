#include <stdio.h>
#include <assert.h>
#include "../layer/mul.h"

// Helper function to print errors
void check(int8_t actual, int8_t expected, const char* msg) {
    if (actual != expected) {
        printf("FAIL: %s | Expected %d, got %d\n", msg, expected, actual);
    } else {
        // printf("PASS: %s\n", msg);
    }
}

void test_mul_elementwise() {
    printf("Testing MulElementwiseInt8...\n");

    // Setup Params
    ArithmeticParams params;
    params.input1_offset = 128; // Zero point
    params.input2_offset = 128; // Zero point
    params.output_offset = -128;
    params.output_multiplier = 1073741824; // 0.5 in Q31
    params.output_shift = -1;
    params.quantized_activation_min = -128;
    params.quantized_activation_max = 127;

    // Data
    int8_t input1[] = {-128, 0, 127};
    int8_t input2[] = {0, -128, 127}; 
    int8_t output[3];
    
    // input1_val = {-128+128=0, 128, 255}
    // input2_val = {128, 0, 255}
    // prod = {0, 0, 65025}
    // scale 0.5 => {0, 0, 32512}
    // output - 128 => {-128, -128, 127(sat)}

    // Run
    MulElementwiseInt8(3, &params, input1, input2, output);

    // Verify
    check(output[0], -128, "Item 0");
    check(output[1], -128, "Item 1");
    // 65025 * 0.5 = 32512. 32512 - 128 = 32384 -> Clamp to 127
    check(output[2], 127, "Item 2 (saturated)"); 
    
    printf("MulElementwiseInt8 Test Finished.\n");
}

int main() {
    test_mul_elementwise();
    return 0;
}