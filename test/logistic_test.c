// Not need any more because we are using conv rule6 fused
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include "../layer/logistic.h"

// Helper to count integers in a file
int count_elements(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    int count = 0;
    int val;
    while (fscanf(fp, "%d", &val) == 1) {
        count++;
    }
    fclose(fp);
    return count;
}

// Helper to read int8 array from file
void read_int8_array(const char* filename, int8_t* buffer, int size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        int val;
        if (fscanf(fp, "%d", &val) != 1) {
            printf("Error reading element %d from %s\n", i, filename);
            break;
        }
        buffer[i] = (int8_t)val;
    }
    fclose(fp);
}

// Helper to read single float
float read_float(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    float val;
    fscanf(fp, "%f", &val);
    fclose(fp);
    return val;
}

// Helper to read single int
int32_t read_int(const char* filename) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    int32_t val;
    fscanf(fp, "%d", &val);
    fclose(fp);
    return val;
}

int main() {
    // Paths
    const char* io_dir = "all_layer_io/layer_35_LOGISTIC";
    const char* param_dir = "extracted_params/layer035_LOGISTIC_1_block4a_se_expand_1_Sigmoid";

    printf("=== Testing Logistic Layer (Layer 35) ===\n");
    printf("IO Directory: %s\n", io_dir);
    printf("Param Directory: %s\n", param_dir);

    // 1. Read Parameters
    char path[256];
    
    sprintf(path, "%s/ifm_scale.txt", param_dir);
    float input_scale = read_float(path);

    sprintf(path, "%s/ifm_zp.txt", param_dir);
    int32_t input_zp = read_int(path);

    sprintf(path, "%s/ofm_scale.txt", param_dir);
    float output_scale = read_float(path);

    sprintf(path, "%s/ofm_zp.txt", param_dir);
    int32_t output_zp = read_int(path);

    printf("Params: Input Scale=%.6f, ZP=%d | Output Scale=%.6f, ZP=%d\n", 
           input_scale, input_zp, output_scale, output_zp);

    // 2. Determine Data Size
    sprintf(path, "%s/ifm_0.txt", io_dir);
    int flat_size = count_elements(path);
    printf("Data Size (flat): %d\n", flat_size);

    // 3. Allocate Buffers
    int8_t* input_data = (int8_t*)malloc(flat_size * sizeof(int8_t));
    int8_t* output_data = (int8_t*)malloc(flat_size * sizeof(int8_t));
    int8_t* golden_data = (int8_t*)malloc(flat_size * sizeof(int8_t));

    // 4. Read Data
    read_int8_array(path, input_data, flat_size);
    
    sprintf(path, "%s/ofm_0.txt", io_dir);
    read_int8_array(path, golden_data, flat_size);

    // 5. Run Logistic
    printf("Running LogisticInt8...\n");

    // Calculate Multiplier and Shift
    // Logic matches TFLite reference: double real_multiplier = input_scale * (1 << 27); or similar
    // We use the same logic as the wrapper in logistic.h for consistency
    double effective_scale = input_scale * (1 << 27);
    int exponent;
    double significand = frexp(effective_scale, &exponent);
    int32_t input_multiplier;
    if (significand == 0.5) {
        input_multiplier = 1073741824; // 1 << 30
    } else {
        input_multiplier = (int32_t)(significand * 2147483648.0); // Q31
    }
    int32_t input_shift = exponent; // input_shift = left shift amount for QuantizedMultiplier(HighMul) + RightShift logic.
    // TFLite uses input_multiplier * input * 2^shift.
    // With HighMul (>>31), effective multiplier is M/2^31.
    // We want M * 2^-8.
    // M/2^31 * 2^shift = M * 2^-8 -> shift - 31 = -8 -> shift = 23.
    // frexp exponent is 23. So shift = exponent.

    printf("Computed Params: Multiplier=%d, Shift=%d, InputRadius=127\n", 
           input_multiplier, input_shift);

    LogisticInt8(
        input_zp,
        127, // input_range_radius (covers full int8)
        input_multiplier,
        input_shift,
        flat_size,
        input_data,
        output_data
    );

    // 6. Verification
    int mismatches = 0;
    for (int i = 0; i < flat_size; i++) {
        if (output_data[i] != golden_data[i]) {
            if (mismatches < 10) {
                printf("Mismatch at %d: Expected %d, Got %d (Diff: %d)\n", 
                       i, golden_data[i], output_data[i], output_data[i] - golden_data[i]);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("Test PASSED! All %d elements match.\n", flat_size);
    } else {
        printf("Test FAILED! Total mismatches: %d / %d (%.2f%%)\n", 
               mismatches, flat_size, (float)mismatches / flat_size * 100.0f);
    }

    // Cleanup
    free(input_data);
    free(output_data);
    free(golden_data);

    return (mismatches == 0) ? 0 : 1;
}
