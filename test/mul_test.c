#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#include "../layer/mul.h"



// Helper function to count elements in a file
int count_elements(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error opening file for counting: %s\n", filename);
        return -1;
    }
    int count = 0;
    int temp;
    while (fscanf(f, "%d", &temp) == 1) {
        count++;
    }
    fclose(f);
    return count;
}

// Helper function to read int32 array from text file
void read_int_array(const char* filename, int32_t* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    int temp;
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%d", &temp) != 1) {
             printf("Error reading int32 at index %d from %s\n", i, filename);
             break;
        }
        buffer[i] = temp;
    }
    fclose(f);
}

// Helper function to read float array from text file
void read_float_array(const char* filename, float* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%f", &buffer[i]) != 1) {
             printf("Error reading float at index %d from %s\n", i, filename);
             break;
        }
    }
    fclose(f);
}

// Helper function to read double array from text file (Higher precision)
void read_double_array(const char* filename, double* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%lf", &buffer[i]) != 1) {
             printf("Error reading double at index %d from %s\n", i, filename);
             break;
        }
    }
    fclose(f);
}

// Helper function to read int8 array from text file
void read_int8_array(const char* filename, int8_t* buffer, int size) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }
    int temp;
    for (int i = 0; i < size; i++) {
        if (fscanf(f, "%d", &temp) != 1) {
             printf("Error reading int8 at index %d from %s\n", i, filename);
             break;
        }
        buffer[i] = (int8_t)temp;
    }
    fclose(f);
}

// Helper to write output
void write_int8_file(const char* filename, int8_t* data, int size) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        printf("Error opening file for writing: %s\n", filename);
        return;
    }
    for (int i = 0; i < size; ++i) {
        fprintf(f, "%d\n", data[i]);
    }
    fclose(f);
    printf("Wrote output to %s\n", filename);
}

int main() {
    printf("Starting MUL Test ...\n");

    // Paths
    char params_dir[] = "extracted_params/layer051_MUL_1_block4b_se_excite_1_Mul";
    char input_dir[] = "all_layer_io/layer_51_MUL";
    char path_buf[512];

    // 1. Read Quantization Parameters
    double ifm0_scale, ifm1_scale, ofm_scale; // Changed to double for better precision
    int32_t ifm0_zp, ifm1_zp, ofm_zp;

    // Input 0 (ifm)
    sprintf(path_buf, "%s/ifm_scale.txt", params_dir);
    read_double_array(path_buf, &ifm0_scale, 1);
    sprintf(path_buf, "%s/ifm_zp.txt", params_dir);
    read_int_array(path_buf, &ifm0_zp, 1);

    // Input 1 (ifm_1)
    sprintf(path_buf, "%s/ifm_1_scale.txt", params_dir);
    read_double_array(path_buf, &ifm1_scale, 1);
    sprintf(path_buf, "%s/ifm_1_zp.txt", params_dir);
    read_int_array(path_buf, &ifm1_zp, 1);

    // Output
    sprintf(path_buf, "%s/ofm_scale.txt", params_dir);
    read_double_array(path_buf, &ofm_scale, 1);
    sprintf(path_buf, "%s/ofm_zp.txt", params_dir);
    read_int_array(path_buf, &ofm_zp, 1);

    // 2. Determine sizes and read Input Data
    sprintf(path_buf, "%s/ifm_0.txt", input_dir);
    int size0 = count_elements(path_buf);
    
    sprintf(path_buf, "%s/ifm_1.txt", input_dir);
    int size1 = count_elements(path_buf);

    int8_t* input0_data = (int8_t*)malloc(size0 * sizeof(int8_t));
    int8_t* input1_data = (int8_t*)malloc(size1 * sizeof(int8_t));

    sprintf(path_buf, "%s/ifm_0.txt", input_dir);
    read_int8_array(path_buf, input0_data, size0);

    sprintf(path_buf, "%s/ifm_1.txt", input_dir);
    read_int8_array(path_buf, input1_data, size1);

    // 3. Configure ArithmeticParams
    ArithmeticParams params;
    
    // Calculate Multiplier and Shift
    // Double Multiplier = (S1 * S2) / Sout
    double real_multiplier = (double)ifm0_scale * (double)ifm1_scale / (double)ofm_scale;
    
    int32_t quantized_multiplier;
    int shift;
    QuantizeMultiplier(real_multiplier, &quantized_multiplier, &shift);

    params.output_multiplier = quantized_multiplier;
    params.output_shift = shift;

    // Offsets are negated zero points for (q - zp) formula
    params.input1_offset = -ifm0_zp; 
    params.input2_offset = -ifm1_zp;
    params.output_offset = ofm_zp; 

    params.quantized_activation_min = -128;
    params.quantized_activation_max = 127;

    printf("Params: M=%d, Shift=%d, In1Off=%d, In2Off=%d, OutOff=%d\n", 
           params.output_multiplier, params.output_shift, 
           params.input1_offset, params.input2_offset, params.output_offset);

    // 4. Run MUL
    int8_t* output_data = NULL;
    int output_size = 0;

    if (size0 == size1) {
        output_size = size0;
        output_data = (int8_t*)malloc(output_size * sizeof(int8_t));
        MulElementwiseInt8(size0, &params, input0_data, input1_data, output_data);
    } else {
        printf("Size mismatch: %d vs %d. Attempting broadcast...\n", size0, size1);
        
        // Trường hợp đặc biệt: Input0 lớn (tensor), Input1 nhỏ (vector kênh)
        // Ví dụ: [14, 14, 384] * [384]
        if (size0 > size1 && size0 % size1 == 0) {
             output_size = size0;
             output_data = (int8_t*)malloc(output_size * sizeof(int8_t));
             
             // Gọi hàm BroadcastMulInt8 mới (đã sửa trong mul.h)
             // Lưu ý: Hàm này không dùng NdArrayDesc nữa mà dùng size trực tiếp
             BroadcastMulInt8(&params, 
                              input0_data, size0, 
                              input1_data, size1, 
                              output_data);
                              
        } else if (size1 > size0 && size1 % size0 == 0) {
            // Trường hợp ngược lại: Input1 lớn, Input0 nhỏ
             output_size = size1;
             output_data = (int8_t*)malloc(output_size * sizeof(int8_t));
             
             // Swap inputs logic 
             // arithmetic is commutative for mul, so just swap data pointers and sizes
             // BUT params offsets correspond to input1/input2. Need to swap offsets temporarily?
             // Multiplication: (in1+off1) * (in2+off2). 
             // Swapping data => (in2+off1) * (in1+off2). WRONG.
             // We need to tell the function to use correct offsets.
             // But our simple function assumes Input1 is the big one.
             // So we should construct a temporary struct with swapped offsets.
             
             ArithmeticParams params_swapped = params;
             params_swapped.input1_offset = params.input2_offset;
             params_swapped.input2_offset = params.input1_offset;
             
             BroadcastMulInt8(&params_swapped, 
                              input1_data, size1, 
                              input0_data, size0, 
                              output_data);

        } else {
             printf("Error: Unsupported broadcast shapes (must be multiple of each other).\n");
             free(input0_data); free(input1_data);
             return 1;
        }
    }

    // 5. Write Output
    // Write to "param folder" => extracted_params/...
    sprintf(path_buf, "%s/ofm.txt", params_dir);
    write_int8_file(path_buf, output_data, output_size);

    // Also write to a debug file "ofm_computed.txt" in params?
    // User asked "in ra output vào folder param".
    
    free(input0_data);
    free(input1_data);
    free(output_data);

    return 0;
}
