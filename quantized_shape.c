#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "cnn_layers.h" // Thêm thư viện CNN

// =================================================================================================
// 1. ĐỊNH NGHĨA KÍCH THƯỚC TENSOR ĐẦU VÀO CHO LỚP 58
// Dựa trên model_profile.txt, đầu vào của Lớp 58 (Shape) là Tensor[257] có shape [1, 1, 384]
// =================================================================================================
#define INPUT_DIMS {1, 1, 384}
#define INPUT_RANK 3

// Đầu ra của lớp Shape là một vector 1D có kích thước bằng số chiều của đầu vào.
#define OUTPUT_SIZE INPUT_RANK

// =================================================================================================
// 2. HÀM GHI FILE (CHO MỤC ĐÍCH KIỂM THỬ)
// =================================================================================================

/**
 * @brief Ghi một mảng số nguyên 32-bit ra file text trên một dòng.
 */
void write_int32_array_to_file(const char* path, int32_t* arr, int size) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    for (int i = 0; i < size; ++i) {
        fprintf(fp, "%d ", arr[i]); // Ghi cách nhau bởi dấu cách
    }
    fprintf(fp, "\n");
    fclose(fp);
}

// =================================================================================================
// 3. HÀM MAIN - MÔ PHỎNG LỚP SHAPE (LỚP 58)
// =================================================================================================
int main() {
    printf("--- Shape Layer 58 Simulation (C version) ---\n");

    // --- Cấp phát bộ nhớ cho tensor đầu ra ---
    int32_t* output_shape_tensor = (int32_t*)malloc(OUTPUT_SIZE * sizeof(int32_t));
    if (!output_shape_tensor) {
        fprintf(stderr, "Failed to allocate memory for output tensor.\n");
        return 1;
    }

    // --- Định nghĩa các chiều đầu vào ---
    const int32_t input_dims[INPUT_RANK] = INPUT_DIMS;

    // --- Logic chính: Gọi hàm get_shape từ thư viện ---
    printf("Performing Shape operation by calling library function...\n");
    get_shape(output_shape_tensor, input_dims, INPUT_RANK);
    
    printf("Input Shape (for Layer 58): [%d, %d, %d]\n", input_dims[0], input_dims[1], input_dims[2]);
    printf("Output Tensor (Shape): [%d, %d, %d]\n", 
           output_shape_tensor[0], output_shape_tensor[1], 
           output_shape_tensor[2]);

    // --- Đường dẫn file output ---
    const char* output_filename = "c:\\Code c\\Tensorflow\\golden_verilog\\op058_shape_ofm_c_sim.txt";

    // --- Ghi kết quả ---
    printf("Writing output tensor to %s...\n", output_filename);
    write_int32_array_to_file(output_filename, output_shape_tensor, OUTPUT_SIZE);

    printf("Shape layer 58 simulation finished successfully!\n");

    // --- Giải phóng bộ nhớ ---
    free(output_shape_tensor);

    return 0;
}