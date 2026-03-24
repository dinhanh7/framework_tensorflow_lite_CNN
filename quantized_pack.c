#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "cnn_layers.h" // Thêm thư viện CNN của chúng ta

// =================================================================================================
// 1. ĐỊNH NGHĨA KÍCH THƯỚC TENSOR
// Dựa trên model_profile.txt cho lớp PACK (LỚP 62)
// =================================================================================================
#define INPUT0_SIZE 2 // Shape của op060_strided_slice_ofm
#define INPUT1_SIZE 1 // Shape của op061_strided_slice_ofm
#define NUM_INPUTS 2

// Output shape là [2, 2]
#define OUTPUT_ROWS 2
#define OUTPUT_COLS 2
#define OUTPUT_SIZE (OUTPUT_ROWS * OUTPUT_COLS)

// =================================================================================================
// 2. CÁC HÀM ĐỌC/GHI FILE (Tái sử dụng từ các file trước)
// =================================================================================================

/**
 * @brief Đọc một mảng số nguyên 32-bit từ file text.
 */
void read_int32_array_from_file(const char* path, int32_t* arr, int size) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    for (int i = 0; i < size; ++i) {
        if (fscanf(fp, "%d", &arr[i]) != 1) {
            fprintf(stderr, "Error: Failed to read element %d from %s\n", i, path);
            fclose(fp);
            exit(1);
        }
    }
    fclose(fp);
}

/**
 * @brief Ghi một mảng số nguyên 32-bit ra file text.
 */
void write_int32_array_to_file(const char* path, int32_t* arr, int size) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    // Ghi dưới dạng ma trận 2x2 để dễ đọc
    for (int i = 0; i < size; ++i) {
        fprintf(fp, "%d ", arr[i]);
        if ((i + 1) % OUTPUT_COLS == 0) {
            fprintf(fp, "\n");
        }
    }
    fclose(fp);
}

// =================================================================================================
// 3. HÀM MAIN (TRÌNH CHẠY KIỂM THỬ)
// =================================================================================================
int main() {
    printf("--- Pack Layer Simulation (C version) ---\n");
    printf("--- NOTE: This is an INT32 operation with padding, no quantization. ---\n");

    // --- Cấp phát bộ nhớ ---
    int32_t* input0_tensor = (int32_t*)malloc(INPUT0_SIZE * sizeof(int32_t));
    int32_t* input1_tensor = (int32_t*)malloc(INPUT1_SIZE * sizeof(int32_t));
    int32_t* output_tensor = (int32_t*)malloc(OUTPUT_SIZE * sizeof(int32_t));

    // --- Đường dẫn file ---
    // LƯU Ý: File op061 chưa được tạo, chúng ta sẽ giả lập nó.
    const char* input0_filename = "c:\\Code c\\Tensorflow\\golden_verilog\\op060_strided_slice_ofm_c_sim.txt";
    // const char* input1_filename = "c:\\Code c\\Tensorflow\\golden_verilog\\op061_strided_slice_ofm_c_sim.txt";
    const char* output_filename = "c:\\Code c\\Tensorflow\\golden_verilog\\op062_pack_ofm_c_sim.txt";

    // --- Đọc dữ liệu đầu vào ---
    printf("Reading input tensor 0 from %s...\n", input0_filename);
    read_int32_array_from_file(input0_filename, input0_tensor, INPUT0_SIZE);
    
    // Do file cho op061 chưa tồn tại, ta hardcode giá trị của nó theo model_profile
    // Layer 61: STRIDED_SLICE(input=op058, begin=[3], end=[4]) -> trích xuất phần tử cuối cùng là 3
    printf("Simulating input tensor 1 with value [3]\n");
    input1_tensor[0] = 3;
    // read_int32_array_from_file(input1_filename, input1_tensor, INPUT1_SIZE);

    printf("Input Tensor 0: [%d, %d]\n", input0_tensor[0], input0_tensor[1]);
    printf("Input Tensor 1: [%d]\n", input1_tensor[0]);

    // --- Thực thi phép tính Pack bằng cách gọi hàm từ thư viện ---
    printf("Performing Pack operation by calling the library function...\n");
    pack_int32_with_padding(
        output_tensor,
        input0_tensor,
        INPUT0_SIZE,
        input1_tensor,
        INPUT1_SIZE,
        OUTPUT_COLS
    );

    printf("Output Tensor (2x2):\n%d %d\n%d %d\n", 
           output_tensor[0], output_tensor[1], 
           output_tensor[2], output_tensor[3]);

    // --- Ghi kết quả ---
    printf("Writing output tensor to %s...\n", output_filename);
    write_int32_array_to_file(output_filename, output_tensor, OUTPUT_SIZE);

    printf("Pack layer simulation finished successfully!\n");

    // --- Giải phóng bộ nhớ ---
    free(input0_tensor);
    free(input1_tensor);
    free(output_tensor);

    return 0;
}