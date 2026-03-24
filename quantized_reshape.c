#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

// =================================================================================================
// 1. ĐỊNH NGHĨA KÍCH THƯỚC TENSOR (Dựa trên Layer 30 từ model_profile.txt)
// =================================================================================================
// Input: Tensor[100] từ lớp Mean, shape [1, 1, 48]
#define INPUT_DIM0 1
#define INPUT_DIM1 1
#define INPUT_DIM2 48
#define INPUT_SIZE (INPUT_DIM0 * INPUT_DIM1 * INPUT_DIM2)

// Output: Tensor[102], shape [1, 1, 1, 48]
#define OUTPUT_DIM0 1
#define OUTPUT_DIM1 1
#define OUTPUT_DIM2 1
#define OUTPUT_DIM3 48
#define OUTPUT_SIZE (OUTPUT_DIM0 * OUTPUT_DIM1 * OUTPUT_DIM2 * OUTPUT_DIM3)

// =================================================================================================
// 2. CÁC HÀM ĐỌC/GHI FILE (Tái sử dụng từ các file trước)
// =================================================================================================

/**
 * @brief Đọc một mảng số nguyên 8-bit từ file text.
 */
void read_int8_array_from_file(const char* path, int8_t* arr, int size) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    int temp_val;
    for (int i = 0; i < size; ++i) {
        if (fscanf(fp, "%d", &temp_val) != 1) {
            fprintf(stderr, "Error: Failed to read element %d from %s\n", i, path);
            fclose(fp);
            exit(1);
        }
        arr[i] = (int8_t)temp_val;
    }
    fclose(fp);
}

/**
 * @brief Ghi một mảng số nguyên 8-bit ra file text.
 */
void write_int8_array_to_file(const char* path, int8_t* arr, int size) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open file %s\n", path);
        exit(1);
    }
    for (int i = 0; i < size; ++i) {
        fprintf(fp, "%d\n", arr[i]);
    }
    fclose(fp);
}

// =================================================================================================
// 3. HÀM LÕI QUANTIZED RESHAPE (Sẵn sàng để đóng gói)
// =================================================================================================

/**
 * @brief Thực hiện phép Reshape trên một tensor lượng tử hóa.
 *
 * Vì các tham số lượng tử hóa (scale, zero-point) của đầu vào và đầu ra
 * là giống hệt nhau, phép toán này chỉ đơn giản là sao chép dữ liệu.
 * Việc thay đổi shape được xử lý bởi cách các mảng được cấp phát và truy cập
 * ở bên ngoài hàm này.
 *
 * @param output_data Con trỏ tới buffer dữ liệu đầu ra.
 * @param input_data Con trỏ tới dữ liệu đầu vào.
 * @param num_elements Tổng số phần tử trong tensor (phải bằng nhau cho cả input và output).
 */
void quantized_reshape(
    int8_t* output_data,
    const int8_t* input_data,
    int num_elements
) {
    memcpy(output_data, input_data, num_elements * sizeof(int8_t));
}

// =================================================================================================
// 4. HÀM MAIN (TRÌNH CHẠY KIỂM THỬ CHO LỚP 30)
// =================================================================================================
int main() {
    printf("--- Quantized Reshape Layer 30 Simulation (C version) ---\n");
    printf("--- NOTE: This is a data copy operation, quantization params are preserved. ---\n");

    // --- Kiểm tra tính hợp lệ của Reshape ---
    if (INPUT_SIZE != OUTPUT_SIZE) {
        fprintf(stderr, "Error: Input and output tensor sizes must be the same for Reshape. (%d != %d)\n", INPUT_SIZE, OUTPUT_SIZE);
        return 1;
    }

    // --- Cấp phát bộ nhớ ---
    int8_t* input_tensor = (int8_t*)malloc(INPUT_SIZE * sizeof(int8_t));
    int8_t* output_tensor = (int8_t*)malloc(OUTPUT_SIZE * sizeof(int8_t));

    // --- Đường dẫn file ---
    // Đầu vào là output của lớp Mean (Layer 29)
    const char* input_filename = "c:\\Code c\\Tensorflow\\golden_verilog\\op029_mean_ofm_c_sim.txt";
    const char* output_filename = "c:\\Code c\\Tensorflow\\golden_verilog\\op030_reshape_ofm_c_sim.txt";

    // --- Đọc dữ liệu đầu vào ---
    // Lưu ý: Cần phải có file op029...txt trước khi chạy.
    printf("Reading input tensor from %s...\n", input_filename);
    read_int8_array_from_file(input_filename, input_tensor, INPUT_SIZE);

    // --- Thực thi Reshape bằng cách gọi hàm lõi ---
    printf("Performing Reshape operation (memcpy)...\n");
    quantized_reshape(
        output_tensor,
        input_tensor,
        INPUT_SIZE
    );

    // --- Ghi kết quả ---
    printf("Writing output tensor to %s...\n", output_filename);
    write_int8_array_to_file(output_filename, output_tensor, OUTPUT_SIZE);

    printf("Reshape layer 30 simulation finished successfully!\n");

    // --- Giải phóng bộ nhớ ---
    free(input_tensor);
    free(output_tensor);

    return 0;
}