#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// =================================================================================================
// 1. ĐỊNH NGHĨA KÍCH THƯỚC TENSOR
// Dựa trên model_profile.txt cho lớp STRIDED_SLICE (LỚP 60)
// =================================================================================================
// Đầu vào là kết quả của lớp Shape, là một vector 1D
#define INPUT_SIZE 4 

// Các tham số begin, end, strides đều là vector 1D có 1 phần tử
#define PARAMS_SIZE 1

// Đầu ra của phép cắt này là một vector 1D có 2 phần tử
#define OUTPUT_SIZE 2

// =================================================================================================
// 2. CÁC HÀM ĐỌC/GHI FILE (Tái sử dụng từ các file trước)
// =================================================================================================

/**
 * @brief Đọc một mảng số nguyên 32-bit từ file text.
 * 
 * @param path Đường dẫn tới file input.
 * @param arr Con trỏ tới mảng để lưu dữ liệu.
 * @param size Số lượng phần tử cần đọc.
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
            exit(1);
        }
    }
    fclose(fp);
}

/**
 * @brief Ghi một mảng số nguyên 32-bit ra file text.
 * 
 * @param path Đường dẫn tới file output.
 * @param arr Mảng int32_t cần ghi.
 * @param size Số lượng phần tử trong mảng.
 */
void write_int32_array_to_file(const char* path, int32_t* arr, int size) {
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
// 3. LOGIC CHÍNH CỦA LỚP STRIDED_SLICE (ĐÃ TÁCH RA HÀM RIÊNG)
// =================================================================================================

/**
 * @brief Thực hiện phép cắt (slice) trên một tensor 1D int32.
 * 
 * @param output_tensor Con trỏ tới mảng để lưu kết quả đầu ra.
 * @param input_tensor Con trỏ tới mảng dữ liệu đầu vào.
 * @param begin_params Con trỏ tới mảng chứa chỉ số bắt đầu.
 * @param end_params Con trỏ tới mảng chứa chỉ số kết thúc.
 * @param strides_params Con trỏ tới mảng chứa bước nhảy.
 * @param output_size Kích thước của mảng đầu ra.
 */
void strided_slice_int32(
    int32_t* output_tensor,
    const int32_t* input_tensor,
    const int32_t* begin_params,
    const int32_t* end_params,
    const int32_t* strides_params,
    int output_size
) {
    int output_idx = 0;
    // Logic này giả định các tham số chỉ có 1 phần tử, đúng với model_profile.txt
    for (int i = begin_params[0]; i < end_params[0]; i += strides_params[0]) {
        if (output_idx < output_size) {
            output_tensor[output_idx] = input_tensor[i];
            output_idx++;
        }
    }
}


// =================================================================================================
// 4. HÀM MAIN - BỘ KIỂM THỬ (TEST HARNESS)
// =================================================================================================
int main() {
    printf("--- Strided Slice Layer Simulation (C version) ---\n");
    printf("--- NOTE: This is an INT32 operation, no quantization is performed. ---\n");

    // --- Cấp phát bộ nhớ ---
    int32_t* input_tensor = (int32_t*)malloc(INPUT_SIZE * sizeof(int32_t));
    int32_t* begin_params = (int32_t*)malloc(PARAMS_SIZE * sizeof(int32_t));
    int32_t* end_params = (int32_t*)malloc(PARAMS_SIZE * sizeof(int32_t));
    int32_t* strides_params = (int32_t*)malloc(PARAMS_SIZE * sizeof(int32_t));
    int32_t* output_tensor = (int32_t*)malloc(OUTPUT_SIZE * sizeof(int32_t));

    // --- Đường dẫn file ---
    const char* golden_verilog_path = "c:\\Code c\\Tensorflow\\golden_verilog\\";
    const char* params_path = "c:\\Code c\\Tensorflow\\framework_tensorflow_lite_CNN\\extracted_params\\layer060_STRIDED_SLICE_1_block4c_se_reshape_1_strided_slice\\";
    
    char input_file[256], begin_file[256], end_file[256], strides_file[256], output_file[256];

    // Đầu vào là output của lớp Shape 58
    sprintf(input_file, "%sop058_shape_ofm_c_sim.txt", golden_verilog_path); 
    
    // Các tham số của Strided Slice
    sprintf(begin_file, "%sweight_values.txt", params_path); // 'begin' được lưu trong weight_values
    sprintf(end_file, "%sparam_idx2_values.txt", params_path);   // 'end'
    sprintf(strides_file, "%sparam_idx3_values.txt", params_path); // 'strides'

    // File output
    sprintf(output_file, "%sop060_strided_slice_ofm_c_sim.txt", golden_verilog_path);

    // --- Đọc dữ liệu từ file ---
    printf("Reading input and parameter files...\n");
    read_int32_array_from_file(input_file, input_tensor, INPUT_SIZE);
    read_int32_array_from_file(begin_file, begin_params, PARAMS_SIZE);
    read_int32_array_from_file(end_file, end_params, PARAMS_SIZE);
    read_int32_array_from_file(strides_file, strides_params, PARAMS_SIZE);

    printf("Input Tensor: [%d, %d, %d, %d]\n", input_tensor[0], input_tensor[1], input_tensor[2], input_tensor[3]);
    printf("Slice Params: begin=[%d], end=[%d], strides=[%d]\n", begin_params[0], end_params[0], strides_params[0]);

    // --- Gọi hàm xử lý chính ---
    printf("Performing Strided Slice operation...\n");
    strided_slice_int32(
        output_tensor,
        input_tensor,
        begin_params,
        end_params,
        strides_params,
        OUTPUT_SIZE
    );
    
    printf("Output Tensor: [%d, %d]\n", output_tensor[0], output_tensor[1]);

    // --- Ghi kết quả ---
    printf("Writing output tensor to %s...\n", output_file);
    write_int32_array_to_file(output_file, output_tensor, OUTPUT_SIZE);

    printf("Strided Slice layer simulation finished successfully!\n");

    // --- Giải phóng bộ nhớ ---
    free(input_tensor);
    free(begin_params);
    free(end_params);
    free(strides_params);
    free(output_tensor);

    return 0;
}