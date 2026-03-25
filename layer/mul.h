#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "utils.h"
#include "requantize_utils.h"


#define K_MAX_MUL_BROADCAST_DIM 6

// Cấu trúc mô phỏng ArithmeticParams của TensorFlow Lite
typedef struct {
    int32_t input1_offset;
    int32_t input2_offset;
    int32_t output_offset;
    int32_t output_multiplier;
    int32_t output_shift;
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;
} ArithmeticParams;


// ----------------------------------------------------------------------------
// Fixed-point Arithmetic Simulator
// ----------------------------------------------------------------------------
// Sử dụng hàm MultiplyByQuantizedMultiplier từ requantize_utils.h
// thay vì định nghĩa lại ở đây.


// ----------------------------------------------------------------------------
// Luồng 1: Linear Memory Element-wise Multiplication
// ----------------------------------------------------------------------------
// Dùng khi shape của 2 tensor giống hệt nhau (flat memory array).
void MulElementwiseInt8(int size, const ArithmeticParams* params,
                        const int8_t* input1_data, const int8_t* input2_data,
                        int8_t* output_data) {
    for (int i = 0; i < size; ++i) {
        // 1. Phục hồi giá trị theo Zero-point offset
        const int32_t input1_val = params->input1_offset + input1_data[i];
        const int32_t input2_val = params->input2_offset + input2_data[i];

        // 2. Nhân raw integer & Rescale fixed-point
        const int32_t unclamped_result = params->output_offset +
            MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                          params->output_multiplier,
                                          params->output_shift);

        // 3. Fused Activation (Clamp Min/Max)
        const int32_t clamped_output = MIN(params->quantized_activation_max,
                                           MAX(params->quantized_activation_min, unclamped_result));
        
        // 4. Ép kiểu về kích thước thanh ghi lưu trữ
        output_data[i] = (int8_t)clamped_output;
    }
}

// ----------------------------------------------------------------------------
// Luồng 2: Strided Broadcast Multiplication (6D Slow)
// ----------------------------------------------------------------------------
// Dùng khi 2 tensor lệch shape (vd: [1, 64, 64, 3] x [3]). Yêu cầu padding ảo.
void BroadcastMulInt8(const ArithmeticParams* params,
                            const int8_t* input1_data,
                            int input1_size,
                            const int8_t* input2_data,
                            int input2_size,
                            int8_t* output_data) {
    if (input1_size % input2_size != 0) {
        // Chỉ hỗ trợ broadcast đơn giản: input1 là tensor lớn, input2 là vector kênh
        return; 
    }
    int channel_count = input2_size;
    int spatial_elements = input1_size / channel_count;

    for (int i = 0; i < spatial_elements; ++i) {
        for (int c = 0; c < channel_count; ++c) {
            int idx = i * channel_count + c;
            
            const int32_t input1_val = params->input1_offset + input1_data[idx];
            const int32_t input2_val = params->input2_offset + input2_data[c]; // Broadcasted input2

            const int32_t unclamped_result = params->output_offset +
                MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                              params->output_multiplier,
                                              params->output_shift);

            const int32_t clamped_output = MIN(params->quantized_activation_max,
                                               MAX(params->quantized_activation_min, unclamped_result));
            
            output_data[idx] = (int8_t)clamped_output;
        }
    }
}