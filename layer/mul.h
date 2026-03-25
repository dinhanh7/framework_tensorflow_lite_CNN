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

// Cấu trúc mô phỏng RuntimeShape/NdArrayDesc để xử lý Strides
typedef struct {
    int32_t ext_shape[K_MAX_MUL_BROADCAST_DIM];
    int32_t strides[K_MAX_MUL_BROADCAST_DIM];
} NdArrayDesc;

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
void BroadcastMul6DSlowInt8(const ArithmeticParams* params,
                            const NdArrayDesc* desc1, const int8_t* input1_data,
                            const NdArrayDesc* desc2, const int8_t* input2_data,
                            const int32_t* extended_output_shape_dims, // Array of 6 integers
                            int8_t* output_data) {
    
    size_t input1_offset_a = 0, input2_offset_a = 0, output_offset_a = 0;
    
    for (int a = 0; a < extended_output_shape_dims[0]; ++a) {
        size_t input1_offset_d = input1_offset_a;
        size_t input2_offset_d = input2_offset_a;
        size_t output_offset_d = output_offset_a;
        for (int d = 0; d < extended_output_shape_dims[1]; ++d) {
            size_t input1_offset_b = input1_offset_d;
            size_t input2_offset_b = input2_offset_d;
            size_t output_offset_b = output_offset_d;
            for (int b = 0; b < extended_output_shape_dims[2]; ++b) {
                size_t input1_offset_y = input1_offset_b;
                size_t input2_offset_y = input2_offset_b;
                size_t output_offset_y = output_offset_b;
                for (int y = 0; y < extended_output_shape_dims[3]; ++y) {
                    size_t input1_offset_x = input1_offset_y;
                    size_t input2_offset_x = input2_offset_y;
                    size_t output_offset_x = output_offset_y;
                    for (int x = 0; x < extended_output_shape_dims[4]; ++x) {
                        size_t input1_offset_c = input1_offset_x;
                        size_t input2_offset_c = input2_offset_x;
                        size_t output_offset_c = output_offset_x;
                        
                        // Chiều sâu nhất trong bộ nhớ ( innermost loop )
                        for (int c = 0; c < extended_output_shape_dims[5]; ++c) {
                            
                            // Datapath tương tự Elementwise
                            int32_t input1_val = params->input1_offset + input1_data[input1_offset_c];
                            int32_t input2_val = params->input2_offset + input2_data[input2_offset_c];
                            
                            int32_t unclamped_result = params->output_offset +
                                MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                                              params->output_multiplier,
                                                              params->output_shift);
                            
                            int32_t clamped_output = MIN(params->quantized_activation_max,
                                                         MAX(params->quantized_activation_min, unclamped_result));
                            
                            output_data[output_offset_c] = (int8_t)clamped_output;
                            
                            // Cập nhật con trỏ theo Strides
                            // Lưu ý: Nếu chiều bị broadcast, strides[5] sẽ bằng 0, 
                            // CPU sẽ lặp lại pointer cũ mà không tốn lệnh fetch vật lý mới.
                            input1_offset_c += desc1->strides[5];
                            input2_offset_c += desc2->strides[5];
                            ++output_offset_c;
                        }
                        input1_offset_x += desc1->strides[4];
                        input2_offset_x += desc2->strides[4];
                        output_offset_x += extended_output_shape_dims[5];
                    }
                    input1_offset_y += desc1->strides[3];
                    input2_offset_y += desc2->strides[3];
                    output_offset_y += extended_output_shape_dims[4] * extended_output_shape_dims[5];
                }
                input1_offset_b += desc1->strides[2];
                input2_offset_b += desc2->strides[2];
                output_offset_b += extended_output_shape_dims[3] * extended_output_shape_dims[4] * extended_output_shape_dims[5];
            }
            input1_offset_d += desc1->strides[1];
            input2_offset_d += desc2->strides[1];
            output_offset_d += extended_output_shape_dims[2] * extended_output_shape_dims[3] * extended_output_shape_dims[4] * extended_output_shape_dims[5];
        }
        input1_offset_a += desc1->strides[0];
        input2_offset_a += desc2->strides[0];
        output_offset_a += extended_output_shape_dims[1] * extended_output_shape_dims[2] * extended_output_shape_dims[3] * extended_output_shape_dims[4] * extended_output_shape_dims[5];
    }
}