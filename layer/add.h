#ifndef ADD_H_
#define ADD_H_

#include <stdint.h>
#include <stddef.h>
#include "requantize_utils.h"

// Các cấu trúc dữ liệu mô phỏng TensorFlow Lite Micro
struct RuntimeShape {
    int32_t dimensions_count;
    const int32_t* dims_data;
};

struct AddArithmeticParams {
    int32_t input1_zp;
    int32_t input2_zp;
    int32_t output_zp;
    
    int32_t input1_multiplier;
    int input1_shift;
    
    int32_t input2_multiplier;
    int input2_shift;
    
    int32_t output_multiplier;
    int output_shift;
    
    int left_shift;
    
    int32_t quantized_activation_min;
    int32_t quantized_activation_max;
};

// Helper: Calculate Arithmetic Params for ADD
void CalculateAddArithmeticParams(float input1_scale, int32_t input1_zp,
                               float input2_scale, int32_t input2_zp,
                               float output_scale, int32_t output_zp,
                               struct AddArithmeticParams* params) {
    params->input1_zp = input1_zp;
    params->input2_zp = input2_zp;
    params->output_zp = output_zp;
    
    params->left_shift = 20;

    double twice_max_input_scale = 2 * (input1_scale > input2_scale ? input1_scale : input2_scale);
    double real_input1_multiplier = input1_scale / twice_max_input_scale;
    double real_input2_multiplier = input2_scale / twice_max_input_scale;
    double real_output_multiplier = twice_max_input_scale / ((1 << params->left_shift) * output_scale);

    QuantizeMultiplier(real_input1_multiplier, &params->input1_multiplier, &params->input1_shift);
    QuantizeMultiplier(real_input2_multiplier, &params->input2_multiplier, &params->input2_shift);
    QuantizeMultiplier(real_output_multiplier, &params->output_multiplier, &params->output_shift);

    params->quantized_activation_min = -128;
    params->quantized_activation_max = 127;
}
// Hàm này tương đương MultiplyByQuantizedMultiplier của TFLite
// Đã được thay thế bằng include "requantize_utils.h"

// 1. Hàm tính toán tổng số phần tử (MatchingElementsSize)
static inline int MatchingElementsSize(const struct RuntimeShape* shape1,
                                       const struct RuntimeShape* shape2,
                                       const struct RuntimeShape* output_shape) {
    int num_elements = 1;
    for (int i = 0; i < output_shape->dimensions_count; ++i) {
        num_elements *= output_shape->dims_data[i];
    }
    return num_elements;
}


// 3. Hàm AddFunc: Trái tim toán học của toàn bộ file
static inline void AddFunc(const struct AddArithmeticParams* params,
                           int8_t input1_val, int8_t input2_val,
                           int8_t* output_val) {
    // Bước 1: Thêm zp cho đầu vào
    int32_t input1_shifted = (int32_t)input1_val + params->input1_zp;
    int32_t input2_shifted = (int32_t)input2_val + params->input2_zp;

    // Bước 2: Dịch bit và nhân với tham số tỷ lệ (scale multiplier)
    if (params->left_shift > 0) {
        input1_shifted <<= params->left_shift;
        input2_shifted <<= params->left_shift;
    }
    
    int32_t input1_scaled = MultiplyByQuantizedMultiplier(input1_shifted, params->input1_multiplier, params->input1_shift);
    int32_t input2_scaled = MultiplyByQuantizedMultiplier(input2_shifted, params->input2_multiplier, params->input2_shift);

    // Bước 3: Cộng kết quả lại, sau đó scale về đầu ra
    int32_t raw_sum = input1_scaled + input2_scaled;
    
    int32_t output = MultiplyByQuantizedMultiplier(raw_sum, params->output_multiplier, params->output_shift);
    output += params->output_zp;

    // Bước 4: Giới hạn giá trị đầu ra (clamping) trong phạm vi từ -128 đến 127
    if (output < params->quantized_activation_min) {
        output = params->quantized_activation_min;
    }
    if (output > params->quantized_activation_max) {
        output = params->quantized_activation_max;
    }

    *output_val = (int8_t)output;
}

// 4. Hàm AddElementwise: Thiết lập vòng lặp duyệt qua tất cả các phần tử
static inline void AddElementwise(int total_elements,
                                  const struct AddArithmeticParams* params,
                                  const int8_t* input1_data,
                                  const int8_t* input2_data,
                                  int8_t* output_data) {
    // Vòng lặp áp dụng AddFunc lên từng cặp số nguyên
    for (int i = 0; i < total_elements; ++i) {
        AddFunc(params, input1_data[i], input2_data[i], &output_data[i]);
    }
}

// 5. Hàm Add: Giao diện chính
static inline void Add(const struct AddArithmeticParams* params,
                       const struct RuntimeShape* input1_shape, const int8_t* input1_data,
                       const struct RuntimeShape* input2_shape, const int8_t* input2_data,
                       const struct RuntimeShape* output_shape, int8_t* output_data) {
    
    // Tính toán tổng số phần tử của tensor
    int total_elements = MatchingElementsSize(input1_shape, input2_shape, output_shape);
    
    // Chuyển việc cho AddElementwise
    AddElementwise(total_elements, params, input1_data, input2_data, output_data);
}

#endif  // TENSORFLOW_LITE_MICRO_ADD_H_ 
