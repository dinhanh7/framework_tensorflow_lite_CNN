#ifndef ADD_H_
#define ADD_H_

#include <stdint.h>
#include <stddef.h>

// Các cấu trúc dữ liệu mô phỏng TensorFlow Lite Micro
struct RuntimeShape {
    int32_t dimensions_count;
    const int32_t* dims_data;
};

struct ArithmeticParams {
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

// Hàm phụ trợ cho việc dịch và nhân bit dùng chung (cho C)
// Hàm này tương đương MultiplyByQuantizedMultiplier của TFLite
static inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    
    int64_t total = (int64_t)x * (int64_t)quantized_multiplier;
    int32_t result = (int32_t)((total + (1LL << 30)) >> 31);
    
    if (left_shift > 0) {
        result <<= left_shift;
    }
    if (right_shift > 0) {
        result = (int32_t)(((int64_t)result + (1LL << (right_shift - 1))) >> right_shift);
    }
    return result;
}

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
static inline void AddFunc(const struct ArithmeticParams* params,
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
                                  const struct ArithmeticParams* params,
                                  const int8_t* input1_data,
                                  const int8_t* input2_data,
                                  int8_t* output_data) {
    // Vòng lặp áp dụng AddFunc lên từng cặp số nguyên
    for (int i = 0; i < total_elements; ++i) {
        AddFunc(params, input1_data[i], input2_data[i], &output_data[i]);
    }
}

// 5. Hàm Add: Giao diện chính
static inline void Add(const struct ArithmeticParams* params,
                       const struct RuntimeShape* input1_shape, const int8_t* input1_data,
                       const struct RuntimeShape* input2_shape, const int8_t* input2_data,
                       const struct RuntimeShape* output_shape, int8_t* output_data) {
    
    // Tính toán tổng số phần tử của tensor
    int total_elements = MatchingElementsSize(input1_shape, input2_shape, output_shape);
    
    // Chuyển việc cho AddElementwise
    AddElementwise(total_elements, params, input1_data, input2_data, output_data);
}

#endif  // TENSORFLOW_LITE_MICRO_ADD_H_
