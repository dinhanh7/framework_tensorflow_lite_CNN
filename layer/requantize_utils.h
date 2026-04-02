#ifndef REQUANTIZE_UTILS_H
#define REQUANTIZE_UTILS_H
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <math.h>

// Tái hiện hàm `SaturatingRoundingDoublingHighMul` (Đã đồng bộ với phiên bản TFLite chính xác)
static int32_t SaturatingRoundingDoublingHighMul(int32_t a, int32_t b) {
    const int64_t a_64 = a;
    const int64_t b_64 = b;
    const int64_t ab_64 = a_64 * b_64;
    const int overflow = (a == INT32_MIN && b == INT32_MIN);
    const int64_t nudge = (ab_64 >= 0) ? (1LL << 30) : (1LL - (1LL << 30));
    const int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) >> 31);
    return overflow ? INT32_MAX : ab_x2_high32;
}

static int32_t SaturatingRoundingDoublingHighMulConvDw(int32_t a, int32_t b) {
    const int64_t a_64 = a;
    const int64_t b_64 = b;
    const int64_t ab_64 = a_64 * b_64;
    const int overflow = (a == INT32_MIN && b == INT32_MIN);
    const int64_t nudge = (1LL << 30);
    const int32_t ab_x2_high32 = (int32_t)((ab_64 + nudge) >> 31);
    return overflow ? INT32_MAX : ab_x2_high32;
}


static int32_t RoundingRightShift(int32_t x, int shift) {
    if (shift <= 0) return x;
    int32_t mask = (1 << shift) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1);
    return (x >> shift) + (remainder > threshold ? 1 : 0);
}

// Tái hiện hàm `RoundingRightShift` (Symmetric Rounding - giống gemmlowp::RoundingDivideByPOT)
// Giúp xử lý các trường hợp làm tròn số âm chính xác hơn so với (x + nudge) >> shift
static int32_t RoundingRightShiftDWConv(int32_t x, int shift) {
    if (shift <= 0) return x;
    int32_t mask = (1 << shift) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    return (x >> shift) + (remainder > threshold ? 1 : 0);
}

static int32_t RoundingRightShiftConv(int32_t x, int8_t shift) {
    if (shift <= 0) {
        return x;
    }
    int64_t nudge = (int64_t)1 << (shift - 1);
    int64_t result_64 = (x + nudge) >> shift;
    return (int32_t)result_64;
}

static inline int32_t MultiplyByQuantizedMultiplierDWConv(int32_t x, int32_t quantized_multiplier, int32_t shift) {
    int32_t left_shift = (shift > 0) ? shift : 0;
    int32_t right_shift = (shift < 0) ? -shift : 0;
    int64_t x_shifted_64 = (int64_t)x << left_shift;
    int32_t x_shifted_32;
    if (x_shifted_64 > 2147483647LL) {
        x_shifted_32 = 2147483647;
    } else if (x_shifted_64 < -2147483648LL) {
        x_shifted_32 = -2147483648LL;
    } else {
        x_shifted_32 = (int32_t)x_shifted_64;
    }
    int32_t high_mul = SaturatingRoundingDoublingHighMulConvDw(x_shifted_32, quantized_multiplier);
    return RoundingRightShiftDWConv(high_mul, right_shift);
}

static inline int32_t MultiplyByQuantizedMultiplierConv(int32_t x, int32_t quantized_multiplier, int32_t shift) {
    int32_t left_shift = (shift > 0) ? shift : 0;
    int32_t right_shift = (shift < 0) ? -shift : 0;
    int64_t x_shifted_64 = (int64_t)x << left_shift;
    int32_t x_shifted_32;
    if (x_shifted_64 > 2147483647LL) {
        x_shifted_32 = 2147483647;
    } else if (x_shifted_64 < -2147483648LL) {
        x_shifted_32 = -2147483648LL;
    } else {
        x_shifted_32 = (int32_t)x_shifted_64;
    }
    int32_t high_mul = SaturatingRoundingDoublingHighMulConvDw(x_shifted_32, quantized_multiplier);
    return RoundingRightShiftConv(high_mul, right_shift);
}

static inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int32_t shift) {
    int32_t left_shift = (shift > 0) ? shift : 0;
    int32_t right_shift = (shift < 0) ? -shift : 0;
    int64_t x_shifted_64 = (int64_t)x << left_shift;
    int32_t x_shifted_32;
    if (x_shifted_64 > 2147483647LL) {
        x_shifted_32 = 2147483647;
    } else if (x_shifted_64 < -2147483648LL) {
        x_shifted_32 = -2147483648LL;
    } else {
        x_shifted_32 = (int32_t)x_shifted_64;
    }
    int32_t high_mul = SaturatingRoundingDoublingHighMul(x_shifted_32, quantized_multiplier);
    return RoundingRightShift(high_mul, right_shift);
}

// Hàm kẹp giá trị trong khoảng int8_t
static inline int8_t clip_int8(int32_t val) {
    if (val > 127) return 127;
    if (val < -128) return -128;
    return (int8_t)val;
}

// Hàm kẹp giá trị trong khoảng int16_t
static inline int16_t clamp_int16(int32_t val) {
    if (val > 32767) return 32767;
    if (val < -32768) return -32768;
    return (int16_t)val;
}

// Hàm nhân bão hòa KHÔNG LÀM TRÒN (Truncating) cho int16
static inline int16_t SaturatingDoublingHighMul_Int16_trunc(int16_t a, int16_t b) {
    if (a == -32768 && b == -32768) return 32767;
    int32_t ab = (int32_t)a * (int32_t)b;
    int32_t result = ab / 32768;
    return clamp_int16(result);
}

// Hàm nhân bão hòa CÓ LÀM TRÒN (Rounding) cho int16
static inline int16_t SaturatingRoundingDoublingHighMul_Int16_round(int16_t a, int16_t b) {
    if (a == -32768 && b == -32768) return 32767;
    int32_t ab = (int32_t)a * (int32_t)b;
    int32_t nudge = 16384;
    int32_t result = (ab + nudge) >> 15;
    return clamp_int16(result);
}

// Hàm chia cho lũy thừa của 2 CÓ LÀM TRÒN
static inline int16_t RoundingDivideByPOT(int16_t x, int exponent) {
    if (exponent == 0) return x;
    if (exponent < 0) return clamp_int16((int32_t)x << -exponent);
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t threshold = (mask >> 1) + (x < 0 ? 1 : 0);
    int32_t result = (x >> exponent) + (remainder > threshold ? 1 : 0);
    return clamp_int16(result);
}

static void QuantizeMultiplier(double double_multiplier, int32_t* quantized_multiplier, int* shift) {
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }
    // frexp trả về q trong khoảng [0.5, 1) sao cho double_multiplier = q * 2^shift
    const double q = frexp(double_multiplier, shift);
    
    // Chuyển q về dạng fixed-point 31 bit (Q0.31)
    // q * 2^31 sẽ nằm trong khoảng [2^30, 2^31)
    int64_t q_fixed = (int64_t)(round(q * (1LL << 31)));
    
    // Trường hợp đặc biệt: Nếu làm tròn lên chạm ngưỡng 2^31
    if (q_fixed == (1LL << 31)) {
        q_fixed /= 2;
        *shift += 1;
    }
    
    // Đảm bảo không tràn int32 (mặc dù q_fixed <= 2^31-1 sau bước trên)
    if (q_fixed > 2147483647LL) q_fixed = 2147483647LL; 
    
    *quantized_multiplier = (int32_t)q_fixed;
}

static inline int CountLeadingZeros64(uint64_t x) {
    if (x == 0) return 64;
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#else
    int count = 0;
    while ((x & (1ULL << 63)) == 0) {
        x <<= 1;
        ++count;
    }
    return count;
#endif
}
int16_t SaturatingLeftShift(int16_t val, int shift) {
    int32_t res = (int32_t)val * (1 << shift);
    return clamp_int16(res);
}

// Hàm helper mô phỏng phép chia cho lũy thừa của 2 với làm tròn cho int32
static inline int32_t RoundingDivideByPOT_32(int32_t x, int exponent) {
    if (exponent <= 0) return x;
    int32_t mask = (1 << exponent) - 1;
    int32_t remainder = x & mask;
    int32_t bias = (1 << (exponent - 1));
    int32_t result = x >> exponent;
    if (remainder > bias || (remainder == bias && (result & 1))) {
        result++;
    }
    return result;
}
#endif
