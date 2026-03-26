#ifndef FC_H
#define FC_H

#include <stdint.h>
#include <assert.h>
#include "requantize_utils.h"
#include "utils.h"


typedef struct FullyConnectedParams {
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
} FullyConnectedParams;

#ifndef RUNTIME_SHAPE_DEFINED
#define RUNTIME_SHAPE_DEFINED
typedef struct RuntimeShape {
  int32_t dimensions_count;
  const int32_t* dims_data;
} RuntimeShape;
#endif

// Helper functions for shape access
static inline int32_t ShapeDims(const RuntimeShape* shape, int i) {
  return shape->dims_data[i];
}

static inline int ShapeDimensionsCount(const RuntimeShape* shape) {
  return shape->dimensions_count;
}

static inline int FlatSizeSkipDim(const RuntimeShape* shape, int skip_dim) {
  int result = 1;
  for (int i = 0; i < ShapeDimensionsCount(shape); ++i) {
    if (i != skip_dim) {
      result *= ShapeDims(shape, i);
    }
  }
  return result;
}

// For per-channel functions, since it is defined in quantization spec that
// weights are symmetric
// (https://www.tensorflow.org/lite/performance/quantization_spec#symmetric_vs_asymmetric),
// zero_point (params.weights_offset) is always 0.
// However, for per-tensor functions, params.weights_offset is still applied for
// backward compatibility.
static void FullyConnectedPerChannel(
    const FullyConnectedParams* params, const int32_t* output_multiplier,
    const int* output_shift, const RuntimeShape* input_shape,
    const int8_t* input_data, const RuntimeShape* filter_shape,
    const int8_t* filter_data, const RuntimeShape* bias_shape,
    const int32_t* bias_data, const RuntimeShape* output_shape,
    int8_t* output_data) {
  const int32_t input_offset = params->input_offset;
  const int32_t output_offset = params->output_offset;
  const int32_t output_activation_min = params->quantized_activation_min;
  const int32_t output_activation_max = params->quantized_activation_max;
  TFLITE_DCHECK_GE(ShapeDimensionsCount(filter_shape), 2);
  TFLITE_DCHECK_EQ(ShapeDimensionsCount(output_shape), 2);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = ShapeDimensionsCount(filter_shape);
  const int batches = ShapeDims(output_shape, 0);
  const int output_depth = ShapeDims(output_shape, 1);
  TFLITE_DCHECK_LE(output_depth, ShapeDims(filter_shape, filter_dim_count - 2));
  const int accum_depth = ShapeDims(filter_shape, filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += filter_val * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled = MultiplyByQuantizedMultiplier(
          acc, output_multiplier[out_c], output_shift[out_c]);
      acc_scaled += output_offset;
      acc_scaled = MAX(acc_scaled, output_activation_min);
      acc_scaled = MIN(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = (int8_t)acc_scaled;
    }
  }
}

static void FullyConnected(const FullyConnectedParams* params,
                           const RuntimeShape* input_shape,
                           const int8_t* input_data,
                           const RuntimeShape* filter_shape,
                           const int8_t* filter_data,
                           const RuntimeShape* bias_shape, const int32_t* bias_data,
                           const RuntimeShape* output_shape, int8_t* output_data) {
  const int32_t input_offset = params->input_offset;
  const int32_t filter_offset = params->weights_offset;
  const int32_t output_offset = params->output_offset;
  const int32_t output_multiplier = params->output_multiplier;
  const int output_shift = params->output_shift;
  const int32_t output_activation_min = params->quantized_activation_min;
  const int32_t output_activation_max = params->quantized_activation_max;
  TFLITE_DCHECK_GE(ShapeDimensionsCount(filter_shape), 2);
  TFLITE_DCHECK_GE(ShapeDimensionsCount(output_shape), 1);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int filter_dim_count = ShapeDimensionsCount(filter_shape);
  const int output_dim_count = ShapeDimensionsCount(output_shape);
  const int batches = FlatSizeSkipDim(output_shape, output_dim_count - 1);
  const int output_depth = ShapeDims(output_shape, output_dim_count - 1);
  TFLITE_DCHECK_LE(output_depth, ShapeDims(filter_shape, filter_dim_count - 2));
  const int accum_depth = ShapeDims(filter_shape, filter_dim_count - 1);
  for (int b = 0; b < batches; ++b) {
    for (int out_c = 0; out_c < output_depth; ++out_c) {
      int32_t acc = 0;
      for (int d = 0; d < accum_depth; ++d) {
        int32_t input_val = input_data[b * accum_depth + d];
        int32_t filter_val = filter_data[out_c * accum_depth + d];
        acc += (filter_val + filter_offset) * (input_val + input_offset);
      }
      if (bias_data) {
        acc += bias_data[out_c];
      }
      int32_t acc_scaled =
          MultiplyByQuantizedMultiplier(acc, output_multiplier, output_shift);
      acc_scaled += output_offset;
      acc_scaled = MAX(acc_scaled, output_activation_min);
      acc_scaled = MIN(acc_scaled, output_activation_max);
      output_data[out_c + output_depth * b] = (int8_t)acc_scaled;
    }
  }
}

#endif
