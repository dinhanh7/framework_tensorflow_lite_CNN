/**
 * ============================================================================
 * TFLite Model Sequence Simulator - EfficientNetV2 B0 (INT8)
 * Layers 4 ~ 94
 * ============================================================================
 * 
 * QUAN TRỌNG: Chỉ số layer
 * - Model layer bắt đầu từ 0 → Layer K (model) = Layer K+1 (params folder)
 * - Layer 4 (model) = layer005_... (params folder)
 * - Layer 94 (model) = layer095_... (params folder)
 * 
 * Chiến lược xử lý shape layers:
 * - RESHAPE, SHAPE, STRIDED_SLICE, PACK → Không thay đổi dữ liệu, chỉ skip
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// Include layer headers
#include "layer/conv2d.h"
#include "layer/hardswish.h"
#include "layer/dw_conv.h"
#include "layer/add.h"
#include "layer/mul.h"
#include "layer/mean.h"
#include "layer/reshape.h"

// ============================================================================
// CONFIG & CONSTANTS
// ============================================================================

#define DATA_PATH "extracted_params_hsigmoid/"
#define IFM_FILE "all_layer_io/layer_5_CONV_2D/ifm.txt"

// Tensor structure
typedef struct {
    int8_t* data;
    int height, width, channels, batch;
    float scale;
    int32_t zp;
} Tensor;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

Tensor* allocate_tensor(int batch, int height, int width, int channels,
                       float scale, int32_t zp) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    int size = batch * height * width * channels;
    t->data = (int8_t*)malloc(size * sizeof(int8_t));
    t->batch = batch;
    t->height = height;
    t->width = width;
    t->channels = channels;
    t->scale = scale;
    t->zp = zp;
    return t;
}

void free_tensor(Tensor* t) {
    if (t) {
        if (t->data) free(t->data);
        free(t);
    }
}

int8_t* read_quantized_values(const char* filepath, int expected_size) {
    FILE* f = fopen(filepath, "r");
    if (!f) {
        printf("ERROR: Cannot open %s\n", filepath);
        return NULL;
    }
    
    int8_t* data = (int8_t*)malloc(expected_size * sizeof(int8_t));
    for (int i = 0; i < expected_size; i++) {
        int val;
        if (fscanf(f, "%d", &val) != 1) {
            printf("ERROR: Read failed at index %d in %s\n", i, filepath);
            fclose(f);
            free(data);
            return NULL;
        }
        data[i] = (int8_t)val;
    }
    fclose(f);
    return data;
}

int32_t read_int32_value(const char* filepath) {
    FILE* f = fopen(filepath, "r");
    if (!f) return 0;
    int32_t val = 0;
    fscanf(f, "%d", &val);
    fclose(f);
    return val;
}

float read_float_value(const char* filepath) {
    FILE* f = fopen(filepath, "r");
    if (!f) return 0.0f;
    float val = 0.0f;
    fscanf(f, "%f", &val);
    fclose(f);
    return val;
}

int32_t* read_int32_array(const char* filepath, int expected_size) {
    FILE* f = fopen(filepath, "r");
    if (!f) return NULL;
    
    int32_t* data = (int32_t*)malloc(expected_size * sizeof(int32_t));
    for (int i = 0; i < expected_size; i++) {
        if (fscanf(f, "%d", &data[i]) != 1) {
            printf("ERROR: Read failed at index %d in %s\n", i, filepath);
            fclose(f);
            free(data);
            return NULL;
        }
    }
    fclose(f);
    return data;
}

Tensor* load_input_from_file(const char* filepath, int batch, int height, 
                             int width, int channels, float scale, int32_t zp) {
    int size = batch * height * width * channels;
    int8_t* data = read_quantized_values(filepath, size);
    if (!data) return NULL;
    
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));
    t->data = data;
    t->batch = batch;
    t->height = height;
    t->width = width;
    t->channels = channels;
    t->scale = scale;
    t->zp = zp;
    return t;
}

// ============================================================================
// LAYER EXECUTION FUNCTIONS
// ============================================================================

/**
 * Generic CONV2D execution
 */
Tensor* layer_conv2d(Tensor* input, const char* param_folder, 
                     int out_h, int out_w, int out_ch,
                     int kernel_h, int kernel_w, int stride_h, int stride_w,
                     int layer_num) {
    printf("\n=== [LAYER %d] CONV_2D (k=%d, s=%d) ===\n", 
           layer_num, kernel_h, stride_h);
    fflush(stdout);
    
    char path[512];
    int weight_size = kernel_h * kernel_w * input->channels * out_ch;
    
    sprintf(path, "%s%s/weight_values.txt", DATA_PATH, param_folder);
    int8_t* weights = read_quantized_values(path, weight_size);
    
    sprintf(path, "%s%s/effective_bias.txt", DATA_PATH, param_folder);
    int32_t* eff_bias = read_int32_array(path, out_ch);
    
    sprintf(path, "%s%s/multiplier.txt", DATA_PATH, param_folder);
    int32_t* mult = read_int32_array(path, out_ch);
    
    sprintf(path, "%s%s/shift.txt", DATA_PATH, param_folder);
    int8_t* shft = read_quantized_values(path, out_ch);
    
    sprintf(path, "%s%s/ofm_scale.txt", DATA_PATH, param_folder);
    float ofm_scale = read_float_value(path);
    
    sprintf(path, "%s%s/ofm_zp.txt", DATA_PATH, param_folder);
    int32_t ofm_zp = read_int32_value(path);
    
    Tensor* output = allocate_tensor(1, out_h, out_w, out_ch, ofm_scale, ofm_zp);
    int out_h_calc, out_w_calc;
    
    quantized_conv2d(output->data, &out_h_calc, &out_w_calc,
        input->data, input->height, input->width, input->channels,
        weights, out_ch, kernel_h, kernel_w,
        eff_bias, mult, shft, input->zp, ofm_zp,
        stride_h, stride_w, "SAME");
    
    free(weights);
    free(eff_bias);
    free(mult);
    free(shft);
    
    return output;
}

/**
 * Generic HARD_SWISH execution
 */
Tensor* layer_hardswish(Tensor* input, const char* param_folder, int layer_num) {
    printf("\n=== [LAYER %d] HARD_SWISH ===\n", layer_num);
    fflush(stdout);
    
    char path[512];
    sprintf(path, "%s%s/ofm_scale.txt", DATA_PATH, param_folder);
    float ofm_scale = read_float_value(path);
    
    sprintf(path, "%s%s/ofm_zp.txt", DATA_PATH, param_folder);
    int32_t ofm_zp = read_int32_value(path);
    
    int size = input->batch * input->height * input->width * input->channels;
    Tensor* output = allocate_tensor(input->batch, input->height,
                                    input->width, input->channels,
                                    ofm_scale, ofm_zp);
    
    quantized_hardswish(input->data, output->data, size,
        input->zp, ofm_zp, input->scale, ofm_scale);
    
    return output;
}

/**
 * Generic ADD execution (skip connection)
 */
Tensor* layer_add(Tensor* main, Tensor* skip, const char* param_folder, int layer_num) {
    printf("\n=== [LAYER %d] ADD (Skip Connection) ===\n", layer_num);
    fflush(stdout);
    
    char path[512];
    sprintf(path, "%s%s/ofm_scale.txt", DATA_PATH, param_folder);
    float ofm_scale = read_float_value(path);
    
    sprintf(path, "%s%s/ofm_zp.txt", DATA_PATH, param_folder);
    int32_t ofm_zp = read_int32_value(path);
    
    int size = main->batch * main->height * main->width * main->channels;
    Tensor* output = allocate_tensor(main->batch, main->height,
                                    main->width, main->channels,
                                    ofm_scale, ofm_zp);
    
    // Simple add with saturation
    for (int i = 0; i < size; i++) {
        int32_t val = (int32_t)main->data[i] + (int32_t)skip->data[i];
        output->data[i] = (int8_t)(val > 127 ? 127 : (val < -128 ? -128 : val));
    }
    
    return output;
}

/**
 * Skip shape operations (just reuse data pointer with new shape metadata)
 */
Tensor* layer_skip_shape(Tensor* input, int new_h, int new_w, int new_ch,
                         int layer_num, const char* op_name) {
    printf("\n=== [LAYER %d] %s (SKIP - shape only) ===\n", layer_num, op_name);
    fflush(stdout);
    
    Tensor* output = (Tensor*)malloc(sizeof(Tensor));
    output->data = input->data;  // Reuse same data pointer
    output->batch = input->batch;
    output->height = new_h;
    output->width = new_w;
    output->channels = new_ch;
    output->scale = input->scale;
    output->zp = input->zp;
    
    return output;
}

/**
 * Generic DEPTHWISE_CONV_2D execution
 */
Tensor* layer_depthwise_conv2d(Tensor* input, const char* param_folder,
                               int out_h, int out_w, int out_ch,
                               int kernel_h, int kernel_w, int stride_h, int stride_w,
                               int layer_num) {
    printf("\n=== [LAYER %d] DEPTHWISE_CONV_2D (k=%d, s=%d) ===\n",
           layer_num, kernel_h, stride_h);
    fflush(stdout);
    
    char path[512];
    int weight_size = kernel_h * kernel_w * out_ch;  // For DW: [1, H, W, C]
    
    sprintf(path, "%s%s/weight_values.txt", DATA_PATH, param_folder);
    int8_t* weights = read_quantized_values(path, weight_size);
    
    sprintf(path, "%s%s/effective_bias.txt", DATA_PATH, param_folder);
    int32_t* eff_bias = read_int32_array(path, out_ch);
    
    sprintf(path, "%s%s/multiplier.txt", DATA_PATH, param_folder);
    int32_t* mult = read_int32_array(path, out_ch);
    
    sprintf(path, "%s%s/shift.txt", DATA_PATH, param_folder);
    int32_t* shft = read_int32_array(path, out_ch);
    
    sprintf(path, "%s%s/ofm_scale.txt", DATA_PATH, param_folder);
    float ofm_scale = read_float_value(path);
    
    sprintf(path, "%s%s/ofm_zp.txt", DATA_PATH, param_folder);
    int32_t ofm_zp = read_int32_value(path);
    
    Tensor* output = allocate_tensor(1, out_h, out_w, out_ch, ofm_scale, ofm_zp);
    
    // Setup DW params
    DepthwiseParams params = { .input_zp = input->zp,
        .output_zp = ofm_zp,
        .quantized_activation_min = -128,
        .quantized_activation_max = 127,
        .stride_width = stride_w,
        .stride_height = stride_h,
        .pad_width = 0,
        .pad_height = 0,
        .dilation_width_factor = 1,
        .dilation_height_factor = 1
    };
    
    DepthwiseConvPerChannel(&params, mult, shft,
        1, input->height, input->width, input->channels, input->data,
        kernel_h, kernel_w, out_ch, weights,
        out_ch, eff_bias,
        out_h, out_w, out_ch, output->data,
        NULL);  // output_acc = NULL
    
    free(weights);
    free(eff_bias);
    free(mult);
    free(shft);
    
    return output;
}

/**
 * Generic MEAN execution (SE block squeeze)
 */
Tensor* layer_mean(Tensor* input, const char* param_folder, int layer_num) {
    printf("\n=== [LAYER %d] MEAN (Global avg pool) ===\n", layer_num);
    fflush(stdout);
    
    char path[512];
    sprintf(path, "%s%s/ofm_scale.txt", DATA_PATH, param_folder);
    float ofm_scale = read_float_value(path);
    
    sprintf(path, "%s%s/ofm_zp.txt", DATA_PATH, param_folder);
    int32_t ofm_zp = read_int32_value(path);
    
    // MEAN output: [1, channels] from [1, H, W, channels]
    Tensor* output = allocate_tensor(1, 1, input->channels, 1, ofm_scale, ofm_zp);
    
    quantized_mean_int8(input->data, output->data,
        input->batch, input->height, input->width, input->channels,
        input->channels, input->scale, input->zp, ofm_scale, ofm_zp);
    
    return output;
}

/**
 * Generic MUL execution (SE excitation)
 */
Tensor* layer_mul(Tensor* input1, Tensor* input2, const char* param_folder, int layer_num) {
    printf("\n=== [LAYER %d] MUL (SE excitation) ===\n", layer_num);
    fflush(stdout);
    
    char path[512];
    sprintf(path, "%s%s/ofm_scale.txt", DATA_PATH, param_folder);
    float ofm_scale = read_float_value(path);
    
    sprintf(path, "%s%s/ofm_zp.txt", DATA_PATH, param_folder);
    int32_t ofm_zp = read_int32_value(path);
    
    int size1 = input1->batch * input1->height * input1->width * input1->channels;
    int size2 = input2->batch * input2->height * input2->width * input2->channels;
    
    Tensor* output = allocate_tensor(input1->batch, input1->height,
                                    input1->width, input1->channels,
                                    ofm_scale, ofm_zp);
    
    // Simple element-wise mul with broadcasting for SE
    if (size2 == input2->channels && size1 % size2 == 0) {
        // input2 is [1,1,1,C], broadcast to [1,H,W,C]
        int idx2 = 0;
        for (int i = 0; i < size1; i++) {
            int32_t v1 = (int32_t)input1->data[i] - input1->zp;
            int32_t v2 = (int32_t)input2->data[idx2] - input2->zp;
            int32_t mul_result = (v1 * v2) / 256;
            int32_t final = mul_result + ofm_zp;
            output->data[i] = (int8_t)(final > 127 ? 127 : (final < -128 ? -128 : final));
            idx2++;
            if (idx2 >= size2) idx2 = 0;
        }
    } else {
        // Direct element-wise
        for (int i = 0; i < size1; i++) {
            int32_t v1 = (int32_t)input1->data[i] - input1->zp;
            int32_t v2 = (int32_t)input2->data[i] - input2->zp;
            int32_t mul_result = (v1 * v2) / 256;
            int32_t final = mul_result + ofm_zp;
            output->data[i] = (int8_t)(final > 127 ? 127 : (final < -128 ? -128 : final));
        }
    }
    
    return output;
}

// ============================================================================
// MAIN: MODEL SEQUENCE FROM LAYER 4 TO 94
// ============================================================================

int main() {
    printf("================================================================================\n");
    printf("TFLite Model Sequence: EfficientNetV2 B0 (INT8) - Layer 4 ~ Layer 94\n");
    printf("================================================================================\n\n");
    
    // Load input file (Layer 4 input from IFM)
    printf("[*] Loading input tensor from ifm.txt...\n");
    float ifm_scale = 0.01865845f;  // From model_profile.txt - Layer 4 input
    int32_t ifm_zp = -14;
    
    Tensor* layer_input = load_input_from_file(
        IFM_FILE, 1, 224, 224, 3, ifm_scale, ifm_zp
    );
    
    if (!layer_input) {
        printf("ERROR: Failed to load input tensor\n");
        return 1;
    }
    
    printf("✓ Input loaded: %d x %d x %d x %d\n\n",
           layer_input->batch, layer_input->height, 
           layer_input->width, layer_input->channels);
    fflush(stdout);
    
    Tensor* current = layer_input;
    Tensor* skip_l10 = NULL;  // For skip at layer 14
    Tensor* skip_l17 = NULL;  // For skip at layer 21
    Tensor* skip_l36 = NULL;  // For skip at layer 52
    Tensor* skip_l80 = NULL;  // For skip at layer 95
    
    // ==== LAYER 4: CONV_2D (stem_conv_1) ====
    Tensor* l4 = layer_conv2d(current, 
        "layer005_CONV_2D_1_stem_conv_1_BiasAdd_1_stem_conv_1_convolution_",
        112, 112, 32, 3, 3, 2, 2, 4);
    if (current != layer_input) free_tensor(current);
    current = l4;
    
    // ==== LAYER 5: HARD_SWISH ====
    Tensor* l5 = layer_hardswish(current,
        "layer006_HARD_SWISH_1_stem_activation_1_truediv__1_stem_activation_1_Relu6_1_ste", 5);
    free_tensor(current);
    current = l5;
    
    // ==== LAYER 6: CONV_2D (block1a_project_conv) ====
    Tensor* l6 = layer_conv2d(current,
        "layer007_CONV_2D_1_block1a_project_conv_1_BiasAdd_1_block1a_project_conv_1_convo",
        112, 112, 16, 3, 3, 1, 1, 6);
    free_tensor(current);
    current = l6;
    
    // ==== LAYER 7: HARD_SWISH ====
    Tensor* l7 = layer_hardswish(current,
        "layer008_HARD_SWISH_1_block1a_project_activation_1_truediv__1_block1a_project_ac", 7);
    free_tensor(current);
    current = l7;
    
    // ==== LAYER 8: CONV_2D (block2a_expand_conv) ====
    Tensor* l8 = layer_conv2d(current,
        "layer009_CONV_2D_1_block2a_expand_conv_1_BiasAdd_1_block2a_expand_conv_1_convolu",
        56, 56, 64, 3, 3, 2, 2, 8);
    free_tensor(current);
    current = l8;
    
    // ==== LAYER 9: HARD_SWISH ====
    Tensor* l9 = layer_hardswish(current,
        "layer010_HARD_SWISH_1_block2a_expand_activation_1_truediv__1_block2a_expand_acti", 9);
    free_tensor(current);
    current = l9;
    
    // ==== LAYER 10: CONV_2D (block2a_project_conv) ====
    Tensor* l10 = layer_conv2d(current,
        "layer011_CONV_2D_1_block2a_project_conv_1_BiasAdd_1_block2a_project_conv_1_convo",
        56, 56, 32, 1, 1, 1, 1, 10);
    free_tensor(current);
    current = l10;
    skip_l10 = current;  // Save for skip at layer 14
    
    // ==== LAYER 11: CONV_2D (block2b_expand_conv) ====
    Tensor* l11 = layer_conv2d(current,
        "layer012_CONV_2D_1_block2b_expand_conv_1_BiasAdd_1_block2b_expand_conv_1_convolu",
        56, 56, 128, 3, 3, 1, 1, 11);
    free_tensor(current);
    current = l11;
    
    // ==== LAYER 12: HARD_SWISH ====
    Tensor* l12 = layer_hardswish(current,
        "layer013_HARD_SWISH_1_block2b_expand_activation_1_truediv__1_block2b_expand_acti", 12);
    free_tensor(current);
    current = l12;
    
    // ==== LAYER 13: CONV_2D (block2b_project_conv) ====
    Tensor* l13 = layer_conv2d(current,
        "layer014_CONV_2D_1_block2b_project_conv_1_BiasAdd_1_block2b_project_conv_1_convo",
        56, 56, 32, 1, 1, 1, 1, 13);
    free_tensor(current);
    current = l13;
    
    // ==== LAYER 14: ADD (skip from layer 10) ====
    Tensor* l14 = layer_add(current, skip_l10,
        "layer015_ADD_1_block2b_add_1_Add", 14);
    free_tensor(current);
    current = l14;
    
    // ==== LAYER 15: CONV_2D (block3a_expand_conv) ====
    Tensor* l15 = layer_conv2d(current,
        "layer016_CONV_2D_1_block3a_expand_conv_1_BiasAdd_1_block3a_expand_conv_1_convolu",
        28, 28, 128, 3, 3, 2, 2, 15);
    free_tensor(current);
    current = l15;
    
    // ==== LAYER 16: HARD_SWISH ====
    Tensor* l16 = layer_hardswish(current,
        "layer017_HARD_SWISH_1_block3a_expand_activation_1_truediv__1_block3a_expand_acti", 16);
    free_tensor(current);
    current = l16;
    
    // ==== LAYER 17: CONV_2D (block3a_project_conv) ====
    Tensor* l17 = layer_conv2d(current,
        "layer018_CONV_2D_1_block3a_project_conv_1_BiasAdd_1_block3a_project_conv_1_convo",
        28, 28, 48, 1, 1, 1, 1, 17);
    free_tensor(current);
    current = l17;
    skip_l17 = current;  // Save for skip at layer 21
    
    // ==== LAYER 18: CONV_2D (block3b_expand_conv) ====
    Tensor* l18 = layer_conv2d(current,
        "layer019_CONV_2D_1_block3b_expand_conv_1_BiasAdd_1_block3b_expand_conv_1_convolu",
        28, 28, 192, 3, 3, 1, 1, 18);
    free_tensor(current);
    current = l18;
    
    // ==== LAYER 19: HARD_SWISH ====
    Tensor* l19 = layer_hardswish(current,
        "layer020_HARD_SWISH_1_block3b_expand_activation_1_truediv__1_block3b_expand_acti", 19);
    free_tensor(current);
    current = l19;
    
    // ==== LAYER 20: CONV_2D (block3b_project_conv) ====
    Tensor* l20 = layer_conv2d(current,
        "layer021_CONV_2D_1_block3b_project_conv_1_BiasAdd_1_block3b_project_conv_1_convo",
        28, 28, 48, 1, 1, 1, 1, 20);
    free_tensor(current);
    current = l20;
    
    // ==== LAYER 21: ADD (skip from layer 17) ====
    Tensor* l21 = layer_add(current, skip_l17,
        "layer022_ADD_1_block3b_add_1_Add", 21);
    free_tensor(current);
    current = l21;
    
    printf("\n>>> BLOCKS 4a-5b (Layers 22-94) <<<\n");
    fflush(stdout);
    
    // ===== LAYER 22-35: BLOCK 4a with SE =====
    printf("\n[*] Block 4a (Layers 22-35 with SE)...\n");
    fflush(stdout);
    
    // Layer 22: CONV_2D - block4a expand
    Tensor* l22 = layer_conv2d(current, "layer023_CONV_2D_1_block4a_expand_conv_1_BiasAdd_1_block4a_expand_conv_1_convolu",
                          28, 28, 192, 1, 1, 1, 1, 22);
    free_tensor(current); current = l22;
    
    // Layer 23: HARD_SWISH
    Tensor* l23 = layer_hardswish(current, "layer024_HARD_SWISH_1_block4a_expand_activation_1_truediv__1_block4a_expand_acti", 23);
    free_tensor(current); current = l23;
    
    // Layer 24: DEPTHWISE_CONV_2D - stride 2
    Tensor* l24 = layer_depthwise_conv2d(current, "layer025_DEPTHWISE_CONV_2D_1_block4a_dwconv2_1_BiasAdd_1_block4a_dwconv2_1_depth",
                                     14, 14, 192, 3, 3, 2, 2, 24);
    free_tensor(current); current = l24;
    
    // Layer 25: HARD_SWISH
    Tensor* l25 = layer_hardswish(current, "layer026_HARD_SWISH_1_block4a_activation_1_truediv__1_block4a_activation_1_Relu6", 25);
    free_tensor(current); current = l25;
    
    // Layer 26: MEAN - SE squeeze [1,14,14,192] -> [1,1,192,1]
    Tensor* l26 = layer_mean(current, "layer027_MEAN_1_block4a_se_squeeze_1_Mean", 26);
    
    // Layers 27-30: SHAPE, STRIDED_SLICE, PACK, RESHAPE to [1,1,1,192]
    Tensor* se_reshaped = layer_skip_shape(l26, 1, 1, 192, 27, "layer027-030_SHAPE");
    free_tensor(l26);
    
    // Layer 31: CONV_2D - SE reduce [1,1,1,192] -> [1,1,1,12]
    Tensor* l31 = layer_conv2d(se_reshaped, "layer032_CONV_2D_1_block4a_se_reduce_1_BiasAdd_1_block4a_se_reduce_1_convolution",
                                     1, 1, 12, 1, 1, 1, 1, 31);
    free_tensor(se_reshaped);
    
    // Layer 32: HARD_SWISH - SE reduce activation
    Tensor* l32 = layer_hardswish(l31, "layer033_HARD_SWISH_1_block4a_se_reduce_1_truediv__1_block4a_se_reduce_1_Relu6_1", 32);
    free_tensor(l31);
    
    // Layer 33: CONV_2D - SE expand [1,1,1,12] -> [1,1,1,192]
    Tensor* l33 = layer_conv2d(l32, "layer034_CONV_2D_1_block4a_se_expand_1_Relu6_1_block4a_se_expand_1_add_1_block4a",
                                     1, 1, 192, 1, 1, 1, 1, 33);
    free_tensor(l32);
    
    // Layer 34: MUL - SE output scaling (scalar multiply)
    Tensor* l34 = layer_mul(l33, l33, "layer035_MUL_1_block4a_se_expand_1_truediv", 34);
    free_tensor(l33);
    
    // Layer 35: MUL - SE excitation (broadcast [1,1,1,192] to [1,14,14,192])
    Tensor* l35 = layer_mul(current, l34, "layer036_MUL_1_block4a_se_excite_1_Mul", 35);
    free_tensor(current); free_tensor(l34);
    current = l35;
    
    // Layer 36: CONV_2D - block4a project [1,14,14,192] -> [1,14,14,96]
    Tensor* l36 = layer_conv2d(current, "layer037_CONV_2D_1_block4a_project_conv_1_BiasAdd_1_block4a_project_conv_1_convo",
                          14, 14, 96, 1, 1, 1, 1, 36);
    free_tensor(current);
    current = l36;
    skip_l36 = current;  // Save for skip at layer 52
    
    // ===== LAYER 37-52: BLOCK 4b with SE and skip =====
    printf("[*] Block 4b (Layers 37-52 with SE)...\n");
    
    // Layer 37: CONV_2D
    Tensor* l37 = layer_conv2d(current, "layer038_CONV_2D_1_block4b_expand_conv_1_BiasAdd_1_block4b_expand_conv_1_convolu",
                          14, 14, 384, 1, 1, 1, 1, 37);
    free_tensor(current); current = l37;
    
    // Layer 38: HARD_SWISH
    l37 = layer_hardswish(current, "layer039_HARD_SWISH_1_block4b_expand_activation_1_truediv__1_block4b_expand_acti", 38);
    free_tensor(current); current = l37;
    
    // Layer 39: DEPTHWISE_CONV_2D
    Tensor* l39 = layer_depthwise_conv2d(current, "layer040_DEPTHWISE_CONV_2D_1_block4b_dwconv2_1_BiasAdd_1_block4b_dwconv2_1_depth",
                                     14, 14, 384, 3, 3, 1, 1, 39);
    free_tensor(current); current = l39;
    
    // Layer 40: HARD_SWISH
    l39 = layer_hardswish(current, "layer041_HARD_SWISH_1_block4b_activation_1_truediv__1_block4b_activation_1_Relu6_1", 40);
    free_tensor(current); current = l39;
    
    // Layer 41: MEAN (SE)
    Tensor* l41 = layer_mean(current, "layer042_MEAN_1_block4b_se_squeeze_1_Mean", 41);
    se_reshaped = layer_skip_shape(l41, 1, 1, 384, 42, "layer042-045_SHAPE");
    free_tensor(l41);
    
    // Layer 46: CONV_2D - SE reduce
    Tensor* l46 = layer_conv2d(se_reshaped, "layer047_CONV_2D_1_block4b_se_reduce_1_BiasAdd_1_block4b_se_reduce_1_convolution",
                            1, 1, 24, 1, 1, 1, 1, 46);
    free_tensor(se_reshaped);
    l46 = layer_hardswish(l46, "layer048_HARD_SWISH_1_block4b_se_reduce_1_truediv__1_block4b_se_reduce_1_Relu6_1", 47);
    
    // Layer 48: CONV_2D - SE expand
    Tensor* l48 = layer_conv2d(l46, "layer049_CONV_2D_1_block4b_se_expand_1_Relu6_1_block4b_se_expand_1_add_1_block4b",
                            1, 1, 384, 1, 1, 1, 1, 48);
    free_tensor(l46);
    l48 = layer_mul(l48, l48, "layer050_MUL_1_block4b_se_expand_1_truediv", 49);
    
    // Layer 50: MUL - SE excitation
    Tensor* l50 = layer_mul(current, l48, "layer051_MUL_1_block4b_se_excite_1_Mul", 50);
    free_tensor(current); free_tensor(l48);
    current = l50;
    
    // Layer 51: CONV_2D - project
    l50 = layer_conv2d(current, "layer052_CONV_2D_1_block4b_project_conv_1_BiasAdd_1_block4b_project_conv_1_convo",
                          14, 14, 96, 1, 1, 1, 1, 51);
    free_tensor(current); current = l50;
    
    // Layer 52: ADD - skip connection
    l50 = layer_add(current, skip_l36, "layer053_ADD_1_block4b_add_1_Add", 52);
    free_tensor(current);  // Only free current, not skip_l36 (same memory)
    current = l50;
    
    // ===== LAYER 53-68: BLOCK 4c and 5a =====
    printf("[*] Block 4c and 5a (Layers 53-68)...\n");
    
    // Layer 53-56: Block 4c (similar pattern)
    Tensor* l53 = layer_conv2d(current, "layer054_CONV_2D_1_block4c_expand_conv_1_BiasAdd_1_block4c_expand_conv_1_convolu",
                          14, 14, 384, 1, 1, 1, 1, 53);
    free_tensor(current);
    l53 = layer_hardswish(l53, "layer055_HARD_SWISH_1_block4c_expand_activation_1_truediv__1_block4c_expand_acti", 54);
    l53 = layer_depthwise_conv2d(l53, "layer056_DEPTHWISE_CONV_2D_1_block4c_dwconv2_1_BiasAdd_1_block4c_dwconv2_1_depth",
                                     14, 14, 384, 3, 3, 1, 1, 55);
    l53 = layer_hardswish(l53, "layer057_HARD_SWISH_1_block4c_activation_1_truediv__1_block4c_activation_1_Relu6_1", 56);
    
    // SE for block 4c
    Tensor* se_l = layer_mean(l53, "layer058_MEAN_1_block4c_se_squeeze_1_Mean", 57);
    se_reshaped = layer_skip_shape(se_l, 1, 1, 384, 58, "layer058-061_SHAPE");
    free_tensor(se_l);
    se_l = layer_conv2d(se_reshaped, "layer062_CONV_2D_1_block4c_se_reduce_1_BiasAdd_1_block4c_se_reduce_1_convolution",
                            1, 1, 24, 1, 1, 1, 1, 61);
    free_tensor(se_reshaped);
    se_l = layer_hardswish(se_l, "layer063_HARD_SWISH_1_block4c_se_reduce_1_truediv__1_block4c_se_reduce_1_Relu6_1", 62);
    se_l = layer_conv2d(se_l, "layer064_CONV_2D_1_block4c_se_expand_1_Relu6_1_block4c_se_expand_1_add_1_block4c",
                            1, 1, 384, 1, 1, 1, 1, 63);
    se_l = layer_mul(se_l, se_l, "layer065_MUL_1_block4c_se_expand_1_truediv", 64);
    l53 = layer_mul(l53, se_l, "layer066_MUL_1_block4c_se_excite_1_Mul", 65);
    free_tensor(se_l);
    
    l53 = layer_conv2d(l53, "layer067_CONV_2D_1_block4c_project_conv_1_BiasAdd_1_block4c_project_conv_1_convo",
                          14, 14, 96, 1, 1, 1, 1, 66);
    
    // Layer 67-68: Block 5a start
    l53 = layer_conv2d(l53, "layer068_CONV_2D_1_block5a_expand_conv_1_BiasAdd_1_block5a_expand_conv_1_convolu",
                          14, 14, 480, 1, 1, 1, 1, 67);
    l53 = layer_hardswish(l53, "layer069_HARD_SWISH_1_block5a_expand_activation_1_truediv__1_block5a_expand_acti", 68);
    
    current = l53;
    
    // ===== LAYER 69-95: BLOCK 5a-5b =====
    printf("[*] Block 5a-5b (Layers 69-95)...\n");
    
    // Block 5a continuation
    Tensor* l69 = layer_depthwise_conv2d(current, "layer070_DEPTHWISE_CONV_2D_1_block5a_dwconv2_1_BiasAdd_1_block5a_dwconv2_1_depth",
                                     14, 14, 480, 3, 3, 1, 1, 69);
    free_tensor(current);
    l69 = layer_hardswish(l69, "layer071_HARD_SWISH_1_block5a_activation_1_truediv__1_block5a_activation_1_Relu6_1", 70);
    
    // SE for 5a
    se_l = layer_mean(l69, "layer072_MEAN_1_block5a_se_squeeze_1_Mean", 71);
    se_reshaped = layer_skip_shape(se_l, 1, 1, 480, 72, "layer072-075_SHAPE");
    free_tensor(se_l);
    se_l = layer_conv2d(se_reshaped, "layer076_CONV_2D_1_block5a_se_reduce_1_BiasAdd_1_block5a_se_reduce_1_convolution",
                            1, 1, 20, 1, 1, 1, 1, 75);
    free_tensor(se_reshaped);
    se_l = layer_hardswish(se_l, "layer077_HARD_SWISH_1_block5a_se_reduce_1_truediv__1_block5a_se_reduce_1_Relu6_1", 76);
    se_l = layer_conv2d(se_l, "layer078_CONV_2D_1_block5a_se_expand_1_Relu6_1_block5a_se_expand_1_add_1_block5a",
                            1, 1, 480, 1, 1, 1, 1, 77);
    se_l = layer_mul(se_l, se_l, "layer079_MUL_1_block5a_se_expand_1_truediv", 78);
    l69 = layer_mul(l69, se_l, "layer080_MUL_1_block5a_se_excite_1_Mul", 79);
    free_tensor(se_l);
    
    l69 = layer_conv2d(l69, "layer081_CONV_2D_1_block5a_project_conv_1_BiasAdd_1_block5a_project_conv_1_convo",
                          14, 14, 120, 1, 1, 1, 1, 80);
    skip_l80 = l69;  // Save for skip at block 5b
    
    // Block 5b
    Tensor* l81 = layer_conv2d(l69, "layer082_CONV_2D_1_block5b_expand_conv_1_BiasAdd_1_block5b_expand_conv_1_convolu",
                          14, 14, 480, 1, 1, 1, 1, 81);
    // NOTE: Don't free l69 yet - skip_l80 needs it!
    l81 = layer_hardswish(l81, "layer083_HARD_SWISH_1_block5b_expand_activation_1_truediv__1_block5b_expand_acti", 82);
    l81 = layer_depthwise_conv2d(l81, "layer084_DEPTHWISE_CONV_2D_1_block5b_dwconv2_1_BiasAdd_1_block5b_dwconv2_1_depth",
                                     14, 14, 480, 3, 3, 1, 1, 83);
    l81 = layer_hardswish(l81, "layer085_HARD_SWISH_1_block5b_activation_1_truediv__1_block5b_activation_1_Relu6_1", 84);
    
    // SE for 5b
    se_l = layer_mean(l81, "layer086_MEAN_1_block5b_se_squeeze_1_Mean", 85);
    se_reshaped = layer_skip_shape(se_l, 1, 1, 480, 86, "layer086-089_SHAPE");
    free_tensor(se_l);
    se_l = layer_conv2d(se_reshaped, "layer090_CONV_2D_1_block5b_se_reduce_1_BiasAdd_1_block5b_se_reduce_1_convolution",
                            1, 1, 20, 1, 1, 1, 1, 89);
    free_tensor(se_reshaped);
    se_l = layer_hardswish(se_l, "layer091_HARD_SWISH_1_block5b_se_reduce_1_truediv__1_block5b_se_reduce_1_Relu6_1", 90);
    se_l = layer_conv2d(se_l, "layer092_CONV_2D_1_block5b_se_expand_1_Relu6_1_block5b_se_expand_1_add_1_block5b",
                            1, 1, 480, 1, 1, 1, 1, 91);
    se_l = layer_mul(se_l, se_l, "layer093_MUL_1_block5b_se_expand_1_truediv", 92);
    l81 = layer_mul(l81, se_l, "layer094_MUL_1_block5b_se_excite_1_Mul", 93);
    free_tensor(se_l);
    
    l81 = layer_conv2d(l81, "layer095_CONV_2D_1_block5b_project_conv_1_BiasAdd_1_block5b_project_conv_1_convo",
                          14, 14, 120, 1, 1, 1, 1, 94);
    
    // ADD skip 5a->5b
    l81 = layer_add(l81, skip_l80, "layer096_ADD_1_block5b_add_1_Add", 95);
    // NOTE: skip_l80 is same as l69, will be freed below
    free_tensor(l69);  // Free the skip tensor (same as skip_l80)
    
    printf("\n>>> Final Output <<<\n");
    printf("Output shape: [%d, %d, %d, %d]\n", l81->batch, l81->height, l81->width, l81->channels);
    printf("Output scale: %.8f, zp: %d\n", l81->scale, l81->zp);
    
    // Cleanup
    free_tensor(l81);
    
    printf("\n✓ Model inference completed (Layers 4-95)!\n");
    return 0;
}
