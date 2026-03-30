/**
 * ============================================================================
 * TFLite Model Sequence Simulator - EfficientNetV2 B0 (INT8)
 * Layers 4 ~ 94 using the run_*_layer refactor pattern
 * ============================================================================
 * 
 * QUAN TRỌNG: Chỉ số layer
 * - Model layer bắt đầu từ 0 → Layer K (model) = Layer K+1 (params folder)
 * - Layer 4 (model) uses layer005_... (params folder)
 * - Layer 94 (model) uses layer095_... (params folder)
 * 
 * Chiến lược xử lý shape layers:
 * - RESHAPE, SHAPE, STRIDED_SLICE, PACK → Không thay đổi dữ liệu, chỉ skip
 * ============================================================================
 */


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "layer/layer_helper.h"

static int8_t* load_input_tensor(const char* path, int* input_size) {
    int size = count_elements(path);
    int8_t* input_data = (int8_t*)malloc(size * sizeof(int8_t));

    if (!input_data) {
        return NULL;
    }

    read_int8_array(path, input_data, size);
    if (input_size) {
        *input_size = size;
    }

    return input_data;
}

int main() {
    printf("================================================================================\n");
    printf("TFLite Model Sequence: EfficientNetV2 B0 (INT8) - Layer 4 ~ Layer 94\n");
    printf("================================================================================\n\n");
    
    // Load input from file
    printf("[*] Loading input tensor from ifm.txt...\n");
    int input_size;
    int8_t* layer_4_out = load_input_tensor("all_layer_io/layer_5_CONV_2D/ifm.txt", &input_size);
    if (!layer_4_out) {
        printf("ERROR: Failed to load input tensor\n");
        return 1;
    }
    printf("✓ Input loaded: size = %d\n\n", input_size);
    fflush(stdout);
    
    // LAYER 4: CONV_2D (stem_conv_1) [1,224,224,3] -> [1,112,112,32]
    printf("Starting Layer 4 CONV_2D...\n");
    int conv5_out_h, conv5_out_w;
    int8_t* layer_5_out = run_conv2d_layer("extracted_params_hsigmoid/layer005_CONV_2D_1_stem_conv_1_BiasAdd_1_stem_conv_1_convolution_", "all_layer_io/layer_5_CONV_2D",
                                           layer_4_out, 224, 224, 3, 32, 3, 3, 2, 2, "SAME", &conv5_out_h, &conv5_out_w);
    
    // LAYER 5: HARD_SWISH
    printf("Starting Layer 5 HARD_SWISH...\n");
    int layer_6_size = conv5_out_h * conv5_out_w * 32;
    int8_t* layer_6_out = run_hardswish_layer("extracted_params_hsigmoid/layer006_HARD_SWISH_1_stem_activation_1_truediv__1_stem_activation_1_Relu6_1_ste", "all_layer_io/layer_6_HARD_SWISH",
                                             layer_5_out, &layer_6_size);
    
    // LAYER 6: CONV_2D (block1a_project_conv) [1,112,112,32] -> [1,112,112,16]
    printf("Starting Layer 6 CONV_2D...\n");
    int conv7_out_h, conv7_out_w;
    int8_t* layer_7_out = run_conv2d_layer("extracted_params_hsigmoid/layer007_CONV_2D_1_block1a_project_conv_1_BiasAdd_1_block1a_project_conv_1_convo", "all_layer_io/layer_7_CONV_2D",
                                           layer_6_out, 112, 112, 32, 16, 3, 3, 1, 1, "SAME", &conv7_out_h, &conv7_out_w);
    
    // LAYER 7: HARD_SWISH
    printf("Starting Layer 7 HARD_SWISH...\n");
    int layer_8_size = conv7_out_h * conv7_out_w * 16;
    int8_t* layer_8_out = run_hardswish_layer("extracted_params_hsigmoid/layer008_HARD_SWISH_1_block1a_project_activation_1_truediv__1_block1a_project_ac", "all_layer_io/layer_8_HARD_SWISH",
                                             layer_7_out, &layer_8_size);
    
    // LAYER 8: CONV_2D (block2a_expand_conv) [1,112,112,16] -> [1,56,56,64]
    printf("Starting Layer 8 CONV_2D...\n");
    int conv9_out_h, conv9_out_w;
    int8_t* layer_9_out = run_conv2d_layer("extracted_params_hsigmoid/layer009_CONV_2D_1_block2a_expand_conv_1_BiasAdd_1_block2a_expand_conv_1_convolu", "all_layer_io/layer_9_CONV_2D",
                                           layer_8_out, 112, 112, 16, 64, 3, 3, 2, 2, "SAME", &conv9_out_h, &conv9_out_w);
    
    // LAYER 9: HARD_SWISH
    printf("Starting Layer 9 HARD_SWISH...\n");
    int layer_10_size = conv9_out_h * conv9_out_w * 64;
    int8_t* layer_10_out = run_hardswish_layer("extracted_params_hsigmoid/layer010_HARD_SWISH_1_block2a_expand_activation_1_truediv__1_block2a_expand_acti", "all_layer_io/layer_10_HARD_SWISH",
                                              layer_9_out, &layer_10_size);
    
    // LAYER 10: CONV_2D (block2a_project_conv) [1,56,56,64] -> [1,56,56,32]
    printf("Starting Layer 10 CONV_2D...\n");
    int conv11_out_h, conv11_out_w;
    int8_t* layer_11_out = run_conv2d_layer("extracted_params_hsigmoid/layer011_CONV_2D_1_block2a_project_conv_1_BiasAdd_1_block2a_project_conv_1_convo", "all_layer_io/layer_11_CONV_2D",
                                            layer_10_out, 56, 56, 64, 32, 1, 1, 1, 1, "SAME", &conv11_out_h, &conv11_out_w);
    
    // LAYER 11: CONV_2D (block2b_expand_conv) [1,56,56,32] -> [1,56,56,128]
    printf("Starting Layer 11 CONV_2D...\n");
    int conv12_out_h, conv12_out_w;
    int8_t* layer_12_out = run_conv2d_layer("extracted_params_hsigmoid/layer012_CONV_2D_1_block2b_expand_conv_1_BiasAdd_1_block2b_expand_conv_1_convolu", "all_layer_io/layer_12_CONV_2D",
                                            layer_11_out, 56, 56, 32, 128, 3, 3, 1, 1, "SAME", &conv12_out_h, &conv12_out_w);
    
    // LAYER 12: HARD_SWISH
    printf("Starting Layer 12 HARD_SWISH...\n");
    int layer_13_size = conv12_out_h * conv12_out_w * 128;
    int8_t* layer_13_out = run_hardswish_layer("extracted_params_hsigmoid/layer013_HARD_SWISH_1_block2b_expand_activation_1_truediv__1_block2b_expand_acti", "all_layer_io/layer_13_HARD_SWISH",
                                              layer_12_out, &layer_13_size);
    
    // LAYER 13: CONV_2D (block2b_project_conv) [1,56,56,128] -> [1,56,56,32]
    printf("Starting Layer 13 CONV_2D...\n");
    int conv14_out_h, conv14_out_w;
    int8_t* layer_14_out = run_conv2d_layer("extracted_params_hsigmoid/layer014_CONV_2D_1_block2b_project_conv_1_BiasAdd_1_block2b_project_conv_1_convo", "all_layer_io/layer_14_CONV_2D",
                                            layer_13_out, 56, 56, 128, 32, 1, 1, 1, 1, "SAME", &conv14_out_h, &conv14_out_w);
    
    // LAYER 14: ADD (skip connection from layer 10)
    printf("Starting Layer 14 ADD...\n");
    int8_t* layer_15_out = run_add_layer("extracted_params_hsigmoid/layer015_ADD_1_block2b_add_1_Add", "all_layer_io/layer_15_ADD",
                                         layer_14_out, conv14_out_h * conv14_out_w * 32, layer_11_out, conv14_out_h * conv14_out_w * 32);
    
    // LAYER 15: CONV_2D (block3a_expand_conv) [1,56,56,32] -> [1,28,28,128]
    printf("Starting Layer 15 CONV_2D...\n");
    int conv16_out_h, conv16_out_w;
    int8_t* layer_16_out = run_conv2d_layer("extracted_params_hsigmoid/layer016_CONV_2D_1_block3a_expand_conv_1_BiasAdd_1_block3a_expand_conv_1_convolu", "all_layer_io/layer_16_CONV_2D",
                                            layer_15_out, 56, 56, 32, 128, 3, 3, 2, 2, "SAME", &conv16_out_h, &conv16_out_w);
    
    // LAYER 16: HARD_SWISH
    printf("Starting Layer 16 HARD_SWISH...\n");
    int layer_17_size = conv16_out_h * conv16_out_w * 128;
    int8_t* layer_17_out = run_hardswish_layer("extracted_params_hsigmoid/layer017_HARD_SWISH_1_block3a_expand_activation_1_truediv__1_block3a_expand_acti", "all_layer_io/layer_17_HARD_SWISH",
                                              layer_16_out, &layer_17_size);
    
    // LAYER 17: CONV_2D (block3a_project_conv) [1,28,28,128] -> [1,28,28,48]
    printf("Starting Layer 17 CONV_2D...\n");
    int conv18_out_h, conv18_out_w;
    int8_t* layer_18_out = run_conv2d_layer("extracted_params_hsigmoid/layer018_CONV_2D_1_block3a_project_conv_1_BiasAdd_1_block3a_project_conv_1_convo", "all_layer_io/layer_18_CONV_2D",
                                            layer_17_out, 28, 28, 128, 48, 1, 1, 1, 1, "SAME", &conv18_out_h, &conv18_out_w);
    
    // LAYER 18: CONV_2D (block3b_expand_conv) [1,28,28,48] -> [1,28,28,192]
    printf("Starting Layer 18 CONV_2D...\n");
    int conv19_out_h, conv19_out_w;
    int8_t* layer_19_out = run_conv2d_layer("extracted_params_hsigmoid/layer019_CONV_2D_1_block3b_expand_conv_1_BiasAdd_1_block3b_expand_conv_1_convolu", "all_layer_io/layer_19_CONV_2D",
                                            layer_18_out, 28, 28, 48, 192, 3, 3, 1, 1, "SAME", &conv19_out_h, &conv19_out_w);
    
    // LAYER 19: HARD_SWISH
    printf("Starting Layer 19 HARD_SWISH...\n");
    int layer_20_size = conv19_out_h * conv19_out_w * 192;
    int8_t* layer_20_out = run_hardswish_layer("extracted_params_hsigmoid/layer020_HARD_SWISH_1_block3b_expand_activation_1_truediv__1_block3b_expand_acti", "all_layer_io/layer_20_HARD_SWISH",
                                              layer_19_out, &layer_20_size);
    
    // LAYER 20: CONV_2D (block3b_project_conv) [1,28,28,192] -> [1,28,28,48]
    printf("Starting Layer 20 CONV_2D...\n");
    int conv21_out_h, conv21_out_w;
    int8_t* layer_21_out = run_conv2d_layer("extracted_params_hsigmoid/layer021_CONV_2D_1_block3b_project_conv_1_BiasAdd_1_block3b_project_conv_1_convo", "all_layer_io/layer_21_CONV_2D",
                                            layer_20_out, 28, 28, 192, 48, 1, 1, 1, 1, "SAME", &conv21_out_h, &conv21_out_w);
    
    // LAYER 21: ADD (skip connection from layer 17)
    printf("Starting Layer 21 ADD...\n");
    int8_t* layer_22_out = run_add_layer("extracted_params_hsigmoid/layer022_ADD_1_block3b_add_1_Add", "all_layer_io/layer_22_ADD",
                                         layer_21_out, conv21_out_h * conv21_out_w * 48, layer_18_out, conv21_out_h * conv21_out_w * 48);
    
    printf("\n>>> BLOCKS 4a-5a (Layers 22-94) <<<\n\n");
    
    // ===== LAYER 22-35: BLOCK 4a with SE =====
    printf("[*] Block 4a (Layers 22-35 with SE)...\n");
    
    // LAYER 22: CONV_2D (block4a_expand_conv) [1,28,28,48] -> [1,28,28,192]
    printf("Starting Layer 22 CONV_2D...\n");
    int conv23_out_h, conv23_out_w;
    int8_t* layer_23_out = run_conv2d_layer("extracted_params_hsigmoid/layer023_CONV_2D_1_block4a_expand_conv_1_BiasAdd_1_block4a_expand_conv_1_convolu", "all_layer_io/layer_23_CONV_2D",
                                            layer_22_out, 28, 28, 48, 192, 3, 3, 1, 1, "SAME", &conv23_out_h, &conv23_out_w);
    
    // LAYER 23: HARD_SWISH
    printf("Starting Layer 23 HARD_SWISH...\n");
    int layer_24_size = conv23_out_h * conv23_out_w * 192;
    int8_t* layer_24_out = run_hardswish_layer("extracted_params_hsigmoid/layer024_HARD_SWISH_1_block4a_expand_activation_1_truediv__1_block4a_expand_acti", "all_layer_io/layer_24_HARD_SWISH",
                                              layer_23_out, &layer_24_size);
    
    // LAYER 24: DEPTHWISE_CONV_2D (block4a_dwconv2) [1,28,28,192] -> [1,14,14,192]
    printf("Starting Layer 24 DEPTHWISE_CONV_2D...\n");
    int conv25_out_h, conv25_out_w;
    int8_t* layer_25_out = run_dw_conv_layer("extracted_params_hsigmoid/layer025_DEPTHWISE_CONV_2D_1_block4a_dwconv2_1_BiasAdd_1_block4a_dwconv2_1_depth",
                                             layer_24_out, 28, 28, 192, 192, 3, 3, 2, 2, "SAME", &conv25_out_h, &conv25_out_w);
    
    // LAYER 25: HARD_SWISH
    printf("Starting Layer 25 HARD_SWISH...\n");
    int layer_26_size = conv25_out_h * conv25_out_w * 192;
    int8_t* layer_26_out = run_hardswish_layer("extracted_params_hsigmoid/layer026_HARD_SWISH_1_block4a_activation_1_truediv__1_block4a_activation_1_Relu6", "all_layer_io/layer_26_HARD_SWISH",
                                              layer_25_out, &layer_26_size);
    
    // LAYER 26: MEAN (SE squeeze) [1,14,14,192] -> [1,1,192]
    printf("Starting Layer 26 MEAN...\n");
    int8_t* layer_27_out = run_mean_layer("extracted_params_hsigmoid/layer027_MEAN_1_block4a_se_squeeze_1_Mean", "all_layer_io/layer_27_MEAN",
                                          layer_26_out, 14, 14, 192, 1, 0);
    
    printf("Skip layer 27-30 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 31: CONV_2D (SE reduce) [1,1,192] -> [1,1,12]
    printf("Starting Layer 31 CONV_2D...\n");
    int conv32_out_h, conv32_out_w;
    int8_t* layer_32_out = run_conv2d_layer("extracted_params_hsigmoid/layer032_CONV_2D_1_block4a_se_reduce_1_BiasAdd_1_block4a_se_reduce_1_convolution", "all_layer_io/layer_32_CONV_2D",
                                            layer_27_out, 1, 1, 192, 12, 1, 1, 1, 1, "SAME", &conv32_out_h, &conv32_out_w);
    
    // LAYER 32: HARD_SWISH (SE reduce activation)
    printf("Starting Layer 32 HARD_SWISH...\n");
    int layer_33_size = conv32_out_h * conv32_out_w * 12;
    int8_t* layer_33_out = run_hardswish_layer("extracted_params_hsigmoid/layer033_HARD_SWISH_1_block4a_se_reduce_1_truediv__1_block4a_se_reduce_1_Relu6_1", "all_layer_io/layer_33_HARD_SWISH",
                                              layer_32_out, &layer_33_size);
    
    // LAYER 33: CONV_2D (SE expand) [1,1,12] -> [1,1,192]
    printf("Starting Layer 33 CONV_2D...\n");
    int conv34_out_h, conv34_out_w;
    int8_t* layer_34_out = run_conv2d_layer("extracted_params_hsigmoid/layer034_CONV_2D_1_block4a_se_expand_1_Relu6_1_block4a_se_expand_1_add_1_block4a", "all_layer_io/layer_34_CONV_2D",
                                            layer_33_out, 1, 1, 12, 192, 1, 1, 1, 1, "SAME", &conv34_out_h, &conv34_out_w);
    
    // LAYER 34: MUL (SE output scaling)
    printf("Starting Layer 34 MUL...\n");
    int8_t* layer_35_out = run_mul_layer("extracted_params_hsigmoid/layer035_MUL_1_block4a_se_expand_1_truediv", "all_layer_io/layer_34_MUL",
                                        layer_34_out, 192, NULL, 1, 0);
    
    // LAYER 35: MUL (SE excitation - broadcast)
    printf("Starting Layer 35 MUL...\n");
    int8_t* layer_36_out = run_mul_layer("extracted_params_hsigmoid/layer036_MUL_1_block4a_se_excite_1_Mul", "all_layer_io/layer_35_MUL",
                                        NULL, 14*14*192, layer_35_out, 192, 1);
    
    // LAYER 36: CONV_2D (block4a project) [1,14,14,192] -> [1,14,14,96]
    printf("Starting Layer 36 CONV_2D...\n");
    int conv37_out_h, conv37_out_w;
    int8_t* layer_37_out = run_conv2d_layer("extracted_params_hsigmoid/layer037_CONV_2D_1_block4a_project_conv_1_BiasAdd_1_block4a_project_conv_1_convo", "all_layer_io/layer_36_CONV_2D",
                                            layer_36_out, 14, 14, 192, 96, 1, 1, 1, 1, "SAME", &conv37_out_h, &conv37_out_w);
    
    // ===== LAYER 37-52: BLOCK 4b with SE and skip =====
    printf("[*] Block 4b (Layers 37-52 with SE)...\n");
    
    // LAYER 37: CONV_2D (block4b_expand_conv) [1,14,14,96] -> [1,14,14,384]
    printf("Starting Layer 37 CONV_2D...\n");
    int conv38_out_h, conv38_out_w;
    int8_t* layer_38_out = run_conv2d_layer("extracted_params_hsigmoid/layer038_CONV_2D_1_block4b_expand_conv_1_BiasAdd_1_block4b_expand_conv_1_convolu", "all_layer_io/layer_38_CONV_2D",
                                            layer_37_out, 14, 14, 96, 384, 1, 1, 1, 1, "SAME", &conv38_out_h, &conv38_out_w);
    
    // LAYER 38: HARD_SWISH
    printf("Starting Layer 38 HARD_SWISH...\n");
    int layer_39_size = conv38_out_h * conv38_out_w * 384;
    int8_t* layer_39_out = run_hardswish_layer("extracted_params_hsigmoid/layer039_HARD_SWISH_1_block4b_expand_activation_1_truediv__1_block4b_expand_acti", "all_layer_io/layer_39_HARD_SWISH",
                                              layer_38_out, &layer_39_size);
    
    // LAYER 39: DEPTHWISE_CONV_2D (block4b_dwconv2) [1,14,14,384] -> [1,14,14,384]
    printf("Starting Layer 39 DEPTHWISE_CONV_2D...\n");
    int conv40_out_h, conv40_out_w;
    int8_t* layer_40_out = run_dw_conv_layer("extracted_params_hsigmoid/layer040_DEPTHWISE_CONV_2D_1_block4b_dwconv2_1_BiasAdd_1_block4b_dwconv2_1_depth",
                                             layer_39_out, 14, 14, 384, 384, 3, 3, 1, 1, "SAME", &conv40_out_h, &conv40_out_w);
    
    // LAYER 40: HARD_SWISH
    printf("Starting Layer 40 HARD_SWISH...\n");
    int layer_41_size = conv40_out_h * conv40_out_w * 384;
    int8_t* layer_41_out = run_hardswish_layer("extracted_params_hsigmoid/layer041_HARD_SWISH_1_block4b_activation_1_truediv__1_block4b_activation_1_Relu6", "all_layer_io/layer_41_HARD_SWISH",
                                              layer_40_out, &layer_41_size);
    
    // LAYER 41: MEAN (SE squeeze) [1,14,14,384] -> [1,1,384]
    printf("Starting Layer 41 MEAN...\n");
    int8_t* layer_42_out = run_mean_layer("extracted_params_hsigmoid/layer042_MEAN_1_block4b_se_squeeze_1_Mean", "all_layer_io/layer_42_MEAN",
                                          layer_41_out, 14, 14, 384, 1, 0);
    
    printf("Skip layer 42-45 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 46: CONV_2D (SE reduce) [1,1,384] -> [1,1,16]
    printf("Starting Layer 46 CONV_2D...\n");
    int conv47_out_h, conv47_out_w;
    int8_t* layer_47_out = run_conv2d_layer("extracted_params_hsigmoid/layer047_CONV_2D_1_block4b_se_reduce_1_BiasAdd_1_block4b_se_reduce_1_convolution", "all_layer_io/layer_47_CONV_2D",
                                            layer_42_out, 1, 1, 384, 16, 1, 1, 1, 1, "SAME", &conv47_out_h, &conv47_out_w);
    
    // LAYER 47: HARD_SWISH (SE reduce activation)
    printf("Starting Layer 47 HARD_SWISH...\n");
    int layer_48_size = conv47_out_h * conv47_out_w * 16;
    int8_t* layer_48_out = run_hardswish_layer("extracted_params_hsigmoid/layer048_HARD_SWISH_1_block4b_se_reduce_1_truediv__1_block4b_se_reduce_1_Relu6_1", "all_layer_io/layer_48_HARD_SWISH",
                                              layer_47_out, &layer_48_size);
    
    // LAYER 48: CONV_2D (SE expand) [1,1,16] -> [1,1,384]
    printf("Starting Layer 48 CONV_2D...\n");
    int conv49_out_h, conv49_out_w;
    int8_t* layer_49_out = run_conv2d_layer("extracted_params_hsigmoid/layer049_CONV_2D_1_block4b_se_expand_1_Relu6_1_block4b_se_expand_1_add_1_block4b", "all_layer_io/layer_49_CONV_2D",
                                            layer_48_out, 1, 1, 16, 384, 1, 1, 1, 1, "SAME", &conv49_out_h, &conv49_out_w);
    
    // LAYER 49: MUL (SE output scaling)
    printf("Starting Layer 49 MUL...\n");
    int8_t* layer_50_out = run_mul_layer("extracted_params_hsigmoid/layer050_MUL_1_block4b_se_expand_1_truediv", "all_layer_io/layer_50_MUL",
                                        layer_49_out, 384, NULL, 1, 0);
    
    // LAYER 50: MUL (SE excitation - broadcast)
    printf("Starting Layer 50 MUL...\n");
    int8_t* layer_51_out = run_mul_layer("extracted_params_hsigmoid/layer051_MUL_1_block4b_se_excite_1_Mul", "all_layer_io/layer_51_MUL",
                                        NULL, 14*14*384, layer_50_out, 384, 1);
    
    // LAYER 51: CONV_2D (block4b project) [1,14,14,384] -> [1,14,14,96]
    printf("Starting Layer 51 CONV_2D...\n");
    int conv52_out_h, conv52_out_w;
    int8_t* layer_52_out = run_conv2d_layer("extracted_params_hsigmoid/layer052_CONV_2D_1_block4b_project_conv_1_BiasAdd_1_block4b_project_conv_1_convo", "all_layer_io/layer_52_CONV_2D",
                                            layer_51_out, 14, 14, 384, 96, 1, 1, 1, 1, "SAME", &conv52_out_h, &conv52_out_w);
    
    // LAYER 52: ADD (skip connection from layer 36)
    printf("Starting Layer 52 ADD...\n");
    int8_t* layer_53_out = run_add_layer("extracted_params_hsigmoid/layer053_ADD_1_block4b_add_1_Add", "all_layer_io/layer_53_ADD",
                                         layer_52_out, conv52_out_h * conv52_out_w * 96, layer_37_out, conv52_out_h * conv52_out_w * 96);
    
    // ===== LAYER 53-68: BLOCK 4c with SE and skip =====
    printf("[*] Block 4c (Layers 53-68 with SE)...\n");
    
    // LAYER 53: CONV_2D (block4c_expand_conv) [1,14,14,96] -> [1,14,14,384]
    printf("Starting Layer 53 CONV_2D...\n");
    int conv54_out_h, conv54_out_w;
    int8_t* layer_54_out = run_conv2d_layer("extracted_params_hsigmoid/layer054_CONV_2D_1_block4c_expand_conv_1_BiasAdd_1_block4c_expand_conv_1_convolu", "all_layer_io/layer_54_CONV_2D",
                                            layer_53_out, 14, 14, 96, 384, 1, 1, 1, 1, "SAME", &conv54_out_h, &conv54_out_w);
    
    // LAYER 54: HARD_SWISH
    printf("Starting Layer 54 HARD_SWISH...\n");
    int layer_55_size = conv54_out_h * conv54_out_w * 384;
    int8_t* layer_55_out = run_hardswish_layer("extracted_params_hsigmoid/layer055_HARD_SWISH_1_block4c_expand_activation_1_truediv__1_block4c_expand_acti", "all_layer_io/layer_55_HARD_SWISH",
                                              layer_54_out, &layer_55_size);
    
    // LAYER 55: DEPTHWISE_CONV_2D (block4c_dwconv2) [1,14,14,384] -> [1,14,14,384]
    printf("Starting Layer 55 DEPTHWISE_CONV_2D...\n");
    int conv56_out_h, conv56_out_w;
    int8_t* layer_56_out = run_dw_conv_layer("extracted_params_hsigmoid/layer056_DEPTHWISE_CONV_2D_1_block4c_dwconv2_1_BiasAdd_1_block4c_dwconv2_1_depth",
                                             layer_55_out, 14, 14, 384, 384, 3, 3, 1, 1, "SAME", &conv56_out_h, &conv56_out_w);
    
    // LAYER 56: HARD_SWISH
    printf("Starting Layer 56 HARD_SWISH...\n");
    int layer_57_size = conv56_out_h * conv56_out_w * 384;
    int8_t* layer_57_out = run_hardswish_layer("extracted_params_hsigmoid/layer057_HARD_SWISH_1_block4c_activation_1_truediv__1_block4c_activation_1_Relu6", "all_layer_io/layer_57_HARD_SWISH",
                                              layer_56_out, &layer_57_size);
    
    // LAYER 57: MEAN (SE squeeze) [1,14,14,384] -> [1,1,384]
    printf("Starting Layer 57 MEAN...\n");
    int8_t* layer_58_out = run_mean_layer("extracted_params_hsigmoid/layer058_MEAN_1_block4c_se_squeeze_1_Mean", "all_layer_io/layer_58_MEAN",
                                          layer_57_out, 14, 14, 384, 1, 0);
    
    printf("Skip layer 58-61 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 62: CONV_2D (SE reduce) [1,1,384] -> [1,1,16]
    printf("Starting Layer 62 CONV_2D...\n");
    int conv63_out_h, conv63_out_w;
    int8_t* layer_63_out = run_conv2d_layer("extracted_params_hsigmoid/layer063_CONV_2D_1_block4c_se_reduce_1_BiasAdd_1_block4c_se_reduce_1_convolution", "all_layer_io/layer_63_CONV_2D",
                                            layer_58_out, 1, 1, 384, 16, 1, 1, 1, 1, "SAME", &conv63_out_h, &conv63_out_w);
    
    // LAYER 63: HARD_SWISH (SE reduce activation)
    printf("Starting Layer 63 HARD_SWISH...\n");
    int layer_64_size = conv63_out_h * conv63_out_w * 16;
    int8_t* layer_64_out = run_hardswish_layer("extracted_params_hsigmoid/layer064_HARD_SWISH_1_block4c_se_reduce_1_truediv__1_block4c_se_reduce_1_Relu6_1", "all_layer_io/layer_64_HARD_SWISH",
                                              layer_63_out, &layer_64_size);
    
    // LAYER 64: CONV_2D (SE expand) [1,1,16] -> [1,1,384]
    printf("Starting Layer 64 CONV_2D...\n");
    int conv65_out_h, conv65_out_w;
    int8_t* layer_65_out = run_conv2d_layer("extracted_params_hsigmoid/layer065_CONV_2D_1_block4c_se_expand_1_Relu6_1_block4c_se_expand_1_add_1_block4c", "all_layer_io/layer_65_CONV_2D",
                                            layer_64_out, 1, 1, 16, 384, 1, 1, 1, 1, "SAME", &conv65_out_h, &conv65_out_w);
    
    // LAYER 65: MUL (SE output scaling)
    printf("Starting Layer 65 MUL...\n");
    int8_t* layer_66_out = run_mul_layer("extracted_params_hsigmoid/layer066_MUL_1_block4c_se_expand_1_truediv", "all_layer_io/layer_66_MUL",
                                        layer_65_out, 384, NULL, 1, 0);
    
    // LAYER 66: MUL (SE excitation - broadcast)
    printf("Starting Layer 66 MUL...\n");
    int8_t* layer_67_out = run_mul_layer("extracted_params_hsigmoid/layer067_MUL_1_block4c_se_excite_1_Mul", "all_layer_io/layer_67_MUL",
                                        NULL, 14*14*384, layer_66_out, 384, 1);
    
    // LAYER 67: CONV_2D (block4c project) [1,14,14,384] -> [1,14,14,96]
    printf("Starting Layer 67 CONV_2D...\n");
    int conv68_out_h, conv68_out_w;
    int8_t* layer_68_out = run_conv2d_layer("extracted_params_hsigmoid/layer068_CONV_2D_1_block4c_project_conv_1_BiasAdd_1_block4c_project_conv_1_convo", "all_layer_io/layer_68_CONV_2D",
                                            layer_67_out, 14, 14, 384, 96, 1, 1, 1, 1, "SAME", &conv68_out_h, &conv68_out_w);
    
    // LAYER 68: ADD (skip connection from layer 52)
    printf("Starting Layer 68 ADD...\n");
    int8_t* layer_69_out = run_add_layer("extracted_params_hsigmoid/layer069_ADD_1_block4c_add_1_Add", "all_layer_io/layer_69_ADD",
                                         layer_68_out, conv68_out_h * conv68_out_w * 96, layer_53_out, conv68_out_h * conv68_out_w * 96);
    
    // ===== LAYER 69-84: BLOCK 5a with SE =====
    printf("[*] Block 5a (Layers 69-84 with SE)...\n");
    
    // LAYER 69: CONV_2D (block5a_expand_conv) [1,14,14,96] -> [1,14,14,384]
    printf("Starting Layer 69 CONV_2D...\n");
    int conv70_out_h, conv70_out_w;
    int8_t* layer_70_out = run_conv2d_layer("extracted_params_hsigmoid/layer070_CONV_2D_1_block5a_expand_conv_1_BiasAdd_1_block5a_expand_conv_1_convolu", "all_layer_io/layer_70_CONV_2D",
                                            layer_69_out, 14, 14, 96, 384, 1, 1, 1, 1, "SAME", &conv70_out_h, &conv70_out_w);
    
    // LAYER 70: HARD_SWISH
    printf("Starting Layer 70 HARD_SWISH...\n");
    int layer_71_size = conv70_out_h * conv70_out_w * 384;
    int8_t* layer_71_out = run_hardswish_layer("extracted_params_hsigmoid/layer071_HARD_SWISH_1_block5a_expand_activation_1_truediv__1_block5a_expand_acti", "all_layer_io/layer_71_HARD_SWISH",
                                              layer_70_out, &layer_71_size);
    
    // LAYER 71: DEPTHWISE_CONV_2D (block5a_dwconv2) [1,14,14,384] -> [1,7,7,384]
    printf("Starting Layer 71 DEPTHWISE_CONV_2D...\n");
    int conv72_out_h, conv72_out_w;
    int8_t* layer_72_out = run_dw_conv_layer("extracted_params_hsigmoid/layer072_DEPTHWISE_CONV_2D_1_block5a_dwconv2_1_BiasAdd_1_block5a_dwconv2_1_depth",
                                             layer_71_out, 14, 14, 384, 384, 3, 3, 2, 2, "SAME", &conv72_out_h, &conv72_out_w);
    
    // LAYER 72: HARD_SWISH
    printf("Starting Layer 72 HARD_SWISH...\n");
    int layer_73_size = conv72_out_h * conv72_out_w * 384;
    int8_t* layer_73_out = run_hardswish_layer("extracted_params_hsigmoid/layer073_HARD_SWISH_1_block5a_activation_1_truediv__1_block5a_activation_1_Relu6", "all_layer_io/layer_73_HARD_SWISH",
                                              layer_72_out, &layer_73_size);
    
    // LAYER 73: MEAN (SE squeeze) [1,14,14,384] -> [1,1,384]
    printf("Starting Layer 73 MEAN...\n");
    int8_t* layer_74_out = run_mean_layer("extracted_params_hsigmoid/layer074_MEAN_1_block5a_se_squeeze_1_Mean", "all_layer_io/layer_74_MEAN",
                                          layer_73_out, 14, 14, 384, 1, 0);
    
    printf("Skip layer 74-77 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 78: CONV_2D (SE reduce) [1,1,384] -> [1,1,16]
    printf("Starting Layer 78 CONV_2D...\n");
    int conv79_out_h, conv79_out_w;
    int8_t* layer_79_out = run_conv2d_layer("extracted_params_hsigmoid/layer079_CONV_2D_1_block5a_se_reduce_1_BiasAdd_1_block5a_se_reduce_1_convolution", "all_layer_io/layer_79_CONV_2D",
                                            layer_74_out, 1, 1, 384, 16, 1, 1, 1, 1, "SAME", &conv79_out_h, &conv79_out_w);
    
    // LAYER 79: HARD_SWISH (SE reduce activation)
    printf("Starting Layer 79 HARD_SWISH...\n");
    int layer_80_size = conv79_out_h * conv79_out_w * 16;
    int8_t* layer_80_out = run_hardswish_layer("extracted_params_hsigmoid/layer080_HARD_SWISH_1_block5a_se_reduce_1_truediv__1_block5a_se_reduce_1_Relu6_1", "all_layer_io/layer_80_HARD_SWISH",
                                              layer_79_out, &layer_80_size);
    
    // LAYER 80: CONV_2D (SE expand) [1,1,16] -> [1,1,384]
    printf("Starting Layer 80 CONV_2D...\n");
    int conv81_out_h, conv81_out_w;
    int8_t* layer_81_out = run_conv2d_layer("extracted_params_hsigmoid/layer081_CONV_2D_1_block5a_se_expand_1_Relu6_1_block5a_se_expand_1_add_1_block5a", "all_layer_io/layer_81_CONV_2D",
                                            layer_80_out, 1, 1, 16, 384, 1, 1, 1, 1, "SAME", &conv81_out_h, &conv81_out_w);
    
    // LAYER 81: MUL (SE output scaling)
    printf("Starting Layer 81 MUL...\n");
    int8_t* layer_82_out = run_mul_layer("extracted_params_hsigmoid/layer082_MUL_1_block5a_se_expand_1_truediv", "all_layer_io/layer_82_MUL",
                                        layer_81_out, 384, NULL, 1, 0);
    
    // LAYER 82: MUL (SE excitation - broadcast)
    printf("Starting Layer 82 MUL...\n");
    int8_t* layer_83_out = run_mul_layer("extracted_params_hsigmoid/layer083_MUL_1_block5a_se_excite_1_Mul", "all_layer_io/layer_83_MUL",
                                        NULL, 14*14*384, layer_82_out, 384, 1);
    
    // LAYER 83: CONV_2D (block5a project) [1,7,7,384] -> [1,7,7,112]
    printf("Starting Layer 83 CONV_2D...\n");
    int conv84_out_h, conv84_out_w;
    int8_t* layer_84_out = run_conv2d_layer("extracted_params_hsigmoid/layer084_CONV_2D_1_block5a_project_conv_1_BiasAdd_1_block5a_project_conv_1_convo", "all_layer_io/layer_84_CONV_2D",
                                            layer_83_out, 7, 7, 384, 112, 1, 1, 1, 1, "SAME", &conv84_out_h, &conv84_out_w);
    
    // ===== LAYER 85-94: BLOCK 5b with SE and skip =====
    printf("[*] Block 5b (Layers 85-94 with SE)...\n");
    
    // LAYER 85: CONV_2D (block5b_expand_conv) [1,7,7,112] -> [1,7,7,672]
    printf("Starting Layer 85 CONV_2D...\n");
    int conv86_out_h, conv86_out_w;
    int8_t* layer_86_out = run_conv2d_layer("extracted_params_hsigmoid/layer085_CONV_2D_1_block5b_expand_conv_1_BiasAdd_1_block5b_expand_conv_1_convolu", "all_layer_io/layer_85_CONV_2D",
                                            layer_84_out, 7, 7, 112, 672, 1, 1, 1, 1, "SAME", &conv86_out_h, &conv86_out_w);
    
    // LAYER 86: HARD_SWISH
    printf("Starting Layer 86 HARD_SWISH...\n");
    int layer_87_size = conv86_out_h * conv86_out_w * 672;
    int8_t* layer_87_out = run_hardswish_layer("extracted_params_hsigmoid/layer086_HARD_SWISH_1_block5b_expand_activation_1_truediv__1_block5b_expand_acti", "all_layer_io/layer_86_HARD_SWISH",
                                              layer_86_out, &layer_87_size);
    
    // LAYER 87: DEPTHWISE_CONV_2D (block5b_dwconv2) [1,7,7,672] -> [1,7,7,672]
    printf("Starting Layer 87 DEPTHWISE_CONV_2D...\n");
    int conv88_out_h, conv88_out_w;
    int8_t* layer_88_out = run_dw_conv_layer("extracted_params_hsigmoid/layer087_DEPTHWISE_CONV_2D_1_block5b_dwconv2_1_BiasAdd_1_block5b_dwconv2_1_depth",
                                             layer_87_out, 7, 7, 672, 672, 3, 3, 1, 1, "SAME", &conv88_out_h, &conv88_out_w);
    
    // LAYER 88: HARD_SWISH
    printf("Starting Layer 88 HARD_SWISH...\n");
    int layer_89_size = conv88_out_h * conv88_out_w * 672;
    int8_t* layer_89_out = run_hardswish_layer("extracted_params_hsigmoid/layer088_HARD_SWISH_1_block5b_activation_1_truediv__1_block5b_activation_1_Relu6", "all_layer_io/layer_88_HARD_SWISH",
                                              layer_88_out, &layer_89_size);
    
    // LAYER 89: MEAN (SE squeeze) [1,7,7,672] -> [1,1,672]
    printf("Starting Layer 89 MEAN...\n");
    int8_t* layer_90_out = run_mean_layer("extracted_params_hsigmoid/layer089_MEAN_1_block5b_se_squeeze_1_Mean", "all_layer_io/layer_89_MEAN",
                                          layer_89_out, 7, 7, 672, 1, 0);
    
    printf("Skip layer 90-93 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 94: CONV_2D (SE reduce) [1,1,672] -> [1,1,28]
    printf("Starting Layer 94 CONV_2D...\n");
    int conv95_out_h, conv95_out_w;
    int8_t* layer_94_se_reduce_out = run_conv2d_layer("extracted_params_hsigmoid/layer094_CONV_2D_1_block5b_se_reduce_1_BiasAdd_1_block5b_se_reduce_1_convolution", "all_layer_io/layer_94_CONV_2D",
                                                      layer_90_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv95_out_h, &conv95_out_w);
    
    (void)layer_94_se_reduce_out;

    printf("\n=== COMPLETED LAYERS 4-94 ===\n");
    printf("Final output: Layer 94 SE reduce output ready for the next sequence step\n");
    printf("Output shape: [%d, %d, %d, %d]\n", 1, conv95_out_h, conv95_out_w, 28);
    
    // Cleanup
    // Note: In this pattern, memory management is handled internally by run_*_layer functions
    
    return 0;
}
