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
#include "layer_paths.h"

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
    
    // LAYER 5: CONV_2D (stem_conv_1) [1,224,224,3] -> [1,112,112,32]
    printf("Starting Layer 5 CONV_2D...\n");
    int conv5_out_h, conv5_out_w;
    int8_t* layer_5_out = run_conv2d_layer(LAYER_PARAMS[5], LAYER_IO_PATHS[5],
                                           layer_4_out, 224, 224, 3, 32, 3, 3, 2, 2, "SAME", &conv5_out_h, &conv5_out_w);
    
    // LAYER 6: HARD_SWISH
    printf("Starting Layer 6 HARD_SWISH...\n");
    int layer_6_size = conv5_out_h * conv5_out_w * 32;
    int8_t* layer_6_out = run_hardswish_layer(LAYER_PARAMS[6], LAYER_IO_PATHS[6],
                                             layer_5_out, &layer_6_size);
    
    // LAYER 7: CONV_2D (block1a_project_conv) [1,112,112,32] -> [1,112,112,16]
    printf("Starting Layer 7 CONV_2D...\n");
    int conv7_out_h, conv7_out_w;
    int8_t* layer_7_out = run_conv2d_layer(LAYER_PARAMS[7], LAYER_IO_PATHS[7],
                                           layer_6_out, 112, 112, 32, 16, 3, 3, 1, 1, "SAME", &conv7_out_h, &conv7_out_w);
    
    // LAYER 8: HARD_SWISH
    printf("Starting Layer 8 HARD_SWISH...\n");
    int layer_8_size = conv7_out_h * conv7_out_w * 16;
    int8_t* layer_8_out = run_hardswish_layer(LAYER_PARAMS[8], LAYER_IO_PATHS[8],
                                             layer_7_out, &layer_8_size);
    
    // LAYER 9: CONV_2D (block2a_expand_conv) [1,112,112,16] -> [1,56,56,64]
    printf("Starting Layer 9 CONV_2D...\n");
    int conv9_out_h, conv9_out_w;
    int8_t* layer_9_out = run_conv2d_layer(LAYER_PARAMS[9], LAYER_IO_PATHS[9],
                                           layer_8_out, 112, 112, 16, 64, 3, 3, 2, 2, "SAME", &conv9_out_h, &conv9_out_w);
    
    // LAYER 10: HARD_SWISH
    printf("Starting Layer 10 HARD_SWISH...\n");
    int layer_10_size = conv9_out_h * conv9_out_w * 64;
    int8_t* layer_10_out = run_hardswish_layer(LAYER_PARAMS[10], LAYER_IO_PATHS[10],
                                              layer_9_out, &layer_10_size);
    
    // LAYER 11: CONV_2D (block2a_project_conv) [1,56,56,64] -> [1,56,56,32]
    printf("Starting Layer 11 CONV_2D...\n");
    int conv11_out_h, conv11_out_w;
    int8_t* layer_11_out = run_conv2d_layer(LAYER_PARAMS[11], LAYER_IO_PATHS[11],
                                            layer_10_out, 56, 56, 64, 32, 1, 1, 1, 1, "SAME", &conv11_out_h, &conv11_out_w);
    
    // LAYER 12: CONV_2D (block2b_expand_conv) [1,56,56,32] -> [1,56,56,128]
    printf("Starting Layer 12 CONV_2D...\n");
    int conv12_out_h, conv12_out_w;
    int8_t* layer_12_out = run_conv2d_layer(LAYER_PARAMS[12], LAYER_IO_PATHS[12],
                                            layer_11_out, 56, 56, 32, 128, 3, 3, 1, 1, "SAME", &conv12_out_h, &conv12_out_w);
    
    // LAYER 13: HARD_SWISH
    printf("Starting Layer 13 HARD_SWISH...\n");
    int layer_13_size = conv12_out_h * conv12_out_w * 128;
    int8_t* layer_13_out = run_hardswish_layer(LAYER_PARAMS[13], LAYER_IO_PATHS[13],
                                              layer_12_out, &layer_13_size);
    
    // LAYER 14: CONV_2D (block2b_project_conv) [1,56,56,128] -> [1,56,56,32]
    printf("Starting Layer 14 CONV_2D...\n");
    int conv14_out_h, conv14_out_w;
    int8_t* layer_14_out = run_conv2d_layer(LAYER_PARAMS[14], LAYER_IO_PATHS[14],
                                            layer_13_out, 56, 56, 128, 32, 1, 1, 1, 1, "SAME", &conv14_out_h, &conv14_out_w);
    
    // LAYER 15: ADD (skip connection from layer 10)
    printf("Starting Layer 15 ADD...\n");
    int8_t* layer_15_out = run_add_layer(LAYER_PARAMS[15], LAYER_IO_PATHS[15],
                                         layer_14_out, conv14_out_h * conv14_out_w * 32, layer_11_out, conv14_out_h * conv14_out_w * 32);
    
    // LAYER 16: CONV_2D (block3a_expand_conv) [1,56,56,32] -> [1,28,28,128]
    printf("Starting Layer 16 CONV_2D...\n");
    int conv16_out_h, conv16_out_w;
    int8_t* layer_16_out = run_conv2d_layer(LAYER_PARAMS[16], LAYER_IO_PATHS[16],
                                            layer_15_out, 56, 56, 32, 128, 3, 3, 2, 2, "SAME", &conv16_out_h, &conv16_out_w);
    
    // LAYER 17: HARD_SWISH
    printf("Starting Layer 17 HARD_SWISH...\n");
    int layer_17_size = conv16_out_h * conv16_out_w * 128;
    int8_t* layer_17_out = run_hardswish_layer(LAYER_PARAMS[17], LAYER_IO_PATHS[17],
                                              layer_16_out, &layer_17_size);
    
    // LAYER 18: CONV_2D (block3a_project_conv) [1,28,28,128] -> [1,28,28,48]
    printf("Starting Layer 18 CONV_2D...\n");
    int conv18_out_h, conv18_out_w;
    int8_t* layer_18_out = run_conv2d_layer(LAYER_PARAMS[18], LAYER_IO_PATHS[18],
                                            layer_17_out, 28, 28, 128, 48, 1, 1, 1, 1, "SAME", &conv18_out_h, &conv18_out_w);
    
    // LAYER 19: CONV_2D (block3b_expand_conv) [1,28,28,48] -> [1,28,28,192]
    printf("Starting Layer 19 CONV_2D...\n");
    int conv19_out_h, conv19_out_w;
    int8_t* layer_19_out = run_conv2d_layer(LAYER_PARAMS[19], LAYER_IO_PATHS[19],
                                            layer_18_out, 28, 28, 48, 192, 3, 3, 1, 1, "SAME", &conv19_out_h, &conv19_out_w);
    
    // LAYER 20: HARD_SWISH
    printf("Starting Layer 20 HARD_SWISH...\n");
    int layer_20_size = conv19_out_h * conv19_out_w * 192;
    int8_t* layer_20_out = run_hardswish_layer(LAYER_PARAMS[20], LAYER_IO_PATHS[20],
                                              layer_19_out, &layer_20_size);
    
    // LAYER 21: CONV_2D (block3b_project_conv) [1,28,28,192] -> [1,28,28,48]
    printf("Starting Layer 21 CONV_2D...\n");
    int conv21_out_h, conv21_out_w;
    int8_t* layer_21_out = run_conv2d_layer(LAYER_PARAMS[21], LAYER_IO_PATHS[21],
                                            layer_20_out, 28, 28, 192, 48, 1, 1, 1, 1, "SAME", &conv21_out_h, &conv21_out_w);
    
    // LAYER 22: ADD (skip connection from layer 17)
    printf("Starting Layer 22 ADD...\n");
    int8_t* layer_22_out = run_add_layer(LAYER_PARAMS[22], LAYER_IO_PATHS[22],
                                         layer_21_out, conv21_out_h * conv21_out_w * 48, layer_18_out, conv21_out_h * conv21_out_w * 48);
    
    printf("\n>>> BLOCKS 4a-5a (Layers 23-94) <<<\n\n");
    
    // ===== LAYER 23-35: BLOCK 4a with SE =====
    printf("[*] Block 4a (Layers 23-35 with SE)...\n");
    
    // LAYER 23: CONV_2D (block4a_expand_conv) [1,28,28,48] -> [1,28,28,192]
    printf("Starting Layer 23 CONV_2D...\n");
    int conv23_out_h, conv23_out_w;
    int8_t* layer_23_out = run_conv2d_layer(LAYER_PARAMS[23], LAYER_IO_PATHS[23],
                                            layer_22_out, 28, 28, 48, 192, 3, 3, 1, 1, "SAME", &conv23_out_h, &conv23_out_w);
    
    // LAYER 24: HARD_SWISH
    printf("Starting Layer 24 HARD_SWISH...\n");
    int layer_24_size = conv23_out_h * conv23_out_w * 192;
    int8_t* layer_24_out = run_hardswish_layer(LAYER_PARAMS[24], LAYER_IO_PATHS[24],
                                              layer_23_out, &layer_24_size);
    
    // LAYER 25: DEPTHWISE_CONV_2D (block4a_dwconv2) [1,28,28,192] -> [1,14,14,192]
    printf("Starting Layer 25 DEPTHWISE_CONV_2D...\n");
    int conv25_out_h, conv25_out_w;
    int8_t* layer_25_out = run_dw_conv_layer(LAYER_PARAMS[25],
                                             layer_24_out, 28, 28, 192, 192, 3, 3, 2, 2, "SAME", &conv25_out_h, &conv25_out_w);
    
    // LAYER 26: HARD_SWISH
    printf("Starting Layer 26 HARD_SWISH...\n");
    int layer_26_size = conv25_out_h * conv25_out_w * 192;
    int8_t* layer_26_out = run_hardswish_layer(LAYER_PARAMS[26], LAYER_IO_PATHS[26],
                                              layer_25_out, &layer_26_size);
    
    // LAYER 27: MEAN (SE squeeze) [1,14,14,192] -> [1,1,192]
    printf("Starting Layer 27 MEAN...\n");
    int8_t* layer_27_out = run_mean_layer(LAYER_PARAMS[27], LAYER_IO_PATHS[27],
                                          layer_26_out, 14, 14, 192, 1, 0);
    
    printf("Skip layer 27-30 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 32: CONV_2D (SE reduce) [1,1,192] -> [1,1,12]
    printf("Starting Layer 32 CONV_2D...\n");
    int conv32_out_h, conv32_out_w;
    int8_t* layer_32_out = run_conv2d_layer(LAYER_PARAMS[32], LAYER_IO_PATHS[32],
                                            layer_27_out, 1, 1, 192, 12, 1, 1, 1, 1, "SAME", &conv32_out_h, &conv32_out_w);
    
    // LAYER 33: HARD_SWISH (SE reduce activation)
    printf("Starting Layer 33 HARD_SWISH...\n");
    int layer_33_size = conv32_out_h * conv32_out_w * 12;
    int8_t* layer_33_out = run_hardswish_layer(LAYER_PARAMS[33], LAYER_IO_PATHS[33],
                                              layer_32_out, &layer_33_size);
    
    // LAYER 34: CONV_2D (SE expand) [1,1,12] -> [1,1,192]
    printf("Starting Layer 34 CONV_2D...\n");
    int conv34_out_h, conv34_out_w;
    int8_t* layer_34_out = run_conv2d_layer(LAYER_PARAMS[34], LAYER_IO_PATHS[34],
                                            layer_33_out, 1, 1, 12, 192, 1, 1, 1, 1, "SAME", &conv34_out_h, &conv34_out_w);
    
    // LAYER 35: MUL (SE output scaling)
    printf("Starting Layer 35 MUL...\n");
    int8_t* layer_35_out = run_mul_layer(LAYER_PARAMS[35], LAYER_IO_PATHS[35],
                                        layer_34_out, 192, NULL, 1, 0);
    
    // LAYER 36: MUL (SE excitation - broadcast)
    printf("Starting Layer 36 MUL...\n");
    int8_t* layer_36_out = run_mul_layer(LAYER_PARAMS[36], LAYER_IO_PATHS[36],
                                        NULL, 14*14*192, layer_35_out, 192, 1);
    
    // LAYER 37: CONV_2D (block4a project) [1,14,14,192] -> [1,14,14,96]
    printf("Starting Layer 37 CONV_2D...\n");
    int conv37_out_h, conv37_out_w;
    int8_t* layer_37_out = run_conv2d_layer(LAYER_PARAMS[37], LAYER_IO_PATHS[37],
                                            layer_36_out, 14, 14, 192, 96, 1, 1, 1, 1, "SAME", &conv37_out_h, &conv37_out_w);
    
    // ===== LAYER 37-52: BLOCK 4b with SE and skip =====
    printf("[*] Block 4b (Layers 37-52 with SE)...\n");
    
    // LAYER 38: CONV_2D (block4b_expand_conv) [1,14,14,96] -> [1,14,14,384]
    printf("Starting Layer 38 CONV_2D...\n");
    int conv38_out_h, conv38_out_w;
    int8_t* layer_38_out = run_conv2d_layer(LAYER_PARAMS[38], LAYER_IO_PATHS[38],
                                            layer_37_out, 14, 14, 96, 384, 1, 1, 1, 1, "SAME", &conv38_out_h, &conv38_out_w);
    
    // LAYER 39: HARD_SWISH
    printf("Starting Layer 39 HARD_SWISH...\n");
    int layer_39_size = conv38_out_h * conv38_out_w * 384;
    int8_t* layer_39_out = run_hardswish_layer(LAYER_PARAMS[39], LAYER_IO_PATHS[39],
                                              layer_38_out, &layer_39_size);
    
    // LAYER 40: DEPTHWISE_CONV_2D (block4b_dwconv2) [1,14,14,384] -> [1,14,14,384]
    printf("Starting Layer 40 DEPTHWISE_CONV_2D...\n");
    int conv40_out_h, conv40_out_w;
    int8_t* layer_40_out = run_dw_conv_layer(LAYER_PARAMS[40],
                                             layer_39_out, 14, 14, 384, 384, 3, 3, 1, 1, "SAME", &conv40_out_h, &conv40_out_w);
    
    // LAYER 41: HARD_SWISH
    printf("Starting Layer 41 HARD_SWISH...\n");
    int layer_41_size = conv40_out_h * conv40_out_w * 384;
    int8_t* layer_41_out = run_hardswish_layer(LAYER_PARAMS[41], LAYER_IO_PATHS[41],
                                              layer_40_out, &layer_41_size);
    
    // LAYER 42: MEAN (SE squeeze) [1,14,14,384] -> [1,1,384]
    printf("Starting Layer 42 MEAN...\n");
    int8_t* layer_42_out = run_mean_layer(LAYER_PARAMS[42], LAYER_IO_PATHS[42],
                                          layer_41_out, 14, 14, 384, 1, 0);
    
    printf("Skip layer 42-45 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 47: CONV_2D (SE reduce) [1,1,384] -> [1,1,16]
    printf("Starting Layer 47 CONV_2D...\n");
    int conv47_out_h, conv47_out_w;
    int8_t* layer_47_out = run_conv2d_layer(LAYER_PARAMS[47], LAYER_IO_PATHS[47],
                                            layer_42_out, 1, 1, 384, 16, 1, 1, 1, 1, "SAME", &conv47_out_h, &conv47_out_w);
    
    // LAYER 48: HARD_SWISH (SE reduce activation)
    printf("Starting Layer 48 HARD_SWISH...\n");
    int layer_48_size = conv47_out_h * conv47_out_w * 16;
    int8_t* layer_48_out = run_hardswish_layer(LAYER_PARAMS[48], LAYER_IO_PATHS[48],
                                              layer_47_out, &layer_48_size);
    
    // LAYER 49: CONV_2D (SE expand) [1,1,16] -> [1,1,384]
    printf("Starting Layer 49 CONV_2D...\n");
    int conv49_out_h, conv49_out_w;
    int8_t* layer_49_out = run_conv2d_layer(LAYER_PARAMS[49], LAYER_IO_PATHS[49],
                                            layer_48_out, 1, 1, 16, 384, 1, 1, 1, 1, "SAME", &conv49_out_h, &conv49_out_w);
    
    // LAYER 50: MUL (SE output scaling)
    printf("Starting Layer 50 MUL...\n");
    int8_t* layer_50_out = run_mul_layer(LAYER_PARAMS[50], LAYER_IO_PATHS[50],
                                        layer_49_out, 384, NULL, 1, 0);
    
    // LAYER 51: MUL (SE excitation - broadcast)
    printf("Starting Layer 51 MUL...\n");
    int8_t* layer_51_out = run_mul_layer(LAYER_PARAMS[51], LAYER_IO_PATHS[51],
                                        NULL, 14*14*384, layer_50_out, 384, 1);
    
    // LAYER 52: CONV_2D (block4b project) [1,14,14,384] -> [1,14,14,96]
    printf("Starting Layer 52 CONV_2D...\n");
    int conv52_out_h, conv52_out_w;
    int8_t* layer_52_out = run_conv2d_layer(LAYER_PARAMS[52], LAYER_IO_PATHS[52],
                                            layer_51_out, 14, 14, 384, 96, 1, 1, 1, 1, "SAME", &conv52_out_h, &conv52_out_w);
    
    // LAYER 53: ADD (skip connection from layer 36)
    printf("Starting Layer 53 ADD...\n");
    int8_t* layer_53_out = run_add_layer(LAYER_PARAMS[53], LAYER_IO_PATHS[53],
                                         layer_52_out, conv52_out_h * conv52_out_w * 96, layer_37_out, conv52_out_h * conv52_out_w * 96);
    
    // ===== LAYER 53-68: BLOCK 4c with SE and skip =====
    printf("[*] Block 4c (Layers 53-68 with SE)...\n");
    
    // LAYER 54: CONV_2D (block4c_expand_conv) [1,14,14,96] -> [1,14,14,384]
    printf("Starting Layer 54 CONV_2D...\n");
    int conv54_out_h, conv54_out_w;
    int8_t* layer_54_out = run_conv2d_layer(LAYER_PARAMS[54], LAYER_IO_PATHS[54],
                                            layer_53_out, 14, 14, 96, 384, 1, 1, 1, 1, "SAME", &conv54_out_h, &conv54_out_w);
    
    // LAYER 55: HARD_SWISH
    printf("Starting Layer 55 HARD_SWISH...\n");
    int layer_55_size = conv54_out_h * conv54_out_w * 384;
    int8_t* layer_55_out = run_hardswish_layer(LAYER_PARAMS[55], LAYER_IO_PATHS[55],
                                              layer_54_out, &layer_55_size);
    
    // LAYER 56: DEPTHWISE_CONV_2D (block4c_dwconv2) [1,14,14,384] -> [1,14,14,384]
    printf("Starting Layer 56 DEPTHWISE_CONV_2D...\n");
    int conv56_out_h, conv56_out_w;
    int8_t* layer_56_out = run_dw_conv_layer(LAYER_PARAMS[56],
                                             layer_55_out, 14, 14, 384, 384, 3, 3, 1, 1, "SAME", &conv56_out_h, &conv56_out_w);
    
    // LAYER 57: HARD_SWISH
    printf("Starting Layer 57 HARD_SWISH...\n");
    int layer_57_size = conv56_out_h * conv56_out_w * 384;
    int8_t* layer_57_out = run_hardswish_layer(LAYER_PARAMS[57], LAYER_IO_PATHS[57],
                                              layer_56_out, &layer_57_size);
    
    // LAYER 58: MEAN (SE squeeze) [1,14,14,384] -> [1,1,384]
    printf("Starting Layer 58 MEAN...\n");
    int8_t* layer_58_out = run_mean_layer(LAYER_PARAMS[58], LAYER_IO_PATHS[58],
                                          layer_57_out, 14, 14, 384, 1, 0);
    
    printf("Skip layer 58-61 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 63: CONV_2D (SE reduce) [1,1,384] -> [1,1,16]
    printf("Starting Layer 63 CONV_2D...\n");
    int conv63_out_h, conv63_out_w;
    int8_t* layer_63_out = run_conv2d_layer(LAYER_PARAMS[63], LAYER_IO_PATHS[63],
                                            layer_58_out, 1, 1, 384, 16, 1, 1, 1, 1, "SAME", &conv63_out_h, &conv63_out_w);
    
    // LAYER 64: HARD_SWISH (SE reduce activation)
    printf("Starting Layer 64 HARD_SWISH...\n");
    int layer_64_size = conv63_out_h * conv63_out_w * 16;
    int8_t* layer_64_out = run_hardswish_layer(LAYER_PARAMS[64], LAYER_IO_PATHS[64],
                                              layer_63_out, &layer_64_size);
    
    // LAYER 65: CONV_2D (SE expand) [1,1,16] -> [1,1,384]
    printf("Starting Layer 65 CONV_2D...\n");
    int conv65_out_h, conv65_out_w;
    int8_t* layer_65_out = run_conv2d_layer(LAYER_PARAMS[65], LAYER_IO_PATHS[65],
                                            layer_64_out, 1, 1, 16, 384, 1, 1, 1, 1, "SAME", &conv65_out_h, &conv65_out_w);
    
    // LAYER 66: MUL (SE output scaling)
    printf("Starting Layer 66 MUL...\n");
    int8_t* layer_66_out = run_mul_layer(LAYER_PARAMS[66], LAYER_IO_PATHS[66],
                                        layer_65_out, 384, NULL, 1, 0);
    
    // LAYER 67: MUL (SE excitation - broadcast)
    printf("Starting Layer 67 MUL...\n");
    int8_t* layer_67_out = run_mul_layer(LAYER_PARAMS[67], LAYER_IO_PATHS[67],
                                        NULL, 14*14*384, layer_66_out, 384, 1);
    
    // LAYER 68: CONV_2D (block4c project) [1,14,14,384] -> [1,14,14,96]
    printf("Starting Layer 68 CONV_2D...\n");
    int conv68_out_h, conv68_out_w;
    int8_t* layer_68_out = run_conv2d_layer(LAYER_PARAMS[68], LAYER_IO_PATHS[68],
                                            layer_67_out, 14, 14, 384, 96, 1, 1, 1, 1, "SAME", &conv68_out_h, &conv68_out_w);
    
    // LAYER 69: ADD (skip connection from layer 52)
    printf("Starting Layer 69 ADD...\n");
    int8_t* layer_69_out = run_add_layer(LAYER_PARAMS[69], LAYER_IO_PATHS[69],
                                         layer_68_out, conv68_out_h * conv68_out_w * 96, layer_53_out, conv68_out_h * conv68_out_w * 96);
    
    // ===== LAYER 69-84: BLOCK 5a with SE =====
    printf("[*] Block 5a (Layers 69-84 with SE)...\n");
    
    // LAYER 70: CONV_2D (block5a_expand_conv) [1,14,14,96] -> [1,14,14,384]
    printf("Starting Layer 70 CONV_2D...\n");
    int conv70_out_h, conv70_out_w;
    int8_t* layer_70_out = run_conv2d_layer(LAYER_PARAMS[70], LAYER_IO_PATHS[70],
                                            layer_69_out, 14, 14, 96, 384, 1, 1, 1, 1, "SAME", &conv70_out_h, &conv70_out_w);
    
    // LAYER 71: HARD_SWISH
    printf("Starting Layer 71 HARD_SWISH...\n");
    int layer_71_size = conv70_out_h * conv70_out_w * 384;
    int8_t* layer_71_out = run_hardswish_layer(LAYER_PARAMS[71], LAYER_IO_PATHS[71],
                                              layer_70_out, &layer_71_size);
    
    // LAYER 72: DEPTHWISE_CONV_2D (block5a_dwconv2) [1,14,14,384] -> [1,7,7,384]
    printf("Starting Layer 72 DEPTHWISE_CONV_2D...\n");
    int conv72_out_h, conv72_out_w;
    int8_t* layer_72_out = run_dw_conv_layer(LAYER_PARAMS[72],
                                             layer_71_out, 14, 14, 384, 384, 3, 3, 2, 2, "SAME", &conv72_out_h, &conv72_out_w);
    
    // LAYER 73: HARD_SWISH
    printf("Starting Layer 73 HARD_SWISH...\n");
    int layer_73_size = conv72_out_h * conv72_out_w * 384;
    int8_t* layer_73_out = run_hardswish_layer(LAYER_PARAMS[73], LAYER_IO_PATHS[73],
                                              layer_72_out, &layer_73_size);
    
    // LAYER 74: MEAN (SE squeeze) [1,14,14,384] -> [1,1,384]
    printf("Starting Layer 74 MEAN...\n");
    int8_t* layer_74_out = run_mean_layer(LAYER_PARAMS[74], LAYER_IO_PATHS[74],
                                          layer_73_out, 14, 14, 384, 1, 0);
    
    printf("Skip layer 74-77 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 79: CONV_2D (SE reduce) [1,1,384] -> [1,1,16]
    printf("Starting Layer 79 CONV_2D...\n");
    int conv79_out_h, conv79_out_w;
    int8_t* layer_79_out = run_conv2d_layer(LAYER_PARAMS[79], LAYER_IO_PATHS[79],
                                            layer_74_out, 1, 1, 384, 16, 1, 1, 1, 1, "SAME", &conv79_out_h, &conv79_out_w);
    
    // LAYER 80: HARD_SWISH (SE reduce activation)
    printf("Starting Layer 80 HARD_SWISH...\n");
    int layer_80_size = conv79_out_h * conv79_out_w * 16;
    int8_t* layer_80_out = run_hardswish_layer(LAYER_PARAMS[80], LAYER_IO_PATHS[80],
                                              layer_79_out, &layer_80_size);
    
    // LAYER 81: CONV_2D (SE expand) [1,1,16] -> [1,1,384]
    printf("Starting Layer 81 CONV_2D...\n");
    int conv81_out_h, conv81_out_w;
    int8_t* layer_81_out = run_conv2d_layer(LAYER_PARAMS[81], LAYER_IO_PATHS[81],
                                            layer_80_out, 1, 1, 16, 384, 1, 1, 1, 1, "SAME", &conv81_out_h, &conv81_out_w);
    
    // LAYER 82: MUL (SE output scaling)
    printf("Starting Layer 82 MUL...\n");
    int8_t* layer_82_out = run_mul_layer(LAYER_PARAMS[82], LAYER_IO_PATHS[82],
                                        layer_81_out, 384, NULL, 1, 0);
    
    // LAYER 83: MUL (SE excitation - broadcast)
    printf("Starting Layer 83 MUL...\n");
    int8_t* layer_83_out = run_mul_layer(LAYER_PARAMS[83], LAYER_IO_PATHS[83],
                                        NULL, 14*14*384, layer_82_out, 384, 1);
    
    // LAYER 84: CONV_2D (block5a project) [1,7,7,384] -> [1,7,7,112]
    printf("Starting Layer 84 CONV_2D...\n");
    int conv84_out_h, conv84_out_w;
    int8_t* layer_84_out = run_conv2d_layer(LAYER_PARAMS[84], LAYER_IO_PATHS[84],
                                            layer_83_out, 7, 7, 384, 112, 1, 1, 1, 1, "SAME", &conv84_out_h, &conv84_out_w);
    
    // ===== LAYER 85-94: BLOCK 5b with SE and skip =====
    printf("[*] Block 5b (Layers 85-94 with SE)...\n");
    
    // LAYER 85: CONV_2D (block5b_expand_conv) [1,7,7,112] -> [1,7,7,672]
    printf("Starting Layer 85 CONV_2D...\n");
    int conv86_out_h, conv86_out_w;
    int8_t* layer_86_out = run_conv2d_layer(LAYER_PARAMS[85], LAYER_IO_PATHS[85],
                                            layer_84_out, 7, 7, 112, 672, 1, 1, 1, 1, "SAME", &conv86_out_h, &conv86_out_w);
    
    // LAYER 86: HARD_SWISH
    printf("Starting Layer 86 HARD_SWISH...\n");
    int layer_87_size = conv86_out_h * conv86_out_w * 672;
    int8_t* layer_87_out = run_hardswish_layer(LAYER_PARAMS[86], LAYER_IO_PATHS[86],
                                              layer_86_out, &layer_87_size);
    
    // LAYER 87: DEPTHWISE_CONV_2D (block5b_dwconv2) [1,7,7,672] -> [1,7,7,672]
    printf("Starting Layer 87 DEPTHWISE_CONV_2D...\n");
    int conv88_out_h, conv88_out_w;
    int8_t* layer_88_out = run_dw_conv_layer(LAYER_PARAMS[87],
                                             layer_87_out, 7, 7, 672, 672, 3, 3, 1, 1, "SAME", &conv88_out_h, &conv88_out_w);
    
    // LAYER 88: HARD_SWISH
    printf("Starting Layer 88 HARD_SWISH...\n");
    int layer_89_size = conv88_out_h * conv88_out_w * 672;
    int8_t* layer_89_out = run_hardswish_layer(LAYER_PARAMS[88], LAYER_IO_PATHS[88],
                                              layer_88_out, &layer_89_size);
    
    // LAYER 89: MEAN (SE squeeze) [1,7,7,672] -> [1,1,672]
    printf("Starting Layer 89 MEAN...\n");
    int8_t* layer_90_out = run_mean_layer(LAYER_PARAMS[89], LAYER_IO_PATHS[89],
                                          layer_89_out, 7, 7, 672, 1, 0);
    
    printf("Skip layer 90-93 SHAPE, STRIDED_SLICE, PACK, RESHAPE...\n");
    
    // LAYER 94: CONV_2D (SE reduce) [1,1,672] -> [1,1,28]
    printf("Starting Layer 94 CONV_2D...\n");
    int conv95_out_h, conv95_out_w;
    int8_t* layer_94_se_reduce_out = run_conv2d_layer(LAYER_PARAMS[94], LAYER_IO_PATHS[94],
                                                      layer_90_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv95_out_h, &conv95_out_w);
    
    printf("Starting Layer 95 HardSwish...\n");
    int layer_95_size;
    int8_t* layer_95_out = run_hardswish_layer(LAYER_PARAMS[95], LAYER_IO_PATHS[95], layer_94_se_reduce_out, &layer_95_size);

    printf("Starting Layer 96 CONV_2D...\n");
    int conv_out_h, conv_out_w;
    int8_t* layer_96_out = run_conv2d_layer(LAYER_PARAMS[96], LAYER_IO_PATHS[96],
                                            layer_95_out, 1, 1, 28, 672, 1, 1, 1, 1, "SAME", &conv_out_h, &conv_out_w);

    printf("Starting Layer 97 MUL...\n");
    int8_t* layer_97_out = run_mul_layer(LAYER_PARAMS[97], LAYER_IO_PATHS[97],
                                         layer_96_out, 672, NULL, 1, 0);

    printf("Starting Layer 98 MUL...\n");
    int8_t* layer_98_out = run_mul_layer(LAYER_PARAMS[98], LAYER_IO_PATHS[98],
                                         NULL, 14*14*672, layer_97_out, 672, 1);
    printf("Starting Layer 99 CONV_2D...\n");
    int conv99_out_h, conv99_out_w;
    int8_t* layer_99_out = run_conv2d_layer(LAYER_PARAMS[99], LAYER_IO_PATHS[99],
                                            layer_98_out, 14, 14, 672, 112, 1, 1, 1, 1, "SAME", &conv99_out_h, &conv99_out_w);
    printf("Starting Layer 100 ADD...\n");
    int8_t* layer_100_out = run_add_layer(LAYER_PARAMS[100], LAYER_IO_PATHS[100],
                                        layer_99_out, conv99_out_h * conv99_out_w * 112, NULL, conv99_out_h * conv99_out_w * 112);
    
    printf("Starting Layer 101 Conv_2D...\n");
    int conv101_out_h, conv101_out_w;
    int8_t* layer_101_out = run_conv2d_layer(LAYER_PARAMS[101], LAYER_IO_PATHS[101],
                                            layer_100_out, conv99_out_h, conv99_out_w, 112, 672, 1, 1, 1, 1, "SAME", &conv101_out_h, &conv101_out_w);
    printf("Starting Layer 102 HardSwish...\n");
    int layer_102_size = conv101_out_h * conv101_out_w * 672;
    int8_t* layer_102_out = run_hardswish_layer(LAYER_PARAMS[102], LAYER_IO_PATHS[102],
                                                layer_101_out, &layer_102_size);
    printf("Starting Layer 103 DepthwiseConv2D...\n");
    int conv103_out_h, conv103_out_w;
    int8_t* layer_103_out = run_dw_conv_layer(LAYER_PARAMS[103],
                                            layer_102_out, conv101_out_h, conv101_out_w, 672, 672, 3, 3, 1, 1, "SAME", &conv103_out_h, &conv103_out_w);
    printf("Starting Layer 104 HardSwish...\n");
    int layer_104_size = conv103_out_h * conv103_out_w * 672;
    int8_t* layer_104_out = run_hardswish_layer(LAYER_PARAMS[104], LAYER_IO_PATHS[104],
                                                layer_103_out, &layer_104_size);
    printf("Starting Layer 105 Mean...\n");
    int8_t* layer_105_out = run_mean_layer(LAYER_PARAMS[105], LAYER_IO_PATHS[105],
                                            layer_104_out, conv103_out_h, conv103_out_w, 672, 1, 0);
    printf("Skip layer 106 shape 107 stride_slice 108 pack 109 reshape...\n");
    printf("Starting Layer 110 Conv_2D...\n");
    int conv110_out_h, conv110_out_w;
    int8_t* layer_110_out = run_conv2d_layer(LAYER_PARAMS[110], LAYER_IO_PATHS[110],
                                            layer_105_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv110_out_h, &conv110_out_w);
    printf("Starting Layer 111 HardSwish...\n");
    int layer_111_size = conv110_out_h * conv110_out_w * 28;
    int8_t* layer_111_out = run_hardswish_layer(LAYER_PARAMS[111], LAYER_IO_PATHS[111],
                                                layer_110_out, &layer_111_size);
    printf("Starting Layer 112 Conv_2D...\n");
    int conv112_out_h, conv112_out_w;
    int8_t* layer_112_out = run_conv2d_layer(LAYER_PARAMS[112], LAYER_IO_PATHS[112],
                                            layer_111_out, conv110_out_h, conv110_out_w, 28, 672, 1, 1, 1, 1, "SAME", &conv112_out_h, &conv112_out_w);
    printf("Starting Layer 113 MUL...\n");
    int8_t* layer_113_out = run_mul_layer(LAYER_PARAMS[113], LAYER_IO_PATHS[113],
                                         layer_112_out, 672, NULL, 1, 0);
    printf("Starting Layer 114 MUL...\n");
    int8_t* layer_114_out = run_mul_layer(LAYER_PARAMS[114], LAYER_IO_PATHS[114],
                                         NULL, 14*14*672, layer_113_out, 672, 1);
    printf("Starting Layer 115 CONV_2D...\n");
    int conv115_out_h, conv115_out_w;
    int8_t* layer_115_out = run_conv2d_layer(LAYER_PARAMS[115], LAYER_IO_PATHS[115],
                                            layer_114_out, 14, 14, 672, 112, 1, 1, 1, 1, "SAME", &conv115_out_h, &conv115_out_w);
    printf("Starting Layer 116 ADD...\n");
    int8_t* layer_116_out = run_add_layer(LAYER_PARAMS[116], LAYER_IO_PATHS[116],
                                        layer_115_out, conv115_out_h * conv115_out_w * 112, layer_100_out, conv115_out_h * conv115_out_w * 112);
    
    printf("Starting Layer 117 Conv_2D...\n");
    int conv117_out_h, conv117_out_w;
    int8_t* layer_117_out = run_conv2d_layer(LAYER_PARAMS[117], LAYER_IO_PATHS[117],
                                            layer_116_out, conv115_out_h, conv115_out_w, 112, 672, 1, 1, 1, 1, "SAME", &conv117_out_h, &conv117_out_w);
    printf("Starting Layer 118 HardSwish...\n");
    int layer_118_size = conv117_out_h * conv117_out_w * 672;
    int8_t* layer_118_out = run_hardswish_layer(LAYER_PARAMS[118], LAYER_IO_PATHS[118],
                                                layer_117_out, &layer_118_size);
    printf("Starting Layer 119 DepthwiseConv2D...\n");
    int conv119_out_h, conv119_out_w;
    int8_t* layer_119_out = run_dw_conv_layer(LAYER_PARAMS[119],
                                            layer_118_out, conv117_out_h, conv117_out_w, 672, 672, 3, 3, 1, 1, "SAME", &conv119_out_h, &conv119_out_w);
    printf("Starting Layer 120 HardSwish...\n");
    int layer_120_size = conv119_out_h * conv119_out_w * 672;
    int8_t* layer_120_out = run_hardswish_layer(LAYER_PARAMS[120], LAYER_IO_PATHS[120],
                                                layer_119_out, &layer_120_size);
    printf("Starting Layer 121 Mean...\n");
    int8_t* layer_121_out = run_mean_layer(LAYER_PARAMS[121], LAYER_IO_PATHS[121],
                                            layer_120_out, conv119_out_h, conv119_out_w, 672, 1, 0);
    printf("Skip layer 122 shape 123 stride_slice 124 pack 125 reshape...\n");
    printf("Starting Layer 126 Conv_2D...\n");
    int conv126_out_h, conv126_out_w;
    int8_t* layer_126_out = run_conv2d_layer(LAYER_PARAMS[126], LAYER_IO_PATHS[126],
                                            layer_121_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv126_out_h, &conv126_out_w);
    printf("Starting Layer 127 HardSwish...\n");
    int layer_127_size = conv126_out_h * conv126_out_w * 28;
    int8_t* layer_127_out = run_hardswish_layer(LAYER_PARAMS[127], LAYER_IO_PATHS[127],
                                                layer_126_out, &layer_127_size);
    printf("Starting Layer 128 Conv_2D...\n");
    int conv128_out_h, conv128_out_w;
    int8_t* layer_128_out = run_conv2d_layer(LAYER_PARAMS[128], LAYER_IO_PATHS[128],
                                            layer_127_out, conv126_out_h, conv126_out_w, 28, 672, 1, 1, 1, 1, "SAME", &conv128_out_h, &conv128_out_w);
    printf("Starting Layer 129 MUL...\n");
    int8_t* layer_129_out = run_mul_layer(LAYER_PARAMS[129], LAYER_IO_PATHS[129],
                                         layer_128_out, 672, NULL, 1, 0);
    printf("Starting Layer 130 MUL...\n");
    int8_t* layer_130_out = run_mul_layer(LAYER_PARAMS[130], LAYER_IO_PATHS[130],
                                         NULL, 14*14*672, layer_129_out, 672, 1);
    printf("Starting Layer 131 CONV_2D...\n");
    int conv131_out_h, conv131_out_w;
    int8_t* layer_131_out = run_conv2d_layer(LAYER_PARAMS[131], LAYER_IO_PATHS[131],
                                            layer_130_out, 14, 14, 672, 112, 1, 1, 1, 1, "SAME", &conv131_out_h, &conv131_out_w);
    printf("Starting Layer 132 ADD...\n");
    int8_t* layer_132_out = run_add_layer(LAYER_PARAMS[132], LAYER_IO_PATHS[132],
                                        layer_131_out, conv131_out_h * conv131_out_w * 112, layer_116_out, conv131_out_h * conv131_out_w * 112);
    printf("Starting Layer 133 Conv_2D...\n");
    int conv133_out_h, conv133_out_w;
    int8_t* layer_133_out = run_conv2d_layer(LAYER_PARAMS[133], LAYER_IO_PATHS[133],
                                            layer_132_out, conv131_out_h, conv131_out_w, 112, 672, 1, 1, 1, 1, "SAME", &conv133_out_h, &conv133_out_w);
    printf("Starting Layer 134 HardSwish...\n");
    int layer_134_size = conv133_out_h * conv133_out_w * 672;
    int8_t* layer_134_out = run_hardswish_layer(LAYER_PARAMS[134], LAYER_IO_PATHS[134],
                                                layer_133_out, &layer_134_size);
    printf("Starting Layer 135 DepthwiseConv2D...\n");
    int conv135_out_h, conv135_out_w;
    int8_t* layer_135_out = run_dw_conv_layer(LAYER_PARAMS[135], 
                                            layer_134_out, conv133_out_h, conv133_out_w, 672, 672, 3, 3, 1, 1, "SAME", &conv135_out_h, &conv135_out_w);
    printf("Starting Layer 136 HardSwish...\n");
    int layer_136_size = conv135_out_h * conv135_out_w * 672;
    int8_t* layer_136_out = run_hardswish_layer(LAYER_PARAMS[136], LAYER_IO_PATHS[136],
                                                layer_135_out, &layer_136_size);
    printf("Starting Layer 137 Mean...\n");
    int8_t* layer_137_out = run_mean_layer(LAYER_PARAMS[137], LAYER_IO_PATHS[137],
                                            layer_136_out, conv135_out_h, conv135_out_w, 672, 1, 0);
    printf("Skip layer 138 shape 139 stride_slice 140 pack 141 reshape...\n");
    printf("Starting Layer 142 Conv_2D...\n");
    int conv142_out_h, conv142_out_w;
    int8_t* layer_142_out = run_conv2d_layer(LAYER_PARAMS[142], LAYER_IO_PATHS[142],
                                            layer_137_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv142_out_h, &conv142_out_w);
    printf("Starting Layer 143 HardSwish...\n");
    int layer_143_size = conv142_out_h * conv142_out_w * 28;
    int8_t* layer_143_out = run_hardswish_layer(LAYER_PARAMS[143], LAYER_IO_PATHS[143],
                                                layer_142_out, &layer_143_size);
    printf("Starting Layer 144 Conv_2D...\n");
    int conv144_out_h, conv144_out_w;
    int8_t* layer_144_out = run_conv2d_layer(LAYER_PARAMS[144], LAYER_IO_PATHS[144],
                                            layer_143_out, conv142_out_h, conv142_out_w, 28, 672, 1, 1, 1, 1, "SAME", &conv144_out_h, &conv144_out_w);
    printf("Starting Layer 145 MUL...\n");
    int8_t* layer_145_out = run_mul_layer(LAYER_PARAMS[145], LAYER_IO_PATHS[145],
                                         layer_144_out, 672, NULL, 1, 0);
    printf("Starting Layer 146 MUL...\n");
    int8_t* layer_146_out = run_mul_layer(LAYER_PARAMS[146], LAYER_IO_PATHS[146],
                                         NULL, 14*14*672, layer_145_out, 672, 1);
    printf("Starting Layer 147 CONV_2D...\n");
    int conv147_out_h, conv147_out_w;
    int8_t* layer_147_out = run_conv2d_layer(LAYER_PARAMS[147], LAYER_IO_PATHS[147],
                                            layer_146_out, 14, 14, 672, 112, 1, 1, 1, 1, "SAME", &conv147_out_h, &conv147_out_w);
    printf("Starting Layer 148 ADD...\n");
    int8_t* layer_148_out = run_add_layer(LAYER_PARAMS[148], LAYER_IO_PATHS[148],
                                        layer_147_out, conv147_out_h * conv147_out_w * 112, layer_132_out, conv147_out_h * conv147_out_w * 112);
    printf("Starting Layer 149 Conv_2D...\n");
    int conv149_out_h, conv149_out_w;
    int8_t* layer_149_out = run_conv2d_layer(LAYER_PARAMS[149], LAYER_IO_PATHS[149],
                                            layer_148_out, conv147_out_h, conv147_out_w, 112, 672, 1, 1, 1, 1, "SAME", &conv149_out_h, &conv149_out_w);
    printf("Starting Layer 150 HardSwish...\n");
    int layer_150_size = conv149_out_h * conv149_out_w * 672;
    int8_t* layer_150_out = run_hardswish_layer(LAYER_PARAMS[150], LAYER_IO_PATHS[150],
                                                layer_149_out, &layer_150_size);
    printf("Starting Layer 151 DepthwiseConv2D...\n");
    int conv151_out_h, conv151_out_w;
    int8_t* layer_151_out = run_dw_conv_layer(LAYER_PARAMS[151],
                                            layer_150_out, conv149_out_h, conv149_out_w, 672, 672, 3, 3, 2, 2, "SAME", &conv151_out_h, &conv151_out_w);
    printf("Starting Layer 152 HardSwish...\n");
    int layer_152_size = conv151_out_h * conv151_out_w * 672;
    int8_t* layer_152_out = run_hardswish_layer(LAYER_PARAMS[152], LAYER_IO_PATHS[152],
                                                layer_151_out, &layer_152_size);
    printf("Starting Layer 153 Mean...\n");
    int8_t* layer_153_out = run_mean_layer(LAYER_PARAMS[153], LAYER_IO_PATHS[153],
                                            layer_152_out, conv151_out_h, conv151_out_w, 672, 1, 0);
    printf("Skip layer 154 shape 155 stride_slice 156 pack 157 reshape...\n");
    printf("Starting Layer 158 Conv_2D...\n");
    int conv158_out_h, conv158_out_w;
    int8_t* layer_158_out = run_conv2d_layer(LAYER_PARAMS[158], LAYER_IO_PATHS[158],
                                            layer_153_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv158_out_h, &conv158_out_w);
    printf("Starting Layer 159 HardSwish...\n");
    int layer_159_size = conv158_out_h * conv158_out_w * 28;
    int8_t* layer_159_out = run_hardswish_layer(LAYER_PARAMS[159], LAYER_IO_PATHS[159],
                                                layer_158_out, &layer_159_size);
    printf("Starting Layer 160 Conv_2D...\n");
    int conv160_out_h, conv160_out_w;
    int8_t* layer_160_out = run_conv2d_layer(LAYER_PARAMS[160], LAYER_IO_PATHS[160],
                                            layer_159_out, conv158_out_h, conv158_out_w, 28, 672, 1, 1, 1, 1, "SAME", &conv160_out_h, &conv160_out_w);
    printf("Starting Layer 161 MUL...\n");
    int8_t* layer_161_out = run_mul_layer(LAYER_PARAMS[161], LAYER_IO_PATHS[161],
                                         layer_160_out, 672, NULL, 1, 0);
    printf("Starting Layer 162 MUL...\n");
    int8_t* layer_162_out = run_mul_layer(LAYER_PARAMS[162], LAYER_IO_PATHS[162],
                                         NULL, 7*7*672, layer_161_out, 672, 1);
    printf("Starting Layer 163 CONV_2D...\n");
    int conv163_out_h, conv163_out_w;
    int8_t* layer_163_out = run_conv2d_layer(LAYER_PARAMS[163], LAYER_IO_PATHS[163],
                                            layer_162_out, 7, 7, 672, 192, 1, 1, 1, 1, "SAME", &conv163_out_h, &conv163_out_w);
    printf("Starting Layer 164 CONV_2D...\n");
    int conv164_out_h, conv164_out_w;
    int8_t* layer_164_out = run_conv2d_layer(LAYER_PARAMS[164], LAYER_IO_PATHS[164],
                                            layer_163_out, conv163_out_h, conv163_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv164_out_h, &conv164_out_w);
    printf("Starting Layer 165 HardSwish...\n");
    int layer_165_size = conv164_out_h * conv164_out_w * 1152;
    int8_t* layer_165_out = run_hardswish_layer(LAYER_PARAMS[165], LAYER_IO_PATHS[165],
                                                layer_164_out, &layer_165_size);
    printf("Starting Layer 166 DepthwiseConv2D...\n");
    int conv166_out_h, conv166_out_w;
    int8_t* layer_166_out = run_dw_conv_layer(LAYER_PARAMS[166],
                                            layer_165_out, conv164_out_h, conv164_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv166_out_h, &conv166_out_w);
    printf("Starting Layer 167 HardSwish...\n");
    int layer_167_size = conv166_out_h * conv166_out_w * 1152;
    int8_t* layer_167_out = run_hardswish_layer(LAYER_PARAMS[167], LAYER_IO_PATHS[167],
                                                layer_166_out, &layer_167_size);
    // printf("Layer 167 output:\n");
    // for(int i = 0; i < conv166_out_h * conv166_out_w * 1152; ++i) {
    //     printf("%d ", layer_167_out[i]);
    //     if ((i + 1) % 1152 == 0) printf("\n");
    // }
    printf("Starting Layer 168 Mean...\n");
    int8_t* layer_168_out = run_mean_layer(LAYER_PARAMS[168], LAYER_IO_PATHS[168],
                                            layer_167_out, conv166_out_h, conv166_out_w, 1152, 1, 1);
    
    
    printf("Skip layer 169 shape 170 stride_slice 171 pack 172 reshape...\n");
    printf("Starting Layer 173 Conv_2D...\n");
    int conv173_out_h, conv173_out_w;
    int8_t* layer_173_out = run_conv2d_layer(LAYER_PARAMS[173], LAYER_IO_PATHS[173],
                                            layer_168_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv173_out_h, &conv173_out_w);
    printf("Starting Layer 174 HardSwish...\n");
    int layer_174_size = conv173_out_h * conv173_out_w * 48;
    int8_t* layer_174_out = run_hardswish_layer(LAYER_PARAMS[174], LAYER_IO_PATHS[174],
                                                layer_173_out, &layer_174_size);
    printf("Starting layer 175 Conv_2D...\n");
    int conv175_out_h, conv175_out_w;
    int8_t* layer_175_out = run_conv2d_layer(LAYER_PARAMS[175], LAYER_IO_PATHS[175],
                                            layer_174_out, conv173_out_h, conv173_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv175_out_h, &conv175_out_w);
    printf("Starting Layer 176 MUL...\n");
    int8_t* layer_176_out = run_mul_layer(LAYER_PARAMS[176], LAYER_IO_PATHS[176],
                                         layer_175_out, 1152, NULL, 1, 0);
    printf("Starting Layer 177 MUL...\n");
    int8_t* layer_177_out = run_mul_layer(LAYER_PARAMS[177], LAYER_IO_PATHS[177],
                                         NULL, 7*7*1152, layer_176_out, 1152, 1);
    printf("Starting layer 178 Conv_2D...\n");
    int conv178_out_h, conv178_out_w;
    int8_t* layer_178_out = run_conv2d_layer(LAYER_PARAMS[178], LAYER_IO_PATHS[178],
                                            layer_177_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv178_out_h, &conv178_out_w);
    printf("Starting Layer 179 ADD...\n");
    int8_t* layer_179_out = run_add_layer(LAYER_PARAMS[179], LAYER_IO_PATHS[179],
                                        layer_178_out, conv178_out_h * conv178_out_w * 192, layer_163_out, conv178_out_h * conv178_out_w * 192);
    printf("Output of layer 179 ADD:\n");
    for(int i = 0; i < conv178_out_h * conv178_out_w * 192; ++i) {
        printf("%d ", layer_179_out[i]);
        if ((i + 1) % 192 == 0) printf("\n");
    }
    printf("Starting Layer 180 Conv_2D...\n");
    int conv180_out_h, conv180_out_w;
    int8_t* layer_180_out = run_conv2d_layer(LAYER_PARAMS[180], LAYER_IO_PATHS[180],
                                            layer_179_out, conv178_out_h, conv178_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv180_out_h, &conv180_out_w);
    printf("Starting Layer 181 HardSwish...\n");
    int layer_181_size = conv180_out_h * conv180_out_w * 1152;
    int8_t* layer_181_out = run_hardswish_layer(LAYER_PARAMS[181], LAYER_IO_PATHS[181],
                                                layer_180_out, &layer_181_size);
    printf("Starting Layer 182 DepthwiseConv2D...\n");
    int conv182_out_h, conv182_out_w;
    int8_t* layer_182_out = run_dw_conv_layer(LAYER_PARAMS[182],
                                            layer_181_out, conv180_out_h, conv180_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv182_out_h, &conv182_out_w);
    printf("Starting Layer 183 HardSwish...\n");
    int layer_183_size = conv182_out_h * conv182_out_w * 1152;
    int8_t* layer_183_out = run_hardswish_layer(LAYER_PARAMS[183], LAYER_IO_PATHS[183],
                                                layer_182_out, &layer_183_size);
    printf("Starting Layer 184 Mean...\n");
    int8_t* layer_184_out = run_mean_layer(LAYER_PARAMS[184], LAYER_IO_PATHS[184],
                                            layer_183_out, conv182_out_h, conv182_out_w, 1152, 1, 1);
    printf("Skip layer 185 shape 186 stride_slice 187 pack 188 reshape...\n");
    printf("Starting Layer 189 Conv_2D...\n");
    int conv189_out_h, conv189_out_w;
    int8_t* layer_189_out = run_conv2d_layer(LAYER_PARAMS[189], LAYER_IO_PATHS[189],
                                            layer_184_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv189_out_h, &conv189_out_w);
    printf("Starting Layer 190 HardSwish...\n");
    int layer_190_size = conv189_out_h * conv189_out_w * 48;
    int8_t* layer_190_out = run_hardswish_layer(LAYER_PARAMS[190], LAYER_IO_PATHS[190],
                                                layer_189_out, &layer_190_size);
    printf("Starting Layer 191 Conv_2D...\n");
    int conv191_out_h, conv191_out_w;
    int8_t* layer_191_out = run_conv2d_layer(LAYER_PARAMS[191], LAYER_IO_PATHS[191],
                                            layer_190_out, conv189_out_h, conv189_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv191_out_h, &conv191_out_w);
    printf("Starting Layer 192 MUL...\n");
    int8_t* layer_192_out = run_mul_layer(LAYER_PARAMS[192], LAYER_IO_PATHS[192],
                                         layer_191_out, 1152, NULL, 1, 0);
    printf("Starting Layer 193 MUL...\n");
    int8_t* layer_193_out = run_mul_layer(LAYER_PARAMS[193], LAYER_IO_PATHS[193],
                                         layer_183_out, 7*7*1152, layer_192_out, 1152, 1);
    printf("Starting Layer 194 Conv_2D...\n");
    int conv194_out_h, conv194_out_w;
    int8_t* layer_194_out = run_conv2d_layer(LAYER_PARAMS[194], LAYER_IO_PATHS[194],
                                            layer_193_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv194_out_h, &conv194_out_w);
    printf("Starting Layer 195 ADD...\n");
    int8_t* layer_195_out = run_add_layer(LAYER_PARAMS[195], LAYER_IO_PATHS[195],
                                        layer_194_out, conv194_out_h * conv194_out_w * 192, layer_179_out, conv194_out_h * conv194_out_w * 192);
    printf("Starting Layer 196 Conv_2D...\n");
    int conv196_out_h, conv196_out_w;
    int8_t* layer_196_out = run_conv2d_layer(LAYER_PARAMS[196], LAYER_IO_PATHS[196],
                                            layer_195_out, conv194_out_h, conv194_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv196_out_h, &conv196_out_w);
    printf("Starting Layer 197 HardSwish...\n");
    int layer_197_size = conv196_out_h * conv196_out_w * 1152;
    int8_t* layer_197_out = run_hardswish_layer(LAYER_PARAMS[197], LAYER_IO_PATHS[197],
                                                layer_196_out, &layer_197_size);
    printf("Starting Layer 198 DepthwiseConv2D...\n");
    int conv198_out_h, conv198_out_w;
    int8_t* layer_198_out = run_dw_conv_layer(LAYER_PARAMS[198],
                                            layer_197_out, conv196_out_h, conv196_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv198_out_h, &conv198_out_w);
    printf("Starting Layer 199 HardSwish...\n");
    int layer_199_size = conv198_out_h * conv198_out_w * 1152;
    int8_t* layer_199_out = run_hardswish_layer(LAYER_PARAMS[199], LAYER_IO_PATHS[199],
                                                layer_198_out, &layer_199_size);
    printf("Starting Layer 200 Mean...\n");
    int8_t* layer_200_out = run_mean_layer(LAYER_PARAMS[200], LAYER_IO_PATHS[200],
                                            layer_199_out, conv198_out_h, conv198_out_w, 1152, 1, 1);
    printf("Skip layer 201 shape 202 stride_slice 203 pack 204 reshape...\n");
    printf("Starting Layer 205 Conv_2D...\n");
    int conv205_out_h, conv205_out_w;
    int8_t* layer_205_out = run_conv2d_layer(LAYER_PARAMS[205], LAYER_IO_PATHS[205],
                                            layer_200_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv205_out_h, &conv205_out_w);
    printf("Starting Layer 206 HardSwish...\n");
    int layer_206_size = conv205_out_h * conv205_out_w * 48;
    int8_t* layer_206_out = run_hardswish_layer(LAYER_PARAMS[206], LAYER_IO_PATHS[206],
                                                layer_205_out, &layer_206_size);
    printf("Starting Layer 207 Conv_2D...\n");
    int conv207_out_h, conv207_out_w;
    int8_t* layer_207_out = run_conv2d_layer(LAYER_PARAMS[207], LAYER_IO_PATHS[207],
                                            layer_206_out, conv205_out_h, conv205_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv207_out_h, &conv207_out_w);
    printf("Starting Layer 208 MUL...\n");
    int8_t* layer_208_out = run_mul_layer(LAYER_PARAMS[208], LAYER_IO_PATHS[208],
                                         layer_207_out, 1152, NULL, 1, 0);
    printf("Starting Layer 209 MUL...\n");
    int8_t* layer_209_out = run_mul_layer(LAYER_PARAMS[209], LAYER_IO_PATHS[209],
                                         layer_199_out, 7*7*1152, layer_208_out, 1152, 1);
    printf("Starting Layer 210 Conv_2D...\n");
    int conv210_out_h, conv210_out_w;
    int8_t* layer_210_out = run_conv2d_layer(LAYER_PARAMS[210], LAYER_IO_PATHS[210],
                                            layer_209_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv210_out_h, &conv210_out_w);
    printf("Starting Layer 211 ADD...\n");
    int8_t* layer_211_out = run_add_layer(LAYER_PARAMS[211], LAYER_IO_PATHS[211],
                                        layer_210_out, conv210_out_h * conv210_out_w * 192, layer_195_out, conv210_out_h * conv210_out_w * 192);
    printf("Starting Layer 212 Conv_2D...\n");
    int conv212_out_h, conv212_out_w;
    int8_t* layer_212_out = run_conv2d_layer(LAYER_PARAMS[212], LAYER_IO_PATHS[212],
                                            layer_211_out, conv210_out_h, conv210_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv212_out_h, &conv212_out_w);
    printf("Output of layer 212 Conv_2D:\n");
    for(int i = 0; i < conv212_out_h * conv212_out_w * 1152; ++i) {
        printf("%d ", layer_212_out[i]);
        if ((i + 1) % 1152 == 0) printf("\n");
    }
    printf("Starting Layer 213 HardSwish...\n");
    int layer_213_size = conv212_out_h * conv212_out_w * 1152;
    int8_t* layer_213_out = run_hardswish_layer(LAYER_PARAMS[213], LAYER_IO_PATHS[213],
                                                layer_212_out, &layer_213_size);
    printf("Starting Layer 214 DepthwiseConv2D...\n");
    int conv214_out_h, conv214_out_w;
    int8_t* layer_214_out = run_dw_conv_layer(LAYER_PARAMS[214],
                                            layer_213_out, conv212_out_h, conv212_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv214_out_h, &conv214_out_w);
    printf("Starting Layer 215 HardSwish...\n");
    int layer_215_size = conv214_out_h * conv214_out_w * 1152;
    int8_t* layer_215_out = run_hardswish_layer(LAYER_PARAMS[215], LAYER_IO_PATHS[215],
                                                layer_214_out, &layer_215_size);
    printf("Output of layer 215 HardSwish:\n");
    for(int i = 0; i < conv214_out_h * conv214_out_w * 1152; ++i) {
        printf("%d ", layer_215_out[i]);
        if ((i + 1) % 1152 == 0) printf("\n");
    }
    printf("Starting Layer 216 Mean...\n");
    int8_t* layer_216_out = run_mean_layer(LAYER_PARAMS[216], LAYER_IO_PATHS[216],
                                            layer_215_out, conv214_out_h, conv214_out_w, 1152, 1, 1);
    printf("Skip layer 217 shape 218 stride_slice 219 pack 220 reshape...\n");
    printf("Starting Layer 221 Conv_2D...\n");
    int conv221_out_h, conv221_out_w;
    int8_t* layer_221_out = run_conv2d_layer(LAYER_PARAMS[221], LAYER_IO_PATHS[221],
                                            layer_216_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv221_out_h, &conv221_out_w);
    printf("Starting Layer 222 HardSwish...\n");
    int layer_222_size = conv221_out_h * conv221_out_w * 48;
    int8_t* layer_222_out = run_hardswish_layer(LAYER_PARAMS[222], LAYER_IO_PATHS[222],
                                                layer_221_out, &layer_222_size);
    printf("Starting Layer 223 Conv_2D...\n");
    int conv223_out_h, conv223_out_w;
    int8_t* layer_223_out = run_conv2d_layer(LAYER_PARAMS[223], LAYER_IO_PATHS[223],
                                            layer_222_out, conv221_out_h, conv221_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv223_out_h, &conv223_out_w);
    printf("Starting Layer 224 MUL...\n");
    int8_t* layer_224_out = run_mul_layer(LAYER_PARAMS[224], LAYER_IO_PATHS[224],
                                         layer_223_out, 1152, NULL, 1, 0);
    printf("Starting Layer 225 MUL...\n");
    int8_t* layer_225_out = run_mul_layer(LAYER_PARAMS[225], LAYER_IO_PATHS[225],
                                         layer_215_out, 7*7*1152, layer_224_out, 1152, 1);
    printf("Output of layer 225 MUL:\n");
    for(int i = 0; i < conv214_out_h * conv214_out_w * 1152; ++i) {
        printf("%d ", layer_225_out[i]);
        if ((i + 1) % 1152 == 0) printf("\n");
    }
    printf("Starting Layer 226 Conv_2D...\n");
    int conv226_out_h, conv226_out_w;
    int8_t* layer_226_out = run_conv2d_layer(LAYER_PARAMS[226], LAYER_IO_PATHS[226],
                                            layer_225_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv226_out_h, &conv226_out_w);
    printf("Starting Layer 227 ADD...\n");
    int8_t* layer_227_out = run_add_layer(LAYER_PARAMS[227], LAYER_IO_PATHS[227],
                                        layer_226_out, conv226_out_h * conv226_out_w * 192, layer_211_out, conv226_out_h * conv226_out_w * 192);
    printf("Starting Layer 228 Conv_2D...\n");
    int conv228_out_h, conv228_out_w;
    int8_t* layer_228_out = run_conv2d_layer(LAYER_PARAMS[228], LAYER_IO_PATHS[228],
                                            layer_227_out, conv226_out_h, conv226_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv228_out_h, &conv228_out_w);
    printf("Starting Layer 229 HardSwish...\n");
    int layer_229_size = conv228_out_h * conv228_out_w * 1152;
    int8_t* layer_229_out = run_hardswish_layer(LAYER_PARAMS[229], LAYER_IO_PATHS[229],
                                                layer_228_out, &layer_229_size);
    printf("Starting Layer 230 DepthwiseConv2D...\n");
    int conv230_out_h, conv230_out_w;
    int8_t* layer_230_out = run_dw_conv_layer(LAYER_PARAMS[230],
                                            layer_229_out, conv228_out_h, conv228_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv230_out_h, &conv230_out_w);
    printf("Starting Layer 231 HardSwish...\n");
    int layer_231_size = conv230_out_h * conv230_out_w * 1152;
    int8_t* layer_231_out = run_hardswish_layer(LAYER_PARAMS[231], LAYER_IO_PATHS[231],
                                                layer_230_out, &layer_231_size);
    printf("Starting Layer 232 Mean...\n");
    int8_t* layer_232_out = run_mean_layer(LAYER_PARAMS[232], LAYER_IO_PATHS[232],
                                            layer_231_out, conv230_out_h, conv230_out_w, 1152, 1, 1);
    printf("Skip layer 233 shape 234 stride_slice 235 pack 236 reshape...\n");
    printf("Starting Layer 237 Conv_2D...\n");
    int conv237_out_h, conv237_out_w;
    int8_t* layer_237_out = run_conv2d_layer(LAYER_PARAMS[237], LAYER_IO_PATHS[237],
                                            layer_232_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv237_out_h, &conv237_out_w);
    printf("Starting Layer 238 HardSwish...\n");
    int layer_238_size = conv237_out_h * conv237_out_w * 48;
    int8_t* layer_238_out = run_hardswish_layer(LAYER_PARAMS[238], LAYER_IO_PATHS[238],
                                                layer_237_out, &layer_238_size);
    printf("Starting Layer 239 Conv_2D...\n");
    int conv239_out_h, conv239_out_w;
    int8_t* layer_239_out = run_conv2d_layer(LAYER_PARAMS[239], LAYER_IO_PATHS[239],
                                            layer_238_out, conv237_out_h, conv237_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv239_out_h, &conv239_out_w);
    printf("Starting Layer 240 MUL...\n");
    int8_t* layer_240_out = run_mul_layer(LAYER_PARAMS[240], LAYER_IO_PATHS[240],
                                         layer_239_out, 1152, NULL, 1, 0);
    printf("Starting Layer 241 MUL...\n");
    int8_t* layer_241_out = run_mul_layer(LAYER_PARAMS[241], LAYER_IO_PATHS[241],
                                         layer_231_out, 7*7*1152, layer_240_out, 1152, 1);
    printf("Starting Layer 242 Conv_2D...\n");
    int conv242_out_h, conv242_out_w;
    int8_t* layer_242_out = run_conv2d_layer(LAYER_PARAMS[242], LAYER_IO_PATHS[242],
                                            layer_241_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv242_out_h, &conv242_out_w);
    printf("Starting Layer 243 ADD...\n");
    int8_t* layer_243_out = run_add_layer(LAYER_PARAMS[243], LAYER_IO_PATHS[243],
                                        layer_242_out, conv242_out_h * conv242_out_w * 192, layer_227_out, conv242_out_h * conv242_out_w * 192);
    printf("Starting Layer 244 Conv_2D...\n");
    int conv244_out_h, conv244_out_w;
    int8_t* layer_244_out = run_conv2d_layer(LAYER_PARAMS[244], LAYER_IO_PATHS[244],
                                            layer_243_out, conv242_out_h, conv242_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv244_out_h, &conv244_out_w);
    printf("Starting Layer 245 HardSwish...\n");
    int layer_245_size = conv244_out_h * conv244_out_w * 1152;
    int8_t* layer_245_out = run_hardswish_layer(LAYER_PARAMS[245], LAYER_IO_PATHS[245],
                                                layer_244_out, &layer_245_size);
    printf("Starting Layer 246 DepthwiseConv2D...\n");
    int conv246_out_h, conv246_out_w;
    int8_t* layer_246_out = run_dw_conv_layer(LAYER_PARAMS[246],
                                            layer_245_out, conv244_out_h, conv244_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv246_out_h, &conv246_out_w);
    printf("Starting Layer 247 HardSwish...\n");
    int layer_247_size = conv246_out_h * conv246_out_w * 1152;
    int8_t* layer_247_out = run_hardswish_layer(LAYER_PARAMS[247], LAYER_IO_PATHS[247],
                                                layer_246_out, &layer_247_size);
    printf("Starting Layer 248 Mean...\n");
    int8_t* layer_248_out = run_mean_layer(LAYER_PARAMS[248], LAYER_IO_PATHS[248],
                                            layer_247_out, conv246_out_h, conv246_out_w, 1152, 1, 1);
    printf("Skip layer 249 shape 250 stride_slice 251 pack 252 reshape...\n");
    printf("Starting Layer 253 Conv_2D...\n");
    int conv253_out_h, conv253_out_w;
    int8_t* layer_253_out = run_conv2d_layer(LAYER_PARAMS[253], LAYER_IO_PATHS[253],
                                            layer_248_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv253_out_h, &conv253_out_w);
    printf("Starting Layer 254 HardSwish...\n");
    int layer_254_size = conv253_out_h * conv253_out_w * 48;
    int8_t* layer_254_out = run_hardswish_layer(LAYER_PARAMS[254], LAYER_IO_PATHS[254],
                                                layer_253_out, &layer_254_size);
    printf("Starting Layer 255 Conv_2D...\n");
    int conv255_out_h, conv255_out_w;
    int8_t* layer_255_out = run_conv2d_layer(LAYER_PARAMS[255], LAYER_IO_PATHS[255],
                                            layer_254_out, conv253_out_h, conv253_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv255_out_h, &conv255_out_w);
    printf("Starting Layer 256 MUL...\n");
    int8_t* layer_256_out = run_mul_layer(LAYER_PARAMS[256], LAYER_IO_PATHS[256],
                                         layer_255_out, 1152, NULL, 1, 0);
    printf("Starting Layer 257 MUL...\n");
    int8_t* layer_257_out = run_mul_layer(LAYER_PARAMS[257], LAYER_IO_PATHS[257],
                                         layer_247_out, 7*7*1152, layer_256_out, 1152, 1);
    printf("Starting layer 258 Conv_2D...\n");
    int conv258_out_h, conv258_out_w;
    int8_t* layer_258_out = run_conv2d_layer(LAYER_PARAMS[258], LAYER_IO_PATHS[258],
                                            layer_257_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv258_out_h, &conv258_out_w);
    printf("Starting layer 259 ADD...\n");
    int8_t* layer_259_out = run_add_layer(LAYER_PARAMS[259], LAYER_IO_PATHS[259],
                                        layer_258_out, conv258_out_h * conv258_out_w * 192, layer_243_out, conv258_out_h * conv258_out_w * 192);
    printf("Starting layer 260 Conv_2D...\n");
    int conv260_out_h, conv260_out_w;
    int8_t* layer_260_out = run_conv2d_layer(LAYER_PARAMS[260], LAYER_IO_PATHS[260],
                                            layer_259_out, conv258_out_h, conv258_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv260_out_h, &conv260_out_w);
    printf("Starting layer 261 HardSwish...\n");
    int layer_261_size = conv260_out_h * conv260_out_w * 1152;
    int8_t* layer_261_out = run_hardswish_layer(LAYER_PARAMS[261], LAYER_IO_PATHS[261],
                                                layer_260_out, &layer_261_size);
    printf("Starting layer 262 DepthwiseConv2D...\n");
    int conv262_out_h, conv262_out_w;
    int8_t* layer_262_out = run_dw_conv_layer(LAYER_PARAMS[262],
                                            layer_261_out, conv260_out_h, conv260_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv262_out_h, &conv262_out_w);
    printf("Starting layer 263 HardSwish...\n");
    int layer_263_size = conv262_out_h * conv262_out_w * 1152;
    int8_t* layer_263_out = run_hardswish_layer(LAYER_PARAMS[263], LAYER_IO_PATHS[263],
                                                layer_262_out, &layer_263_size);
    printf("Starting layer 264 Mean...\n");
    int8_t* layer_264_out = run_mean_layer(LAYER_PARAMS[264], LAYER_IO_PATHS[264],
                                            layer_263_out, conv262_out_h, conv262_out_w, 1152, 1, 1);
    printf("Skip layer 265 shape 266 stride_slice 267 pack 268 reshape...\n");
    printf("Starting layer 269 Conv_2D...\n");
    int conv269_out_h, conv269_out_w;
    int8_t* layer_269_out = run_conv2d_layer(LAYER_PARAMS[269], LAYER_IO_PATHS[269],
                                            layer_264_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv269_out_h, &conv269_out_w);
    printf("Starting layer 270 HardSwish...\n");
    int layer_270_size = conv269_out_h * conv269_out_w * 48;
    int8_t* layer_270_out = run_hardswish_layer(LAYER_PARAMS[270], LAYER_IO_PATHS[270],
                                                layer_269_out, &layer_270_size);
    printf("Starting layer 271 Conv_2D...\n");
    int conv271_out_h, conv271_out_w;
    int8_t* layer_271_out = run_conv2d_layer(LAYER_PARAMS[271], LAYER_IO_PATHS[271],
                                            layer_270_out, conv269_out_h, conv269_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv271_out_h, &conv271_out_w);
    printf("Starting layer 272 MUL...\n");
    int8_t* layer_272_out = run_mul_layer(LAYER_PARAMS[272], LAYER_IO_PATHS[272],
                                         layer_271_out, 1152, NULL, 1, 0);
    printf("Starting layer 273 MUL...\n");
    int8_t* layer_273_out = run_mul_layer(LAYER_PARAMS[273], LAYER_IO_PATHS[273],
                                         layer_263_out, 7*7*1152, layer_272_out, 1152, 1);
    printf("Starting layer 274 Conv_2D...\n");
    int conv274_out_h, conv274_out_w;
    int8_t* layer_274_out = run_conv2d_layer(LAYER_PARAMS[274], LAYER_IO_PATHS[274],
                                            layer_273_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv274_out_h, &conv274_out_w);
    printf("Starting layer 275 ADD...\n");
    int8_t* layer_275_out = run_add_layer(LAYER_PARAMS[275], LAYER_IO_PATHS[275],
                                        layer_274_out, conv274_out_h * conv274_out_w * 192, layer_259_out, conv274_out_h * conv274_out_w * 192);
    printf("Starting layer 276 Conv_2D...\n");
    int conv276_out_h, conv276_out_w;
    int8_t* layer_276_out = run_conv2d_layer(LAYER_PARAMS[276], LAYER_IO_PATHS[276],
                                            layer_275_out, conv274_out_h, conv274_out_w, 192, 1280, 1, 1, 1, 1, "SAME", &conv276_out_h, &conv276_out_w);
    printf("Starting layer 277 HardSwish...\n");
    int layer_277_size = conv276_out_h * conv276_out_w * 1280;
    int8_t* layer_277_out = run_hardswish_layer(LAYER_PARAMS[277], LAYER_IO_PATHS[277],
                                                layer_276_out, &layer_277_size);
    printf("Starting layer 278 Mean...\n");
    int8_t* layer_278_out = run_mean_layer(LAYER_PARAMS[278], LAYER_IO_PATHS[278],
                                            layer_277_out, conv276_out_h, conv276_out_w, 1280, 1, 1);
    printf("Output of layer 278 Mean:\n");
    for(int i = 0; i < 1280; ++i) {
        printf("%d ", layer_278_out[i]);
    }
    printf("\n");

    return 0;
}
