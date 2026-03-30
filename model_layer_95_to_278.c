#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "layer/layer_helper.h"

int main(){
    printf("Starting Layer 95 HardSwish...\n");
    int layer_95_size;
    int8_t* layer_95_out = run_hardswish_layer("extracted_params_hsigmoid/layer095_HARD_SWISH_1_block5b_se_reduce_1_truediv__1_block5b_se_reduce_1_Relu6_1", "all_layer_io/layer_95_HARD_SWISH", NULL, &layer_95_size);

    printf("Starting Layer 96 CONV_2D...\n");
    int conv_out_h, conv_out_w;
    int8_t* layer_96_out = run_conv2d_layer("extracted_params_hsigmoid/layer096_CONV_2D_1_block5b_se_expand_1_Relu6_1_block5b_se_expand_1_add_1_block5b", "all_layer_io/layer_96_CONV_2D",
                                            layer_95_out, 1, 1, 28, 672, 1, 1, 1, 1, "SAME", &conv_out_h, &conv_out_w);

    printf("Starting Layer 97 MUL...\n");
    int8_t* layer_97_out = run_mul_layer("extracted_params_hsigmoid/layer097_MUL_1_block5b_se_expand_1_truediv", "all_layer_io/layer_97_MUL",
                                         layer_96_out, 672, NULL, 1, 0);

    printf("Starting Layer 98 MUL...\n");
    int8_t* layer_98_out = run_mul_layer("extracted_params_hsigmoid/layer098_MUL_1_block5b_se_excite_1_Mul", "all_layer_io/layer_98_MUL",
                                         NULL, 14*14*672, layer_97_out, 672, 1);
    printf("Starting Layer 99 CONV_2D...\n");
    int conv99_out_h, conv99_out_w;
    int8_t* layer_99_out = run_conv2d_layer("extracted_params_hsigmoid/layer099_CONV_2D_1_block5b_project_conv_1_BiasAdd_1_block5b_project_conv_1_convo", "all_layer_io/layer_99_CONV_2D",
                                            layer_98_out, 14, 14, 672, 112, 1, 1, 1, 1, "SAME", &conv99_out_h, &conv99_out_w);
    printf("Starting Layer 100 ADD...\n");
    int8_t* layer_100_out = run_add_layer("extracted_params_hsigmoid/layer100_ADD_1_block5b_add_1_Add", "all_layer_io/layer_100_ADD",
                                        layer_99_out, conv99_out_h * conv99_out_w * 112, NULL, conv99_out_h * conv99_out_w * 112);
    
    printf("Starting Layer 101 Conv_2D...\n");
    int conv101_out_h, conv101_out_w;
    int8_t* layer_101_out = run_conv2d_layer("extracted_params_hsigmoid/layer101_CONV_2D_1_block5c_expand_conv_1_BiasAdd_1_block5c_expand_conv_1_convolu", "all_layer_io/layer_101_CONV_2D",
                                            layer_100_out, conv99_out_h, conv99_out_w, 112, 672, 1, 1, 1, 1, "SAME", &conv101_out_h, &conv101_out_w);
    printf("Starting Layer 102 HardSwish...\n");
    int layer_102_size = conv101_out_h * conv101_out_w * 672;
    int8_t* layer_102_out = run_hardswish_layer("extracted_params_hsigmoid/layer102_HARD_SWISH_1_block5c_expand_activation_1_truediv__1_block5c_expand_acti", "all_layer_io/layer_102_HARD_SWISH",
                                                layer_101_out, &layer_102_size);
    printf("Starting Layer 103 DepthwiseConv2D...\n");
    int conv103_out_h, conv103_out_w;
    int8_t* layer_103_out = run_dw_conv_layer("extracted_params_hsigmoid/layer103_DEPTHWISE_CONV_2D_1_block5c_dwconv2_1_BiasAdd_1_block5c_dwconv2_1_depth",
                                            layer_102_out, conv101_out_h, conv101_out_w, 672, 672, 3, 3, 1, 1, "SAME", &conv103_out_h, &conv103_out_w);
    printf("Starting Layer 104 HardSwish...\n");
    int layer_104_size = conv103_out_h * conv103_out_w * 672;
    int8_t* layer_104_out = run_hardswish_layer("extracted_params_hsigmoid/layer104_HARD_SWISH_1_block5c_activation_1_truediv__1_block5c_activation_1_Relu6", "all_layer_io/layer_104_HARD_SWISH",
                                                layer_103_out, &layer_104_size);
    printf("Starting Layer 105 Mean...\n");
    int8_t* layer_105_out = run_mean_layer("extracted_params_hsigmoid/layer105_MEAN_1_block5c_se_squeeze_1_Mean", "all_layer_io/layer_105_MEAN",
                                            layer_104_out, conv103_out_h, conv103_out_w, 672, 1, 0);
    printf("Skip layer 106 shape 107 stride_slice 108 pack 109 reshape...\n");
    printf("Starting Layer 110 Conv_2D...\n");
    int conv110_out_h, conv110_out_w;
    int8_t* layer_110_out = run_conv2d_layer("extracted_params_hsigmoid/layer110_CONV_2D_1_block5c_se_reduce_1_BiasAdd_1_block5c_se_reduce_1_convolution", "all_layer_io/layer_110_CONV_2D",
                                            layer_105_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv110_out_h, &conv110_out_w);
    printf("Starting Layer 111 HardSwish...\n");
    int layer_111_size = conv110_out_h * conv110_out_w * 28;
    int8_t* layer_111_out = run_hardswish_layer("extracted_params_hsigmoid/layer111_HARD_SWISH_1_block5c_se_reduce_1_truediv__1_block5c_se_reduce_1_Relu6_1", "all_layer_io/layer_111_HARD_SWISH",
                                                layer_110_out, &layer_111_size);
    printf("Starting Layer 112 Conv_2D...\n");
    int conv112_out_h, conv112_out_w;
    int8_t* layer_112_out = run_conv2d_layer("extracted_params_hsigmoid/layer112_CONV_2D_1_block5c_se_expand_1_Relu6_1_block5c_se_expand_1_add_1_block5c", "all_layer_io/layer_112_CONV_2D",
                                            layer_111_out, conv110_out_h, conv110_out_w, 28, 672, 1, 1, 1, 1, "SAME", &conv112_out_h, &conv112_out_w);
    printf("Starting Layer 113 MUL...\n");
    int8_t* layer_113_out = run_mul_layer("extracted_params_hsigmoid/layer113_MUL_1_block5c_se_expand_1_truediv", "all_layer_io/layer_113_MUL",
                                         layer_112_out, 672, NULL, 1, 0);
    printf("Starting Layer 114 MUL...\n");
    int8_t* layer_114_out = run_mul_layer("extracted_params_hsigmoid/layer114_MUL_1_block5c_se_excite_1_Mul", "all_layer_io/layer_114_MUL",
                                         NULL, 14*14*672, layer_113_out, 672, 1);
    printf("Starting Layer 115 CONV_2D...\n");
    int conv115_out_h, conv115_out_w;
    int8_t* layer_115_out = run_conv2d_layer("extracted_params_hsigmoid/layer115_CONV_2D_1_block5c_project_conv_1_BiasAdd_1_block5c_project_conv_1_convo", "all_layer_io/layer_115_CONV_2D",
                                            layer_114_out, 14, 14, 672, 112, 1, 1, 1, 1, "SAME", &conv115_out_h, &conv115_out_w);
    printf("Starting Layer 116 ADD...\n");
    int8_t* layer_116_out = run_add_layer("extracted_params_hsigmoid/layer116_ADD_1_block5c_add_1_Add", "all_layer_io/layer_116_ADD",
                                        layer_115_out, conv115_out_h * conv115_out_w * 112, layer_100_out, conv115_out_h * conv115_out_w * 112);
    
    printf("Starting Layer 117 Conv_2D...\n");
    int conv117_out_h, conv117_out_w;
    int8_t* layer_117_out = run_conv2d_layer("extracted_params_hsigmoid/layer117_CONV_2D_1_block5d_expand_conv_1_BiasAdd_1_block5d_expand_conv_1_convolu", "all_layer_io/layer_117_CONV_2D",
                                            layer_116_out, conv115_out_h, conv115_out_w, 112, 672, 1, 1, 1, 1, "SAME", &conv117_out_h, &conv117_out_w);
    printf("Starting Layer 118 HardSwish...\n");
    int layer_118_size = conv117_out_h * conv117_out_w * 672;
    int8_t* layer_118_out = run_hardswish_layer("extracted_params_hsigmoid/layer118_HARD_SWISH_1_block5d_expand_activation_1_truediv__1_block5d_expand_acti", "all_layer_io/layer_118_HARD_SWISH",
                                                layer_117_out, &layer_118_size);
    printf("Starting Layer 119 DepthwiseConv2D...\n");
    int conv119_out_h, conv119_out_w;
    int8_t* layer_119_out = run_dw_conv_layer("extracted_params_hsigmoid/layer119_DEPTHWISE_CONV_2D_1_block5d_dwconv2_1_BiasAdd_1_block5d_dwconv2_1_depth",
                                            layer_118_out, conv117_out_h, conv117_out_w, 672, 672, 3, 3, 1, 1, "SAME", &conv119_out_h, &conv119_out_w);
    printf("Starting Layer 120 HardSwish...\n");
    int layer_120_size = conv119_out_h * conv119_out_w * 672;
    int8_t* layer_120_out = run_hardswish_layer("extracted_params_hsigmoid/layer120_HARD_SWISH_1_block5d_activation_1_truediv__1_block5d_activation_1_Relu6", "all_layer_io/layer_120_HARD_SWISH",
                                                layer_119_out, &layer_120_size);
    printf("Starting Layer 121 Mean...\n");
    int8_t* layer_121_out = run_mean_layer("extracted_params_hsigmoid/layer121_MEAN_1_block5d_se_squeeze_1_Mean", "all_layer_io/layer_121_MEAN",
                                            layer_120_out, conv119_out_h, conv119_out_w, 672, 1, 0);
    printf("Skip layer 122 shape 123 stride_slice 124 pack 125 reshape...\n");
    printf("Starting Layer 126 Conv_2D...\n");
    int conv126_out_h, conv126_out_w;
    int8_t* layer_126_out = run_conv2d_layer("extracted_params_hsigmoid/layer126_CONV_2D_1_block5d_se_reduce_1_BiasAdd_1_block5d_se_reduce_1_convolution", "all_layer_io/layer_126_CONV_2D",
                                            layer_121_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv126_out_h, &conv126_out_w);
    printf("Starting Layer 127 HardSwish...\n");
    int layer_127_size = conv126_out_h * conv126_out_w * 28;
    int8_t* layer_127_out = run_hardswish_layer("extracted_params_hsigmoid/layer127_HARD_SWISH_1_block5d_se_reduce_1_truediv__1_block5d_se_reduce_1_Relu6_1", "all_layer_io/layer_127_HARD_SWISH",
                                                layer_126_out, &layer_127_size);
    printf("Starting Layer 128 Conv_2D...\n");
    int conv128_out_h, conv128_out_w;
    int8_t* layer_128_out = run_conv2d_layer("extracted_params_hsigmoid/layer128_CONV_2D_1_block5d_se_expand_1_Relu6_1_block5d_se_expand_1_add_1_block5d", "all_layer_io/layer_128_CONV_2D",
                                            layer_127_out, conv126_out_h, conv126_out_w, 28, 672, 1, 1, 1, 1, "SAME", &conv128_out_h, &conv128_out_w);
    printf("Starting Layer 129 MUL...\n");
    int8_t* layer_129_out = run_mul_layer("extracted_params_hsigmoid/layer129_MUL_1_block5d_se_expand_1_truediv", "all_layer_io/layer_129_MUL",
                                         layer_128_out, 672, NULL, 1, 0);
    printf("Starting Layer 130 MUL...\n");
    int8_t* layer_130_out = run_mul_layer("extracted_params_hsigmoid/layer130_MUL_1_block5d_se_excite_1_Mul", "all_layer_io/layer_130_MUL",
                                         NULL, 14*14*672, layer_129_out, 672, 1);
    printf("Starting Layer 131 CONV_2D...\n");
    int conv131_out_h, conv131_out_w;
    int8_t* layer_131_out = run_conv2d_layer("extracted_params_hsigmoid/layer131_CONV_2D_1_block5d_project_conv_1_BiasAdd_1_block5d_project_conv_1_convo", "all_layer_io/layer_131_CONV_2D",
                                            layer_130_out, 14, 14, 672, 112, 1, 1, 1, 1, "SAME", &conv131_out_h, &conv131_out_w);
    printf("Starting Layer 132 ADD...\n");
    int8_t* layer_132_out = run_add_layer("extracted_params_hsigmoid/layer132_ADD_1_block5d_add_1_Add", "all_layer_io/layer_132_ADD",
                                        layer_131_out, conv131_out_h * conv131_out_w * 112, layer_116_out, conv131_out_h * conv131_out_w * 112);
    printf("Starting Layer 133 Conv_2D...\n");
    int conv133_out_h, conv133_out_w;
    int8_t* layer_133_out = run_conv2d_layer("extracted_params_hsigmoid/layer133_CONV_2D_1_block5e_expand_conv_1_BiasAdd_1_block5e_expand_conv_1_convolu", "all_layer_io/layer_133_CONV_2D",
                                            layer_132_out, conv131_out_h, conv131_out_w, 112, 672, 1, 1, 1, 1, "SAME", &conv133_out_h, &conv133_out_w);
    printf("Starting Layer 134 HardSwish...\n");
    int layer_134_size = conv133_out_h * conv133_out_w * 672;
    int8_t* layer_134_out = run_hardswish_layer("extracted_params_hsigmoid/layer134_HARD_SWISH_1_block5e_expand_activation_1_truediv__1_block5e_expand_acti", "all_layer_io/layer_134_HARD_SWISH",
                                                layer_133_out, &layer_134_size);
    printf("Starting Layer 135 DepthwiseConv2D...\n");
    int conv135_out_h, conv135_out_w;
    int8_t* layer_135_out = run_dw_conv_layer("extracted_params_hsigmoid/layer135_DEPTHWISE_CONV_2D_1_block5e_dwconv2_1_BiasAdd_1_block5e_dwconv2_1_depth", 
                                            layer_134_out, conv133_out_h, conv133_out_w, 672, 672, 3, 3, 1, 1, "SAME", &conv135_out_h, &conv135_out_w);
    printf("Starting Layer 136 HardSwish...\n");
    int layer_136_size = conv135_out_h * conv135_out_w * 672;
    int8_t* layer_136_out = run_hardswish_layer("extracted_params_hsigmoid/layer136_HARD_SWISH_1_block5e_activation_1_truediv__1_block5e_activation_1_Relu6", "all_layer_io/layer_136_HARD_SWISH",
                                                layer_135_out, &layer_136_size);
    printf("Starting Layer 137 Mean...\n");
    int8_t* layer_137_out = run_mean_layer("extracted_params_hsigmoid/layer137_MEAN_1_block5e_se_squeeze_1_Mean", "all_layer_io/layer_137_MEAN",
                                            layer_136_out, conv135_out_h, conv135_out_w, 672, 1, 0);
    printf("Skip layer 138 shape 139 stride_slice 140 pack 141 reshape...\n");
    printf("Starting Layer 142 Conv_2D...\n");
    int conv142_out_h, conv142_out_w;
    int8_t* layer_142_out = run_conv2d_layer("extracted_params_hsigmoid/layer142_CONV_2D_1_block5e_se_reduce_1_BiasAdd_1_block5e_se_reduce_1_convolution", "all_layer_io/layer_142_CONV_2D",
                                            layer_137_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv142_out_h, &conv142_out_w);
    printf("Starting Layer 143 HardSwish...\n");
    int layer_143_size = conv142_out_h * conv142_out_w * 28;
    int8_t* layer_143_out = run_hardswish_layer("extracted_params_hsigmoid/layer143_HARD_SWISH_1_block5e_se_reduce_1_truediv__1_block5e_se_reduce_1_Relu6_1", "all_layer_io/layer_143_HARD_SWISH",
                                                layer_142_out, &layer_143_size);
    printf("Starting Layer 144 Conv_2D...\n");
    int conv144_out_h, conv144_out_w;
    int8_t* layer_144_out = run_conv2d_layer("extracted_params_hsigmoid/layer144_CONV_2D_1_block5e_se_expand_1_Relu6_1_block5e_se_expand_1_add_1_block5e", "all_layer_io/layer_144_CONV_2D",
                                            layer_143_out, conv142_out_h, conv142_out_w, 28, 672, 1, 1, 1, 1, "SAME", &conv144_out_h, &conv144_out_w);
    printf("Starting Layer 145 MUL...\n");
    int8_t* layer_145_out = run_mul_layer("extracted_params_hsigmoid/layer145_MUL_1_block5e_se_expand_1_truediv", "all_layer_io/layer_145_MUL",
                                         layer_144_out, 672, NULL, 1, 0);
    printf("Starting Layer 146 MUL...\n");
    int8_t* layer_146_out = run_mul_layer("extracted_params_hsigmoid/layer146_MUL_1_block5e_se_excite_1_Mul", "all_layer_io/layer_146_MUL",
                                         NULL, 14*14*672, layer_145_out, 672, 1);
    printf("Starting Layer 147 CONV_2D...\n");
    int conv147_out_h, conv147_out_w;
    int8_t* layer_147_out = run_conv2d_layer("extracted_params_hsigmoid/layer147_CONV_2D_1_block5e_project_conv_1_BiasAdd_1_block5e_project_conv_1_convo", "all_layer_io/layer_147_CONV_2D",
                                            layer_146_out, 14, 14, 672, 112, 1, 1, 1, 1, "SAME", &conv147_out_h, &conv147_out_w);
    printf("Starting Layer 148 ADD...\n");
    int8_t* layer_148_out = run_add_layer("extracted_params_hsigmoid/layer148_ADD_1_block5e_add_1_Add", "all_layer_io/layer_148_ADD",
                                        layer_147_out, conv147_out_h * conv147_out_w * 112, layer_132_out, conv147_out_h * conv147_out_w * 112);
    printf("Starting Layer 149 Conv_2D...\n");
    int conv149_out_h, conv149_out_w;
    int8_t* layer_149_out = run_conv2d_layer("extracted_params_hsigmoid/layer149_CONV_2D_1_block6a_expand_conv_1_BiasAdd_1_block6a_expand_conv_1_convolu", "all_layer_io/layer_149_CONV_2D",
                                            layer_148_out, conv147_out_h, conv147_out_w, 112, 672, 1, 1, 1, 1, "SAME", &conv149_out_h, &conv149_out_w);
    printf("Starting Layer 150 HardSwish...\n");
    int layer_150_size = conv149_out_h * conv149_out_w * 672;
    int8_t* layer_150_out = run_hardswish_layer("extracted_params_hsigmoid/layer150_HARD_SWISH_1_block6a_expand_activation_1_truediv__1_block6a_expand_acti", "all_layer_io/layer_150_HARD_SWISH",
                                                layer_149_out, &layer_150_size);
    printf("Starting Layer 151 DepthwiseConv2D...\n");
    int conv151_out_h, conv151_out_w;
    int8_t* layer_151_out = run_dw_conv_layer("extracted_params_hsigmoid/layer151_DEPTHWISE_CONV_2D_1_block6a_dwconv2_1_BiasAdd_1_block6a_dwconv2_1_depth",
                                            layer_150_out, conv149_out_h, conv149_out_w, 672, 672, 3, 3, 2, 2, "SAME", &conv151_out_h, &conv151_out_w);
    printf("Starting Layer 152 HardSwish...\n");
    int layer_152_size = conv151_out_h * conv151_out_w * 672;
    int8_t* layer_152_out = run_hardswish_layer("extracted_params_hsigmoid/layer152_HARD_SWISH_1_block6a_activation_1_truediv__1_block6a_activation_1_Relu6", "all_layer_io/layer_152_HARD_SWISH",
                                                layer_151_out, &layer_152_size);
    printf("Starting Layer 153 Mean...\n");
    int8_t* layer_153_out = run_mean_layer("extracted_params_hsigmoid/layer153_MEAN_1_block6a_se_squeeze_1_Mean", "all_layer_io/layer_153_MEAN",
                                            layer_152_out, conv151_out_h, conv151_out_w, 672, 1, 0);
    printf("Skip layer 154 shape 155 stride_slice 156 pack 157 reshape...\n");
    printf("Starting Layer 158 Conv_2D...\n");
    int conv158_out_h, conv158_out_w;
    int8_t* layer_158_out = run_conv2d_layer("extracted_params_hsigmoid/layer158_CONV_2D_1_block6a_se_reduce_1_BiasAdd_1_block6a_se_reduce_1_convolution", "all_layer_io/layer_158_CONV_2D",
                                            layer_153_out, 1, 1, 672, 28, 1, 1, 1, 1, "SAME", &conv158_out_h, &conv158_out_w);
    printf("Starting Layer 159 HardSwish...\n");
    int layer_159_size = conv158_out_h * conv158_out_w * 28;
    int8_t* layer_159_out = run_hardswish_layer("extracted_params_hsigmoid/layer159_HARD_SWISH_1_block6a_se_reduce_1_truediv__1_block6a_se_reduce_1_Relu6_1", "all_layer_io/layer_159_HARD_SWISH",
                                                layer_158_out, &layer_159_size);
    printf("Starting Layer 160 Conv_2D...\n");
    int conv160_out_h, conv160_out_w;
    int8_t* layer_160_out = run_conv2d_layer("extracted_params_hsigmoid/layer160_CONV_2D_1_block6a_se_expand_1_Relu6_1_block6a_se_expand_1_add_1_block6a", "all_layer_io/layer_160_CONV_2D",
                                            layer_159_out, conv158_out_h, conv158_out_w, 28, 672, 1, 1, 1, 1, "SAME", &conv160_out_h, &conv160_out_w);
    printf("Starting Layer 161 MUL...\n");
    int8_t* layer_161_out = run_mul_layer("extracted_params_hsigmoid/layer161_MUL_1_block6a_se_expand_1_truediv", "all_layer_io/layer_161_MUL",
                                         layer_160_out, 672, NULL, 1, 0);
    printf("Starting Layer 162 MUL...\n");
    int8_t* layer_162_out = run_mul_layer("extracted_params_hsigmoid/layer162_MUL_1_block6a_se_excite_1_Mul", "all_layer_io/layer_162_MUL",
                                         NULL, 7*7*672, layer_161_out, 672, 1);
    printf("Starting Layer 163 CONV_2D...\n");
    int conv163_out_h, conv163_out_w;
    int8_t* layer_163_out = run_conv2d_layer("extracted_params_hsigmoid/layer163_CONV_2D_1_block6a_project_conv_1_BiasAdd_1_block6a_project_conv_1_convo", "all_layer_io/layer_163_CONV_2D",
                                            layer_162_out, 7, 7, 672, 192, 1, 1, 1, 1, "SAME", &conv163_out_h, &conv163_out_w);
    printf("Starting Layer 164 CONV_2D...\n");
    int conv164_out_h, conv164_out_w;
    int8_t* layer_164_out = run_conv2d_layer("extracted_params_hsigmoid/layer164_CONV_2D_1_block6b_expand_conv_1_BiasAdd_1_block6b_expand_conv_1_convolu", "all_layer_io/layer_164_CONV_2D",
                                            layer_163_out, conv163_out_h, conv163_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv164_out_h, &conv164_out_w);
    printf("Starting Layer 165 HardSwish...\n");
    int layer_165_size = conv164_out_h * conv164_out_w * 1152;
    int8_t* layer_165_out = run_hardswish_layer("extracted_params_hsigmoid/layer165_HARD_SWISH_1_block6b_expand_activation_1_truediv__1_block6b_expand_acti", "all_layer_io/layer_165_HARD_SWISH",
                                                layer_164_out, &layer_165_size);
    printf("Starting Layer 166 DepthwiseConv2D...\n");
    int conv166_out_h, conv166_out_w;
    int8_t* layer_166_out = run_dw_conv_layer("extracted_params_hsigmoid/layer166_DEPTHWISE_CONV_2D_1_block6b_dwconv2_1_BiasAdd_1_block6b_dwconv2_1_depth",
                                            layer_165_out, conv164_out_h, conv164_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv166_out_h, &conv166_out_w);
    printf("Starting Layer 167 HardSwish...\n");
    int layer_167_size = conv166_out_h * conv166_out_w * 1152;
    int8_t* layer_167_out = run_hardswish_layer("extracted_params_hsigmoid/layer167_HARD_SWISH_1_block6b_activation_1_truediv__1_block6b_activation_1_Relu6", "all_layer_io/layer_167_HARD_SWISH",
                                                layer_166_out, &layer_167_size);
    // printf("Layer 167 output:\n");
    // for(int i = 0; i < conv166_out_h * conv166_out_w * 1152; ++i) {
    //     printf("%d ", layer_167_out[i]);
    //     if ((i + 1) % 1152 == 0) printf("\n");
    // }
    printf("Starting Layer 168 Mean...\n");
    int8_t* layer_168_out = run_mean_layer("extracted_params_hsigmoid/layer168_MEAN_1_block6b_se_squeeze_1_Mean", "all_layer_io/layer_168_MEAN",
                                            layer_167_out, conv166_out_h, conv166_out_w, 1152, 1, 1);
    
    
    printf("Skip layer 169 shape 170 stride_slice 171 pack 172 reshape...\n");
    printf("Starting Layer 173 Conv_2D...\n");
    int conv173_out_h, conv173_out_w;
    int8_t* layer_173_out = run_conv2d_layer("extracted_params_hsigmoid/layer173_CONV_2D_1_block6b_se_reduce_1_BiasAdd_1_block6b_se_reduce_1_convolution", "all_layer_io/layer_173_CONV_2D",
                                            layer_168_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv173_out_h, &conv173_out_w);
    printf("Starting Layer 174 HardSwish...\n");
    int layer_174_size = conv173_out_h * conv173_out_w * 48;
    int8_t* layer_174_out = run_hardswish_layer("extracted_params_hsigmoid/layer174_HARD_SWISH_1_block6b_se_reduce_1_truediv__1_block6b_se_reduce_1_Relu6_1", "all_layer_io/layer_174_HARD_SWISH",
                                                layer_173_out, &layer_174_size);
    printf("Starting layer 175 Conv_2D...\n");
    int conv175_out_h, conv175_out_w;
    int8_t* layer_175_out = run_conv2d_layer("extracted_params_hsigmoid/layer175_CONV_2D_1_block6b_se_expand_1_Relu6_1_block6b_se_expand_1_add_1_block6b", "all_layer_io/layer_175_CONV_2D",
                                            layer_174_out, conv173_out_h, conv173_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv175_out_h, &conv175_out_w);
    printf("Starting Layer 176 MUL...\n");
    int8_t* layer_176_out = run_mul_layer("extracted_params_hsigmoid/layer176_MUL_1_block6b_se_expand_1_truediv", "all_layer_io/layer_176_MUL",
                                         layer_175_out, 1152, NULL, 1, 0);
    printf("Starting Layer 177 MUL...\n");
    int8_t* layer_177_out = run_mul_layer("extracted_params_hsigmoid/layer177_MUL_1_block6b_se_excite_1_Mul", "all_layer_io/layer_177_MUL",
                                         NULL, 7*7*1152, layer_176_out, 1152, 1);
    printf("Starting layer 178 Conv_2D...\n");
    int conv178_out_h, conv178_out_w;
    int8_t* layer_178_out = run_conv2d_layer("extracted_params_hsigmoid/layer178_CONV_2D_1_block6b_project_conv_1_BiasAdd_1_block6b_project_conv_1_convo", "all_layer_io/layer_178_CONV_2D",
                                            layer_177_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv178_out_h, &conv178_out_w);
    printf("Starting Layer 179 ADD...\n");
    int8_t* layer_179_out = run_add_layer("extracted_params_hsigmoid/layer179_ADD_1_block6b_add_1_Add", "all_layer_io/layer_179_ADD",
                                        layer_178_out, conv178_out_h * conv178_out_w * 192, layer_163_out, conv178_out_h * conv178_out_w * 192);
    printf("Output of layer 179 ADD:\n");
    for(int i = 0; i < conv178_out_h * conv178_out_w * 192; ++i) {
        printf("%d ", layer_179_out[i]);
        if ((i + 1) % 192 == 0) printf("\n");
    }
    printf("Starting Layer 180 Conv_2D...\n");
    int conv180_out_h, conv180_out_w;
    int8_t* layer_180_out = run_conv2d_layer("extracted_params_hsigmoid/layer180_CONV_2D_1_block6c_expand_conv_1_BiasAdd_1_block6c_expand_conv_1_convolu", "all_layer_io/layer_180_CONV_2D",
                                            layer_179_out, conv178_out_h, conv178_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv180_out_h, &conv180_out_w);
    printf("Starting Layer 181 HardSwish...\n");
    int layer_181_size = conv180_out_h * conv180_out_w * 1152;
    int8_t* layer_181_out = run_hardswish_layer("extracted_params_hsigmoid/layer181_HARD_SWISH_1_block6c_expand_activation_1_truediv__1_block6c_expand_acti", "all_layer_io/layer_181_HARD_SWISH",
                                                layer_180_out, &layer_181_size);
    printf("Starting Layer 182 DepthwiseConv2D...\n");
    int conv182_out_h, conv182_out_w;
    int8_t* layer_182_out = run_dw_conv_layer("extracted_params_hsigmoid/layer182_DEPTHWISE_CONV_2D_1_block6c_dwconv2_1_BiasAdd_1_block6c_dwconv2_1_depth",
                                            layer_181_out, conv180_out_h, conv180_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv182_out_h, &conv182_out_w);
    printf("Starting Layer 183 HardSwish...\n");
    int layer_183_size = conv182_out_h * conv182_out_w * 1152;
    int8_t* layer_183_out = run_hardswish_layer("extracted_params_hsigmoid/layer183_HARD_SWISH_1_block6c_activation_1_truediv__1_block6c_activation_1_Relu6", "all_layer_io/layer_183_HARD_SWISH",
                                                layer_182_out, &layer_183_size);
    printf("Starting Layer 184 Mean...\n");
    int8_t* layer_184_out = run_mean_layer("extracted_params_hsigmoid/layer184_MEAN_1_block6c_se_squeeze_1_Mean", "all_layer_io/layer_184_MEAN",
                                            layer_183_out, conv182_out_h, conv182_out_w, 1152, 1, 1);
    printf("Skip layer 185 shape 186 stride_slice 187 pack 188 reshape...\n");
    printf("Starting Layer 189 Conv_2D...\n");
    int conv189_out_h, conv189_out_w;
    int8_t* layer_189_out = run_conv2d_layer("extracted_params_hsigmoid/layer189_CONV_2D_1_block6c_se_reduce_1_BiasAdd_1_block6c_se_reduce_1_convolution", "all_layer_io/layer_189_CONV_2D",
                                            layer_184_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv189_out_h, &conv189_out_w);
    printf("Starting Layer 190 HardSwish...\n");
    int layer_190_size = conv189_out_h * conv189_out_w * 48;
    int8_t* layer_190_out = run_hardswish_layer("extracted_params_hsigmoid/layer190_HARD_SWISH_1_block6c_se_reduce_1_truediv__1_block6c_se_reduce_1_Relu6_1", "all_layer_io/layer_190_HARD_SWISH",
                                                layer_189_out, &layer_190_size);
    printf("Starting Layer 191 Conv_2D...\n");
    int conv191_out_h, conv191_out_w;
    int8_t* layer_191_out = run_conv2d_layer("extracted_params_hsigmoid/layer191_CONV_2D_1_block6c_se_expand_1_Relu6_1_block6c_se_expand_1_add_1_block6c", "all_layer_io/layer_191_CONV_2D",
                                            layer_190_out, conv189_out_h, conv189_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv191_out_h, &conv191_out_w);
    printf("Starting Layer 192 MUL...\n");
    int8_t* layer_192_out = run_mul_layer("extracted_params_hsigmoid/layer192_MUL_1_block6c_se_expand_1_truediv", "all_layer_io/layer_192_MUL",
                                         layer_191_out, 1152, NULL, 1, 0);
    printf("Starting Layer 193 MUL...\n");
    int8_t* layer_193_out = run_mul_layer("extracted_params_hsigmoid/layer193_MUL_1_block6c_se_excite_1_Mul", "all_layer_io/layer_193_MUL",
                                         layer_183_out, 7*7*1152, layer_192_out, 1152, 1);
    printf("Starting Layer 194 Conv_2D...\n");
    int conv194_out_h, conv194_out_w;
    int8_t* layer_194_out = run_conv2d_layer("extracted_params_hsigmoid/layer194_CONV_2D_1_block6c_project_conv_1_BiasAdd_1_block6c_project_conv_1_convo", "all_layer_io/layer_194_CONV_2D",
                                            layer_193_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv194_out_h, &conv194_out_w);
    printf("Starting Layer 195 ADD...\n");
    int8_t* layer_195_out = run_add_layer("extracted_params_hsigmoid/layer195_ADD_1_block6c_add_1_Add", "all_layer_io/layer_195_ADD",
                                        layer_194_out, conv194_out_h * conv194_out_w * 192, layer_179_out, conv194_out_h * conv194_out_w * 192);
    printf("Starting Layer 196 Conv_2D...\n");
    int conv196_out_h, conv196_out_w;
    int8_t* layer_196_out = run_conv2d_layer("extracted_params_hsigmoid/layer196_CONV_2D_1_block6d_expand_conv_1_BiasAdd_1_block6d_expand_conv_1_convolu", "all_layer_io/layer_196_CONV_2D",
                                            layer_195_out, conv194_out_h, conv194_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv196_out_h, &conv196_out_w);
    printf("Starting Layer 197 HardSwish...\n");
    int layer_197_size = conv196_out_h * conv196_out_w * 1152;
    int8_t* layer_197_out = run_hardswish_layer("extracted_params_hsigmoid/layer197_HARD_SWISH_1_block6d_expand_activation_1_truediv__1_block6d_expand_acti", "all_layer_io/layer_197_HARD_SWISH",
                                                layer_196_out, &layer_197_size);
    printf("Starting Layer 198 DepthwiseConv2D...\n");
    int conv198_out_h, conv198_out_w;
    int8_t* layer_198_out = run_dw_conv_layer("extracted_params_hsigmoid/layer198_DEPTHWISE_CONV_2D_1_block6d_dwconv2_1_BiasAdd_1_block6d_dwconv2_1_depth",
                                            layer_197_out, conv196_out_h, conv196_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv198_out_h, &conv198_out_w);
    printf("Starting Layer 199 HardSwish...\n");
    int layer_199_size = conv198_out_h * conv198_out_w * 1152;
    int8_t* layer_199_out = run_hardswish_layer("extracted_params_hsigmoid/layer199_HARD_SWISH_1_block6d_activation_1_truediv__1_block6d_activation_1_Relu6", "all_layer_io/layer_199_HARD_SWISH",
                                                layer_198_out, &layer_199_size);
    printf("Starting Layer 200 Mean...\n");
    int8_t* layer_200_out = run_mean_layer("extracted_params_hsigmoid/layer200_MEAN_1_block6d_se_squeeze_1_Mean", "all_layer_io/layer_200_MEAN",
                                            layer_199_out, conv198_out_h, conv198_out_w, 1152, 1, 1);
    printf("Skip layer 201 shape 202 stride_slice 203 pack 204 reshape...\n");
    printf("Starting Layer 205 Conv_2D...\n");
    int conv205_out_h, conv205_out_w;
    int8_t* layer_205_out = run_conv2d_layer("extracted_params_hsigmoid/layer205_CONV_2D_1_block6d_se_reduce_1_BiasAdd_1_block6d_se_reduce_1_convolution", "all_layer_io/layer_205_CONV_2D",
                                            layer_200_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv205_out_h, &conv205_out_w);
    printf("Starting Layer 206 HardSwish...\n");
    int layer_206_size = conv205_out_h * conv205_out_w * 48;
    int8_t* layer_206_out = run_hardswish_layer("extracted_params_hsigmoid/layer206_HARD_SWISH_1_block6d_se_reduce_1_truediv__1_block6d_se_reduce_1_Relu6_1", "all_layer_io/layer_206_HARD_SWISH",
                                                layer_205_out, &layer_206_size);
    printf("Starting Layer 207 Conv_2D...\n");
    int conv207_out_h, conv207_out_w;
    int8_t* layer_207_out = run_conv2d_layer("extracted_params_hsigmoid/layer207_CONV_2D_1_block6d_se_expand_1_Relu6_1_block6d_se_expand_1_add_1_block6d", "all_layer_io/layer_207_CONV_2D",
                                            layer_206_out, conv205_out_h, conv205_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv207_out_h, &conv207_out_w);
    printf("Starting Layer 208 MUL...\n");
    int8_t* layer_208_out = run_mul_layer("extracted_params_hsigmoid/layer208_MUL_1_block6d_se_expand_1_truediv", "all_layer_io/layer_208_MUL",
                                         layer_207_out, 1152, NULL, 1, 0);
    printf("Starting Layer 209 MUL...\n");
    int8_t* layer_209_out = run_mul_layer("extracted_params_hsigmoid/layer209_MUL_1_block6d_se_excite_1_Mul", "all_layer_io/layer_209_MUL",
                                         layer_199_out, 7*7*1152, layer_208_out, 1152, 1);
    printf("Starting Layer 210 Conv_2D...\n");
    int conv210_out_h, conv210_out_w;
    int8_t* layer_210_out = run_conv2d_layer("extracted_params_hsigmoid/layer210_CONV_2D_1_block6d_project_conv_1_BiasAdd_1_block6d_project_conv_1_convo", "all_layer_io/layer_210_CONV_2D",
                                            layer_209_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv210_out_h, &conv210_out_w);
    printf("Starting Layer 211 ADD...\n");
    int8_t* layer_211_out = run_add_layer("extracted_params_hsigmoid/layer211_ADD_1_block6d_add_1_Add", "all_layer_io/layer_211_ADD",
                                        layer_210_out, conv210_out_h * conv210_out_w * 192, layer_195_out, conv210_out_h * conv210_out_w * 192);
    printf("Starting Layer 212 Conv_2D...\n");
    int conv212_out_h, conv212_out_w;
    int8_t* layer_212_out = run_conv2d_layer("extracted_params_hsigmoid/layer212_CONV_2D_1_block6e_expand_conv_1_BiasAdd_1_block6e_expand_conv_1_convolu", "all_layer_io/layer_212_CONV_2D",
                                            layer_211_out, conv210_out_h, conv210_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv212_out_h, &conv212_out_w);
    printf("Output of layer 212 Conv_2D:\n");
    for(int i = 0; i < conv212_out_h * conv212_out_w * 1152; ++i) {
        printf("%d ", layer_212_out[i]);
        if ((i + 1) % 1152 == 0) printf("\n");
    }
    printf("Starting Layer 213 HardSwish...\n");
    int layer_213_size = conv212_out_h * conv212_out_w * 1152;
    int8_t* layer_213_out = run_hardswish_layer("extracted_params_hsigmoid/layer213_HARD_SWISH_1_block6e_expand_activation_1_truediv__1_block6e_expand_acti", "all_layer_io/layer_213_HARD_SWISH",
                                                layer_212_out, &layer_213_size);
    printf("Starting Layer 214 DepthwiseConv2D...\n");
    int conv214_out_h, conv214_out_w;
    int8_t* layer_214_out = run_dw_conv_layer("extracted_params_hsigmoid/layer214_DEPTHWISE_CONV_2D_1_block6e_dwconv2_1_BiasAdd_1_block6e_dwconv2_1_depth",
                                            layer_213_out, conv212_out_h, conv212_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv214_out_h, &conv214_out_w);
    printf("Starting Layer 215 HardSwish...\n");
    int layer_215_size = conv214_out_h * conv214_out_w * 1152;
    int8_t* layer_215_out = run_hardswish_layer("extracted_params_hsigmoid/layer215_HARD_SWISH_1_block6e_activation_1_truediv__1_block6e_activation_1_Relu6", "all_layer_io/layer_215_HARD_SWISH",
                                                layer_214_out, &layer_215_size);
    printf("Output of layer 215 HardSwish:\n");
    for(int i = 0; i < conv214_out_h * conv214_out_w * 1152; ++i) {
        printf("%d ", layer_215_out[i]);
        if ((i + 1) % 1152 == 0) printf("\n");
    }
    printf("Starting Layer 216 Mean...\n");
    int8_t* layer_216_out = run_mean_layer("extracted_params_hsigmoid/layer216_MEAN_1_block6e_se_squeeze_1_Mean", "all_layer_io/layer_216_MEAN",
                                            layer_215_out, conv214_out_h, conv214_out_w, 1152, 1, 1);
    printf("Skip layer 217 shape 218 stride_slice 219 pack 220 reshape...\n");
    printf("Starting Layer 221 Conv_2D...\n");
    int conv221_out_h, conv221_out_w;
    int8_t* layer_221_out = run_conv2d_layer("extracted_params_hsigmoid/layer221_CONV_2D_1_block6e_se_reduce_1_BiasAdd_1_block6e_se_reduce_1_convolution", "all_layer_io/layer_221_CONV_2D",
                                            layer_216_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv221_out_h, &conv221_out_w);
    printf("Starting Layer 222 HardSwish...\n");
    int layer_222_size = conv221_out_h * conv221_out_w * 48;
    int8_t* layer_222_out = run_hardswish_layer("extracted_params_hsigmoid/layer222_HARD_SWISH_1_block6e_se_reduce_1_truediv__1_block6e_se_reduce_1_Relu6_1", "all_layer_io/layer_222_HARD_SWISH",
                                                layer_221_out, &layer_222_size);
    printf("Starting Layer 223 Conv_2D...\n");
    int conv223_out_h, conv223_out_w;
    int8_t* layer_223_out = run_conv2d_layer("extracted_params_hsigmoid/layer223_CONV_2D_1_block6e_se_expand_1_Relu6_1_block6e_se_expand_1_add_1_block6e", "all_layer_io/layer_223_CONV_2D",
                                            layer_222_out, conv221_out_h, conv221_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv223_out_h, &conv223_out_w);
    printf("Starting Layer 224 MUL...\n");
    int8_t* layer_224_out = run_mul_layer("extracted_params_hsigmoid/layer224_MUL_1_block6e_se_expand_1_truediv", "all_layer_io/layer_224_MUL",
                                         layer_223_out, 1152, NULL, 1, 0);
    printf("Starting Layer 225 MUL...\n");
    int8_t* layer_225_out = run_mul_layer("extracted_params_hsigmoid/layer225_MUL_1_block6e_se_excite_1_Mul", "all_layer_io/layer_225_MUL",
                                         layer_215_out, 7*7*1152, layer_224_out, 1152, 1);
    printf("Output of layer 225 MUL:\n");
    for(int i = 0; i < conv214_out_h * conv214_out_w * 1152; ++i) {
        printf("%d ", layer_225_out[i]);
        if ((i + 1) % 1152 == 0) printf("\n");
    }
    printf("Starting Layer 226 Conv_2D...\n");
    int conv226_out_h, conv226_out_w;
    int8_t* layer_226_out = run_conv2d_layer("extracted_params_hsigmoid/layer226_CONV_2D_1_block6e_project_conv_1_BiasAdd_1_block6e_project_conv_1_convo", "all_layer_io/layer_226_CONV_2D",
                                            layer_225_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv226_out_h, &conv226_out_w);
    printf("Starting Layer 227 ADD...\n");
    int8_t* layer_227_out = run_add_layer("extracted_params_hsigmoid/layer227_ADD_1_block6e_add_1_Add", "all_layer_io/layer_227_ADD",
                                        layer_226_out, conv226_out_h * conv226_out_w * 192, layer_211_out, conv226_out_h * conv226_out_w * 192);
    printf("Starting Layer 228 Conv_2D...\n");
    int conv228_out_h, conv228_out_w;
    int8_t* layer_228_out = run_conv2d_layer("extracted_params_hsigmoid/layer228_CONV_2D_1_block6f_expand_conv_1_BiasAdd_1_block6f_expand_conv_1_convolu", "all_layer_io/layer_228_CONV_2D",
                                            layer_227_out, conv226_out_h, conv226_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv228_out_h, &conv228_out_w);
    printf("Starting Layer 229 HardSwish...\n");
    int layer_229_size = conv228_out_h * conv228_out_w * 1152;
    int8_t* layer_229_out = run_hardswish_layer("extracted_params_hsigmoid/layer229_HARD_SWISH_1_block6f_expand_activation_1_truediv__1_block6f_expand_acti", "all_layer_io/layer_229_HARD_SWISH",
                                                layer_228_out, &layer_229_size);
    printf("Starting Layer 230 DepthwiseConv2D...\n");
    int conv230_out_h, conv230_out_w;
    int8_t* layer_230_out = run_dw_conv_layer("extracted_params_hsigmoid/layer230_DEPTHWISE_CONV_2D_1_block6f_dwconv2_1_BiasAdd_1_block6f_dwconv2_1_depth",
                                            layer_229_out, conv228_out_h, conv228_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv230_out_h, &conv230_out_w);
    printf("Starting Layer 231 HardSwish...\n");
    int layer_231_size = conv230_out_h * conv230_out_w * 1152;
    int8_t* layer_231_out = run_hardswish_layer("extracted_params_hsigmoid/layer231_HARD_SWISH_1_block6f_activation_1_truediv__1_block6f_activation_1_Relu6", "all_layer_io/layer_231_HARD_SWISH",
                                                layer_230_out, &layer_231_size);
    printf("Starting Layer 232 Mean...\n");
    int8_t* layer_232_out = run_mean_layer("extracted_params_hsigmoid/layer232_MEAN_1_block6f_se_squeeze_1_Mean", "all_layer_io/layer_232_MEAN",
                                            layer_231_out, conv230_out_h, conv230_out_w, 1152, 1, 1);
    printf("Skip layer 233 shape 234 stride_slice 235 pack 236 reshape...\n");
    printf("Starting Layer 237 Conv_2D...\n");
    int conv237_out_h, conv237_out_w;
    int8_t* layer_237_out = run_conv2d_layer("extracted_params_hsigmoid/layer237_CONV_2D_1_block6f_se_reduce_1_BiasAdd_1_block6f_se_reduce_1_convolution", "all_layer_io/layer_237_CONV_2D",
                                            layer_232_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv237_out_h, &conv237_out_w);
    printf("Starting Layer 238 HardSwish...\n");
    int layer_238_size = conv237_out_h * conv237_out_w * 48;
    int8_t* layer_238_out = run_hardswish_layer("extracted_params_hsigmoid/layer238_HARD_SWISH_1_block6f_se_reduce_1_truediv__1_block6f_se_reduce_1_Relu6_1", "all_layer_io/layer_238_HARD_SWISH",
                                                layer_237_out, &layer_238_size);
    printf("Starting Layer 239 Conv_2D...\n");
    int conv239_out_h, conv239_out_w;
    int8_t* layer_239_out = run_conv2d_layer("extracted_params_hsigmoid/layer239_CONV_2D_1_block6f_se_expand_1_Relu6_1_block6f_se_expand_1_add_1_block6f", "all_layer_io/layer_239_CONV_2D",
                                            layer_238_out, conv237_out_h, conv237_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv239_out_h, &conv239_out_w);
    printf("Starting Layer 240 MUL...\n");
    int8_t* layer_240_out = run_mul_layer("extracted_params_hsigmoid/layer240_MUL_1_block6f_se_expand_1_truediv", "all_layer_io/layer_240_MUL",
                                         layer_239_out, 1152, NULL, 1, 0);
    printf("Starting Layer 241 MUL...\n");
    int8_t* layer_241_out = run_mul_layer("extracted_params_hsigmoid/layer241_MUL_1_block6f_se_excite_1_Mul", "all_layer_io/layer_241_MUL",
                                         layer_231_out, 7*7*1152, layer_240_out, 1152, 1);
    printf("Starting Layer 242 Conv_2D...\n");
    int conv242_out_h, conv242_out_w;
    int8_t* layer_242_out = run_conv2d_layer("extracted_params_hsigmoid/layer242_CONV_2D_1_block6f_project_conv_1_BiasAdd_1_block6f_project_conv_1_convo", "all_layer_io/layer_242_CONV_2D",
                                            layer_241_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv242_out_h, &conv242_out_w);
    printf("Starting Layer 243 ADD...\n");
    int8_t* layer_243_out = run_add_layer("extracted_params_hsigmoid/layer243_ADD_1_block6f_add_1_Add", "all_layer_io/layer_243_ADD",
                                        layer_242_out, conv242_out_h * conv242_out_w * 192, layer_227_out, conv242_out_h * conv242_out_w * 192);
    printf("Starting Layer 244 Conv_2D...\n");
    int conv244_out_h, conv244_out_w;
    int8_t* layer_244_out = run_conv2d_layer("extracted_params_hsigmoid/layer244_CONV_2D_1_block6g_expand_conv_1_BiasAdd_1_block6g_expand_conv_1_convolu", "all_layer_io/layer_244_CONV_2D",
                                            layer_243_out, conv242_out_h, conv242_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv244_out_h, &conv244_out_w);
    printf("Starting Layer 245 HardSwish...\n");
    int layer_245_size = conv244_out_h * conv244_out_w * 1152;
    int8_t* layer_245_out = run_hardswish_layer("extracted_params_hsigmoid/layer245_HARD_SWISH_1_block6g_expand_activation_1_truediv__1_block6g_expand_acti", "all_layer_io/layer_245_HARD_SWISH",
                                                layer_244_out, &layer_245_size);
    printf("Starting Layer 246 DepthwiseConv2D...\n");
    int conv246_out_h, conv246_out_w;
    int8_t* layer_246_out = run_dw_conv_layer("extracted_params_hsigmoid/layer246_DEPTHWISE_CONV_2D_1_block6g_dwconv2_1_BiasAdd_1_block6g_dwconv2_1_depth",
                                            layer_245_out, conv244_out_h, conv244_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv246_out_h, &conv246_out_w);
    printf("Starting Layer 247 HardSwish...\n");
    int layer_247_size = conv246_out_h * conv246_out_w * 1152;
    int8_t* layer_247_out = run_hardswish_layer("extracted_params_hsigmoid/layer247_HARD_SWISH_1_block6g_activation_1_truediv__1_block6g_activation_1_Relu6", "all_layer_io/layer_247_HARD_SWISH",
                                                layer_246_out, &layer_247_size);
    printf("Starting Layer 248 Mean...\n");
    int8_t* layer_248_out = run_mean_layer("extracted_params_hsigmoid/layer248_MEAN_1_block6g_se_squeeze_1_Mean", "all_layer_io/layer_248_MEAN",
                                            layer_247_out, conv246_out_h, conv246_out_w, 1152, 1, 1);
    printf("Skip layer 249 shape 250 stride_slice 251 pack 252 reshape...\n");
    printf("Starting Layer 253 Conv_2D...\n");
    int conv253_out_h, conv253_out_w;
    int8_t* layer_253_out = run_conv2d_layer("extracted_params_hsigmoid/layer253_CONV_2D_1_block6g_se_reduce_1_BiasAdd_1_block6g_se_reduce_1_convolution", "all_layer_io/layer_253_CONV_2D",
                                            layer_248_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv253_out_h, &conv253_out_w);
    printf("Starting Layer 254 HardSwish...\n");
    int layer_254_size = conv253_out_h * conv253_out_w * 48;
    int8_t* layer_254_out = run_hardswish_layer("extracted_params_hsigmoid/layer254_HARD_SWISH_1_block6g_se_reduce_1_truediv__1_block6g_se_reduce_1_Relu6_1", "all_layer_io/layer_254_HARD_SWISH",
                                                layer_253_out, &layer_254_size);
    printf("Starting Layer 255 Conv_2D...\n");
    int conv255_out_h, conv255_out_w;
    int8_t* layer_255_out = run_conv2d_layer("extracted_params_hsigmoid/layer255_CONV_2D_1_block6g_se_expand_1_Relu6_1_block6g_se_expand_1_add_1_block6g", "all_layer_io/layer_255_CONV_2D",
                                            layer_254_out, conv253_out_h, conv253_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv255_out_h, &conv255_out_w);
    printf("Starting Layer 256 MUL...\n");
    int8_t* layer_256_out = run_mul_layer("extracted_params_hsigmoid/layer256_MUL_1_block6g_se_expand_1_truediv", "all_layer_io/layer_256_MUL",
                                         layer_255_out, 1152, NULL, 1, 0);
    printf("Starting Layer 257 MUL...\n");
    int8_t* layer_257_out = run_mul_layer("extracted_params_hsigmoid/layer257_MUL_1_block6g_se_excite_1_Mul", "all_layer_io/layer_257_MUL",
                                         layer_247_out, 7*7*1152, layer_256_out, 1152, 1);
    printf("Starting layer 258 Conv_2D...\n");
    int conv258_out_h, conv258_out_w;
    int8_t* layer_258_out = run_conv2d_layer("extracted_params_hsigmoid/layer258_CONV_2D_1_block6g_project_conv_1_BiasAdd_1_block6g_project_conv_1_convo", "all_layer_io/layer_258_CONV_2D",
                                            layer_257_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv258_out_h, &conv258_out_w);
    printf("Starting layer 259 ADD...\n");
    int8_t* layer_259_out = run_add_layer("extracted_params_hsigmoid/layer259_ADD_1_block6g_add_1_Add", "all_layer_io/layer_259_ADD",
                                        layer_258_out, conv258_out_h * conv258_out_w * 192, layer_243_out, conv258_out_h * conv258_out_w * 192);
    printf("Starting layer 260 Conv_2D...\n");
    int conv260_out_h, conv260_out_w;
    int8_t* layer_260_out = run_conv2d_layer("extracted_params_hsigmoid/layer260_CONV_2D_1_block6h_expand_conv_1_BiasAdd_1_block6h_expand_conv_1_convolu", "all_layer_io/layer_260_CONV_2D",
                                            layer_259_out, conv258_out_h, conv258_out_w, 192, 1152, 1, 1, 1, 1, "SAME", &conv260_out_h, &conv260_out_w);
    printf("Starting layer 261 HardSwish...\n");
    int layer_261_size = conv260_out_h * conv260_out_w * 1152;
    int8_t* layer_261_out = run_hardswish_layer("extracted_params_hsigmoid/layer261_HARD_SWISH_1_block6h_expand_activation_1_truediv__1_block6h_expand_acti", "all_layer_io/layer_261_HARD_SWISH",
                                                layer_260_out, &layer_261_size);
    printf("Starting layer 262 DepthwiseConv2D...\n");
    int conv262_out_h, conv262_out_w;
    int8_t* layer_262_out = run_dw_conv_layer("extracted_params_hsigmoid/layer262_DEPTHWISE_CONV_2D_1_block6h_dwconv2_1_BiasAdd_1_block6h_dwconv2_1_depth",
                                            layer_261_out, conv260_out_h, conv260_out_w, 1152, 1152, 3, 3, 1, 1, "SAME", &conv262_out_h, &conv262_out_w);
    printf("Starting layer 263 HardSwish...\n");
    int layer_263_size = conv262_out_h * conv262_out_w * 1152;
    int8_t* layer_263_out = run_hardswish_layer("extracted_params_hsigmoid/layer263_HARD_SWISH_1_block6h_activation_1_truediv__1_block6h_activation_1_Relu6", "all_layer_io/layer_263_HARD_SWISH",
                                                layer_262_out, &layer_263_size);
    printf("Starting layer 264 Mean...\n");
    int8_t* layer_264_out = run_mean_layer("extracted_params_hsigmoid/layer264_MEAN_1_block6h_se_squeeze_1_Mean", "all_layer_io/layer_264_MEAN",
                                            layer_263_out, conv262_out_h, conv262_out_w, 1152, 1, 1);
    printf("Skip layer 265 shape 266 stride_slice 267 pack 268 reshape...\n");
    printf("Starting layer 269 Conv_2D...\n");
    int conv269_out_h, conv269_out_w;
    int8_t* layer_269_out = run_conv2d_layer("extracted_params_hsigmoid/layer269_CONV_2D_1_block6h_se_reduce_1_BiasAdd_1_block6h_se_reduce_1_convolution", "all_layer_io/layer_269_CONV_2D",
                                            layer_264_out, 1, 1, 1152, 48, 1, 1, 1, 1, "SAME", &conv269_out_h, &conv269_out_w);
    printf("Starting layer 270 HardSwish...\n");
    int layer_270_size = conv269_out_h * conv269_out_w * 48;
    int8_t* layer_270_out = run_hardswish_layer("extracted_params_hsigmoid/layer270_HARD_SWISH_1_block6h_se_reduce_1_truediv__1_block6h_se_reduce_1_Relu6_1", "all_layer_io/layer_270_HARD_SWISH",
                                                layer_269_out, &layer_270_size);
    printf("Starting layer 271 Conv_2D...\n");
    int conv271_out_h, conv271_out_w;
    int8_t* layer_271_out = run_conv2d_layer("extracted_params_hsigmoid/layer271_CONV_2D_1_block6h_se_expand_1_Relu6_1_block6h_se_expand_1_add_1_block6h", "all_layer_io/layer_271_CONV_2D",
                                            layer_270_out, conv269_out_h, conv269_out_w, 48, 1152, 1, 1, 1, 1, "SAME", &conv271_out_h, &conv271_out_w);
    printf("Starting layer 272 MUL...\n");
    int8_t* layer_272_out = run_mul_layer("extracted_params_hsigmoid/layer272_MUL_1_block6h_se_expand_1_truediv", "all_layer_io/layer_272_MUL",
                                         layer_271_out, 1152, NULL, 1, 0);
    printf("Starting layer 273 MUL...\n");
    int8_t* layer_273_out = run_mul_layer("extracted_params_hsigmoid/layer273_MUL_1_block6h_se_excite_1_Mul", "all_layer_io/layer_273_MUL",
                                         layer_263_out, 7*7*1152, layer_272_out, 1152, 1);
    printf("Starting layer 274 Conv_2D...\n");
    int conv274_out_h, conv274_out_w;
    int8_t* layer_274_out = run_conv2d_layer("extracted_params_hsigmoid/layer274_CONV_2D_1_block6h_project_conv_1_BiasAdd_1_block6h_project_conv_1_convo", "all_layer_io/layer_274_CONV_2D",
                                            layer_273_out, 7, 7, 1152, 192, 1, 1, 1, 1, "SAME", &conv274_out_h, &conv274_out_w);
    printf("Starting layer 275 ADD...\n");
    int8_t* layer_275_out = run_add_layer("extracted_params_hsigmoid/layer275_ADD_1_block6h_add_1_Add", "all_layer_io/layer_275_ADD",
                                        layer_274_out, conv274_out_h * conv274_out_w * 192, layer_259_out, conv274_out_h * conv274_out_w * 192);
    printf("Starting layer 276 Conv_2D...\n");
    int conv276_out_h, conv276_out_w;
    int8_t* layer_276_out = run_conv2d_layer("extracted_params_hsigmoid/layer276_CONV_2D_1_top_conv_1_BiasAdd_1_top_conv_1_convolution_", "all_layer_io/layer_276_CONV_2D",
                                            layer_275_out, conv274_out_h, conv274_out_w, 192, 1280, 1, 1, 1, 1, "SAME", &conv276_out_h, &conv276_out_w);
    printf("Starting layer 277 HardSwish...\n");
    int layer_277_size = conv276_out_h * conv276_out_w * 1280;
    int8_t* layer_277_out = run_hardswish_layer("extracted_params_hsigmoid/layer277_HARD_SWISH_1_top_activation_1_truediv__1_top_activation_1_Relu6_1_top_a", "all_layer_io/layer_277_HARD_SWISH",
                                                layer_276_out, &layer_277_size);
    printf("Starting layer 278 Mean...\n");
    int8_t* layer_278_out = run_mean_layer("extracted_params_hsigmoid/layer278_MEAN_1_avg_pool_1_Mean", "all_layer_io/layer_278_MEAN",
                                            layer_277_out, conv276_out_h, conv276_out_w, 1280, 1, 1);
    printf("Output of layer 278 Mean:\n");
    for(int i = 0; i < 1280; ++i) {
        printf("%d ", layer_278_out[i]);
    }
    printf("\n");
    // The final output layer, no need to free after this
    free(layer_95_out);
    free(layer_96_out);
    free(layer_97_out);
    free(layer_98_out);
    free(layer_99_out);
    free(layer_100_out);
    free(layer_101_out);
    free(layer_102_out);
    free(layer_103_out);
    free(layer_104_out);
    free(layer_105_out);
    free(layer_110_out);
    free(layer_111_out);
    free(layer_112_out);
    free(layer_113_out);
    free(layer_114_out);
    free(layer_115_out);
    free(layer_116_out);
    free(layer_117_out);
    free(layer_118_out);
    free(layer_119_out);
    free(layer_120_out);
    free(layer_121_out);
    free(layer_126_out);
    free(layer_127_out);
    free(layer_128_out);
    free(layer_129_out);
    free(layer_130_out);
    free(layer_131_out);
    free(layer_132_out);
    free(layer_133_out);
    free(layer_134_out);
    free(layer_135_out);
    free(layer_136_out);
    free(layer_137_out);
    free(layer_142_out);
    free(layer_143_out);
    free(layer_144_out);
    free(layer_145_out);
    free(layer_146_out);
    free(layer_147_out);
    free(layer_148_out);
    free(layer_149_out);
    free(layer_150_out);
    free(layer_151_out);
    free(layer_152_out);
    free(layer_153_out);
    free(layer_158_out);
    free(layer_160_out);
    free(layer_161_out);
    free(layer_162_out);
    free(layer_163_out);
    free(layer_164_out);
    free(layer_166_out);
    free(layer_167_out);
    free(layer_168_out);
    free(layer_173_out);
    free(layer_174_out);
    free(layer_175_out);
    free(layer_176_out);
    free(layer_177_out);
    free(layer_178_out);
    free(layer_179_out);
    free(layer_180_out);
    free(layer_181_out);
    free(layer_182_out);
    free(layer_183_out);
    free(layer_184_out);
    free(layer_190_out);
    free(layer_191_out);
    free(layer_192_out);
    free(layer_193_out);
    free(layer_194_out);
    free(layer_195_out);
    

    return 0;
}
