import tf2onnx
import onnx
import math, re, numpy as np, tensorflow as tf, tensorflow_datasets as tfds
from tensorflow.keras import layers, models
# ========= 0) ĐỊNH NGHĨA H-SWISH =========
def h_swish(x):
    # Cách 1: Dùng hàm tối ưu sẵn của TF (khuyên dùng cho Jetson/TFLite)
    return x * tf.nn.relu6(x + 3) / 6
    
    # Cách 2: Viết thủ công (nếu TF version cũ)
    # return x * tf.nn.relu6(x + 3) / 6

# def hard_sigmoid(x):
#     return tf.keras.activations.hard_sigmoid(x)

def hard_sigmoid(x):
    x = tf.convert_to_tensor(x)
    return tf.nn.relu6(x + 3) / 6

# ========= 1) Kiến trúc BN-less (không BN) =========
def build_efficientnetv2_b0_full_bnless():
    inputs = layers.Input(shape=(224, 224, 3), name='input_layer_36')
    x = layers.Rescaling(1./255, name='rescaling_14')(inputs)
    normalizer = layers.Normalization(
        axis=-1,
        mean=[0.485, 0.456, 0.406],
        variance=[0.229**2, 0.224**2, 0.225**2],
        name='normalization_14'
    )
    x = normalizer(x)

    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=True, name='stem_conv')(x)
    x = layers.Activation(h_swish, name='stem_activation')(x)

    x = layers.Conv2D(16, 3, padding='same', use_bias=True, name='block1a_project_conv')(x)
    x = layers.Activation(h_swish, name='block1a_project_activation')(x)

    # block2a
    x = layers.Conv2D(64, 3, strides=2, padding='same', use_bias=True, name='block2a_expand_conv')(x)
    x = layers.Activation(h_swish, name='block2a_expand_activation')(x)
    x = layers.Conv2D(32, 1, padding='same', use_bias=True, name='block2a_project_conv')(x)

    # block2b
    x_skip = x
    x = layers.Conv2D(128, 3, padding='same', use_bias=True, name='block2b_expand_conv')(x)
    x = layers.Activation(h_swish, name='block2b_expand_activation')(x)
    x = layers.Conv2D(32, 1, padding='same', use_bias=True, name='block2b_project_conv')(x)
    x = layers.Dropout(0.0, name='block2b_drop')(x)
    x = layers.Add(name='block2b_add')([x, x_skip])

    # block3a
    x = layers.Conv2D(128, 3, strides=2, padding='same', use_bias=True, name='block3a_expand_conv')(x)
    x = layers.Activation(h_swish, name='block3a_expand_activation')(x)
    x = layers.Conv2D(48, 1, padding='same', use_bias=True, name='block3a_project_conv')(x)

    # block3b
    x_skip = x
    x = layers.Conv2D(192, 3, padding='same', use_bias=True, name='block3b_expand_conv')(x)
    x = layers.Activation(h_swish, name='block3b_expand_activation')(x)
    x = layers.Conv2D(48, 1, padding='same', use_bias=True, name='block3b_project_conv')(x)
    x = layers.Dropout(0.0, name='block3b_drop')(x)
    x = layers.Add(name='block3b_add')([x, x_skip])

    # block4a (stride 2)
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block4a_expand_conv')(x)
    x = layers.Activation(h_swish, name='block4a_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=True, name='block4a_dwconv2')(x)
    x = layers.Activation(h_swish, name='block4a_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block4a_se_squeeze')(x)
    se = layers.Reshape((1, 1, 192), name='block4a_se_reshape')(se)
    se = layers.Conv2D(12, 1, activation=h_swish, name='block4a_se_reduce')(se)
    se = layers.Conv2D(192, 1, activation=hard_sigmoid, name='block4a_se_expand')(se)
    x = layers.Multiply(name='block4a_se_excite')([x, se])
    x = layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4a_project_conv')(x)

    # block4b
    x_skip = x
    x = layers.Conv2D(384, 1, padding='same', use_bias=True, name='block4b_expand_conv')(x)
    x = layers.Activation(h_swish, name='block4b_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block4b_dwconv2')(x)
    x = layers.Activation(h_swish, name='block4b_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block4b_se_squeeze')(x)
    se = layers.Reshape((1, 1, 384), name='block4b_se_reshape')(se)
    se = layers.Conv2D(24, 1, activation=h_swish, name='block4b_se_reduce')(se)
    se = layers.Conv2D(384, 1, activation=hard_sigmoid, name='block4b_se_expand')(se)
    x = layers.Multiply(name='block4b_se_excite')([x, se])
    x = layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4b_project_conv')(x)
    x = layers.Dropout(0.0, name='block4b_drop')(x)
    x = layers.Add(name='block4b_add')([x, x_skip])

    # block4c
    x_skip = x
    x = layers.Conv2D(384, 1, padding='same', use_bias=True, name='block4c_expand_conv')(x)
    x = layers.Activation(h_swish, name='block4c_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block4c_dwconv2')(x)
    x = layers.Activation(h_swish, name='block4c_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block4c_se_squeeze')(x)
    se = layers.Reshape((1, 1, 384), name='block4c_se_reshape')(se)
    se = layers.Conv2D(24, 1, activation=h_swish, name='block4c_se_reduce')(se)
    se = layers.Conv2D(384, 1, activation=hard_sigmoid, name='block4c_se_expand')(se)
    x = layers.Multiply(name='block4c_se_excite')([x, se])
    x = layers.Conv2D(96, 1, padding='same', use_bias=True, name='block4c_project_conv')(x)
    x = layers.Dropout(0.0, name='block4c_drop')(x)
    x = layers.Add(name='block4c_add')([x, x_skip])

    # block5a
    x = layers.Conv2D(576, 1, padding='same', use_bias=True, name='block5a_expand_conv')(x)
    x = layers.Activation(h_swish, name='block5a_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block5a_dwconv2')(x)
    x = layers.Activation(h_swish, name='block5a_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block5a_se_squeeze')(x)
    se = layers.Reshape((1, 1, 576), name='block5a_se_reshape')(se)
    se = layers.Conv2D(24, 1, activation=h_swish, name='block5a_se_reduce')(se)
    se = layers.Conv2D(576, 1, activation=hard_sigmoid, name='block5a_se_expand')(se)
    x = layers.Multiply(name='block5a_se_excite')([x, se])
    x = layers.Conv2D(112, 1, padding='same', use_bias=True, name='block5a_project_conv')(x)

    # block5b
    x_skip = x
    x = layers.Conv2D(672, 1, padding='same', use_bias=True, name='block5b_expand_conv')(x)
    x = layers.Activation(h_swish, name='block5b_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block5b_dwconv2')(x)
    x = layers.Activation(h_swish, name='block5b_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block5b_se_squeeze')(x)
    se = layers.Reshape((1, 1, 672), name='block5b_se_reshape')(se)
    se = layers.Conv2D(28, 1, activation=h_swish, name='block5b_se_reduce')(se)
    se = layers.Conv2D(672, 1, activation=hard_sigmoid, name='block5b_se_expand')(se)
    x = layers.Multiply(name='block5b_se_excite')([x, se])
    x = layers.Conv2D(112, 1, padding='same', use_bias=True, name='block5b_project_conv')(x)
    x = layers.Dropout(0.0, name='block5b_drop')(x)
    x = layers.Add(name='block5b_add')([x, x_skip])

    # block5c
    x_skip = x
    x = layers.Conv2D(672, 1, padding='same', use_bias=True, name='block5c_expand_conv')(x)
    x = layers.Activation(h_swish, name='block5c_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block5c_dwconv2')(x)
    x = layers.Activation(h_swish, name='block5c_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block5c_se_squeeze')(x)
    se = layers.Reshape((1, 1, 672), name='block5c_se_reshape')(se)
    se = layers.Conv2D(28, 1, activation=h_swish, name='block5c_se_reduce')(se)
    se = layers.Conv2D(672, 1, activation=hard_sigmoid, name='block5c_se_expand')(se)
    x = layers.Multiply(name='block5c_se_excite')([x, se])
    x = layers.Conv2D(112, 1, padding='same', use_bias=True, name='block5c_project_conv')(x)
    x = layers.Dropout(0.0, name='block5c_drop')(x)
    x = layers.Add(name='block5c_add')([x, x_skip])

    # block5d
    x_skip = x
    x = layers.Conv2D(672, 1, padding='same', use_bias=True, name='block5d_expand_conv')(x)
    x = layers.Activation(h_swish, name='block5d_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block5d_dwconv2')(x)
    x = layers.Activation(h_swish, name='block5d_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block5d_se_squeeze')(x)
    se = layers.Reshape((1, 1, 672), name='block5d_se_reshape')(se)
    se = layers.Conv2D(28, 1, activation=h_swish, name='block5d_se_reduce')(se)
    se = layers.Conv2D(672, 1, activation=hard_sigmoid, name='block5d_se_expand')(se)
    x = layers.Multiply(name='block5d_se_excite')([x, se])
    x = layers.Conv2D(112, 1, padding='same', use_bias=True, name='block5d_project_conv')(x)
    x = layers.Dropout(0.0, name='block5d_drop')(x)
    x = layers.Add(name='block5d_add')([x, x_skip])

    # block5e
    x_skip = x
    x = layers.Conv2D(672, 1, padding='same', use_bias=True, name='block5e_expand_conv')(x)
    x = layers.Activation(h_swish, name='block5e_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block5e_dwconv2')(x)
    x = layers.Activation(h_swish, name='block5e_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block5e_se_squeeze')(x)
    se = layers.Reshape((1, 1, 672), name='block5e_se_reshape')(se)
    se = layers.Conv2D(28, 1, activation=h_swish, name='block5e_se_reduce')(se)
    se = layers.Conv2D(672, 1, activation=hard_sigmoid, name='block5e_se_expand')(se)
    x = layers.Multiply(name='block5e_se_excite')([x, se])
    x = layers.Conv2D(112, 1, padding='same', use_bias=True, name='block5e_project_conv')(x)
    x = layers.Dropout(0.0, name='block5e_drop')(x)
    x = layers.Add(name='block5e_add')([x, x_skip])

    # block6a
    x = layers.Conv2D(672, 1, padding='same', use_bias=True, name='block6a_expand_conv')(x)
    x = layers.Activation(h_swish, name='block6a_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=2, padding='same', use_bias=True, name='block6a_dwconv2')(x)
    x = layers.Activation(h_swish, name='block6a_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block6a_se_squeeze')(x)
    se = layers.Reshape((1, 1, 672), name='block6a_se_reshape')(se)
    se = layers.Conv2D(28, 1, activation=h_swish, name='block6a_se_reduce')(se)
    se = layers.Conv2D(672, 1, activation=hard_sigmoid, name='block6a_se_expand')(se)
    x = layers.Multiply(name='block6a_se_excite')([x, se])
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6a_project_conv')(x)

    # block6b
    x_skip = x
    x = layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6b_expand_conv')(x)
    x = layers.Activation(h_swish, name='block6b_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6b_dwconv2')(x)
    x = layers.Activation(h_swish, name='block6b_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block6b_se_squeeze')(x)
    se = layers.Reshape((1, 1, 1152), name='block6b_se_reshape')(se)
    se = layers.Conv2D(48, 1, activation=h_swish, name='block6b_se_reduce')(se)
    se = layers.Conv2D(1152, 1, activation=hard_sigmoid, name='block6b_se_expand')(se)
    x = layers.Multiply(name='block6b_se_excite')([x, se])
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6b_project_conv')(x)
    x = layers.Dropout(0.0, name='block6b_drop')(x)
    x = layers.Add(name='block6b_add')([x, x_skip])

    # block6c
    x_skip = x
    x = layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6c_expand_conv')(x)
    x = layers.Activation(h_swish, name='block6c_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6c_dwconv2')(x)
    x = layers.Activation(h_swish, name='block6c_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block6c_se_squeeze')(x)
    se = layers.Reshape((1, 1, 1152), name='block6c_se_reshape')(se)
    se = layers.Conv2D(48, 1, activation=h_swish, name='block6c_se_reduce')(se)
    se = layers.Conv2D(1152, 1, activation=hard_sigmoid, name='block6c_se_expand')(se)
    x = layers.Multiply(name='block6c_se_excite')([x, se])
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6c_project_conv')(x)
    x = layers.Dropout(0.0, name='block6c_drop')(x)
    x = layers.Add(name='block6c_add')([x, x_skip])

    # block6d
    x_skip = x
    x = layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6d_expand_conv')(x)
    x = layers.Activation(h_swish, name='block6d_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6d_dwconv2')(x)
    x = layers.Activation(h_swish, name='block6d_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block6d_se_squeeze')(x)
    se = layers.Reshape((1, 1, 1152), name='block6d_se_reshape')(se)
    se = layers.Conv2D(48, 1, activation=h_swish, name='block6d_se_reduce')(se)
    se = layers.Conv2D(1152, 1, activation=hard_sigmoid, name='block6d_se_expand')(se)
    x = layers.Multiply(name='block6d_se_excite')([x, se])
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6d_project_conv')(x)
    x = layers.Dropout(0.0, name='block6d_drop')(x)
    x = layers.Add(name='block6d_add')([x, x_skip])

    # block6e
    x_skip = x
    x = layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6e_expand_conv')(x)
    x = layers.Activation(h_swish, name='block6e_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6e_dwconv2')(x)
    x = layers.Activation(h_swish, name='block6e_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block6e_se_squeeze')(x)
    se = layers.Reshape((1, 1, 1152), name='block6e_se_reshape')(se)
    se = layers.Conv2D(48, 1, activation=h_swish, name='block6e_se_reduce')(se)
    se = layers.Conv2D(1152, 1, activation=hard_sigmoid, name='block6e_se_expand')(se)
    x = layers.Multiply(name='block6e_se_excite')([x, se])
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6e_project_conv')(x)
    x = layers.Dropout(0.0, name='block6e_drop')(x)
    x = layers.Add(name='block6e_add')([x, x_skip])

    # block6f
    x_skip = x
    x = layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6f_expand_conv')(x)
    x = layers.Activation(h_swish, name='block6f_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6f_dwconv2')(x)
    x = layers.Activation(h_swish, name='block6f_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block6f_se_squeeze')(x)
    se = layers.Reshape((1, 1, 1152), name='block6f_se_reshape')(se)
    se = layers.Conv2D(48, 1, activation=h_swish, name='block6f_se_reduce')(se)
    se = layers.Conv2D(1152, 1, activation=hard_sigmoid, name='block6f_se_expand')(se)
    x = layers.Multiply(name='block6f_se_excite')([x, se])
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6f_project_conv')(x)
    x = layers.Dropout(0.0, name='block6f_drop')(x)
    x = layers.Add(name='block6f_add')([x, x_skip])

    # block6g
    x_skip = x
    x = layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6g_expand_conv')(x)
    x = layers.Activation(h_swish, name='block6g_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6g_dwconv2')(x)
    x = layers.Activation(h_swish, name='block6g_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block6g_se_squeeze')(x)
    se = layers.Reshape((1, 1, 1152), name='block6g_se_reshape')(se)
    se = layers.Conv2D(48, 1, activation=h_swish, name='block6g_se_reduce')(se)
    se = layers.Conv2D(1152, 1, activation=hard_sigmoid, name='block6g_se_expand')(se)
    x = layers.Multiply(name='block6g_se_excite')([x, se])
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6g_project_conv')(x)
    x = layers.Dropout(0.0, name='block6g_drop')(x)
    x = layers.Add(name='block6g_add')([x, x_skip])

    # block6h
    x_skip = x
    x = layers.Conv2D(1152, 1, padding='same', use_bias=True, name='block6h_expand_conv')(x)
    x = layers.Activation(h_swish, name='block6h_expand_activation')(x)
    x = layers.DepthwiseConv2D(3, strides=1, padding='same', use_bias=True, name='block6h_dwconv2')(x)
    x = layers.Activation(h_swish, name='block6h_activation')(x)
    se = layers.GlobalAveragePooling2D(name='block6h_se_squeeze')(x)
    se = layers.Reshape((1, 1, 1152), name='block6h_se_reshape')(se)
    se = layers.Conv2D(48, 1, activation=h_swish, name='block6h_se_reduce')(se)
    se = layers.Conv2D(1152, 1, activation=hard_sigmoid, name='block6h_se_expand')(se)
    x = layers.Multiply(name='block6h_se_excite')([x, se])
    x = layers.Conv2D(192, 1, padding='same', use_bias=True, name='block6h_project_conv')(x)
    x = layers.Dropout(0.0, name='block6h_drop')(x)
    x = layers.Add(name='block6h_add')([x, x_skip])

    x = layers.Conv2D(1280, 1, padding='same', use_bias=True, name='top_conv')(x)
    x = layers.Activation(h_swish, name='top_activation')(x)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dropout(0.2, name='top_dropout')(x)
    outputs = layers.Dense(1000, activation='softmax', name='predictions')(x)
    return models.Model(inputs, outputs, name='build_efficientnetv2_b0_full_bnless')

target_model = build_efficientnetv2_b0_full_bnless()
print("Target (BN-less) layers:", len(target_model.layers))

# ========= 2) Tải model tham chiếu (có BN, ImageNet) =========
from tensorflow.keras.applications import EfficientNetV2B0
ref_model = EfficientNetV2B0(weights="imagenet", include_top=True)

# ========= 3) Index depthwise và BN theo block =========
def block_tag_from_name(name: str) -> str:
    m = re.match(r'^(block\d+[a-h])_', name)
    return m.group(1) if m else ""

def build_ref_depthwise_bn_index(ref_model):
    ref_layers_by_name = {l.name: l for l in ref_model.layers}
    ref_layers_seq = list(ref_model.layers)
    depthwise_by_block = {}
    for l in ref_layers_seq:
        if isinstance(l, layers.DepthwiseConv2D):
            tag = block_tag_from_name(l.name)
            if tag:
                depthwise_by_block[tag] = l
    bn_after_by_block = {}
    name_to_idx = {l.name: i for i, l in enumerate(ref_layers_seq)}
    for tag, dw in depthwise_by_block.items():
        i = name_to_idx[dw.name]
        bn_layer = None
        for j in range(i+1, min(i+10, len(ref_layers_seq))):
            cand = ref_layers_seq[j]
            if isinstance(cand, layers.BatchNormalization) and block_tag_from_name(cand.name) == tag:
                bn_layer = cand; break
        if bn_layer is None:
            for j in range(i+1, len(ref_layers_seq)):
                cand = ref_layers_seq[j]
                if isinstance(cand, layers.BatchNormalization) and block_tag_from_name(cand.name) == tag:
                    bn_layer = cand; break
        if bn_layer is not None:
            bn_after_by_block[tag] = bn_layer
    return ref_layers_by_name, ref_layers_seq, depthwise_by_block, bn_after_by_block

ref_layers_by_name, ref_layers_seq, depthwise_by_block, bn_after_by_block = build_ref_depthwise_bn_index(ref_model)

# ========= 4) Fuse conv/depthwise với BN =========
def get_conv_weights_with_bias(layer):
    w = layer.get_weights()
    if not w:
        raise ValueError(f"{layer.name} has no weights")
    W = w[0]
    if isinstance(layer, layers.DepthwiseConv2D):
        out_ch = W.shape[2] * W.shape[3]
    else:
        out_ch = W.shape[-1]
    b = w[1] if len(w) >= 2 else np.zeros(out_ch, dtype=W.dtype)
    return W, b

def fuse_conv_or_depthwise_with_bn(conv_layer, bn_layer):
    W, b = get_conv_weights_with_bias(conv_layer)
    gamma, beta, mean, var = bn_layer.get_weights()
    eps = bn_layer.epsilon
    scale = gamma / np.sqrt(var + eps)
    if isinstance(conv_layer, layers.DepthwiseConv2D):
        in_c, depth_mult = W.shape[2], W.shape[3]
        Wf = W * scale.reshape(1,1,in_c,depth_mult)
    else:
        out_c = W.shape[-1]
        Wf = W * scale.reshape(1,1,1,out_c)
    bf = beta + (b - mean) * scale
    return [Wf, bf]

ref_layers = {l.name: l for l in ref_model.layers}
tgt_layers = {l.name: l for l in target_model.layers}

num_fused = num_direct = 0
missing = []

for name, layer in tgt_layers.items():
    if isinstance(layer, layers.DepthwiseConv2D):
        tag = block_tag_from_name(name)
        ref_dw = depthwise_by_block.get(tag, None)
        ref_bn = bn_after_by_block.get(tag, None)
        if ref_dw is not None and ref_bn is not None:
            Wf, bf = fuse_conv_or_depthwise_with_bn(ref_dw, ref_bn)
            layer.set_weights([Wf, bf]); num_fused += 1
        elif ref_dw is not None:
            W, b = get_conv_weights_with_bias(ref_dw)
            layer.set_weights([W, b]); num_direct += 1
        else:
            missing.append((name, f"no depthwise found in ref for {tag}"))
        continue

    if isinstance(layer, layers.Conv2D):
        ref_conv = ref_layers.get(name, None)
        if ('se_reduce' in name) or ('se_expand' in name):
            if isinstance(ref_conv, layers.Conv2D):
                layer.set_weights(ref_conv.get_weights()); num_direct += 1
            else:
                missing.append((name, "SE conv no match"))
            continue

        if isinstance(ref_conv, layers.Conv2D):
            bn_name_guess = name.replace('_conv', '_bn') if name.endswith('_conv') else None
            ref_bn = ref_layers.get(bn_name_guess, None) if bn_name_guess else None
            if ref_bn is None:
                tag = block_tag_from_name(name)
                idx = [i for i,l in enumerate(ref_layers_seq) if l.name == ref_conv.name]
                if idx:
                    start = idx[0]
                    for j in range(start+1, min(start+10, len(ref_layers_seq))):
                        cand = ref_layers_seq[j]
                        if isinstance(cand, layers.BatchNormalization) and block_tag_from_name(cand.name) == tag:
                            ref_bn = cand; break
            if isinstance(ref_bn, layers.BatchNormalization):
                Wf, bf = fuse_conv_or_depthwise_with_bn(ref_conv, ref_bn)
                layer.set_weights([Wf, bf]); num_fused += 1
            else:
                W, b = get_conv_weights_with_bias(ref_conv)
                layer.set_weights([W, b]); num_direct += 1
        else:
            missing.append((name, "no matching ref conv"))
        continue

    if isinstance(layer, layers.Dense) and layer.name == 'predictions':
        ref_dense = ref_layers.get('predictions', None)
        if ref_dense is None:
            ref_dense = [l for l in ref_model.layers if isinstance(l, layers.Dense)][-1]
        layer.set_weights(ref_dense.get_weights()); num_direct += 1
        continue

print(f"Fused convs: {num_fused} | Direct-copied: {num_direct} | Missing: {len(missing)}")
if missing:
    for n, why in missing[:20]:
        print(" -", n, "→", why)
    if len(missing) > 20:
        print(f" ... and {len(missing)-20} more")

# Khóa Normalization layer (mean/std là hằng)
for l in target_model.layers:
    if isinstance(l, layers.Normalization):
        l.trainable = False

# (Tuỳ chọn) lưu trọng số fused
target_model.save_weights("fused_bnless.weights.h5")
print("Saved weights to fused_bnless.weights.h5")

# ========= 5) ĐÁNH GIÁ trên 300 ảnh ImageNet-V2 =========
def get_imagenet_v2_batch(n=300, resize=(224, 224)):
    ds = tfds.load('imagenet_v2', split='test', as_supervised=True)
    ds = ds.take(n).map(
        lambda img, lab: (tf.image.resize(img, resize), lab),
        num_parallel_calls=tf.data.AUTOTUNE
    ).batch(n).prefetch(1)
    imgs, labels = next(iter(ds))
    imgs = tf.cast(imgs, tf.float32)  # [0..255], vì model đã có Rescaling+Normalization
    return imgs, labels

def top1_top5_from_logits(logits, labels_np):
    top1 = (np.argmax(logits, axis=-1) == labels_np).mean()
    top5_idx = np.argpartition(logits, -5, axis=1)[:, -5:]
    top5 = np.mean([lbl in row for lbl, row in zip(labels_np, top5_idx)])
    return float(top1), float(top5)

# chạy đánh giá cho model BN-less đã fuse
imgs, labels = get_imagenet_v2_batch(n=300)
logits = target_model.predict(imgs, verbose=0)
t1, t5 = top1_top5_from_logits(logits, labels.numpy())
print(f"\nFUSED(BN-less) on 300 ImageNet-V2 images → Top-1: {t1*100:.2f}% | Top-5: {t5*100:.2f}%")

# # (Tuỳ chọn) baseline tham chiếu — bật nếu muốn so sánh:
# ref_logits = ref_model.predict(imgs, verbose=0)
# rt1, rt5 = top1_top5_from_logits(ref_logits, labels.numpy())
# print(f"REF(With BN) on same 300 → Top-1: {rt1*100:.2f}% | Top-5: {rt5*100:.2f}%")
# rt1, rt5 = top1_top5_from_logits(ref_logits, labels.numpy())
# print(f"REF(With BN) on same 300 → Top-1: {rt1*100:.2f}% | Top-5: {rt5*100:.2f}%")

onnx_model_path = "efficientnetv2_b0_bnless.onnx"

print("\n[INFO] Đang chuyển đổi model sang ONNX...")

# Định nghĩa input signature (Batch size động hoặc cố định)
# TensorRT trên Jetson thường thích input cố định (ví dụ batch=1) để tối ưu tốt nhất,
# nhưng để linh hoạt ta có thể để None (dynamic).
# Ở đây ta để dynamic batch size để test, input size 224x224x3.
input_signature = [tf.TensorSpec([None, 224, 224, 3], tf.float32, name='input_layer_36')]

# Thực hiện convert
# opset=13 là version ổn định hỗ trợ hầu hết các ops hiện đại
model_proto, _ = tf2onnx.convert.from_keras(
    target_model, 
    input_signature=input_signature, 
    opset=13
)

# Lưu file
onnx.save(model_proto, onnx_model_path)
print(f"[SUCCESS] Model đã được lưu tại: {onnx_model_path}")

# ========= 7) (Tuỳ chọn) KIỂM TRA MODEL ONNX BẰNG ONNXRUNTIME =========
try:
    import onnxruntime as ort
    
    print("[INFO] Đang kiểm tra model ONNX với onnxruntime...")
    
    # Tạo session
    sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    
    # Lấy input name
    input_name = sess.get_inputs()[0].name
    
    # Tạo dummy input (batch=1) giống ảnh thật đã pre-process
    # Lưu ý: img ở bước 5 (get_imagenet_v2_batch) là Tensor, cần chuyển về numpy
    # Lấy 1 ảnh từ batch test bên trên để so sánh
    test_img = imgs[0:1].numpy() 
    
    # Chạy inference ONNX
    onnx_out = sess.run(None, {input_name: test_img})[0]
    
    # Chạy inference TF gốc
    tf_out = target_model.predict(test_img, verbose=0)
    
    # So sánh sai số
    mse = np.mean((onnx_out - tf_out) ** 2)
    print(f"[CHECK] Mean Squared Error giữa TF và ONNX: {mse:.8f}")
    
    if mse < 1e-5:
        print("=> Model ONNX hoạt động chính xác tương đương TensorFlow!")
    else:
        print("=> Cảnh báo: Có sai số lớn giữa TF và ONNX.")

except ImportError:
    print("Chưa cài đặt onnxruntime, bỏ qua bước kiểm tra.")
except Exception as e:
    print(f"Lỗi khi kiểm tra ONNX: {e}")


# ========= 8) CHUYỂN ĐỔI SANG TFLITE INT8 (FULL INTEGER QUANTIZATION) =========
print("\n[INFO] Đang chuẩn bị dữ liệu Calibration cho TFLite INT8...")

# 1. Tạo hàm sinh dữ liệu mẫu (Representative Dataset Generator)
# TFLite cần khoảng 100-300 ảnh thực tế để tính toán phạm vi (range) cho các lớp INT8.
# Ta tận dụng biến 'imgs' (Tensor) đã load ở bước 5.
def representative_data_gen():
    # imgs shape hiện tại: (300, 224, 224, 3), giá trị [0..255] float32
    # Lấy 100 ảnh đầu tiên để làm mẫu
    num_calibration_steps = 100
    for i in range(num_calibration_steps):
        # Lấy 1 ảnh, thêm dim batch -> (1, 224, 224, 3)
        image = tf.expand_dims(imgs[i], 0)
        yield [image]

# 2. Cấu hình Converter
converter = tf.lite.TFLiteConverter.from_keras_model(target_model)

# Bật cờ tối ưu hóa (Optimizations)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Gán hàm sinh dữ liệu mẫu
converter.representative_dataset = representative_data_gen

# Ép buộc tất cả các ops bên trong model phải dùng INT8
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# (Tuỳ chọn) Cấu hình kiểu dữ liệu Input/Output
# - Nếu muốn Input là float32 (nhưng tính toán bên trong là int8) -> Để mặc định.
# - Nếu muốn Input là int8 hoàn toàn (cho một số NPU kén chọn) -> Bỏ comment 2 dòng dưới:
# converter.inference_input_type = tf.int8
# converter.inference_output_type = tf.int8

print("[INFO] Đang convert model sang TFLite INT8 (quá trình này có thể mất vài phút)...")
tflite_quant_model = converter.convert()

# 3. Lưu file
tflite_filename = "efficientnetv2_b0_bnless_int8.tflite"
with open(tflite_filename, 'wb') as f:
    f.write(tflite_quant_model)

print(f"[SUCCESS] Đã lưu model TFLite INT8 tại: {tflite_filename}")
print(f"Kích thước file: {len(tflite_quant_model) / 1024 / 1024:.2f} MB")

# ========= 9) (Tuỳ chọn) KIỂM TRA MODEL TFLITE INT8 =========
try:
    # Khởi tạo interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_filename)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Chuẩn bị input (lấy ảnh đầu tiên từ bộ test)
    # Lưu ý: Nếu converter.inference_input_type = tf.int8 thì phải cast input sang int8
    test_image = tf.expand_dims(imgs[0], 0).numpy()
    
    # Kiểm tra xem đầu vào model yêu cầu float hay int
    if input_details[0]['dtype'] == np.int8:
        # Nếu model yêu cầu int8, ta cần lượng tử hóa input thủ công dựa trên scale/zero_point
        input_scale, input_zero_point = input_details[0]['quantization']
        test_image = test_image / input_scale + input_zero_point
        test_image = test_image.astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], test_image)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"[CHECK] Inference TFLite thành công. Output shape: {output_data.shape}")
    
    # So sánh độ chính xác cơ bản (Argmax)
    tflite_pred = np.argmax(output_data)
    keras_pred = np.argmax(target_model.predict(tf.expand_dims(imgs[0], 0), verbose=0))
    print(f"Prediction - Keras: {keras_pred}, TFLite INT8: {tflite_pred}")
    
except Exception as e:
    print(f"Lỗi kiểm tra TFLite: {e}")