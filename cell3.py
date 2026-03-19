# ===== Convert Keras -> TFLite (FP32) + Quantize INT8 và đánh giá Top-1/Top-5 =====
import numpy as np
import tensorflow as tf
from cell2 import target_model, imgs, labels, top1_top5_from_logits

# 1) Converter ra TFLite FP32
converter_fp32 = tf.lite.TFLiteConverter.from_keras_model(target_model)
tflite_fp32_model = converter_fp32.convert()
with open("bnless_fp32.tflite", "wb") as f:
    f.write(tflite_fp32_model)

# 2) Converter ra TFLite INT8 (full-integer nếu có thể) với representative dataset
def representative_gen():
    # imgs: float32 trong [0..255], đúng miền đầu vào của model (đã có Rescaling+Normalization ở trong graph)
    for i in range(min(128, imgs.shape[0])):  # dùng ~128 mẫu để hiệu chuẩn
        yield [np.expand_dims(imgs[i].numpy() if hasattr(imgs[i], "numpy") else imgs[i], 0).astype(np.float32)]

converter_int8 = tf.lite.TFLiteConverter.from_keras_model(target_model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
converter_int8.representative_dataset = representative_gen
# Yêu cầu full integer (nếu không hỗ trợ sẽ raise, ta fallback động)
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.int8
converter_int8.inference_output_type = tf.int8

tflite_int8_model = None
try:
    tflite_int8_model = converter_int8.convert()
    with open("bnless_int8.tflite", "wb") as f:
        f.write(tflite_int8_model)
    int8_mode = "full-int8"
except Exception as e:
    print("[INT8] Full-integer thất bại, fallback sang dynamic-range (int8 weights, float activations). Lý do:", e)
    # Fallback: dynamic-range (không cần representative, input vẫn float32)
    converter_dr = tf.lite.TFLiteConverter.from_keras_model(target_model)
    converter_dr.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_int8_model = converter_dr.convert()
    with open("bnless_dynamic_range.tflite", "wb") as f:
        f.write(tflite_int8_model)
    int8_mode = "dynamic-range"

# 3) Hàm inference TFLite (tự xử lý int8/float và trả logits float32)
def tflite_predict_all(tflite_bytes, batch):
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()
    input_info  = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    # Đặt shape theo batch
    interpreter.resize_tensor_input(input_info["index"], batch.shape)
    interpreter.allocate_tensors()

    input_info  = interpreter.get_input_details()[0]  # cập nhật lại sau resize
    output_info = interpreter.get_output_details()[0]

    x = batch
    # Nếu input là int8, lượng tử hóa theo (scale, zero_point) của input tensor
    if input_info["dtype"] == np.int8:
        scale, zp = input_info["quantization"]
        xq = np.round(x / scale + zp).clip(-128, 127).astype(np.int8)
        interpreter.set_tensor(input_info["index"], xq)
    else:
        # float32 hoặc float16 input: cứ nạp float32 (TFLite sẽ cast nếu cần)
        interpreter.set_tensor(input_info["index"], x.astype(np.float32))

    interpreter.invoke()
    y = interpreter.get_tensor(output_info["index"])

    # Nếu output là int8 thì giải lượng tử về float
    if output_info["dtype"] == np.int8:
        oscale, ozp = output_info["quantization"]
        y = (y.astype(np.float32) - ozp) * oscale

    return y.astype(np.float32)

# 4) Chạy đánh giá trên chính (imgs, labels) đã nạp ở cell trước
labels_np = labels.numpy() if hasattr(labels, "numpy") else np.array(labels)

# Keras (nếu bạn đã có t1,t5 từ cell trước thì bỏ qua phần này)
try:
    _ = t1; _ = t5
except NameError:
    logits_keras = target_model.predict(imgs, verbose=0)
    t1, t5 = top1_top5_from_logits(logits_keras, labels_np)

# TFLite FP32
logits_tfl_fp32 = tflite_predict_all(tflite_fp32_model, imgs.numpy() if hasattr(imgs, "numpy") else imgs)
t1_fp32, t5_fp32 = top1_top5_from_logits(logits_tfl_fp32, labels_np)

# TFLite INT8 (full-int8 hoặc dynamic-range fallback)
logits_tfl_int8 = tflite_predict_all(tflite_int8_model, imgs.numpy() if hasattr(imgs, "numpy") else imgs)
t1_int8, t5_int8 = top1_top5_from_logits(logits_tfl_int8, labels_np)

print("\n=== Độ chính xác trên cùng 300 ảnh ImageNet-V2 ===")
print(f"[Keras FP32]            Top-1: {t1*100:.2f}% | Top-5: {t5*100:.2f}%")
print(f"[TFLite FP32]           Top-1: {t1_fp32*100:.2f}% | Top-5: {t5_fp32*100:.2f}%  (file: bnless_fp32.tflite)")
print(f"[TFLite {int8_mode}]    Top-1: {t1_int8*100:.2f}% | Top-5: {t5_int8*100:.2f}%  (file: {'bnless_int8.tflite' if int8_mode=='full-int8' else 'bnless_dynamic_range.tflite'})")
