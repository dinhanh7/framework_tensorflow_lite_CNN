import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# --- CẤU HÌNH ---
TFLITE_MODEL_PATH = r"cell_hswish_hsigmoid\efficientnetv2_b0_bnless_int8.tflite"
NUM_IMAGES = 3000  # Số lượng ảnh bạn muốn kiểm tra (giống với số lượng bạn xuất ra C)

print("="*60)
print("🚀 ĐÁNH GIÁ ACCURACY GỐC CỦA MÔ HÌNH TFLITE")
print("="*60)

print("[*] Khởi tạo TFLite Interpreter...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0] # Lấy Output layer cuối cùng (Classifier)

input_dtype = input_details['dtype']
input_scale, input_zp = input_details['quantization']

print(f"[*] Shape Input: {input_details['shape']}")
print(f"[*] Shape Output: {output_details['shape']}")

print(f"\n[*] Đang load tập dữ liệu ImageNet V2 (lấy {NUM_IMAGES} ảnh đầu tiên)...")
# shuffle_files=False để đảm bảo thứ tự ảnh y hệt như lúc xuất ra cho C
ds = tfds.load('imagenet_v2', split='test', shuffle_files=False)

correct_count = 0
total_count = 0

for i, example in enumerate(ds.take(NUM_IMAGES)):
    img_tensor = example['image']
    true_label = example['label'].numpy()

    # 1. TIỀN XỬ LÝ ẢNH (Phải giống hệt lúc bạn đưa cho C)
    img_resized = tf.image.resize(img_tensor, [224, 224]).numpy()
    
    if input_scale > 0:
        input_data = (img_resized / input_scale) + input_zp
        if input_dtype == np.int8:
            input_data = np.clip(np.round(input_data), -128, 127)
        elif input_dtype == np.uint8:
            input_data = np.clip(np.round(input_data), 0, 255)
        input_data = input_data.astype(input_dtype)
    else:
        input_data = img_resized.astype(input_dtype)

    input_data = np.expand_dims(input_data, axis=0)

    # 2. CHẠY MÔ HÌNH GỐC
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()

    # 3. LẤY KẾT QUẢ TỪ LỚP OUTPUT CUỐI CÙNG
    output_data = interpreter.get_tensor(output_details['index'])[0]
    
    # Xử lý vụ lệch 1 class (1001 vs 1000)
    # Một số mô hình ImageNet có 1001 class (Class 0 là Background/Nền)
    # Nếu output_data có 1001 phần tử, ta lấy argmax từ index 1 trở đi để khớp nhãn 0-999 của TFDS
    if len(output_data) == 1001:
        pred_label = np.argmax(output_data[1:]) 
    else:
        pred_label = np.argmax(output_data)

    # 4. SO SÁNH VỚI NHÃN THẬT
    if pred_label == true_label:
        correct_count += 1
    total_count += 1

    # In tiến độ cho đỡ sốt ruột
    if (i + 1) % 50 == 0:
        print(f" -> Đã chạy {i + 1}/{NUM_IMAGES} ảnh | Acc tạm tính: {(correct_count/total_count)*100:.2f}%")

print("\n" + "="*60)
print("🏆 KẾT QUẢ BASELINE TFLITE CHÍNH THỨC 🏆")
print(f"Top-1 Accuracy: {correct_count}/{total_count} ({(correct_count/total_count)*100:.2f}%)")
print("="*60)