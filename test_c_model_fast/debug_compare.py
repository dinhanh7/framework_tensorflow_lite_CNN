import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# --- CẤU HÌNH ---
TFLITE_MODEL_PATH = r"cell_hswish_hsigmoid\efficientnetv2_b0_bnless_int8.tflite"
MEAN_TENSOR_INDEX = 475 

# SỐ LƯỢNG ẢNH MUỐN TEST (Ví dụ: 3 sẽ chạy img_0000, img_0001, img_0002)
NUM_IMAGES_TO_CHECK = 30

# =========================================================================
# 1. KHỞI TẠO TFLITE INTERPRETER (Chỉ cần làm 1 lần)
# =========================================================================
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH, experimental_preserve_all_tensors=True)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
input_dtype = input_details['dtype']
input_scale, input_zp = input_details['quantization']

print(f"[*] Đang load {NUM_IMAGES_TO_CHECK} ảnh đầu tiên từ tập dữ liệu...")
ds = tfds.load('imagenet_v2', split='test', shuffle_files=False)

# =========================================================================
# 2. VÒNG LẶP QUA TỪNG ẢNH
# =========================================================================
# Dùng enumerate để đếm số thứ tự i (0, 1, 2...) tương ứng với img_0000, img_0001...
for i, example in enumerate(ds.take(NUM_IMAGES_TO_CHECK)):
    img_name = f"img_{i:04d}"
    print(f"\n" + "="*50)
    print(f"[*] ĐANG XỬ LÝ ẢNH: {img_name}")
    print("="*50)

    # 1. ĐỌC KẾT QUẢ TỪ C
    c_out_file = f"c_outputs/out278_{img_name}.txt"
    if not os.path.exists(c_out_file):
        print(f"[CẢNH BÁO] Không tìm thấy file {c_out_file}. Bỏ qua ảnh này!")
        continue

    with open(c_out_file, "r") as f:
        c_out = np.array([int(line.strip()) for line in f.readlines()], dtype=np.int8)

    # 2. TIỀN XỬ LÝ ẢNH CHO TFLITE
    img_tensor = example['image']
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

    # 3. CHẠY TFLITE ĐỂ LẤY GOLDEN OUTPUT
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    tflite_out = interpreter.get_tensor(MEAN_TENSOR_INDEX).flatten()

    # 4. SO SÁNH
    print(f"C Output (10 số đầu)     : {c_out[:10]}")
    print(f"TFLite Output (10 số đầu): {tflite_out[:10]}")

    diff = np.abs(c_out.astype(np.int32) - tflite_out.astype(np.int32))
    max_diff = np.max(diff)
    print(f"\n-> Sai số lớn nhất (Max Diff): {max_diff}")
    print(f"-> Số lượng phần tử sai lệch > 2: {np.sum(diff > 2)} / {len(diff)}")

    # 5. VẼ VÀ LƯU ĐỒ THỊ (Mỗi ảnh một đồ thị riêng)
    plt.figure(figsize=(12, 6))
    plt.plot(diff, label=f'Absolute Difference |C - TFLite| ({img_name})', color='red', alpha=0.8)
    plt.title(f'Absolute Difference (Layer 278) - {img_name}')
    plt.xlabel('Index')
    plt.ylabel('Absolute Difference')
    plt.legend()
    plt.grid(True)
    
    plot_filename = f'abs_diff_plot_layer_278_{img_name}.png'
    plt.savefig(plot_filename)
    plt.close() # Đóng plot lại để giải phóng bộ nhớ cho ảnh tiếp theo
    
    print(f"[✓] Đã lưu đồ thị sai số thành file '{plot_filename}'")