import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# --- CẤU HÌNH ĐƯỜNG DẪN ---
TFLITE_MODEL_PATH = "cell_hswish_hsigmoid/efficientnetv2_b0_bnless_int8.tflite" 
OUTPUT_DIR = "preprocessed_layer5_txt/"             
LABEL_FILE = "ground_truth_labels.json" # File lưu nhãn thật để script 4 đọc
NUM_IMAGES = 300

# THAY BẰNG INDEX CỦA LAYER 5 INPUT TỪ MODEL CỦA BẠN
LAYER_5_INPUT_INDEX = 201

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("[*] Đang load mô hình TFLite...")
interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL_PATH,
    experimental_preserve_all_tensors=True 
)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()[0]
input_dtype = input_details['dtype']

print("[*] Đang load tập ImageNet2012 (Validation) từ TFDS...")
# Tự động tải khoảng 1.2GB, không cần manual_dir
ds = tfds.load('imagenet_v2', split='test', shuffle_files=False)

ground_truth_dict = {}

print(f"[*] Bắt đầu trích xuất tensor cho {NUM_IMAGES} ảnh...")
for i, example in enumerate(ds.take(NUM_IMAGES)):
    img_tensor = example['image']
    label = example['label'].numpy() # Nhãn gốc từ 0 - 999
    
    # Resize ảnh về 224x224 (Dùng bilinear mặc định của TF)
    img_resized = tf.image.resize(img_tensor, [224, 224])
    
    # Cast về đúng kiểu dữ liệu input của TFLite model (thường là uint8 hoặc int8)
    # Nếu TFLite model yêu cầu input normalize (VD int8 từ -128 đến 127), bạn phải trừ 128 ở đây
    input_data = tf.cast(img_resized, input_dtype).numpy()
    input_data = np.expand_dims(input_data, axis=0)
    
    # Chạy inference
    interpreter.set_tensor(input_details['index'], input_data)
    interpreter.invoke()
    
    # Trích xuất Layer 5 IFM
    layer_5_ifm = interpreter.get_tensor(LAYER_5_INPUT_INDEX)
    
    # Định dạng tên file: img_0000.txt, img_0001.txt...
    filename = f"img_{i:04d}"
    out_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    
    # Lưu tensor ra file txt
    layer_5_ifm.flatten().tofile(out_path, sep="\n", format="%d")
    
    # Lưu nhãn vào dict
    ground_truth_dict[filename] = int(label)
    
    if (i + 1) % 50 == 0:
        print(f"  -> Đã xử lý {i + 1}/{NUM_IMAGES} ảnh")

# Lưu từ điển nhãn ra file JSON
with open(LABEL_FILE, 'w') as f:
    json.dump(ground_truth_dict, f, indent=4)

print(f"[✓] Hoàn thành! Đã lưu file C input tại '{OUTPUT_DIR}' và nhãn tại '{LABEL_FILE}'")