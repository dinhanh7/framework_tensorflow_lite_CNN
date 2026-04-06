# TensorFlow Lite CNN in C Framework

## Mô tả
Project này triển khai một số các op trong tflite bằng C, tạo 1 đoạn model sequence và đánh giá acc của model sequence.

## Mô tả cấu trúc dự án
```
.
|---layer/                 # Define các hàm mô phỏng các op trong tflite
|---test/                  # Chứa code để test các hàm được build trong layer, được giữ lại để tham khảo
|---test_c_model_tool_old/ # Cách cũ để test acc model c
|---test_c_model_fast/     # Cách test hiện tại, user manual sẽ được update khi tool hoàn thiện
|---cell_hswish/           # Chứa code python define model với hswish
|---cell_hswish_hsigmoid/  # Chứa code python define model với hswish + hsigmoid
|---reference_python_code/ # Code python để ref của anh Đình Anh
|---tf*lite**/             # Chứa model tflite mới
|
```
Các file không được liệt kê trên là các script python chủ yếu dùng 1 lần sẽ bị xoá trong tương lai
## User manual
Update khi hoàn thành
## Cấu hình Quantization
weight, ifm, ofm: int8
zero-point: int8
scale: fp32
effective_bias: int32
hệ số M: int32
hệ số n (shift): int8


Thư viện tensorflow 2.15:
https://github.com/tensorflow/tensorflow/tree/r2.15