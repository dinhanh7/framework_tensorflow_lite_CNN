import os
import re
import argparse

def generate_multi_layer_paths_header(folder_paths, output_file="layer_paths_multi.h", array_size=300):
    header_lines = []
    header_lines.append("#ifndef LAYER_PATHS_H")
    header_lines.append("#define LAYER_PATHS_H\n")
    
    # Regex để tìm số đằng sau chữ "layer"
    pattern = re.compile(r'^layer_?(\d+)_?(.*)$', re.IGNORECASE)
    
    for folder_path in folder_paths:
        if not os.path.isdir(folder_path):
            print(f"Bỏ qua: Thư mục '{folder_path}' không tồn tại.")
            continue
            
        layer_map = {}
        for item_name in os.listdir(folder_path):
            match = pattern.match(item_name)
            if match:
                layer_idx = int(match.group(1))
                folder_name = os.path.basename(os.path.normpath(folder_path))
                # Đường dẫn chuẩn POSIX
                full_path = f"{folder_name}/{item_name}"
                layer_map[layer_idx] = full_path
                
        if not layer_map:
            print(f"Cảnh báo: Không tìm thấy layer nào trong '{folder_path}'.")
            continue
            
        max_idx = max(layer_map.keys())
        actual_array_size = max(array_size, max_idx + 1)
        
        # Tạo tên biến array từ tên thư mục (Viết hoa, thay ký tự đặc biệt bằng '_')
        array_name = re.sub(r'[^a-zA-Z0-9_]', '_', os.path.basename(os.path.normpath(folder_path))).upper()
        
        header_lines.append(f"static const char* {array_name}[{actual_array_size}] = {{")
        for idx in sorted(layer_map.keys()):
            header_lines.append(f'    [{idx}] = "{layer_map[idx]}",')
        header_lines.append("};\n")

    header_lines.append("#endif\n")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(header_lines))
        
    print(f"Đã tạo thành công '{output_file}' với mảng từ các thư mục: {', '.join(folder_paths)}") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate layer_paths.h from multiple directories")
    parser.add_argument("folders", type=str, nargs='+', help="Danh sách các thư mục (cách nhau bởi dấu cách)")
    parser.add_argument("--output", type=str, default="layer_paths_multi.h", help="Tên file sinh ra")
    parser.add_argument("--size", type=int, default=300, help="Kích thước tối thiểu của mảng")
    
    args = parser.parse_args()
    generate_multi_layer_paths_header(args.folders, args.output, args.size)
