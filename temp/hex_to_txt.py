"""
Convert all .hex files in golden_verilog/ to .txt (same content, new extension).
"""
import shutil
from pathlib import Path

folder = Path("golden_verilog")

hex_files = list(folder.glob("*.hex"))
if not hex_files:
    print("No .hex files found.")
else:
    for hex_file in sorted(hex_files):
        txt_file = hex_file.with_suffix(".txt")
        shutil.copy2(hex_file, txt_file)
        print(f"Copied: {hex_file.name}  ->  {txt_file.name}")
    print(f"\nDone: {len(hex_files)} file(s) converted.")
