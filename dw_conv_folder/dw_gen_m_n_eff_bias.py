# ============= Gen m n cho lop depthwise conv theo TFLite PopulateConvolutionQuantizationParams ====================
# Chọn tham số là folder chứa các giá trị scale và bias của layer
# ========================================================================================================================
import argparse
import math
from pathlib import Path
from typing import List, Tuple

# ========================================================================================================================

def tflite_round(x: float) -> int:
    # C++ std::round: half away from zero
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def quantize_multiplier(double_multiplier: float) -> Tuple[int, int]:
    """
    Mirror TensorFlow Lite QuantizeMultiplier() behavior.
    Returns:
      quantized_multiplier (int32), shift (int)
    """
    if double_multiplier == 0.0:
        return 0, 0

    q, shift = math.frexp(double_multiplier)  # double_multiplier = q * 2^shift, q in [0.5, 1)
    q_fixed = tflite_round(q * (1 << 31))

    if q_fixed == (1 << 31):
        q_fixed //= 2
        shift += 1

    # Flush tiny multipliers to zero (same as TFLite safeguard)
    if shift < -31:
        return 0, 0

    # int32 range check
    if q_fixed > 0x7FFFFFFF:
        q_fixed = 0x7FFFFFFF

    return int(q_fixed), int(shift)


def read_single_float(file_path: Path) -> float:
    txt = file_path.read_text(encoding="utf-8").strip()
    return float(txt.splitlines()[0].strip())


def read_float_list(file_path: Path) -> List[float]:
    vals = []
    if not file_path.exists(): return vals
    for line in file_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        vals.append(float(s))
    return vals

def read_int_list(file_path: Path) -> List[int]:
    vals = []
    if not file_path.exists(): return vals
    for line in file_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        vals.append(int(float(s)))
    return vals


def write_list(file_path: Path, values: List, fmt: str = "{}"):
    with file_path.open("w", encoding="utf-8") as f:
        for v in values:
            f.write(fmt.format(v) + "\n")


def gen_m_n(layer_dir: Path):
    ifm_scale = read_single_float(layer_dir / "ifm_scale.txt")
    ofm_scale = read_single_float(layer_dir / "ofm_scale.txt")
    weight_scales = read_float_list(layer_dir / "weight_scale.txt")

    m_list: List[int] = []
    n_list: List[int] = []
    real_mult_list: List[float] = []
    n_right_shift_list: List[int] = []

    for ws in weight_scales:
        real_multiplier = (ifm_scale * ws) / ofm_scale
        m, n = quantize_multiplier(real_multiplier)  # n = shift theo TFLite
        real_mult_list.append(real_multiplier)
        m_list.append(m)
        n_list.append(n)
        n_right_shift_list.append(-n)  # nếu backend cần right shift dương

    # write_list(layer_dir / "real_multiplier.txt", real_mult_list, "{:.18g}")
    write_list(layer_dir / "multiplier.txt", m_list, "{}")
    write_list(layer_dir / "shift.txt", n_list, "{}")
    # write_list(layer_dir / "n_right_shift.txt", n_right_shift_list, "{}")

    # Tính toán effective_bias
    ifm_zp_path = layer_dir / "ifm_zp.txt"
    bias_path = layer_dir / "bias_value.txt"
    weight_path = layer_dir / "weight_values.txt"
    
    if ifm_zp_path.exists() and bias_path.exists() and weight_path.exists():
        ifm_zp = int(read_single_float(ifm_zp_path))
        bias_values = read_int_list(bias_path)
        weight_values = read_int_list(weight_path)
        
        channels_count = len(weight_scales)
        weight_sums = [0] * channels_count
        for i, w in enumerate(weight_values):
            c = i % channels_count
            weight_sums[c] += w
            
        effective_bias = []
        for c in range(channels_count):
            eb = bias_values[c] - ifm_zp * weight_sums[c]
            effective_bias.append(eb)
            
        write_list(layer_dir / "effective_bias.txt", effective_bias, "{}")
        print("     wrote: effective_bias.txt")

    print(f"[OK] Layer dir: {layer_dir}")
    print(f"     ifm_scale={ifm_scale}, ofm_scale={ofm_scale}")
    print(f"     channels={len(weight_scales)}")
    print(f"     wrote: m.txt, n.txt, real_multiplier.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate per-channel m,n for DepthwiseConv2D by TFLite PopulateConvolutionQuantizationParams logic."
    )
    # ========================================================================================================================
    # Chọn tham số là folder chứa các giá trị scale và bias của layer
    parser.add_argument(
        "--layer_dir",
        type=str,
        default=r"layer025_DEPTHWISE_CONV_2D_1_block4a_dwconv2_1_BiasAdd_1_block4a_dwconv2_1_depthwise_1_block4a_dwconv2_1_Squeeze",
        help="Folder chứa các giá trị scale và bias của layer",
    )
    args = parser.parse_args()
    gen_m_n(Path(args.layer_dir))
