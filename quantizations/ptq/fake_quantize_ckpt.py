import argparse
import os
import shutil
from typing import Iterable, Tuple

import torch

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply uniform fake quantization to all weights in a checkpoint"
    )
    parser.add_argument(
        "ckpt_dir",
        type=str,
        help="Directory containing ckpt.pt and meta.pkl from a previous training run",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to write the quantized checkpoint (defaults to <ckpt_dir>_ptq)",
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=8,
        help="Number of bits for uniform quantization",
    )
    return parser.parse_args()

def fake_quant_tensor(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Uniform quantize then dequantize a tensor."""
    if not torch.is_floating_point(tensor):
        return tensor
    qmax = 2 ** (num_bits - 1) - 1
    if qmax <= 0:
        return tensor
    scale = tensor.abs().max() / qmax
    if scale == 0:
        return tensor
    q = torch.clamp(torch.round(tensor / scale), -qmax - 1, qmax)
    return (q * scale).to(tensor.dtype)


def iter_state_tensors(state_dict) -> Iterable[torch.Tensor]:
    if isinstance(state_dict, torch.nn.Module):
        iterable = state_dict.state_dict().values()
    elif isinstance(state_dict, dict):
        iterable = state_dict.values()
    else:
        iterable = getattr(state_dict, "state_dict", lambda: {})().values()

    for value in iterable:
        if torch.is_tensor(value):
            yield value


def estimate_checkpoint_sizes(state_dict, num_bits: int) -> Tuple[float, float]:
    """Estimate raw and quantized storage requirements for tensors in a state dict."""

    original_bytes = 0.0
    quantized_bytes = 0.0

    for tensor in iter_state_tensors(state_dict):
        numel = tensor.numel()
        elem_bytes = tensor.element_size()
        original_bytes += numel * elem_bytes
        if torch.is_floating_point(tensor):
            quantized_bytes += numel * num_bits / 8.0
        else:
            quantized_bytes += numel * elem_bytes

    return original_bytes, quantized_bytes


def format_size(num_bytes: float) -> str:
    kb = num_bytes / 1024.0
    mb = kb / 1024.0
    gb = mb / 1024.0
    return (
        f"{num_bytes:,.0f} bytes "
        f"({kb:,.2f} KB / {mb:,.2f} MB / {gb:,.4f} GB)"
    )

def main():
    args = parse_args()
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)

    original_bytes, quantized_bytes = estimate_checkpoint_sizes(state_dict, args.num_bits)

    for k, v in state_dict.items():
        if torch.is_tensor(v):
            state_dict[k] = fake_quant_tensor(v, args.num_bits)

    out_dir = args.out_dir or f"{args.ckpt_dir}_ptq"
    os.makedirs(out_dir, exist_ok=True)
    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))

    meta_in = os.path.join(args.ckpt_dir, "meta.pkl")
    meta_out = os.path.join(out_dir, "meta.pkl")
    if os.path.exists(meta_in):
        shutil.copy(meta_in, meta_out)

    print("Quantization summary:")
    print("  Estimated checkpoint size before quantization:")
    print(f"    {format_size(original_bytes)}")
    print("  Estimated checkpoint size after quantization:")
    print(f"    {format_size(quantized_bytes)}")
    if original_bytes > 0:
        if quantized_bytes > 0:
            ratio = original_bytes / quantized_bytes
            pct_of_original = (quantized_bytes / original_bytes) * 100.0
            reduction_pct = 100.0 - pct_of_original
            print(
                "  Estimated compression factor:"
                f" {ratio:.2f}x ({reduction_pct:.2f}% size reduction, "
                f"{pct_of_original:.2f}% of original size)"
            )
        else:
            print("  Estimated compression factor: ∞ (100.00% size reduction)")
    else:
        print("  Estimated compression factor: n/a")

if __name__ == "__main__":
    main()
