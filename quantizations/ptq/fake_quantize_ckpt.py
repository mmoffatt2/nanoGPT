import argparse
import os
import shutil
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

def main():
    args = parse_args()
    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model", checkpoint)

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

if __name__ == "__main__":
    main()
