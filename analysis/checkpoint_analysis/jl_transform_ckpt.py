import argparse
import os
import shutil
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply a Johnson-Lindenstrauss transform to all weights in a checkpoint"
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
        help="Directory to write the transformed checkpoint (defaults to <ckpt_dir>_jl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for the JL transform",
    )
    return parser.parse_args()


def jl_transform_tensor(tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Apply a simple JL style sign flip transform."""
    rnd = torch.randint(0, 2, tensor.shape, generator=generator, device=tensor.device)
    rnd = rnd * 2 - 1
    return tensor * rnd


def main():
    args = parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    meta_path = os.path.join(args.ckpt_dir, "meta.pkl")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    g = torch.Generator()
    g.manual_seed(args.seed)

    for key, tensor in list(state_dict.items()):
        if torch.is_floating_point(tensor):
            state_dict[key] = jl_transform_tensor(tensor, g)

    out_dir = args.out_dir or f"{args.ckpt_dir.rstrip('/').rstrip(os.sep)}_jl"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(out_dir, "meta.pkl"))


if __name__ == "__main__":
    main()
