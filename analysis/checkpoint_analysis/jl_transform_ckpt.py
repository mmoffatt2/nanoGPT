import argparse
import os
import shutil
import math
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
        "--out_embd",
        type=int,
        required=True,
        help="Embedding dimension of the output checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for the JL transform",
    )
    return parser.parse_args()


def sign_matrix(out_dim: int, in_dim: int, generator: torch.Generator, device) -> torch.Tensor:
    """Create a sign-based JL projection matrix of shape (out_dim, in_dim)."""
    mat = torch.randint(0, 2, (out_dim, in_dim), generator=generator, device=device, dtype=torch.float32)
    mat = mat * 2 - 1
    mat /= math.sqrt(out_dim)
    return mat


def jl_project_tensor(tensor: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
    """Project `tensor` along dimensions matching ``proj.shape[1]``."""
    in_dim = proj.shape[1]

    # square matrices that map n_embd->n_embd
    if tensor.ndim == 2 and tensor.shape[0] == in_dim and tensor.shape[1] == in_dim:
        return proj @ tensor @ proj.t()

    # last dimension corresponds to embeddings (e.g. Linear weight or embedding table)
    if tensor.shape[-1] == in_dim:
        tensor = tensor @ proj.t()

    # first dimension could also be embeddings (1D bias or weight matrices where out_features == n_embd)
    if tensor.ndim > 1 and tensor.shape[0] == in_dim:
        tensor = proj @ tensor
    elif tensor.ndim == 1 and tensor.shape[0] == in_dim:
        tensor = (proj @ tensor.unsqueeze(-1)).squeeze(-1)

    return tensor


def main():
    args = parse_args()

    ckpt_path = os.path.join(args.ckpt_dir, "ckpt.pt")
    meta_path = os.path.join(args.ckpt_dir, "meta.pkl")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)

    g = torch.Generator()
    g.manual_seed(args.seed)

    old_embd = checkpoint.get("model_args", {}).get("n_embd")
    if old_embd is None:
        raise ValueError("Could not determine n_embd from checkpoint")

    n_head = checkpoint.get("model_args", {}).get("n_head")
    if n_head is not None:
        old_head_dim = old_embd // n_head
        new_head_dim = args.out_embd // n_head
        if new_head_dim != old_head_dim:
            raise ValueError(
                "out_embd would change per-head dimension; choose a value that keeps n_embd//n_head constant"
            )

    proj = sign_matrix(args.out_embd, old_embd, g, device="cpu")

    for key, tensor in list(state_dict.items()):
        if torch.is_floating_point(tensor):
            state_dict[key] = jl_project_tensor(tensor, proj)

    if "model_args" in checkpoint:
        checkpoint["model_args"]["n_embd"] = args.out_embd
        if (
            "n_embd_wte" in checkpoint["model_args"]
            and checkpoint["model_args"]["n_embd_wte"] == old_embd
        ):
            checkpoint["model_args"]["n_embd_wte"] = args.out_embd
    if "config" in checkpoint and "n_embd" in checkpoint["config"]:
        checkpoint["config"]["n_embd"] = args.out_embd

    out_dir = args.out_dir or f"{args.ckpt_dir.rstrip('/').rstrip(os.sep)}_jl"
    os.makedirs(out_dir, exist_ok=True)

    torch.save(checkpoint, os.path.join(out_dir, "ckpt.pt"))
    if os.path.exists(meta_path):
        shutil.copy2(meta_path, os.path.join(out_dir, "meta.pkl"))


if __name__ == "__main__":
    main()
