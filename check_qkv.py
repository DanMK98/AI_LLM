"""Compare original and packed-QKV checkpoints without training or saving files."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CausalSelfAttention
from transformer_model import TransformerLanguageModel


class OriginalMultiHeadAttention(nn.Module):
    """The separate-head implementation used before packing QKV."""

    def __init__(self, embedding_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalSelfAttention(embedding_size, embedding_size // num_heads)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        q = torch.stack([head.query(x) for head in self.heads], dim=1)
        k = torch.stack([head.key(x) for head in self.heads], dim=1)
        v = torch.stack([head.value(x) for head in self.heads], dim=1)
        output = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=0.0
        )
        batch, heads, positions, head_size = output.shape
        combined = output.transpose(1, 2).reshape(
            batch, positions, heads * head_size
        )
        return self.projection(combined), None


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original", type=Path, default=root / "attention_experiment/best.pt"
    )
    parser.add_argument(
        "--converted", type=Path,
        default=root / "attention_experiment/best_qkv_weights.pt",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu.")

    original = torch.load(args.original, map_location="cpu", weights_only=True)
    converted = torch.load(args.converted, map_location="cpu", weights_only=True)
    if original["architecture"] != converted["architecture"]:
        raise ValueError("The checkpoints have different architectures.")
    if original["characters"] != converted["characters"]:
        raise ValueError("The checkpoints have different character mappings.")
    architecture = original["architecture"]

    old_model = TransformerLanguageModel(**architecture)
    for block in old_model.blocks:
        block.attention = OriginalMultiHeadAttention(
            architecture["embedding_size"], architecture["num_heads"]
        )
    old_model.load_state_dict(original["model_state"], strict=True)
    new_model = TransformerLanguageModel(**architecture)
    new_model.load_state_dict(converted["model_state"], strict=True)

    old_model.to(device).eval()
    new_model.to(device).eval()
    generator = torch.Generator().manual_seed(42)
    lengths = sorted({1, min(32, architecture["context_size"]), architecture["context_size"]})
    print(f"Comparing full-model outputs on {device}, with dropout disabled.")

    with torch.inference_mode():
        for length in lengths:
            inputs = torch.randint(
                architecture["vocab_size"], (2, length), generator=generator
            ).to(device)
            old_scores, _ = old_model(inputs)
            new_scores, _ = new_model(inputs)
            difference = (new_scores - old_scores).abs().max().item()
            print(f"Context {length}: maximum absolute difference = {difference:.8g}")
            torch.testing.assert_close(
                new_scores, old_scores, rtol=1e-4, atol=1e-5
            )

    print("PASS: original and packed-QKV outputs match within tolerance.")


if __name__ == "__main__":
    main()
