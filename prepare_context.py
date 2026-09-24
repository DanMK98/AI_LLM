"""Extend a checkpoint's context length without running training."""

import argparse
import hashlib
import random
from pathlib import Path

import torch

from train_transformer import ROOT, capture_rng, save_checkpoint
from transformer_model import TransformerLanguageModel
from text_tokenizer import tokenizer_from_checkpoint


def prepare(source, output_dir, seed=42, context_size=64):
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    if output_dir == source.parent:
        raise ValueError("The context experiment requires a separate output directory.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new experiment directory.")
    original = torch.load(source, map_location="cpu", weights_only=True)
    architecture = original.get("architecture")
    if architecture is None:
        architecture = {
            key: original[key]
            for key in ("context_size", "embedding_size", "num_heads", "num_layers")
        }
    tokenizer = tokenizer_from_checkpoint(original)
    architecture = dict(architecture, vocab_size=tokenizer.vocab_size)
    old_context_size = architecture["context_size"]
    if context_size <= old_context_size:
        raise ValueError("Target context must be larger than the source context.")
    architecture["context_size"] = context_size
    random.seed(seed)
    torch.manual_seed(seed)
    model = TransformerLanguageModel(**architecture)
    weights = original["model_state"].copy()
    # nn.Embedding initializes all rows with N(0, 1). Keep that initialization
    # for new rows and overwrite the original rows with the trained embeddings.
    positions = model.position_embedding.weight.detach().clone()
    if weights["position_embedding.weight"].shape != (
        old_context_size, architecture["embedding_size"]
    ):
        raise ValueError("Source position embeddings do not match its architecture.")
    positions[:old_context_size].copy_(weights["position_embedding.weight"])
    weights["position_embedding.weight"] = positions
    model.load_state_dict(weights, strict=True)
    config = dict(batch_size=32, eval_every=1000, eval_batches=50, learning_rate=0.0003)
    config.update(original.get("training_config", {}))
    # Deliberately do not load any source optimizer, step, loss, or RNG state.
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    train_rng = torch.Generator().manual_seed(seed)
    payload = dict(
        format_version=3,
        model_state=model.state_dict(),
        optimizer_state=optimizer.state_dict(),
        step=0,
        architecture=architecture,
        training_config=config,
        rng_states=capture_rng(train_rng),
        **tokenizer.checkpoint_fields(),
        seed_tokens=list(original["seed_tokens"]),
        val_loss=None,
        best_val_loss=float("inf"),
        initialization=dict(
            source=str(source),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_step=original.get("step"),
            seed=seed,
            new_position_rows=(
                f"{old_context_size}:{context_size}, normal(mean=0, std=1)"
            ),
        ),
    )
    if "text_sha256" in original:
        payload["text_sha256"] = original["text_sha256"]
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "initial.pt"
    save_checkpoint(path, payload)
    print(f"Prepared {path}\nArchitecture: {architecture}")
    print(
        f"Fresh Adam optimizer; learning rate: {config['learning_rate']}; "
        "step: 0. No training performed."
    )
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "transformer_finetune" / "best.pt"
    )
    parser.add_argument("--context-size", type=int, default=64)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / f"transformer_context{args.context_size}"
    prepare(args.source, output_dir, args.seed, args.context_size)


if __name__ == "__main__":
    main()
