from pathlib import Path

import torch

from transformer_model import TransformerLanguageModel



source = Path("attention_experiment/best.pt")
destination = Path("attention_experiment/best_qkv_weights.pt")

checkpoint = torch.load(
    source,
    map_location=torch.device("cpu"),
    weights_only=True,
)

architecture = checkpoint["architecture"]
state = checkpoint["model_state"].copy()

for block_index in range(architecture["num_layers"]):
    prefix = f"blocks.{block_index}.attention."


    # Ordering must match qkv.chunk(3):
    # all query heads, then all key heads, then all value heads.
    pieces = []

    for projection in ("query", "key", "value"):
        for head_index in range(architecture["num_heads"]):
            name = (
                f"{prefix}heads.{head_index}.{projection}.weight"
            )
            pieces.append(state.pop(name))

    state[f"{prefix}qkv.weight"] = torch.cat(pieces, dim=0)

# Check that every parameter matches the new architecture.
model = TransformerLanguageModel(**architecture)
model.load_state_dict(state, strict=True)

checkpoint["model_state"] = state

# Old Adam state refers to separate parameters, so don't reuse it.
checkpoint.pop("optimizer_state", None)
checkpoint["attention_layout"] = "packed_qkv"

# Exclusive creation prevents overwriting an existing file.
with destination.open("xb") as file:
    torch.save(checkpoint, file)

print(f"Saved converted weights to {destination}")