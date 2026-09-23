"""Sample text from a transformer checkpoint without training or changing it."""

import argparse
import math
from pathlib import Path

import torch

from transformer_model import TransformerLanguageModel


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path(__file__).resolve().parent / "transformer_fresh_context128" / "best.pt"
        ),
    )
    parser.add_argument("--prompt", help="Starting text; defaults to checkpoint seed text")
    parser.add_argument("--length", type=int, default=500, help="Number of new characters")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args(argv)
    if args.length < 0 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Length must be nonnegative and temperature must be finite and positive.")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or auto.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    characters = checkpoint["characters"]
    architecture = checkpoint.get("architecture")
    if architecture is None:
        architecture = {
            key: checkpoint[key]
            for key in ("context_size", "embedding_size", "num_heads", "num_layers")
        }
    architecture = dict(architecture, vocab_size=len(characters))
    mapping = {char: index for index, char in enumerate(characters)}
    if args.prompt is not None:
        unknown = set(args.prompt) - mapping.keys()
        if unknown:
            parser.error(f"Prompt contains characters outside the vocabulary: {sorted(unknown)!r}" )
        generated = [mapping[char] for char in args.prompt]
    else:
        generated = list(checkpoint["seed_tokens"])
    if not generated:
        parser.error("Provide a nonempty prompt.")
    model = TransformerLanguageModel(**architecture).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    generator = torch.Generator(device=device).manual_seed(args.seed)
    with torch.inference_mode():
        for _ in range(args.length):
            context = torch.tensor([generated[-model.context_size:]], dtype=torch.long, device=device)
            scores, _ = model(context)
            probabilities = torch.softmax(scores[0, -1] / args.temperature, dim=-1)
            generated.append(torch.multinomial(probabilities, 1, generator=generator).item())
    print("".join(characters[token] for token in generated))


if __name__ == "__main__":
    main()
