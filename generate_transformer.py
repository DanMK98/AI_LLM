"""Sample text from a transformer checkpoint without training or changing it."""

import argparse
import math
from pathlib import Path

import torch

from transformer_model import TransformerLanguageModel
from text_tokenizer import tokenizer_from_checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            Path(__file__).resolve().parent / "experiments" / "finance_bpe_2048" / "best.pt"
        ),
    )
    parser.add_argument("--prompt", help="Starting text; defaults to checkpoint seed text")
    parser.add_argument("--length", type=int, default=500, help="New tokens (characters for character checkpoints, subwords for BPE)")
    parser.add_argument("--temperature", type=float, default=0.65, help="Higher temperature means more randomness")
    parser.add_argument("--top_k", type=int, default=0, help="0 for no top-k filtering")
    parser.add_argument("--top_p", type=float, default=0.9, help="0.0 for no top-p filtering")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device to run on")
    args = parser.parse_args(argv)
    if not math.isfinite(args.top_p) or not (0.0 <= args.top_p <= 1.0):
        parser.error("Top-p must be a finite number between 0.0 and 1.0.")
    if args.length < 0 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("Length must be nonnegative and temperature must be finite and positive.")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu or auto.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = tokenizer_from_checkpoint(checkpoint)
    architecture = checkpoint.get("architecture")
    if architecture is None:
        architecture = {
            key: checkpoint[key]
            for key in ("context_size", "embedding_size", "num_heads", "num_layers")
        }
    if architecture.get("vocab_size", tokenizer.vocab_size) != tokenizer.vocab_size:
        parser.error("Checkpoint architecture and tokenizer vocabulary sizes differ.")
    architecture = dict(architecture, vocab_size=tokenizer.vocab_size)
    if args.prompt is not None:
        try:
            generated = tokenizer.encode(args.prompt)
        except ValueError as error:
            parser.error(str(error))
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
            if args.top_k > 0:
                k = min(args.top_k, probabilities.size(-1))
                indices_to_remove = probabilities < torch.topk(probabilities, k)[0][..., -1, None]
                probabilities = probabilities.masked_fill(indices_to_remove, 0.0)
            if args.top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                indices_to_remove = cumulative_probs > args.top_p
                indices_to_remove[..., 1:] = indices_to_remove[..., :-1].clone()
                indices_to_remove[..., 0] = False
                sorted_probs = sorted_probs.masked_fill(indices_to_remove, 0.0)
                probabilities = torch.zeros_like(probabilities).scatter_(dim=-1, index=sorted_indices, src=sorted_probs)
            generated.append(torch.multinomial(probabilities, 1, generator=generator).item())
    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
