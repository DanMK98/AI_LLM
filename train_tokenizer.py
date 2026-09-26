import argparse
import hashlib
import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from text_tokenizer import SubwordTokenizer


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "finance_bpe_2048",
    )
    args = parser.parse_args()

    if args.vocab_size < 256:
        parser.error("Byte-level BPE needs at least 256 vocabulary entries.")

    if args.min_frequency < 1:
        parser.error("Minimum frequency must be positive.")

    tokenizer_path = args.output_dir / "tokenizer.json"
    metadata_path = args.output_dir / "tokenizer_metadata.json"

    if tokenizer_path.exists() or metadata_path.exists():
        parser.error("Tokenizer files already exist. Use another output directory.")

    # Match the current trainer's text loading and 80/20 split.
    text = (ROOT / "data" / "input.txt").read_text(encoding="utf-8")
    split = int(0.8 * len(text))
    train_text = text[:split]
    val_text = text[split:]

    if not train_text or not val_text:
        parser.error("Both training and validation text must be nonempty.")

    tokenizer = Tokenizer(models.BPE())

    # Preserve case and whitespace; do not add a leading space.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
    )
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=[],
        show_progress=True,
    )

    # Learn vocabulary and merges from training text only.
    tokenizer.train_from_iterator([train_text], trainer=trainer)

    wrapped = SubwordTokenizer(tokenizer)

    for name, portion in (
        ("Training", train_text),
        ("Validation", val_text),
    ):
        ids = wrapped.encode(portion)

        if wrapped.decode(ids) != portion:
            raise RuntimeError(f"{name} text failed the round-trip check.")

        print(
            f"{name}: {len(portion):,} characters -> "
            f"{len(ids):,} tokens "
            f"({len(portion) / len(ids):.2f} characters/token)"
        )

    # Check characters and whitespace beyond the financial examples.
    example = "Interest: 5.25%\n\t€100 — café 😀"
    assert wrapped.decode(wrapped.encode(example)) == example

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))

    # Confirm the saved tokenizer gives identical token IDs.
    restored = SubwordTokenizer.load(tokenizer_path)
    assert restored.encode(example) == wrapped.encode(example)

    metadata = {
        "type": "byte_level_bpe",
        "vocab_size": wrapped.vocab_size,
        "requested_vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "train_split_character_index": split,
        "text_sha256": hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            tokenizer_path.read_bytes()
        ).hexdigest(),
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    sample = "The interest rate affects future cash flows."
    encoded = tokenizer.encode(sample, add_special_tokens=False)

    print(f"\nActual vocabulary size: {wrapped.vocab_size}")
    print(f"Example text: {sample}")
    print(f"Token pieces: {encoded.tokens}")
    print(f"Token IDs: {encoded.ids}")
    print(f"Decoded: {wrapped.decode(encoded.ids)}")
    print(f"\nSaved: {tokenizer_path}")


if __name__ == "__main__":
    main()
