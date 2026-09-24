"""CUDA-aware transformer training with portable, resumable checkpoints."""

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from transformer_model import TransformerLanguageModel

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT / "best_transformer_2blocks.pt"
DEFAULT_ARCHITECTURE = dict(context_size=128, embedding_size=192, num_heads=4, num_layers=3)


def select_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use --device auto or cpu.")
    return torch.device(choice)


def get_batch(source, generator, context_size, batch_size, device):
    starts = torch.randint(len(source) - context_size, (batch_size,), generator=generator)
    positions = starts[:, None] + torch.arange(context_size)
    return source[positions].to(device), source[positions + 1].to(device)


def calculate_loss(model, inputs, targets):
    scores, _ = model(inputs)
    return F.cross_entropy(scores.reshape(-1, scores.size(-1)), targets.reshape(-1))


@torch.no_grad()
def evaluate(model, source, config, device):
    model.eval()
    # Fixed evaluation samples do not advance the training generator.
    generator = torch.Generator().manual_seed(123)
    losses = [
        calculate_loss(
            model,
            *get_batch(
                source, generator, model.context_size, config["batch_size"], device
            ),
        ).item()
        for _ in range(config["eval_batches"])
    ]
    return sum(losses) / len(losses)


def capture_rng(train_rng):
    return dict(
        python=random.getstate(),
        torch_cpu=torch.get_rng_state(),
        torch_cuda=(
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        train=train_rng.get_state(),
    )


def restore_rng(states, train_rng):
    random.setstate(states["python"])
    torch.set_rng_state(states["torch_cpu"])
    train_rng.set_state(states["train"])
    if torch.cuda.is_available() and len(states["torch_cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(states["torch_cuda"])
    elif states["torch_cuda"] or torch.cuda.is_available():
        print("CUDA topology changed; CUDA random streams cannot be restored exactly.")


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--weights",
        type=Path,
        help="Load weights only; defaults to existing two-block checkpoint",
    )
    source.add_argument("--resume", type=Path, help="Restore a full training checkpoint")
    source.add_argument("--scratch", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--steps", type=int, default=10000, help="Additional optimizer steps")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Override Adam learning rate, including on resume; "
            "new runs default to 0.0003"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "transformer_checkpoints")
    args = parser.parse_args(argv)
    if (
        min(args.steps, args.batch_size, args.eval_batches) < 1 
        or (args.eval_every is not None and args.eval_every < 1)
        or (args.learning_rate is not None and args.learning_rate <= 0)
    ):
        parser.error("Step counts, batch sizes, and learning rate must be positive.")
    return args


def main(argv=None):
    args = parse_args(argv)
    device = select_device(args.device)
    source = args.resume or args.weights or (None if args.scratch else LEGACY)
    checkpoint = (
        torch.load(source, map_location="cpu", weights_only=True) if source else None
    )
    if args.resume and not all(
        key in checkpoint
        for key in (
            "optimizer_state",
            "rng_states",
            "architecture",
            "training_config",
            "best_val_loss",
        )
    ):
        raise ValueError("This checkpoint supports weights only; use --weights instead of --resume.")
    latest = args.output_dir.resolve() / "latest.pt"
    best = args.output_dir.resolve() / "best.pt"
    if source and source.resolve() in (latest, best) and not args.resume:
        raise ValueError("Output paths must not overwrite the input weights checkpoint.")
    if not args.resume and (latest.exists() or best.exists()):
        raise ValueError("Output checkpoints already exist. Use --resume or a new --output-dir.")

    random.seed(42)
    torch.manual_seed(42)
    train_rng = torch.Generator().manual_seed(42)

    text = (ROOT / "input.txt").read_text(encoding="utf-8")
    characters = checkpoint["characters"] if checkpoint else sorted(set(text))
    mapping = {char: index for index, char in enumerate(characters)}
    if set(text) - mapping.keys():
        raise ValueError("Input contains characters absent from the checkpoint vocabulary.")
    tokens = [mapping[char] for char in text]
    architecture = DEFAULT_ARCHITECTURE.copy()
    if checkpoint:
        architecture.update(
            checkpoint.get(
                "architecture",
                {
                    key: checkpoint.get(key, value)
                    for key, value in architecture.items()
                },
            )
        )
    architecture["vocab_size"] = len(characters)
    data = torch.tensor(tokens, dtype=torch.long)
    split = int(0.8 * len(data))
    train_data, val_data = data[:split], data[split:]
    if min(len(train_data), len(val_data)) <= architecture["context_size"]:
        raise ValueError(
            "Training and validation data must each exceed the context length."
        )

    config = dict(
        batch_size=args.batch_size,
        eval_every=200 if args.eval_every is None else args.eval_every,
        eval_batches=args.eval_batches,
        learning_rate=0.0003 if args.learning_rate is None else args.learning_rate,
    )
    if args.resume:
        config = checkpoint["training_config"].copy()

        if args.eval_every is not None:
            config["eval_every"] = args.eval_every

        if args.learning_rate is not None:
            config["learning_rate"] = args.learning_rate

    model = TransformerLanguageModel(**architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    step, best_val_loss = 0, float("inf")
    if checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    if args.resume:
        # Adam loads parameter states onto the corresponding parameter device.
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if args.learning_rate is not None:
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate
        step, best_val_loss = checkpoint["step"], checkpoint["best_val_loss"]
        restore_rng(checkpoint["rng_states"], train_rng)
    print(f"Device: {device}; architecture: {architecture}")
    print(f"Starting at step {step}; running {args.steps} additional steps.")
    print(f"Checkpoint: {source}; learning rate: {optimizer.param_groups[0]['lr']}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(val_loss):
        return dict(
            format_version=2,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            step=step,
            architecture=architecture,
            training_config=config,
            rng_states=capture_rng(train_rng),
            characters=characters,
            seed_tokens=tokens[:model.context_size],
            val_loss=val_loss,
            best_val_loss=best_val_loss,
        )

    # Keep the imported model as the baseline best, even if fine-tuning worsens it.
    if not best.exists():
        best_val_loss = evaluate(model, val_data, config, device)
        save_checkpoint(best, snapshot(best_val_loss))
    end_step = step + args.steps
    while step < end_step:
        model.train()
        inputs, targets = get_batch(train_data, train_rng, model.context_size, config["batch_size"], device)
        optimizer.zero_grad(set_to_none=True)
        loss = calculate_loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
        step += 1
        if step % config["eval_every"] == 0 or step == end_step:
            train_loss = evaluate(model, train_data, config, device)
            val_loss = evaluate(model, val_data, config, device)
            improved = val_loss < best_val_loss
            best_val_loss = min(best_val_loss, val_loss)
            payload = snapshot(val_loss)
            if improved:
                save_checkpoint(best, payload)
            save_checkpoint(latest, payload)
            print(f"Step {step}: train = {train_loss:.4f}, validation = {val_loss:.4f}")
    print(f"Saved latest: {latest}\nBest validation loss: {best_val_loss:.4f}; {best}")


if __name__ == "__main__":
    main()
