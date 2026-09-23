from pathlib import Path

import torch
import torch.nn as nn

from ARCHIVE.tokenizer import characters, tokens
from ARCHIVE.attention_model import AttentionLanguageModel

torch.manual_seed(42)

context_size = 16
embedding_size = 32
head_size = 32
batch_size = 32

data = torch.tensor(tokens, dtype=torch.long)
split = int(0.8 * len(data))
train_data = data[:split]
val_data = data[split:]

assert min(len(train_data), len(val_data)) > context_size

model = AttentionLanguageModel(
    vocab_size=len(characters),
    context_size=context_size,
    embedding_size=embedding_size,
    head_size=head_size,
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
loss_function = nn.CrossEntropyLoss()

# Separate random generators keep evaluation from changing training batches.
train_rng = torch.Generator().manual_seed(42)


def get_batch(source, generator):
    starts = torch.randint(
        len(source) - context_size,
        (batch_size,),
        generator=generator,
    )

    positions = starts[:, None] + torch.arange(context_size)

    inputs = source[positions]
    targets = source[positions + 1]

    return inputs, targets


def calculate_loss(inputs, targets):
    scores, _ = model(inputs)

    return loss_function(
        scores.reshape(-1, len(characters)),
        targets.reshape(-1),
    )


@torch.no_grad()
def evaluate(source):
    model.eval()

    # Reuse the same sampled contexts at every evaluation.
    eval_rng = torch.Generator().manual_seed(123)
    total_loss = 0.0

    for _ in range(50):
        inputs, targets = get_batch(source, eval_rng)
        total_loss += calculate_loss(inputs, targets).item()

    return total_loss / 50


checkpoint_path = Path(__file__).parent / "best_attention.pt"
best_val_loss = float("inf")

for step in range(5000):
    model.train()

    inputs, targets = get_batch(train_data, train_rng)
    loss = calculate_loss(inputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 500 == 0:
        train_loss = evaluate(train_data)
        val_loss = evaluate(val_data)

        print(
            f"Step {step + 1}: "
            f"train = {train_loss:.4f}, "
            f"validation = {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "model_state": model.state_dict(),
                "characters": characters,
                "context_size": context_size,
                "embedding_size": embedding_size,
                "head_size": head_size,
                "seed_tokens": tokens[:context_size],
                "step": step + 1,
                "val_loss": val_loss,
            }, checkpoint_path)

# Generate using the best saved weights.
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)
model.load_state_dict(checkpoint["model_state"])
model.eval()

generated = checkpoint["seed_tokens"].copy()

with torch.no_grad():
    for _ in range(300):
        context = torch.tensor(
            [generated[-context_size:]],
            dtype=torch.long,
        )

        scores, _ = model(context)

        # Only the last position predicts the next character to append.
        probabilities = torch.softmax(scores[0, -1], dim=-1)
        next_id = torch.multinomial(probabilities, 1).item()
        generated.append(next_id)

print(f"\nBest validation loss: {best_val_loss:.4f}")
print("\nGenerated text:")
print("".join(characters[token] for token in generated))