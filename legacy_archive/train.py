import torch
import torch.nn as nn
from pathlib import Path
from copy import deepcopy
from ARCHIVE.tokenizer import characters, tokens, id_to_char

torch.manual_seed(42)

context_size = 16
embedding_size = 16
hidden_size = 64
batch_size = 32
vocab_size = len(characters)

data = torch.tensor(tokens, dtype=torch.long)

# First 80% for training, last 20% for validation.
split = int(0.8 * len(data))
train_data = data[:split]
val_data = data[split:]

assert min(len(train_data), len(val_data)) > context_size, (
    "Add more text: both sections need more than 8 characters."
)

# Convert character IDs into vectors, combine their context,
# and produce a score for each possible next character.
model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_size),
    nn.Flatten(start_dim=1),
    nn.Linear(context_size * embedding_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, vocab_size),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_function = nn.CrossEntropyLoss()

def get_batch(source):
    starts = torch.randint(
        len(source) - context_size, (batch_size,)
    )

    inputs = torch.stack([
        source[i.item():i.item() + context_size]
        for i in starts
    ])
    targets = source[starts + context_size]

    return inputs, targets


@torch.no_grad()
def evaluate(source):
    total_loss = 0.0

    for _ in range(100):
        inputs, targets = get_batch(source)
        loss = loss_function(model(inputs), targets)
        total_loss += loss.item()

    return total_loss / 100

checkpoint_path = Path(__file__).parent / "best_model.pt"
best_val_loss = float("inf")
best_weights = None

for step in range(30000):
    model.train()
    inputs, targets = get_batch(train_data)

    loss = loss_function(model(inputs), targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % 500 == 0:
        model.eval()
        train_loss = evaluate(train_data)
        val_loss = evaluate(val_data)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = deepcopy(model.state_dict())

            torch.save({
                "model_state": best_weights,
                "characters": characters,
                "context_size": context_size,
                "embedding_size": embedding_size,
                "hidden_size": hidden_size,
                "seed_tokens": tokens[:context_size],
                "step": step + 1,
                "val_loss": val_loss,
            }, checkpoint_path)

            print("Saved a new best model.")

        print(
            f"Step {step + 1}: "
            f"train = {train_loss:.4f}, "
            f"validation = {val_loss:.4f}"
        )

        if best_weights is not None:
            model.load_state_dict(best_weights)

print(f"Best validation loss: {best_val_loss:.4f}")

# Start generation with the first 8 characters of the file.
generated = tokens[:context_size].copy()

model.eval()

with torch.no_grad():
    for _ in range(200):
        context = torch.tensor(
            [generated[-context_size:]],
            dtype=torch.long,
        )

        scores = model(context)
        probabilities = torch.softmax(scores, dim=-1)
        next_id = torch.multinomial(
            probabilities[0], num_samples=1
        ).item()

        generated.append(next_id)

result = "".join(id_to_char[token] for token in generated)
print("\nGenerated text:")
print(result)