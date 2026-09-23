from pathlib import Path
import torch
import torch.nn as nn

checkpoint_path = Path(__file__).parent / "best_model.pt"
checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True,
)

characters = checkpoint["characters"]
context_size = checkpoint["context_size"]
embedding_size = checkpoint["embedding_size"]
hidden_size = checkpoint["hidden_size"]
vocab_size = len(characters)

# Recreate the same architecture used during training.
model = nn.Sequential(
    nn.Embedding(vocab_size, embedding_size),
    nn.Flatten(start_dim=1),
    nn.Linear(context_size * embedding_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, vocab_size),
)

# Restore the learned weights.
model.load_state_dict(checkpoint["model_state"])
model.eval()

generated = checkpoint["seed_tokens"].copy()

with torch.no_grad():
    for _ in range(500):
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

print("".join(characters[token] for token in generated))