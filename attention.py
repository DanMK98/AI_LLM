import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()

        self.query = nn.Linear(embedding_size, head_size, bias=False)
        self.key = nn.Linear(embedding_size, head_size, bias=False)
        self.value = nn.Linear(embedding_size, head_size, bias=False)

        self.head_size = head_size

    def forward(self, x):
        # x shape: (batch, characters, embedding_size)
        sequence_length = x.shape[1]

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Compare every position's query with every position's key.
        scores = q @ k.transpose(-2, -1)
        scores = scores / (self.head_size ** 0.5)

        # Allow each position to see itself and earlier positions only.
        allowed = torch.tril(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool,
            )
        )

        scores = scores.masked_fill(~allowed, float("-inf"))

        # Each row becomes a probability distribution over positions.
        weights = torch.softmax(scores, dim=-1)

        # Mix the value vectors using those weights.
        output = weights @ v

        return output, weights


# Run this demonstration only when executing attention.py directly.
if __name__ == "__main__":
    torch.manual_seed(42)

    attention = CausalSelfAttention(
        embedding_size=16,
        head_size=8,
    )

    # One sequence containing four example character vectors.
    x = torch.randn(1, 4, 16)

    with torch.no_grad():
        output, weights = attention(x)

    torch.set_printoptions(precision=3, sci_mode=False)

    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Attention weights:")
    print(weights[0])