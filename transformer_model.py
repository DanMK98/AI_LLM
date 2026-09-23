import torch
import torch.nn as nn

from attention import CausalSelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()

        assert embedding_size % num_heads == 0
        head_size = embedding_size // num_heads

        self.heads = nn.ModuleList([
            CausalSelfAttention(embedding_size, head_size)
            for _ in range(num_heads)
        ])

        self.projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        results = [head(x) for head in self.heads]

        # Join the information gathered by each head.
        combined = torch.cat(
            [output for output, weights in results],
            dim=-1,
        )

        weights = torch.stack(
            [weights for output, weights in results],
            dim=1,
        )

        return self.projection(combined), weights


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()

        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.attention = MultiHeadAttention(
            embedding_size, num_heads
        )

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, 4 * embedding_size),
            nn.ReLU(),
            nn.Linear(4 * embedding_size, embedding_size),
        )

    def forward(self, x):
        attended, weights = self.attention(self.norm1(x))
        x = x + attended

        x = x + self.feed_forward(self.norm2(x))

        return x, weights


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=16,
        embedding_size=32,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_size
        )
        self.position_embedding = nn.Embedding(
            context_size, embedding_size
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_size, num_heads)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Linear(embedding_size, vocab_size)

    def forward(self, token_ids):
        sequence_length = token_ids.shape[1]
        assert sequence_length <= self.context_size

        positions = torch.arange(
            sequence_length, device=token_ids.device
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x, weights = block(x)
        scores = self.output_layer(self.final_norm(x))

        return scores, weights
