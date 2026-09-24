import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import CausalSelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()

        assert embedding_size % num_heads == 0
        head_size = embedding_size // num_heads

        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(embedding_size, head_size)
                for _ in range(num_heads)
            ]
        )

        self.projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        # Each head's existing learned projections are preserved.
        q = torch.stack([head.query(x) for head in self.heads], dim=1)
        k = torch.stack([head.key(x) for head in self.heads], dim=1)
        v = torch.stack([head.value(x) for head in self.heads], dim=1)

        # Shape: [batch, 4 heads, positions, 48 values per head].
        output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,

            # If I wanna add attention dropout, I gotta explicitly disable it during evaluation, otherwise the model will be inconsistent.
            # Code to do it is as follows: dropout_p=0.1 if self.training else 0.0
            dropout_p=0.0,
        )

        # Join the four heads: [batch, positions, 192].
        batch, heads, positions, head_size = output.shape
        combined = output.transpose(1, 2).reshape(
            batch, positions, heads * head_size
        )

        return self.projection(combined), None


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

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        attended, weights = self.attention(self.norm1(x))
        x = x + self.dropout(attended)

        feed_forward_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(feed_forward_output)

        return x, weights


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=128,
        embedding_size=192,
        num_heads=4,
        num_layers=3,
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            vocab_size, embedding_size
        )
        self.position_embedding = nn.Embedding(
            context_size, embedding_size
        )
        self.embedding_dropout = nn.Dropout(0.1)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embedding_size, num_heads)
                for _ in range(num_layers)
            ]
        )
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

        x = self.embedding_dropout(x)

        for block in self.blocks:
            x, weights = block(x)
        scores = self.output_layer(self.final_norm(x))

        return scores, weights
