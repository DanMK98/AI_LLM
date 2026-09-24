import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super().__init__()

        assert embedding_size % num_heads == 0

        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        # Produce all queries, then all keys, then all values.
        self.qkv = nn.Linear(
            embedding_size,
            3 * embedding_size,
            bias=False,
        )

        self.projection = nn.Linear(embedding_size, embedding_size)

    def forward(self, x):
        batch, positions, embedding_size = x.shape

        # [batch, positions, 3 * embedding_size]
        qkv = self.qkv(x)

        # Each becomes [batch, positions, embedding_size].
        q, k, v = qkv.chunk(3, dim=-1)

        # Each becomes [batch, heads, positions, head_size].
        q = q.reshape(
            batch, positions, self.num_heads, self.head_size
        ).transpose(1, 2)

        k = k.reshape(
            batch, positions, self.num_heads, self.head_size
        ).transpose(1, 2)

        v = v.reshape(
            batch, positions, self.num_heads, self.head_size
        ).transpose(1, 2)

        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=0.0,
        )

        combined = output.transpose(1, 2).reshape(
            batch, positions, embedding_size
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
