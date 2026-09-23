import torch
import torch.nn as nn
from attention import CausalSelfAttention


class AttentionLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_size=16,
        embedding_size=32,
        head_size=32,
    ):
        super().__init__()

        self.context_size = context_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(context_size, embedding_size)

        self.attention = CausalSelfAttention(embedding_size, head_size)
        self.output_layer = nn.Linear(head_size, vocab_size)

    def forward(self, token_ids):
        # token_ids: (batch, sequence_length)
        sequence_length = token_ids.shape[1]
        assert sequence_length <= self.context_size

        positions = torch.arange(
            sequence_length,
            device=token_ids.device,
        )

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        attended, weights = self.attention(x)
        scores = self.output_layer(attended)

        return scores, weights


if __name__ == "__main__":
    from ARCHIVE.tokenizer import characters, tokens

    torch.manual_seed(42)

    context_size = 16
    assert len(tokens) > context_size

    model = AttentionLanguageModel(
        vocab_size=len(characters),
        context_size=context_size,
    )

    # Targets are shifted one character ahead.
    inputs = torch.tensor([tokens[:context_size]], dtype=torch.long)
    targets = torch.tensor([tokens[1:context_size + 1]], dtype=torch.long)

    scores, weights = model(inputs)

    loss = nn.CrossEntropyLoss()(
        scores.reshape(-1, len(characters)),
        targets.reshape(-1),
    )

    print("Input shape:", inputs.shape)
    print("Scores shape:", scores.shape)
    print("Attention shape:", weights.shape)
    print("Initial loss:", loss.item())