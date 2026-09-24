"""Shared text encoding and checkpoint serialization for characters and BPE."""

import hashlib


class CharacterTokenizer:
    kind = "character"

    def __init__(self, characters):
        self.characters = list(characters)
        self.mapping = {char: index for index, char in enumerate(characters)}

    @property
    def vocab_size(self):
        return len(self.characters)

    def encode(self, text):
        unknown = set(text) - self.mapping.keys()
        if unknown:
            raise ValueError(f"Characters absent from checkpoint vocabulary: {sorted(unknown)!r}")
        return [self.mapping[char] for char in text]

    def decode(self, token_ids):
        return "".join(self.characters[token] for token in token_ids)

    def checkpoint_fields(self):
        return {"characters": self.characters}


class SubwordTokenizer:
    kind = "byte_level_bpe"

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def load(cls, path):
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    @classmethod
    def from_checkpoint(cls, payload):
        from tokenizers import Tokenizer

        serialized = payload["json"]
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        if digest != payload["sha256"]:
            raise ValueError("Checkpoint tokenizer fingerprint does not match its JSON.")
        return cls(Tokenizer.from_str(serialized))

    def checkpoint_fields(self):
        serialized = self.tokenizer.to_str()
        return {
            "tokenizer": {
                "type": self.kind,
                "json": serialized,
                "sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            }
        }

    @property
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def encode(self, text):
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
        ).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
        )


def tokenizer_from_checkpoint(checkpoint):
    payload = checkpoint.get("tokenizer")
    if payload is None:
        return CharacterTokenizer(checkpoint["characters"])
    if payload.get("type") != SubwordTokenizer.kind:
        raise ValueError(f"Unsupported tokenizer type: {payload.get('type')}")
    return SubwordTokenizer.from_checkpoint(payload)
