"""Short tokenizer integration checks using temporary model checkpoints."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

import generate_transformer as generation
import train_transformer as training
from prepare_context import prepare
from text_tokenizer import CharacterTokenizer, SubwordTokenizer, tokenizer_from_checkpoint


class SubwordTests(unittest.TestCase):
    def test_tokenizer_serialization(self):
        tokenizer = SubwordTokenizer.load(
            training.ROOT / "finance_bpe_2048/tokenizer.json"
        )
        sample = "Interest: 5.25%\n\t€100 — café 😀"
        payload = tokenizer.checkpoint_fields()
        restored = tokenizer_from_checkpoint(payload)
        self.assertEqual(restored.encode(sample), tokenizer.encode(sample))
        self.assertEqual(restored.decode(restored.encode(sample)), sample)
        payload["tokenizer"]["sha256"] = "invalid"
        with self.assertRaises(ValueError):
            tokenizer_from_checkpoint(payload)
        legacy = tokenizer_from_checkpoint({"characters": ["a", "b", " "]})
        self.assertEqual(legacy.decode(legacy.encode("ab ba")), "ab ba")
        with self.assertRaises(ValueError):
            legacy.encode("€")

    def test_bpe_training_resume_generation_and_extension(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            training.main([
                "--scratch", "--tokenizer",
                str(training.ROOT / "finance_bpe_2048/tokenizer.json"),
                "--steps", "1", "--batch-size", "2", "--eval-batches", "1",
                "--eval-every", "1", "--device", "cpu", "--output-dir", str(output),
            ])
            initial = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(initial["architecture"]["vocab_size"], 2048)
            self.assertEqual(initial["step"], 1)
            self.assertNotIn("characters", initial)
            for name in ("best.pt", "latest.pt"):
                saved = torch.load(output / name, weights_only=True)
                self.assertIn("optimizer_state", saved)
                self.assertIn("rng_states", saved)
                tokenizer_from_checkpoint(saved)

            # External tokenizer loading must not be needed after checkpointing.
            with patch.object(SubwordTokenizer, "load", side_effect=AssertionError("External tokenizer used")):
                training.main([
                    "--resume", str(output / "latest.pt"), "--steps", "1",
                    "--device", "cpu", "--output-dir", str(output),
                ])
                generated = io.StringIO()
                with contextlib.redirect_stdout(generated):
                    generation.main([
                        "--checkpoint", str(output / "latest.pt"),
                        "--prompt", "The interest rate", "--length", "3", "--device", "cpu",
                    ])
                self.assertTrue(generated.getvalue().startswith("The interest rate"))
                extended = prepare(
                    output / "latest.pt", output / "extended", context_size=256
                )
                extended_checkpoint = torch.load(extended, weights_only=True)
                self.assertEqual(extended_checkpoint["tokenizer"], initial["tokenizer"])

            resumed = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(resumed["step"], 2)
            self.assertEqual(resumed["tokenizer"], initial["tokenizer"])
            self.assertEqual(resumed["architecture"]["context_size"], 128)

    def test_character_training_and_generation(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            training.main([
                "--scratch", "--steps", "1", "--batch-size", "2",
                "--eval-batches", "1", "--device", "cpu", "--output-dir", directory,
            ])
            checkpoint = torch.load(Path(directory) / "latest.pt", weights_only=True)
            self.assertIsInstance(tokenizer_from_checkpoint(checkpoint), CharacterTokenizer)
            with contextlib.redirect_stdout(io.StringIO()):
                generation.main([
                    "--checkpoint", str(Path(directory) / "latest.pt"),
                    "--prompt", "The", "--length", "2", "--device", "cpu",
                ])


if __name__ == "__main__":
    unittest.main()
