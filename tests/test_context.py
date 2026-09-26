import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from prepare_context import prepare
from train_transformer import ROOT, main as train
from transformer_model import TransformerLanguageModel


class ContextExtensionTests(unittest.TestCase):
    def test_context128(self):
        torch.set_num_threads(1)
        source = ROOT / "experiments" / "transformer_context64_refine" / "best.pt"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        original = torch.load(source, map_location="cpu", weights_only=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "context128"
            initial = prepare(source, output, context_size=128)
            converted = torch.load(initial, weights_only=True)
            self.assertEqual(
                converted["architecture"],
                dict(original["architecture"], context_size=128),
            )
            self.assertEqual(converted["optimizer_state"]["state"], {})
            self.assertEqual(converted["step"], 0)
            for key, expected in original["model_state"].items():
                actual = converted["model_state"][key]
                if key == "position_embedding.weight":
                    self.assertEqual(actual.shape, (128, 192))
                    self.assertTrue(torch.isfinite(actual[64:]).all())
                    self.assertGreater(actual[64:].std().item(), 0)
                    actual = actual[:64]
                self.assertTrue(torch.equal(expected, actual), key)
            model = TransformerLanguageModel(**converted["architecture"]).eval()
            model.load_state_dict(converted["model_state"])
            old_model = TransformerLanguageModel(**original["architecture"]).eval()
            old_model.load_state_dict(original["model_state"])
            with torch.inference_mode():
                inputs = torch.zeros((1, 64), dtype=torch.long)
                self.assertTrue(torch.equal(model(inputs)[0], old_model(inputs)[0]))
                scores, _ = model(torch.zeros((1, 128), dtype=torch.long))
                self.assertEqual(scores.shape[1], 128)
            for invalid_size in (32, 64):
                with self.assertRaises(ValueError):
                    prepare(source, Path(directory) / "invalid", context_size=invalid_size)
            converted["training_config"].update(
                batch_size=2, eval_batches=1, eval_every=1
            )
            torch.save(converted, initial)
            train([
                "--resume", str(initial),
                "--output-dir", str(output),
                "--steps", "1",
                "--device", "cpu",
            ])
            for name in ("best.pt", "latest.pt"):
                result = torch.load(output / name, weights_only=True)
                self.assertEqual(result["architecture"]["context_size"], 128)
            self.assertEqual(result["step"], 1)
            self.assertTrue(result["optimizer_state"]["state"])
        self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())

    def test_conversion_and_training(self):
        torch.set_num_threads(1)
        source = ROOT / "experiments" / "transformer_finetune" / "best.pt"
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        original = torch.load(source, map_location="cpu", weights_only=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "experiment"
            initial = prepare(source, output)
            converted = torch.load(initial, weights_only=True)
            self.assertEqual(converted["step"], 0)
            self.assertEqual(converted["optimizer_state"]["state"], {})
            self.assertEqual(converted["architecture"], dict(original["architecture"], context_size=64))
            for key, value in original["model_state"].items():
                actual = converted["model_state"][key]
                if key == "position_embedding.weight":
                    self.assertEqual(actual.shape, (64, 192))
                    self.assertTrue(torch.isfinite(actual[32:]).all())
                    self.assertGreater(actual[32:].std().item(), 0)
                    self.assertFalse(torch.equal(actual[:32], actual[32:]))
                    actual = actual[:32]
                self.assertTrue(torch.equal(value, actual), key)
            repeat = prepare(source, Path(directory) / "repeat")
            self.assertTrue(
                torch.equal(
                    converted["model_state"]["position_embedding.weight"],
                    torch.load(repeat, weights_only=True)["model_state"][
                        "position_embedding.weight"
                    ],
                )
            )
            old_model = TransformerLanguageModel(**original["architecture"]).eval()
            old_model.load_state_dict(original["model_state"])
            new_model = TransformerLanguageModel(**converted["architecture"]).eval()
            new_model.load_state_dict(converted["model_state"])
            with torch.inference_mode():
                inputs = torch.zeros((1, 32), dtype=torch.long)
                self.assertTrue(torch.equal(old_model(inputs)[0], new_model(inputs)[0]))
                self.assertEqual(new_model(torch.zeros((1, 64), dtype=torch.long))[0].shape[1], 64)
            with self.assertRaises(ValueError):
                prepare(source, output)
            with self.assertRaises(ValueError):
                prepare(source, source.parent)
            # Shorten only the temporary test run, leaving experiment settings unchanged.
            converted["training_config"].update(batch_size=2, eval_batches=1, eval_every=1)
            torch.save(converted, initial)
            train([
                "--resume", str(initial),
                "--output-dir", str(output),
                "--steps", "1",
                "--device", "cpu",
            ])
            for name in ("latest.pt", "best.pt"):
                result = torch.load(output / name, weights_only=True)
                self.assertEqual(result["architecture"]["context_size"], 64)
            latest = torch.load(output / "latest.pt", weights_only=True)
            self.assertEqual(latest["step"], 1)
            self.assertTrue(latest["optimizer_state"]["state"])
        self.assertEqual(digest, hashlib.sha256(source.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
