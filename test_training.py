"""Short checkpoint integration tests; never write production checkpoints."""
import hashlib
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import train_transformer as training


class TrainingTests(unittest.TestCase):
    def test_auto_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(training.select_device("auto"), torch.device("cpu"))

    def test_rng_roundtrip(self):
        generator = torch.Generator().manual_seed(72)
        saved = training.capture_rng(generator)
        expected = (random.random(), torch.rand(4), torch.rand(4, generator=generator))
        training.restore_rng(saved, generator)
        self.assertEqual(expected[0], random.random())
        self.assertTrue(torch.equal(expected[1], torch.rand(4)))
        self.assertTrue(torch.equal(expected[2], torch.rand(4, generator=generator)))

    def test_checkpoint_resume_and_legacy_preservation(self):
        torch.set_num_threads(1)
        before = hashlib.sha256(training.LEGACY.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = ["--device", "cpu", "--batch-size", "2", "--eval-batches", "1", "--eval-every", "1"]
            training.main(common + ["--steps", "2", "--output-dir", str(root / "whole")])
            training.main(common + ["--steps", "1", "--output-dir", str(root / "split")])
            training.main(["--device", "cpu", "--steps", "1", "--resume",
                           str(root / "split/latest.pt"), "--output-dir", str(root / "split")])
            whole = torch.load(root / "whole/latest.pt", weights_only=True)
            split = torch.load(root / "split/latest.pt", weights_only=True)
            self.assertEqual(split["step"], 2)
            for key, value in whole["model_state"].items():
                self.assertTrue(torch.equal(value, split["model_state"][key]), key)
            self.assertTrue(torch.equal(whole["rng_states"]["train"], split["rng_states"]["train"]))
            self.assertEqual(whole["val_loss"], split["val_loss"])
            best = torch.load(root / "split/best.pt", weights_only=True)
            self.assertEqual(best["val_loss"], split["best_val_loss"])
            training.main(["--device", "cpu", "--steps", "1", "--resume",
                           str(root / "split/best.pt"), "--learning-rate", "0.00003",
                           "--output-dir", str(root / "lower_lr")])
            lowered = torch.load(root / "lower_lr/latest.pt", weights_only=True)
            self.assertEqual(lowered["step"], best["step"] + 1)
            self.assertEqual(lowered["training_config"]["learning_rate"], 0.00003)
            for group in lowered["optimizer_state"]["param_groups"]:
                self.assertEqual(group["lr"], 0.00003)
            for key, state in best["optimizer_state"]["state"].items():
                self.assertEqual(lowered["optimizer_state"]["state"][key]["step"], state["step"] + 1)
            for payload in (split, best):
                self.assertTrue(all(key in payload for key in (
                    "model_state", "optimizer_state", "step", "architecture", "rng_states")))
                self.assertEqual(payload["architecture"]["num_layers"], 2)
            if torch.cuda.is_available():
                # Resume CPU optimizer state on GPU and then load GPU state on CPU.
                training.main(["--device", "cuda", "--steps", "1", "--resume",
                               str(root / "split/latest.pt"), "--output-dir", str(root / "cuda")])
                training.main(["--device", "cpu", "--steps", "1", "--resume",
                               str(root / "cuda/latest.pt"), "--output-dir", str(root / "back")])
        self.assertEqual(before, hashlib.sha256(training.LEGACY.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
