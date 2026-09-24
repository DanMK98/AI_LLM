"""Resume override and stable evaluation checks; no training required."""

import contextlib
import io
import unittest
from unittest.mock import Mock, patch

import torch

import train_transformer as training


class BatchConfigTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(
            batch_size=32, eval_every=200, eval_batches=2, learning_rate=0.0003
        )

    def resolve(self, *extra, saved=None):
        args = training.parse_args(["--resume", "unused.pt", *extra])
        return training.resolve_training_config(
            args, {"training_config": self.saved if saved is None else saved}
        )

    def test_override_and_legacy_evaluation(self):
        config = self.resolve("--batch-size", "92")
        self.assertEqual(config["batch_size"], 92)
        self.assertEqual(config["eval_batch_size"], 32)
        self.assertEqual(self.saved["batch_size"], 32)
        self.assertEqual(config["learning_rate"], 0.0003)

    def test_omitted_override_and_next_resume(self):
        self.assertEqual(self.resolve()["batch_size"], 32)
        updated = self.resolve("--batch-size", "92")
        resumed = self.resolve(saved=updated)
        self.assertEqual(resumed["batch_size"], 92)
        self.assertEqual(resumed["eval_batch_size"], 32)
        self.assertEqual(self.resolve("--batch-size", "64", saved=resumed)["eval_batch_size"], 32)

    def test_new_run_default(self):
        config = training.resolve_training_config(training.parse_args(["--scratch"]))
        self.assertEqual(config["batch_size"], 128)
        self.assertEqual(config["eval_batch_size"], 128)

    def test_invalid_batch_size(self):
        for value in ("0", "-1"):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                training.parse_args(["--batch-size", value])

    def test_evaluation_keeps_same_samples(self):
        model = Mock(context_size=4)
        source = torch.arange(100)
        before = []
        after = []

        def loss(record):
            def calculate(model, inputs, targets):
                record.append(inputs.clone())
                return inputs.float().mean()
            return calculate

        with patch.object(training, "calculate_loss", side_effect=loss(before)):
            old_loss = training.evaluate(model, source, self.saved, "cpu")
        with patch.object(training, "calculate_loss", side_effect=loss(after)):
            new_loss = training.evaluate(
                model, source, self.resolve("--batch-size", "92"), "cpu"
            )
        self.assertEqual(old_loss, new_loss)
        for left, right in zip(before, after):
            self.assertTrue(torch.equal(left, right))
            self.assertEqual(right.shape[0], 32)


if __name__ == "__main__":
    unittest.main()
