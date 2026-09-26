# Transformer experiments

This project trains and samples small language models on the text in `data/input.txt`.
Run the commands below from the project root. See [TRAINING.md](TRAINING.md) for
training options and checkpoint details.

## Layout

| Path | Contents |
| --- | --- |
| `data/` | Training corpus |
| `checkpoints/` | Original two-block checkpoint used by the trainer |
| `experiments/` | Saved training runs and tokenizer artifacts |
| `tests/` | Unit and integration tests |
| `legacy_archive/` | Older model scripts and checkpoints |
| Project root | Current training, generation, model, and tokenizer scripts |

`token_test/` is an existing untracked run and has been left in place.

## Quick checks

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_batch_config tests.test_subword.SubwordTests.test_tokenizer_serialization -v
.\.venv\Scripts\python.exe generate_transformer.py --length 100
```

Some context extension tests require older checkpoints that are not currently
present in `experiments/`. Two training tests also encounter corpus mismatches
with saved checkpoints; see [TRAINING.md](TRAINING.md).
