# Transformer training

## Context-64 experiment

`prepare_context64.py` creates `transformer_context64/initial.pt` from
`transformer_finetune/best.pt` without training. It preserves the source file,
copies position embedding rows 0–31 exactly, and initializes rows 32–63 with
PyTorch's embedding default normal distribution (mean 0, standard deviation 1),
using seed 42. Every other model weight and architecture setting stays unchanged:
2 blocks, embedding size 192, and 4 heads. The output directory must be separate
and empty. The source path, SHA-256, step, and initialization seed are recorded.

The new checkpoint starts at step zero with a fresh Adam optimizer and new RNG
states. Training settings, including learning rate, are copied from the source.
Context-32 validation losses are not carried over as a context-64 best score.
Start the prepared experiment when ready:

```powershell
.\.venv\Scripts\python.exe train_transformer.py --resume transformer_context64/initial.pt --steps 1000 --output-dir transformer_context64
```

Here `--resume` loads the prepared context-64 model with its empty optimizer state;
it does not restore the context-32 optimizer. Training first evaluates a context-64
baseline, then writes `best.pt` and `latest.pt` in `transformer_context64`.
For later continuation, resume that directory's `latest.pt`. To sample it, use
`generate_transformer.py --checkpoint transformer_context64/best.pt`.

To prepare another experiment, run `prepare_context64.py --output-dir NEW_DIRECTORY`.

## Context-32 training

The trainer uses CUDA when available and otherwise CPU. `--device cpu` forces
CPU; `--device cuda` requires CUDA. The model and batches use the selected device.
The existing `best_transformer_2blocks.pt` is loaded by default and never overwritten.
Its architecture is 2 blocks, embedding size 192, context length 32, and 4 heads.
`TransformerLanguageModel` now accepts `num_layers` without changing weight names.

From PowerShell in the project directory, start training when ready:

```powershell
.\.venv\Scripts\python.exe train_transformer.py --steps 10000
```

This loads existing weights with a fresh Adam optimizer and starts the new run's
step count at zero. The old file lacks optimizer and RNG state, so its training
trajectory cannot be resumed exactly. Use `--weights PATH` for another saved
transformer checkpoint, or `--scratch` to initialize fresh weights.
Checkpoint architecture and character ordering are used when loading weights.

New files are saved in `transformer_checkpoints/`:

- `latest.pt`: the most recent evaluation checkpoint, also saved at the final step.
- `best.pt`: the lowest validation loss, including an evaluation of the initial
  model before updates. It can therefore contain step zero and an empty Adam
  state if no updates improve on the loaded model.

Both contain model weights, optimizer state, completed training step, architecture
settings, best/current validation losses, vocabulary, seed tokens, training
settings, Python RNG state, PyTorch CPU/CUDA RNG states, and the independent
training-batch generator state. Evaluation uses a separate fixed seed (123).
Files are written through a temporary file and replaced once complete.
The default evaluation/save interval is 2,000 steps. Interruptions can lose work
since the last save. A new run refuses to overwrite existing output checkpoints.

Resume all training state, for example:

```powershell
.\.venv\Scripts\python.exe train_transformer.py --resume transformer_checkpoints/latest.pt --steps 10000
```

`--steps` always means additional steps. Resume restores the saved batch size,
learning rate, evaluation interval, and evaluation batch count. An explicit
`--learning-rate` overrides the restored optimizer's learning rate; other training
settings retain their saved values. New weights-only or scratch runs default to
`0.0003`. Keep `input.txt` unchanged for reproducible continuation.
CPU/GPU transfers are supported, but bit-for-bit reproducibility across devices,
CUDA device counts, or PyTorch versions is not guaranteed. When resuming into a
new output directory, the resumed model establishes that directory's baseline
best; historical best weights are not copied from a different directory.

To continue from your best saved checkpoint with a tenfold lower learning rate,
preserving its optimizer history and writing into a separate directory:

```powershell
.\.venv\Scripts\python.exe train_transformer.py --resume transformer_checkpoints/best.pt --learning-rate 0.0003 --steps 1000 --output-dir transformer_finetune
```

This preserves the original best/latest files. A lower learning rate makes updates
smaller but does not itself prevent overfitting; compare validation loss against
the saved baseline. No training is launched by changing this configuration.

For a one-step smoke check:

```powershell
.\.venv\Scripts\python.exe train_transformer.py --steps 1 --batch-size 2 --eval-batches 1 --output-dir smoke_checkpoints
```

The script does not train when imported, and no longer samples text automatically
after training. `generate.py` still targets the separate older `best_model.pt` model.
To sample the fine-tuned transformer's best checkpoint without training:

```powershell
.\.venv\Scripts\python.exe generate_transformer.py --length 500 --temperature 0.8
```

Use `--prompt "ROMEO:"` to supply starting text, `--seed 123` for another sample,
or `--checkpoint PATH` for a different transformer checkpoint. The script defaults
to `transformer_finetune/best.pt`, uses CUDA when available (CPU otherwise), and
never writes to the checkpoint. Lower positive temperatures concentrate sampling
on more likely characters; higher temperatures increase variety.

Run the short integration tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest test_training -v
```
