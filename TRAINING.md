# Transformer training

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
learning rate, evaluation interval, and evaluation batch count, overriding those
command-line settings. Keep `input.txt` unchanged for reproducible continuation.
CPU/GPU transfers are supported, but bit-for-bit reproducibility across devices,
CUDA device counts, or PyTorch versions is not guaranteed. When resuming into a
new output directory, the resumed model establishes that directory's baseline
best; historical best weights are not copied from a different directory.

For a short check in a separate output directory:

```powershell
.\.venv\Scripts\python.exe train_transformer.py --steps 1 --batch-size 2 --eval-batches 1 --output-dir smoke_checkpoints
```

The script does not train when imported, and no longer samples text automatically
after training. `generate.py` still targets the separate older `best_model.pt` model.
Run the short integration tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest test_training -v
```
