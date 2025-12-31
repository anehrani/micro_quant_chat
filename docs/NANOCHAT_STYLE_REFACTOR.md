# Nanochat-Style Refactor (Develop Branch)

This document explains, in detail, how this repository was refactored to be closer to Karpathy’s **nanochat** repository *in structure and workflow*, while preserving the existing `./scripts/mqc ...` UX.

It covers:
- What changed and why
- The new module layout (`microchat/`)
- The new “scripts entrypoints” (`scripts/base_*.py`)
- Checkpoint format + compatibility rules
- CLI flag compatibility (old vs new)
- How to run train/eval/generate in the new layout

> Scope note
> - This refactor focuses on **code organization and entrypoints**.
> - It does **not** attempt to replicate nanochat’s distributed training, optimizer stack, or large-scale infra.

---

## 1) High-level Goals

### Goal A — nanochat-like structure
Nanochat’s repo style is roughly:
- **core library code** (model, dataset utilities, checkpointing)
- **scripts** as runnable entrypoints that wire arguments → core functions

This repo now follows that pattern:
- `microchat/` contains the core code
- `scripts/base_train.py`, `scripts/base_eval.py`, `scripts/base_sample.py` are runnable entrypoints

### Goal B — keep your current UX
Existing usage patterns like:

- `./scripts/mqc train`
- `./scripts/mqc eval`
- `./scripts/mqc generate`

continue to work, but now dispatch into the nanochat-style entrypoints.

---

## 2) New Repository Layout

### Core package: `microchat/`
This is the nanochat-like “library” layer.

- `microchat/gpt.py`
  - Contains the nanochat-inspired GPT model.
  - Key pieces:
    - `GPTConfig` dataclass
    - `GPT` model class
    - rotary embeddings (RoPE), RMSNorm, QK norm, bias-free Linear layers
    - autoregressive sampling via `GPT.generate(...)`

- `microchat/data.py`
  - Token-stream dataset and loaders.
  - Implements a **time split** (contiguous split) to reduce leakage with overlapping windows.
  - Main functions:
    - `load_tokens(path) -> list[int]`
    - `create_dataloaders(tokens, seq_len, batch_size, ...)`

- `microchat/train.py`
  - Train loop + CLI (`main()`), acting like a nanochat `base_train`.
  - Handles:
    - device selection (`auto` → cuda if available else cpu)
    - optimizer (AdamW)
    - scheduler (cosine annealing)
    - saving best checkpoint

- `microchat/eval.py`
  - Simple evaluation entrypoint computing:
    - average loss
    - perplexity ($e^{loss}$)

- `microchat/sample.py`
  - Sampling / generation entrypoint.
  - Loads checkpoint → runs `model.generate` → prints token sequences.

- `microchat/ckpt.py`
  - Checkpoint IO.
  - Important design choice:
    - config is stored as a **plain dict** (portable, avoids pickling issues)
  - Backward-compatible loader:
    - supports old checkpoints where `config` might be a pickled `GPTConfig`

- `microchat/device.py`
  - `resolve_device("auto"|"cpu"|"cuda") -> torch.device`

### Script entrypoints: `scripts/base_*.py`
These are thin entrypoints (nanochat-style) that call `microchat.*.main()`:

- `scripts/base_train.py`
- `scripts/base_eval.py`
- `scripts/base_sample.py`

Their purpose is to keep a clean separation:
- scripts: argument parsing + wiring
- microchat: reusable logic

---

## 3) Command Routing (What runs when you type `./scripts/mqc ...`)

`scripts/mqc` is a shell helper that always runs `.venv/bin/python` and sets `PYTHONPATH` so imports work.

### New routing
- `./scripts/mqc train ...` → `python scripts/base_train.py ...`
- `./scripts/mqc eval ...` → `python scripts/base_eval.py ...`
- `./scripts/mqc generate ...` → `python scripts/base_sample.py ...`
- `./scripts/mqc examples` → still runs `python src/examples.py`

This keeps the old UX but uses the new nanochat-style entrypoints.

---

## 4) Checkpoint Format (Important)

### New checkpoint format (recommended)
Saved by:
- `microchat/ckpt.py::save_checkpoint`
- `microchat/train.py`

Key properties:
- `config` is a **dict**: `{"sequence_len": ..., "vocab_size": ..., ...}`
- `model_state_dict` is standard PyTorch state dict
- `optimizer_state_dict` optionally saved

This format is *portable* and avoids pickling problems.

### Backward compatibility
Some earlier runs (older branch states) used:
- `config` stored as a pickled `GPTConfig` instance

The loader:
- `microchat/ckpt.py::load_checkpoint`

handles both:
- `config` as dict
- `config` as `GPTConfig`

### PyTorch “weights_only” note
PyTorch changed defaults around `torch.load` safety. For compatibility with legacy objects, we load with:
- `torch.load(..., weights_only=False)`

This is intentional so older checkpoints still load.

---

## 5) CLI Flag Compatibility (Old vs New)

To avoid breaking old commands, the new entrypoints accept both naming styles.

### Training
New entrypoint: `scripts/base_train.py` → `microchat/train.py::main()`

Supported flags:
- Data:
  - `--data` (new)
  - `--data_file` (legacy)
- Output:
  - `--out` (new)
  - `--save_dir` (legacy)
- Epochs:
  - `--epochs` (new)
  - `--num_epochs` (legacy)
- Batch:
  - `--batch` (new)
  - `--batch_size` (legacy)
- Learning rate:
  - `--lr` (new)
  - `--learning_rate` (legacy)

### Evaluation
New entrypoint: `scripts/base_eval.py` → `microchat/eval.py::main()`

Supported flags:
- `--data` and `--data_file`
- `--seq-len` and `--seq_len`

### Sampling / Generation
New entrypoint: `scripts/base_sample.py` → `microchat/sample.py::main()`

Supported flags:
- `--seed` and `--seed_tokens`
- `--num` and `--num_generate`
- `--temp` and `--temperature`
- `--top-k` and `--top_k`
- `--samples` and `--num_samples`

---

## 6) Legacy `src/*` Scripts (What’s still there and why)

This repo still contains `src/train.py`, `src/evaluate.py`, `src/generate.py`, etc.

Reasons:
1. Existing users might call them directly.
2. Some scripts (`src/examples.py`, tokenizer-related tooling) still live under `src/`.

### Important compatibility updates in `src/*`
To align old scripts with the new checkpoint format:
- `src/train.py` now saves `config` as a dict (via `dataclasses.asdict`).
- `src/evaluate.py`, `src/generate.py`, `src/predict.py`, `src/evaluate_predictions.py` now accept both:
  - dict configs
  - dataclass configs

This prevents breakage when mixing older scripts and newer checkpoints.

---

## 7) How to Run Everything (Recommended)

### Train
```bash
./scripts/mqc train --num_epochs 10 --batch_size 32 --learning_rate 5e-4
```
Equivalent nanochat-style direct run:
```bash
./scripts/run.sh scripts/base_train.py --epochs 10 --batch 32 --lr 5e-4
```

### Evaluate
```bash
./scripts/mqc eval --seq_len 256
```
Equivalent:
```bash
./scripts/run.sh scripts/base_eval.py --seq-len 256
```

### Generate
```bash
./scripts/mqc generate --num_generate 50 --num_samples 3
```
Equivalent:
```bash
./scripts/run.sh scripts/base_sample.py --num 50 --samples 3
```

---

## 8) Notes on “How this is like nanochat” (and what it isn’t)

### Similarities (structural)
- Clean separation between:
  - core library code (`microchat/`)
  - script entrypoints (`scripts/base_*.py`)
- A single “base_train/base_eval/base_sample” flow that mirrors nanochat’s script approach.

### Differences (intentional)
This repo does not currently replicate nanochat’s:
- distributed training
- Muon optimizer stack
- large-scale dataset + packing pipeline
- long training runs / logging infra

Those can be added later, but were out of scope for “make it similar” at the code-layout level.
