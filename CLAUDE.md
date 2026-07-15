# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research repository for **ModernBERT** — pre-training and evaluation of modernized BERT encoders. This is the *training/research* codebase, not the HuggingFace inference package. It introduces **FlexBERT**, a modular approach where encoder building blocks (attention, embeddings, layers, MLP, normalization, activation) are selected by name via YAML config. Builds on MosaicML's MosaicBERT + Composer framework.

## Environment & commands

Dependencies are managed with **uv** (Python 3.12); everything lives in `pyproject.toml` + `uv.lock`. There is no conda env or `requirements-*.txt` anymore.

```bash
uv sync                                  # core deps (CPU dev; no flash-attn)
uv sync --extra flash                    # + Flash Attention 2 (Linux/CUDA, prebuilt wheel)
uv sync --extra data --extra colbert     # + data-prep / ColBERT retrieval extras
```

- **Flash Attention is optional.** `flash-attn` lives in the `flash` extra (a prebuilt cp312/torch2.7/cu12 wheel pinned via `[tool.uv.sources]`). FlexBERT imports it behind `try/except` in `src/bert_layers/attention.py` and falls back to PyTorch SDPA when absent, so the core install runs without it. FA3 (H100) is still built separately.
- **torch**: pinned to `2.7.0` (the newest mosaicml/composer 0.32.1 supports). On Linux it resolves from the `pytorch-cu128` index; on macOS from the default PyPI CPU/MPS wheel.
- **Lint/format**: `uv run ruff ...` (config in `ruff.toml` — line length 120, target py311). No separate build step (pure Python).
- **Run tests**: `uv run pytest tests/` — one file `uv run pytest tests/test_main.py`, one case `uv run pytest tests/test_main.py::test_trainer -k flex_bert`. Tests load `yamls/defaults.yaml` + `yamls/models/<model>.yaml` + a `tests/smoketest_config_*.yaml` and merge them; `test_main.py` runs a full tiny train loop against synthetic data (`SynthTextDirectory` in `tests/test_utils.py`).
- **Train**: `uv run composer main.py <config.yaml>` (uses all GPUs). E.g. `uv run composer main.py yamls/modernbert/modernbert-base-pretrain.yaml`. `composer` (from mosaicml) is the launcher — do not use plain `python` for training.

## Key entry points (top-level scripts)

- `main.py` — pre-training (MLM). Dispatches on `model.name` to `src/{flex_bert,hf_bert,mosaic_bert}.py`.
- `eval.py` — run a single fine-tuning eval from a generated config; `python eval.py <config.yaml>`.
- `run_evals.py` + `generate_eval_config.py` — GLUE eval for ModernBERT/FlexBERT checkpoints (auto-download from HF Hub, gen configs, run across GPUs). See `RunEvals.md`.
- `glue.py` — GLUE eval for *non*-ModernBERT baselines (uses `yamls/finetuning/`).
- `sequence_classification.py` — standalone sequence classification training.
- `convert_to_hf.py` — convert a Composer checkpoint to HuggingFace format (typer CLI). `download_artifacts_from_wandb.py` — pull checkpoints from W&B.
- `benchmark.py`, `efficiency/` — throughput/inference benchmarking.

## Architecture

### Model selection is three-layered
1. **`model.name`** in YAML picks the wrapper module: `flex_bert` (ModernBERT), `mosaic_bert` (ALiBi/GLU), or `hf_bert` (vanilla HF). Each `src/*_bert.py` exposes `create_*_mlm` / `create_*_classification` returning a Composer `HuggingFaceModel`.
2. **`model.model_config`** in YAML populates `FlexBertConfig` (`src/bert_layers/configuration_bert.py`), which is a subclass of HF's `BertConfig` with extra fields.
3. **String fields on the config select concrete nn.Module classes.** Each `src/bert_layers/*.py` module owns a registry dict mapping name→class, aggregated in `src/bert_layers/options.py`:
   - `attention_layer` → `attention.py` (`ATTN2CLS`): e.g. `rope`, `base`, padded/unpadded variants
   - `embedding_layer` → `embeddings.py` (`EBB2CLS`): `sans_pos`, `absolute_pos`, alibi
   - `bert_layer` → `layers.py` (`LAYER2CLS`): `prenorm`/`postnorm`, parallel/sequential
   - `mlp_layer` → `mlp.py` (`MLP2CLS`): `mlp`, `glu`
   - `normalization` → `normalization.py` (`NORM2CLS`): `layernorm`, `rmsnorm`
   - `hidden_act` → `activation.py` (`ACT2CLS`)

   `src/bert_layers/model.py` assembles these into `FlexBertModel` / `FlexBertForMaskedLM` / `FlexBertForSequenceClassification` / `FlexBertForMultipleChoice`. To add a new building block, implement the class and register it in the module's `*2CLS` dict — no wiring changes elsewhere.

### Unpadding is central to performance
FlexBERT internally *unpads* batches (removes padding tokens, concatenates sequences) for Flash Attention efficiency. `padding: unpadded` + `unpad_embeddings: true` in config; logic in `src/bert_layers/padding.py` and `src/bert_padding.py`. Some config fields are mutually constraining (e.g. `unpad_embeddings` forces `padding=unpadded` and is incompatible with `absolute_pos`) — see the `__init__`/validation in `configuration_bert.py`.

### Data pipeline
Two dataset classes in `src/text_data.py`, chosen per-loader via `dataset.streaming: true|false`:
- `StreamingTextDataset` (MosaicML StreamingDataset) — MDS/CSV/JSONL, remote or local.
- `NoStreamingDataset` — decompressed MDS only; **preferred for local data** (higher throughput).

Data conversion/prep tools live in `src/data/` (`hf_to_mds.py`, `mds_conversion.py` with `--decompress`, sampling/stats scripts). Sequence packing for training density is in `src/sequence_packer.py`.

### Training loop internals
`main.py` wires: Composer `Trainer` + `Evaluator`s, optimizer (`DecoupledAdamW`/`AdamW`, custom filtered variants in `src/optimizer.py`), schedulers (`src/scheduler.py` adds `WarmupStableDecayScheduler`, `CosineInverseSqrtScheduler`, `OneMinusSqrtScheduler` beyond Composer's built-ins), and custom callbacks in `src/callbacks/` (dataloader speed, grad-norm logging, packing efficiency, scheduled GC). RoPE theta scheduling for context extension is a Composer *algorithm*: `src/algorithms/rope_schedule.py`.

### Config layering
YAMLs are merged with OmegaConf. Base defaults in `yamls/defaults.yaml`; model shells in `yamls/models/`. Full experiment configs: `yamls/main/` (FlexBERT variants), `yamls/modernbert/` (the actual ModernBERT recipe: pretrain → learning-rate-decay → context-extension for base & large), `yamls/baselines/`, `yamls/finetuning/`, `yamls/ablations/`.

## Evals structure
`src/evals/` holds the fine-tuning job definitions (`glue_jobs.py`, `superglue_jobs.py`, `finetuning_jobs.py`, `misc_jobs.py`) and eval data loading (`data.py`). The eval flow is: generate a per-checkpoint YAML (from a checkpoint + its training config or a W&B run) → run `eval.py`. See `RunEvals.md` and `src/evals/README.md`.

## Retrieval (examples/)
Standalone scripts for training/evaluating retrieval models on ModernBERT: dense (Sentence Transformers, `train_st*.py`) and ColBERT (PyLate, `train_pylate.py`). ColBERT-specific BEIR eval code is in `src/colbert_beir/`.

## Conventions
- Scripts append the repo root (and their own dir) to `sys.path` at import time so relative imports work regardless of CWD — keep this pattern when adding entry points.
- License headers: files carry Apache-2.0 headers crediting Answer.AI/LightOn + upstream MosaicML/others. Preserve existing headers on edited files.
