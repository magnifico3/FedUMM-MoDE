# Modality Conflict Traces

This folder contains a self-contained experiment harness for tracing Janus-Pro
LoRA updates across a small FedAvg run and then analyzing pairwise update
conflicts across datasets.

## What It Does

- Runs a single model (`Janus-Pro-1B`) on a single dataset profile at a time
- Runs sequential local training for `R` rounds with configurable epoch / batch settings
- Saves the single-model LoRA gradient and LoRA parameter delta for every round
- Exports flattened vectors for later inner-product / cosine analysis

## Supported Dataset Profiles

- `vqav2`
  Uses the existing `HuggingFaceM4/VQAv2` loader
- `cc3m`
  Expects a local CC3M download produced by `scripts/download_cc3m.py`
- `instruct`
  Defaults to `guyue-wa/instructpix2pix-clip-filtered`
- `text`
  Defaults to `wikitext` with config `wikitext-2-raw-v1`

`text` is the pure text baseline. `instruct` uses the image-editing triplets
`original_image`, `edit_prompt`, and `edited_image`, and trains Janus-Pro with
an autoregressive image-token loss for the edited target image.

## Run One Trace

```bash
python experiments/modality_conflict/run_janus_round_trace.py \
    --dataset_profile vqav2 \
    --output_dir ./outputs/modality_conflict/vqav2 \
    --rounds 5 \
    --lr_scheduler cosine \
    --lora_r 1 \
    --max_samples 256
```

CC3M example:

```bash
python experiments/modality_conflict/run_janus_round_trace.py \
    --dataset_profile cc3m \
    --cc3m_dir ../Datasets/cc3m \
    --output_dir ./outputs/modality_conflict/cc3m \
    --rounds 5 \
    --lora_r 1 \
    --max_samples 256
```

InstructPix2Pix example:

```bash
python experiments/modality_conflict/run_janus_round_trace.py \
    --dataset_profile instruct \
    --output_dir ./outputs/modality_conflict/instruct \
    --rounds 5 \
    --lora_r 1 \
    --max_samples 256
```

## Analyze Conflicts

```bash
python experiments/modality_conflict/analyze_conflicts.py \
    ./outputs/modality_conflict/vqav2 \
    ./outputs/modality_conflict/cc3m \
    ./outputs/modality_conflict/instruct \
    ./outputs/modality_conflict/text \
    --artifact grad \
    --output_csv ./outputs/modality_conflict/gradient_conflict_report.csv
```

The resulting CSV includes:

- `dot`
- `cosine`
- `conflict`

`conflict=True` means the per-round aggregated updates have a negative inner
product, matching the first-order conflict signal described in the paper.

The trace runner now uses a global-step LR scheduler by default
(`--lr_scheduler cosine` with `--warmup_ratio 0.1`). Per-round `summary.json`
files record `lr_start` and `lr_end`, and `metadata.json` records the planned
`total_training_steps` and `warmup_steps`.

## Saved Artifacts Per Round

- `round_grad.pt`
- `round_grad_flat.pt`
- `round_delta.pt`
- `round_delta_flat.pt`
- `summary.json` now also includes task-specific eval metrics when available:
  `eval_loss`, `eval_acc` (VQAv2), `eval_ppl` (text),
  `eval_clipscore` (CC3M / Instruct), and `eval_lpips` (Instruct, when `lpips`
  is installed in the JanusPro environment).

Use `--artifact grad` when you want actual round-level gradient conflict.
Use `--artifact delta` when you want to compare the resulting LoRA updates.
