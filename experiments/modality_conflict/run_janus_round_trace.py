#!/usr/bin/env python3
"""Run a single-dataset Janus-Pro trace and save per-round LoRA gradients."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import src  # noqa: F401
from src.common import (  # noqa: E402
    accelerator_autocast,
    build_hf_download_config,
    clean_generation_text,
    configure_hf_transfer_timeouts,
    count_trainable_params,
    create_accelerator,
    get_default_cc3m_dir,
    get_default_datasets_root,
    get_trainable_params,
    load_dataset_with_local_fallback,
    load_trainable_params,
    maybe_subsample,
    pick_device,
    set_seed,
    unwrap_model,
    vqa_soft_score,
)
from src.januspro_backend import JanusProBackend  # noqa: E402


DEFAULT_DATASETS = {
    "vqav2": {
        "dataset_name": "HuggingFaceM4/VQAv2",
        "train_split": "train",
        "eval_split": "validation[:50%]",
        "eval_max_samples": 256,
    },
    "cc3m": {
        "dataset_name": "",
        "train_split": "train",
        "eval_split": "validation",
        "eval_max_samples": 256,
    },
    "instruct": {
        "dataset_name": "guyue-wa/instructpix2pix-clip-filtered",
        "train_split": "train",
        "eval_max_samples": 128,
    },
    "text": {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "train_split": "train",
        "max_samples": 2048,
        "eval_split": "validation",
        "eval_max_samples": 512,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace Janus-Pro LoRA updates for a single dataset across FedAvg rounds."
    )
    parser.add_argument("--model_name_or_path", type=str, default="")
    parser.add_argument(
        "--dataset_profile",
        type=str,
        required=True,
        choices=sorted(DEFAULT_DATASETS.keys()),
        help="Which dataset/task profile to run.",
    )
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--dataset_config", type=str, default="")
    parser.add_argument("--train_split", type=str, default="")
    parser.add_argument("--eval_split", type=str, default="")
    parser.add_argument(
        "--data_path",
        type=str,
        default=get_default_datasets_root(),
        help="Dataset/cache root. Defaults to the ../Datasets directory next to the repo.",
    )
    parser.add_argument(
        "--cc3m_dir",
        type=str,
        default=get_default_cc3m_dir(),
        help="Directory produced by scripts/download_cc3m.py for the CC3M profile.",
    )
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["constant", "linear", "cosine"],
        help="Learning-rate schedule applied over global optimizer steps.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of total optimizer steps used for LR warmup.",
    )
    parser.add_argument(
        "--min_lr_ratio",
        type=float,
        default=0.0,
        help="Final LR as a fraction of the base LR after decay.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=(
            "Maximum raw records to load. Use 0 for the dataset-profile default "
            "(text defaults to 2048, other profiles to 256)."
        ),
    )
    parser.add_argument("--max_q_len", type=int, default=128)
    parser.add_argument("--max_a_len", type=int, default=64)
    parser.add_argument("--max_text_len", type=int, default=512)
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=0,
        help=(
            "Maximum eval records to load. Use 0 for the dataset-profile default "
            "(text defaults to 512; other profiles disable eval)."
        ),
    )
    parser.add_argument("--gen_image_size", type=int, default=384)
    parser.add_argument("--lora_r", type=int, default=1)
    parser.add_argument("--lora_alpha", type=int, default=1)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument(
        "--eval_holdout_fraction",
        type=float,
        default=0.1,
        help=(
            "When a dataset profile lacks a native eval split, reserve this fraction "
            "of the loaded records as a deterministic eval holdout."
        ),
    )
    parser.add_argument(
        "--clip_model_name_or_path",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP checkpoint used for CLIPScore-style text-image alignment evaluation.",
    )
    parser.add_argument(
        "--lpips_net",
        type=str,
        default="alex",
        choices=["alex", "vgg", "squeeze"],
        help="Backbone used by the optional LPIPS evaluator.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hf_max_retries", type=int, default=8)
    parser.add_argument("--hf_timeout", type=int, default=1800)
    return parser.parse_args()


def _resolve_dataset_setting(args: argparse.Namespace, key: str) -> str:
    if getattr(args, key):
        return getattr(args, key)
    return DEFAULT_DATASETS[args.dataset_profile].get(key, "")


def _resolve_max_samples(args: argparse.Namespace) -> int:
    if args.max_samples != 0:
        return args.max_samples
    return DEFAULT_DATASETS[args.dataset_profile].get("max_samples", 256)


def _resolve_eval_max_samples(args: argparse.Namespace) -> int:
    if args.eval_max_samples != 0:
        return args.eval_max_samples
    return DEFAULT_DATASETS[args.dataset_profile].get("eval_max_samples", -1)


def _train_and_eval_load_limits(args: argparse.Namespace) -> tuple[int, int, int]:
    train_max = _resolve_max_samples(args)
    eval_max = _resolve_eval_max_samples(args)
    eval_target = max(eval_max, 0)
    combined = train_max
    if train_max >= 0:
        combined = train_max + eval_target
    return train_max, eval_max, combined


def _split_records_for_eval(
    records: List[Dict[str, Any]],
    train_max_samples: int,
    eval_max_samples: int,
    seed: int,
    holdout_fraction: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if eval_max_samples < 0:
        train_records = list(records)
        if train_max_samples >= 0:
            train_records = train_records[:train_max_samples]
        return train_records, []

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if not shuffled:
        return [], []

    requested_eval = eval_max_samples
    if requested_eval == 0:
        requested_eval = max(1, int(math.ceil(len(shuffled) * max(holdout_fraction, 0.0))))
    requested_eval = min(requested_eval, len(shuffled))

    eval_records = shuffled[:requested_eval]
    train_records = shuffled[requested_eval:]
    if train_max_samples >= 0:
        train_records = train_records[:train_max_samples]
    return train_records, eval_records


def _clone_param_dict(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in params.items()}


def _subtract_param_dict(
    newer: Dict[str, torch.Tensor], older: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {k: newer[k] - older[k] for k in sorted(newer.keys())}


def _add_param_dict(
    base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {k: base[k] + delta[k] for k in sorted(base.keys())}


def _flatten_param_dict(params: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([params[k].reshape(-1).float() for k in sorted(params.keys())], dim=0)


def _save_update(path: Path, params: Dict[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(params, path)
    torch.save(_flatten_param_dict(params), path.with_name(path.stem + "_flat.pt"))


def _optimizer_steps_per_epoch(num_batches: int, grad_accum: int) -> int:
    return max(int(math.ceil(max(num_batches, 0) / max(grad_accum, 1))), 1)


def _compute_scheduled_lr(
    base_lr: float,
    global_step: int,
    total_steps: int,
    scheduler_name: str,
    warmup_steps: int,
    min_lr_ratio: float,
) -> float:
    total_steps = max(int(total_steps), 1)
    global_step = min(max(int(global_step), 0), total_steps - 1)
    warmup_steps = min(max(int(warmup_steps), 0), total_steps - 1) if total_steps > 1 else 0
    min_lr_ratio = min(max(float(min_lr_ratio), 0.0), 1.0)

    if warmup_steps > 0 and global_step < warmup_steps:
        warmup_scale = float(global_step + 1) / float(warmup_steps)
        return base_lr * warmup_scale

    if scheduler_name == "constant":
        return base_lr

    decay_total = max(total_steps - warmup_steps, 1)
    progress = float(global_step - warmup_steps) / float(decay_total)
    progress = min(max(progress, 0.0), 1.0)

    if scheduler_name == "linear":
        scale = 1.0 - progress * (1.0 - min_lr_ratio)
    elif scheduler_name == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        scale = min_lr_ratio + (1.0 - min_lr_ratio) * cosine
    else:
        scale = 1.0

    return base_lr * scale


def _set_optimizer_lr(optimizer, lr_value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr_value


def _hf_load(
    dataset_name: str,
    split: str,
    cache_dir: str,
    hf_max_retries: int,
    hf_timeout: int,
    dataset_config: str = "",
):
    download_config = build_hf_download_config(
        cache_dir=cache_dir,
        max_retries=hf_max_retries,
        timeout_seconds=hf_timeout,
    )
    return load_dataset(
        dataset_name,
        dataset_config or None,
        split=split,
        cache_dir=cache_dir or None,
        download_config=download_config,
        storage_options=download_config.storage_options,
    )


def _load_vqav2_records(
    args: argparse.Namespace,
    split: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    backend = JanusProBackend()
    cache_dir = args.data_path or os.environ.get("HF_HOME", "/tmp/hf_cache")
    ds = load_dataset_with_local_fallback(
        backend.hf_dataset_name(),
        split,
        data_path=args.data_path,
        cache_dir=cache_dir,
        hf_max_retries=args.hf_max_retries,
        hf_timeout=args.hf_timeout,
    )
    keep = set(backend.keep_columns())
    ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
    ds = maybe_subsample(ds, max_samples, args.seed)
    records = []
    for row in ds:
        records.append(
            {
                "kind": "multimodal",
                "image": row["image"],
                "prompt": (
                    f"{row['question']}\n"
                    "Answer the visual question with the most likely short answer only. "
                    "Use one word or a short phrase when possible."
                ),
                "target": row["multiple_choice_answer"],
                "gt_answers": [a["answer"] for a in row["answers"]],
                "question": row["question"],
                "multiple_choice_answer": row["multiple_choice_answer"],
                "question_id": row.get("question_id"),
                "image_id": row.get("image_id"),
                "question_type": row.get("question_type", ""),
                "answer_type": row.get("answer_type", ""),
            }
        )
    return records


def _load_cc3m_records(
    args: argparse.Namespace,
    split: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    split_dir = Path(args.cc3m_dir) / split
    manifest = split_dir / "downloads.jsonl"
    if not manifest.exists():
        raise FileNotFoundError(
            f"CC3M manifest not found at '{manifest}'. "
            "Run scripts/download_cc3m.py first or pass --cc3m_dir."
        )

    def _resolve_image_path(row: Dict[str, Any]) -> str:
        raw_path = (row.get("path") or "").strip()
        relative_path = (row.get("relative_path") or "").strip()
        candidates: List[Path] = []

        if raw_path:
            raw = Path(raw_path)
            candidates.append(raw)
            if not raw.is_absolute():
                candidates.append(split_dir / raw)
            # Older manifests stored absolute paths, which break after moving the dataset.
            candidates.append(split_dir / "images" / raw.name)

        if relative_path:
            candidates.append(split_dir / relative_path)

        seen = set()
        for candidate in candidates:
            normalized = str(candidate)
            if normalized in seen:
                continue
            seen.add(normalized)
            try:
                if candidate.exists():
                    return str(candidate)
            except OSError:
                continue
        return ""

    candidate_rows = []
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in {"ok", "exists"}:
                continue
            image_path = _resolve_image_path(row)
            if not image_path:
                continue
            caption = (row.get("caption") or "").strip()
            if not caption:
                continue
            candidate_rows.append({"image_path": image_path, "caption": caption})

    random.Random(args.seed).shuffle(candidate_rows)

    def _is_valid_image(image_path: str) -> bool:
        try:
            with Image.open(image_path) as image:
                image.verify()
            return True
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError):
            return False

    records = []
    target_count = max_samples if max_samples >= 0 else None
    for row in candidate_rows:
        if not _is_valid_image(row["image_path"]):
            continue
        records.append(
            {
                "kind": "multimodal",
                "image": row["image_path"],
                "prompt": "Describe this image in one short sentence.",
                "target": row["caption"],
            }
        )
        if target_count is not None and len(records) >= target_count:
            break

    return records


def _load_instruct_records(
    args: argparse.Namespace,
    split: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    cache_dir = args.data_path or os.environ.get("HF_HOME", "/tmp/hf_cache")
    ds = _hf_load(
        dataset_name=_resolve_dataset_setting(args, "dataset_name"),
        dataset_config=_resolve_dataset_setting(args, "dataset_config"),
        split=split,
        cache_dir=cache_dir,
        hf_max_retries=args.hf_max_retries,
        hf_timeout=args.hf_timeout,
    )
    ds = maybe_subsample(ds, max_samples, args.seed)
    records = []
    for row in ds:
        prompt = (row.get("edit_prompt") or "").strip()
        original_image = row.get("original_image")
        edited_image = row.get("edited_image")
        if not prompt or original_image is None or edited_image is None:
            continue
        records.append(
            {
                "kind": "image_edit",
                "prompt": prompt,
                "original_image": original_image,
                "edited_image": edited_image,
            }
        )
    return records


def _load_text_records_for_split(
    args: argparse.Namespace,
    split: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    cache_dir = args.data_path or os.environ.get("HF_HOME", "/tmp/hf_cache")
    ds = _hf_load(
        dataset_name=_resolve_dataset_setting(args, "dataset_name"),
        dataset_config=_resolve_dataset_setting(args, "dataset_config"),
        split=split,
        cache_dir=cache_dir,
        hf_max_retries=args.hf_max_retries,
        hf_timeout=args.hf_timeout,
    )
    ds = ds.filter(lambda x: bool((x.get("text") or "").strip()))
    ds = maybe_subsample(ds, max_samples, args.seed)
    return [{"kind": "text", "text": row["text"].strip()} for row in ds]


def _load_text_records(
    args: argparse.Namespace,
    split: str,
    max_samples: int,
) -> List[Dict[str, Any]]:
    return _load_text_records_for_split(
        args,
        split=split,
        max_samples=max_samples,
    )


def load_train_and_eval_records(
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_max, eval_max, combined_max = _train_and_eval_load_limits(args)
    eval_split = _resolve_dataset_setting(args, "eval_split")

    if args.dataset_profile == "vqav2":
        train_records = _load_vqav2_records(
            args,
            split=_resolve_dataset_setting(args, "train_split"),
            max_samples=train_max,
        )
        eval_records = []
        if eval_split and eval_max >= 0:
            eval_records = _load_vqav2_records(args, split=eval_split, max_samples=eval_max)
        return train_records, eval_records

    if args.dataset_profile == "cc3m":
        if eval_split and (Path(args.cc3m_dir) / eval_split / "downloads.jsonl").exists():
            train_records = _load_cc3m_records(
                args,
                split=_resolve_dataset_setting(args, "train_split"),
                max_samples=train_max,
            )
            eval_records = []
            if eval_max >= 0:
                eval_records = _load_cc3m_records(args, split=eval_split, max_samples=eval_max)
            return train_records, eval_records

        all_records = _load_cc3m_records(
            args,
            split=_resolve_dataset_setting(args, "train_split"),
            max_samples=combined_max,
        )
        return _split_records_for_eval(
            all_records,
            train_max_samples=train_max,
            eval_max_samples=eval_max,
            seed=args.seed,
            holdout_fraction=args.eval_holdout_fraction,
        )

    if args.dataset_profile == "instruct":
        if eval_split:
            train_records = _load_instruct_records(
                args,
                split=_resolve_dataset_setting(args, "train_split"),
                max_samples=train_max,
            )
            eval_records = []
            if eval_max >= 0:
                eval_records = _load_instruct_records(args, split=eval_split, max_samples=eval_max)
            return train_records, eval_records

        all_records = _load_instruct_records(
            args,
            split=_resolve_dataset_setting(args, "train_split"),
            max_samples=combined_max,
        )
        return _split_records_for_eval(
            all_records,
            train_max_samples=train_max,
            eval_max_samples=eval_max,
            seed=args.seed,
            holdout_fraction=args.eval_holdout_fraction,
        )

    if args.dataset_profile == "text":
        train_records = _load_text_records(
            args,
            split=_resolve_dataset_setting(args, "train_split"),
            max_samples=train_max,
        )
        eval_records = []
        if eval_split and eval_max >= 0:
            eval_records = _load_text_records(args, split=eval_split, max_samples=eval_max)
        return train_records, eval_records

    raise ValueError(f"Unsupported dataset profile: {args.dataset_profile}")


class JanusMultimodalTrainDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], processor):
        self.records = records
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_ref):
        if isinstance(image_ref, Image.Image):
            return image_ref.convert("RGB")
        with Image.open(image_ref) as image:
            return image.convert("RGB")

    def _copy_cached_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        cached = dict(example)
        if "raw_image" in cached:
            cached["raw_image"] = cached["raw_image"].copy()
        return cached

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cached = self._cache.get(idx)
        if cached is not None:
            return self._copy_cached_example(cached)

        ex = self.records[idx]
        image = self._load_image(ex["image"])
        prompt = ex["prompt"]
        target = ex["target"]

        prompt_conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        train_conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": target},
        ]

        prompt_prepare = self.processor(
            conversations=prompt_conversation,
            images=[image],
            force_batchify=True,
        )
        train_prepare = self.processor(
            conversations=train_conversation,
            images=[image],
            force_batchify=True,
        )

        input_ids = train_prepare.input_ids.squeeze(0)
        attention_mask = train_prepare.attention_mask.squeeze(0)
        labels = input_ids.clone()
        prompt_len = prompt_prepare.input_ids.shape[-1]
        labels[:prompt_len] = -100
        labels[labels == self.tokenizer.pad_token_id] = -100

        image_token_id = getattr(self.processor, "image_id", None)
        if image_token_id is not None:
            labels[input_ids == image_token_id] = -100

        example = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": train_prepare.pixel_values.squeeze(0),
            "images_seq_mask": train_prepare.images_seq_mask.squeeze(0),
            "images_emb_mask": train_prepare.images_emb_mask.squeeze(0),
            "labels": labels,
            "prompt_input_ids": prompt_prepare.input_ids.squeeze(0),
            "prompt_attention_mask": prompt_prepare.attention_mask.squeeze(0),
            "prompt_images_seq_mask": prompt_prepare.images_seq_mask.squeeze(0),
            "raw_image": image.copy(),
            "prompt_text": prompt,
            "target_text": target,
            "gt_answers": ex.get("gt_answers"),
            "question": ex.get("question"),
            "multiple_choice_answer": ex.get("multiple_choice_answer"),
            "question_id": ex.get("question_id"),
            "image_id": ex.get("image_id"),
            "question_type": ex.get("question_type", ""),
            "answer_type": ex.get("answer_type", ""),
        }
        self._cache[idx] = example
        return self._copy_cached_example(example)


class JanusTextTrainDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], tokenizer, max_text_len: int):
        self.tokenizer = tokenizer
        self.block_size = max_text_len
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.examples = self._build_examples(records)

    def __len__(self) -> int:
        return len(self.examples)

    def _build_examples(self, records: List[Dict[str, Any]]) -> List[torch.Tensor]:
        texts = [ex["text"] for ex in records if ex.get("text")]
        if not texts:
            return []

        tokenized = self.tokenizer(texts, add_special_tokens=False)
        eos_token_id = self.tokenizer.eos_token_id
        all_tokens: List[int] = []
        for token_ids in tokenized["input_ids"]:
            if not token_ids:
                continue
            all_tokens.extend(token_ids)
            if eos_token_id is not None:
                all_tokens.append(eos_token_id)

        if len(all_tokens) < 2:
            return []

        examples: List[torch.Tensor] = []
        for start in range(0, len(all_tokens), self.block_size):
            chunk = all_tokens[start : start + self.block_size]
            if len(chunk) < 2:
                continue
            if len(chunk) < self.block_size and examples:
                break
            examples.append(torch.tensor(chunk, dtype=torch.long))
        return examples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        input_ids = self.examples[idx]
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class JanusImageEditTrainDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], processor, gen_image_size: int):
        self.records = records
        self.processor = processor
        self.gen_image_size = gen_image_size
        self._cache: Dict[int, Dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, image_ref):
        if isinstance(image_ref, Image.Image):
            return image_ref.convert("RGB")
        with Image.open(image_ref) as image:
            return image.convert("RGB")

    def _copy_cached_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        return dict(example)

    def _prepare_generation_target(self, image: Image.Image) -> torch.Tensor:
        image = image.resize((self.gen_image_size, self.gen_image_size), Image.BICUBIC)
        pixels = torch.from_numpy(np.array(image, dtype="float32")).permute(2, 0, 1)
        pixels = pixels / 127.5 - 1.0
        return pixels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        cached = self._cache.get(idx)
        if cached is not None:
            return self._copy_cached_example(cached)

        ex = self.records[idx]
        original_image = self._load_image(ex["original_image"])
        edited_image = self._load_image(ex["edited_image"])
        prompt_text = (
            "<image_placeholder>\n"
            f"{ex['prompt']}\n"
            "Generate the edited image that follows the instruction."
        )
        conversation = [
            {
                "role": "<|User|>",
                "content": prompt_text,
                "images": [original_image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        sft_format = self.processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=self.processor.sft_format,
            system_prompt="",
        )
        prompt = sft_format + self.processor.image_start_tag
        prepare = self.processor(prompt=prompt, images=[original_image], force_batchify=True)
        example = {
            "input_ids": prepare.input_ids.squeeze(0),
            "attention_mask": prepare.attention_mask.squeeze(0),
            "pixel_values": prepare.pixel_values.squeeze(0),
            "images_seq_mask": prepare.images_seq_mask.squeeze(0),
            "images_emb_mask": prepare.images_emb_mask.squeeze(0),
            "edited_pixels": self._prepare_generation_target(edited_image),
            "prompt_text": ex["prompt"],
        }
        self._cache[idx] = example
        return self._copy_cached_example(example)


def collate_multimodal(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    def _pad_1d(tensors, pad_value):
        max_len = max(t.shape[0] for t in tensors)
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, -t.shape[0]:] = t
        return out

    collated = {
        "input_ids": _pad_1d([b["input_ids"] for b in batch], pad_token_id),
        "attention_mask": _pad_1d([b["attention_mask"] for b in batch], 0),
        "images_seq_mask": _pad_1d([b["images_seq_mask"] for b in batch], False),
        "labels": _pad_1d([b["labels"] for b in batch], -100),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "images_emb_mask": torch.stack([b["images_emb_mask"] for b in batch]),
    }
    if "prompt_input_ids" in batch[0]:
        collated["prompt_input_ids"] = _pad_1d([b["prompt_input_ids"] for b in batch], pad_token_id)
        collated["prompt_attention_mask"] = _pad_1d([b["prompt_attention_mask"] for b in batch], 0)
        collated["prompt_images_seq_mask"] = _pad_1d(
            [b["prompt_images_seq_mask"] for b in batch], False
        )
        collated["raw_images"] = [b["raw_image"] for b in batch]
        collated["prompt_texts"] = [b["prompt_text"] for b in batch]
        collated["target_texts"] = [b["target_text"] for b in batch]
        collated["gt_answers"] = [b["gt_answers"] for b in batch]
        collated["questions"] = [b["question"] for b in batch]
        collated["multiple_choice_answers"] = [b["multiple_choice_answer"] for b in batch]
        collated["question_ids"] = [b["question_id"] for b in batch]
        collated["image_ids"] = [b["image_id"] for b in batch]
        collated["question_types"] = [b["question_type"] for b in batch]
        collated["answer_types"] = [b["answer_type"] for b in batch]
    return collated


def collate_text(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    max_len = max(x["input_ids"].shape[0] for x in batch)
    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        attention_mask[i, :seq_len] = item["attention_mask"]
        labels[i, :seq_len] = item["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def collate_image_edit(batch: List[Dict[str, Any]], pad_token_id: int) -> Dict[str, torch.Tensor]:
    def _pad_1d(tensors, pad_value):
        max_len = max(t.shape[0] for t in tensors)
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, -t.shape[0]:] = t
        return out

    return {
        "input_ids": _pad_1d([b["input_ids"] for b in batch], pad_token_id),
        "attention_mask": _pad_1d([b["attention_mask"] for b in batch], 0),
        "images_seq_mask": _pad_1d([b["images_seq_mask"] for b in batch], False),
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "images_emb_mask": torch.stack([b["images_emb_mask"] for b in batch]),
        "edited_pixels": torch.stack([b["edited_pixels"] for b in batch]),
        "prompt_texts": [b["prompt_text"] for b in batch],
    }


def build_task_dataset(args: argparse.Namespace, records: List[Dict[str, Any]], processor):
    if args.dataset_profile in {"vqav2", "cc3m"}:
        ds = JanusMultimodalTrainDataset(records, processor)
        collate_fn = lambda batch: collate_multimodal(batch, processor.pad_id)
        task_kind = "multimodal"
    elif args.dataset_profile == "instruct":
        ds = JanusImageEditTrainDataset(records, processor, args.gen_image_size)
        collate_fn = lambda batch: collate_image_edit(batch, processor.pad_id)
        task_kind = "image_edit"
    else:
        ds = JanusTextTrainDataset(records, processor.tokenizer, args.max_text_len)
        pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
        collate_fn = lambda batch: collate_text(batch, pad_token_id)
        task_kind = "text"
    return ds, collate_fn, task_kind


def janus_multimodal_train_step(model, batch: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    vision_dtype = next(model.vision_model.parameters()).dtype
    inputs_embeds = model.prepare_inputs_embeds(
        input_ids=batch["input_ids"].to(device),
        pixel_values=batch["pixel_values"].to(device=device, dtype=vision_dtype),
        images_seq_mask=batch["images_seq_mask"].to(device),
        images_emb_mask=batch["images_emb_mask"].to(device),
    )
    inputs_embeds = inputs_embeds.requires_grad_()
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    return outputs.loss


def janus_text_train_step(model, batch: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    input_ids = batch["input_ids"].to(device)
    # With gradient checkpointing enabled and the base LM frozen, we need the
    # text inputs to enter the graph explicitly so LoRA weights receive grads.
    inputs_embeds = model.language_model.get_input_embeddings()(input_ids).requires_grad_()
    outputs = model.language_model(
        inputs_embeds=inputs_embeds,
        attention_mask=batch["attention_mask"].to(device),
        labels=batch["labels"].to(device),
    )
    return outputs.loss


def janus_image_edit_train_step(model, batch: Dict[str, torch.Tensor], device: str) -> torch.Tensor:
    vision_dtype = next(model.vision_model.parameters()).dtype
    cond_embeds = model.prepare_inputs_embeds(
        input_ids=batch["input_ids"].to(device),
        pixel_values=batch["pixel_values"].to(device=device, dtype=vision_dtype),
        images_seq_mask=batch["images_seq_mask"].to(device),
        images_emb_mask=batch["images_emb_mask"].to(device),
    )

    gen_dtype = next(model.gen_vision_model.parameters()).dtype
    target_images = batch["edited_pixels"].to(device=device, dtype=gen_dtype)
    with torch.no_grad():
        _, _, info = model.gen_vision_model.encode(target_images)
        target_tokens = info[2].view(target_images.shape[0], -1).long()

    teacher_tokens = target_tokens[:, :-1]
    teacher_embeds = model.prepare_gen_img_embeds(teacher_tokens).to(cond_embeds.dtype)
    full_embeds = torch.cat([cond_embeds, teacher_embeds], dim=1).requires_grad_()
    text_attention = batch["attention_mask"].to(device)
    image_attention = torch.ones(
        target_tokens.shape[0],
        teacher_tokens.shape[1],
        dtype=text_attention.dtype,
        device=device,
    )
    full_attention = torch.cat([text_attention, image_attention], dim=1)

    outputs = model.language_model(
        inputs_embeds=full_embeds,
        attention_mask=full_attention,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    hidden_states = outputs.hidden_states[-1]
    cond_len = cond_embeds.shape[1]
    pred_states = hidden_states[:, cond_len - 1 :, :]
    logits = model.gen_head(pred_states)
    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_tokens.reshape(-1),
    )
    return loss


class ClipScoreEvaluator:
    def __init__(self, model_name_or_path: str, device: str):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name_or_path)
        self.model = CLIPModel.from_pretrained(model_name_or_path).to(device).eval()

    @torch.no_grad()
    def score(self, images: List[Image.Image], texts: List[str]) -> List[float]:
        if not images:
            return []
        inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        image_embeds = F.normalize(outputs.image_embeds.float(), dim=-1)
        text_embeds = F.normalize(outputs.text_embeds.float(), dim=-1)
        sims = (image_embeds * text_embeds).sum(dim=-1).clamp(min=0.0) * 100.0
        return sims.detach().cpu().tolist()


class LpipsEvaluator:
    def __init__(self, device: str, net: str):
        import lpips

        self.device = device
        self.metric = lpips.LPIPS(net=net).to(device).eval()

    @torch.no_grad()
    def score(self, generated: torch.Tensor, target: torch.Tensor) -> List[float]:
        values = self.metric(generated.float(), target.float())
        return values.reshape(-1).detach().cpu().tolist()


def _tensor_to_pil_image(image_tensor: torch.Tensor) -> Image.Image:
    image = image_tensor.detach().cpu().float().clamp(-1.0, 1.0)
    image = ((image + 1.0) / 2.0 * 255.0).round().to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


@torch.no_grad()
def generate_multimodal_text_predictions(
    model,
    batch: Dict[str, Any],
    processor,
    device: str,
    max_new_tokens: int = 32,
) -> List[str]:
    vision_dtype = next(model.vision_model.parameters()).dtype
    inputs_embeds = model.prepare_inputs_embeds(
        input_ids=batch["prompt_input_ids"].to(device),
        pixel_values=batch["pixel_values"].to(device=device, dtype=vision_dtype),
        images_seq_mask=batch["prompt_images_seq_mask"].to(device),
        images_emb_mask=batch["images_emb_mask"].to(device),
    )
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=batch["prompt_attention_mask"].to(device),
        pad_token_id=processor.tokenizer.eos_token_id,
        bos_token_id=processor.tokenizer.bos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    return [
        clean_generation_text(text)
        for text in processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ]


@torch.no_grad()
def generate_image_edit_predictions(
    model,
    batch: Dict[str, Any],
    device: str,
    image_size: int,
    patch_size: int = 16,
    image_token_num_per_image: int = 576,
) -> torch.Tensor:
    vision_dtype = next(model.vision_model.parameters()).dtype
    cond_embeds = model.prepare_inputs_embeds(
        input_ids=batch["input_ids"].to(device),
        pixel_values=batch["pixel_values"].to(device=device, dtype=vision_dtype),
        images_seq_mask=batch["images_seq_mask"].to(device),
        images_emb_mask=batch["images_emb_mask"].to(device),
    )
    inputs_embeds = cond_embeds
    past_key_values = None
    batch_size = cond_embeds.shape[0]
    generated_tokens = torch.zeros(
        (batch_size, image_token_num_per_image),
        dtype=torch.long,
        device=device,
    )

    for token_idx in range(image_token_num_per_image):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None:
            hidden_state_stack = getattr(outputs, "hidden_states", None)
            if hidden_state_stack is None:
                raise RuntimeError(
                    "Janus image generation did not return decoder hidden states."
                )
            hidden_states = hidden_state_stack[-1]
        logits = model.gen_head(hidden_states[:, -1, :])
        next_token = torch.argmax(logits, dim=-1)
        generated_tokens[:, token_idx] = next_token
        next_embeds = model.prepare_gen_img_embeds(next_token).to(cond_embeds.dtype)
        inputs_embeds = next_embeds.unsqueeze(1)

    decoded = model.gen_vision_model.decode_code(
        generated_tokens.to(dtype=torch.int),
        shape=[batch_size, 8, image_size // patch_size, image_size // patch_size],
    )
    return decoded.float().clamp(-1.0, 1.0)


def build_optional_metric_evaluators(
    args: argparse.Namespace,
    device: str,
) -> tuple[Any, Any, Dict[str, str]]:
    clip_evaluator = None
    lpips_evaluator = None
    metric_errors: Dict[str, str] = {}

    if args.dataset_profile in {"cc3m", "instruct"}:
        try:
            clip_evaluator = ClipScoreEvaluator(args.clip_model_name_or_path, device)
        except Exception as exc:  # pragma: no cover - depends on optional env/model download
            metric_errors["clipscore"] = f"{type(exc).__name__}: {exc}"

    if args.dataset_profile == "instruct":
        try:
            lpips_evaluator = LpipsEvaluator(device, args.lpips_net)
        except Exception as exc:  # pragma: no cover - depends on optional env/model download
            metric_errors["lpips"] = f"{type(exc).__name__}: {exc}"

    return clip_evaluator, lpips_evaluator, metric_errors


@torch.no_grad()
def evaluate_vqa_accuracy(
    model,
    dataloader: DataLoader,
    processor,
    device: str,
    accelerator=None,
) -> float:
    model.eval()
    total_score, total = 0.0, 0
    for batch in dataloader:
        with accelerator_autocast(accelerator):
            preds = generate_multimodal_text_predictions(
                model, batch, processor, device, max_new_tokens=10
            )
        for pred, gt_answers in zip(preds, batch["gt_answers"]):
            total_score += vqa_soft_score(pred, gt_answers or [])
            total += 1
    return total_score / max(total, 1)


@torch.no_grad()
def evaluate_caption_clipscore(
    model,
    dataloader: DataLoader,
    processor,
    device: str,
    clip_evaluator,
    accelerator=None,
) -> float | None:
    if clip_evaluator is None:
        return None

    model.eval()
    total_score, total = 0.0, 0
    for batch in dataloader:
        with accelerator_autocast(accelerator):
            preds = generate_multimodal_text_predictions(
                model, batch, processor, device, max_new_tokens=32
            )
        scores = clip_evaluator.score(batch["raw_images"], preds)
        total_score += float(sum(scores))
        total += len(scores)
    return total_score / max(total, 1)


@torch.no_grad()
def evaluate_image_edit_metrics(
    model,
    dataloader: DataLoader,
    device: str,
    image_size: int,
    clip_evaluator,
    lpips_evaluator,
    accelerator=None,
) -> Dict[str, float | None]:
    model.eval()
    total_clip, clip_count = 0.0, 0
    total_lpips, lpips_count = 0.0, 0

    for batch in dataloader:
        with accelerator_autocast(accelerator):
            generated = generate_image_edit_predictions(
                model,
                batch,
                device=device,
                image_size=image_size,
            )
        if clip_evaluator is not None:
            pil_images = [_tensor_to_pil_image(image) for image in generated]
            clip_scores = clip_evaluator.score(pil_images, batch["prompt_texts"])
            total_clip += float(sum(clip_scores))
            clip_count += len(clip_scores)

        if lpips_evaluator is not None:
            target = batch["edited_pixels"].to(device=device, dtype=generated.dtype)
            lpips_scores = lpips_evaluator.score(generated.to(device), target)
            total_lpips += float(sum(lpips_scores))
            lpips_count += len(lpips_scores)

    return {
        "clipscore": total_clip / max(clip_count, 1) if clip_count else None,
        "lpips": total_lpips / max(lpips_count, 1) if lpips_count else None,
    }


def train_client_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
    device: str,
    grad_accum: int,
    task_kind: str,
    base_lr: float,
    total_training_steps: int,
    warmup_steps: int,
    lr_scheduler_name: str,
    min_lr_ratio: float,
    global_optimizer_step: int,
    accelerator=None,
) -> tuple[float, Dict[str, torch.Tensor], int, int, float | None, float | None]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, num_steps = 0.0, len(dataloader)
    grad_sums = {
        name: torch.zeros_like(param.detach().cpu(), dtype=torch.float32)
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    optimizer_steps = 0
    lr_start = None
    lr_end = None

    def _accumulate_current_grads() -> None:
        nonlocal optimizer_steps
        for name, param in model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            grad_sums[name] += param.grad.detach().cpu().float()
        optimizer_steps += 1

    for step, batch in enumerate(dataloader, 1):
        with accelerator_autocast(accelerator):
            if task_kind == "multimodal":
                loss = janus_multimodal_train_step(model, batch, device)
            elif task_kind == "image_edit":
                loss = janus_image_edit_train_step(model, batch, device)
            else:
                loss = janus_text_train_step(model, batch, device)
        total_loss += float(loss.item())
        if accelerator is None:
            (loss / grad_accum).backward()
        else:
            accelerator.backward(loss / grad_accum)
        if step % grad_accum == 0:
            _accumulate_current_grads()
            current_lr = _compute_scheduled_lr(
                base_lr=base_lr,
                global_step=global_optimizer_step,
                total_steps=total_training_steps,
                scheduler_name=lr_scheduler_name,
                warmup_steps=warmup_steps,
                min_lr_ratio=min_lr_ratio,
            )
            _set_optimizer_lr(optimizer, current_lr)
            lr_start = current_lr if lr_start is None else lr_start
            lr_end = current_lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_optimizer_step += 1

    if num_steps % grad_accum != 0:
        _accumulate_current_grads()
        current_lr = _compute_scheduled_lr(
            base_lr=base_lr,
            global_step=global_optimizer_step,
            total_steps=total_training_steps,
            scheduler_name=lr_scheduler_name,
            warmup_steps=warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )
        _set_optimizer_lr(optimizer, current_lr)
        lr_start = current_lr if lr_start is None else lr_start
        lr_end = current_lr
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        global_optimizer_step += 1

    avg_grads = {
        name: grad / max(optimizer_steps, 1)
        for name, grad in grad_sums.items()
    }
    return (
        total_loss / max(num_steps, 1),
        avg_grads,
        optimizer_steps,
        global_optimizer_step,
        lr_start,
        lr_end,
    )


@torch.no_grad()
def evaluate_task_loss(
    model,
    dataloader: DataLoader,
    device: str,
    task_kind: str,
    accelerator=None,
) -> float:
    model.eval()
    total_loss, num_steps = 0.0, len(dataloader)
    for batch in dataloader:
        with accelerator_autocast(accelerator):
            if task_kind == "multimodal":
                loss = janus_multimodal_train_step(model, batch, device)
            elif task_kind == "image_edit":
                loss = janus_image_edit_train_step(model, batch, device)
            else:
                loss = janus_text_train_step(model, batch, device)
        total_loss += float(loss.item())
    return total_loss / max(num_steps, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    configure_hf_transfer_timeouts(args.hf_timeout)
    accelerator = create_accelerator()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records, eval_records = load_train_and_eval_records(args)
    if not records:
        raise RuntimeError(f"No records loaded for dataset profile '{args.dataset_profile}'.")

    device = str(accelerator.device) if accelerator is not None else pick_device()
    backend = JanusProBackend()
    model, processor = backend.build_model_and_processor(
        args.model_name_or_path,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        device,
    )
    print(f"Using device: {device}")
    print(count_trainable_params(model))
    if accelerator is not None:
        model = accelerator.prepare(model)
        print(f"Accelerate enabled (mixed_precision={accelerator.mixed_precision})")
    clip_evaluator, lpips_evaluator, metric_errors = build_optional_metric_evaluators(args, device)

    metadata = {
        "dataset_profile": args.dataset_profile,
        "dataset_name": _resolve_dataset_setting(args, "dataset_name"),
        "dataset_config": _resolve_dataset_setting(args, "dataset_config"),
        "train_split": _resolve_dataset_setting(args, "train_split"),
        "eval_split": _resolve_dataset_setting(args, "eval_split"),
        "num_records": len(records),
        "max_samples": _resolve_max_samples(args),
        "num_eval_records": len(eval_records),
        "eval_max_samples": _resolve_eval_max_samples(args),
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "lr_scheduler": args.lr_scheduler,
        "warmup_ratio": args.warmup_ratio,
        "min_lr_ratio": args.min_lr_ratio,
        "lora_r": args.lora_r,
        "seed": args.seed,
        "device": device,
        "eval_holdout_fraction": args.eval_holdout_fraction,
        "clip_model_name_or_path": args.clip_model_name_or_path,
        "lpips_net": args.lpips_net,
        "metric_errors": metric_errors,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    current_params = _clone_param_dict(get_trainable_params(unwrap_model(model, accelerator)))
    _save_update(output_dir / "initial_params.pt", current_params)
    train_dataset, train_collate_fn, task_kind = build_task_dataset(args, records, processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device != "cpu",
        collate_fn=train_collate_fn,
    )
    if accelerator is not None:
        train_dataloader = accelerator.prepare(train_dataloader)

    eval_dataset = None
    eval_dataloader = None
    eval_task_kind = None
    if eval_records:
        eval_dataset, eval_collate_fn, eval_task_kind = build_task_dataset(args, eval_records, processor)
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device != "cpu",
            collate_fn=eval_collate_fn,
        )
        if accelerator is not None:
            eval_dataloader = accelerator.prepare(eval_dataloader)

    optimizer_steps_per_epoch = _optimizer_steps_per_epoch(len(train_dataloader), args.grad_accum)
    total_training_steps = max(args.rounds * args.local_epochs * optimizer_steps_per_epoch, 1)
    warmup_steps = min(
        max(int(round(total_training_steps * max(args.warmup_ratio, 0.0))), 0),
        max(total_training_steps - 1, 0),
    )
    metadata["optimizer_steps_per_epoch"] = optimizer_steps_per_epoch
    metadata["total_training_steps"] = total_training_steps
    metadata["warmup_steps"] = warmup_steps
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    global_optimizer_step = 0
    for round_idx in range(1, args.rounds + 1):
        round_dir = output_dir / f"round_{round_idx:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        load_trainable_params(unwrap_model(model, accelerator), current_params, device)
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=args.lr,
        )
        if accelerator is not None:
            optimizer = accelerator.prepare(optimizer)

        round_loss = 0.0
        round_grad_sums = {
            name: torch.zeros_like(param, dtype=torch.float32)
            for name, param in current_params.items()
        }
        total_optimizer_steps = 0
        round_lr_start = None
        round_lr_end = None

        for _ in range(args.local_epochs):
            (
                epoch_loss,
                epoch_avg_grads,
                optimizer_steps,
                global_optimizer_step,
                epoch_lr_start,
                epoch_lr_end,
            ) = train_client_epoch(
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                device=device,
                grad_accum=args.grad_accum,
                task_kind=task_kind,
                base_lr=args.lr,
                total_training_steps=total_training_steps,
                warmup_steps=warmup_steps,
                lr_scheduler_name=args.lr_scheduler,
                min_lr_ratio=args.min_lr_ratio,
                global_optimizer_step=global_optimizer_step,
                accelerator=accelerator,
            )
            round_loss = epoch_loss
            for name, grad in epoch_avg_grads.items():
                round_grad_sums[name] += grad
            total_optimizer_steps += optimizer_steps
            if epoch_lr_start is not None and round_lr_start is None:
                round_lr_start = epoch_lr_start
            if epoch_lr_end is not None:
                round_lr_end = epoch_lr_end

        round_avg_grads = {
            name: grad / max(args.local_epochs, 1)
            for name, grad in round_grad_sums.items()
        }
        new_params = _clone_param_dict(get_trainable_params(unwrap_model(model, accelerator)))
        round_delta = _subtract_param_dict(new_params, current_params)
        current_params = new_params
        eval_loss = None
        eval_acc = None
        eval_ppl = None
        eval_clipscore = None
        eval_lpips = None
        if eval_dataloader is not None:
            eval_loss = evaluate_task_loss(
                model=unwrap_model(model, accelerator),
                dataloader=eval_dataloader,
                device=device,
                task_kind=eval_task_kind,
                accelerator=accelerator,
            )
            if eval_task_kind == "multimodal":
                if args.dataset_profile == "vqav2":
                    eval_acc = evaluate_vqa_accuracy(
                        model=unwrap_model(model, accelerator),
                        dataloader=eval_dataloader,
                        processor=processor,
                        device=device,
                        accelerator=accelerator,
                    )
                elif args.dataset_profile == "cc3m":
                    eval_clipscore = evaluate_caption_clipscore(
                        model=unwrap_model(model, accelerator),
                        dataloader=eval_dataloader,
                        processor=processor,
                        device=device,
                        clip_evaluator=clip_evaluator,
                        accelerator=accelerator,
                    )
            elif eval_task_kind == "text" and eval_loss is not None:
                eval_ppl = float(math.exp(min(eval_loss, 20.0)))
            elif eval_task_kind == "image_edit":
                image_metrics = evaluate_image_edit_metrics(
                    model=unwrap_model(model, accelerator),
                    dataloader=eval_dataloader,
                    device=device,
                    image_size=args.gen_image_size,
                    clip_evaluator=clip_evaluator,
                    lpips_evaluator=lpips_evaluator,
                    accelerator=accelerator,
                )
                eval_clipscore = image_metrics["clipscore"]
                eval_lpips = image_metrics["lpips"]

        _save_update(round_dir / "round_grad.pt", round_avg_grads)
        _save_update(round_dir / "round_delta.pt", round_delta)
        _save_update(round_dir / "round_params.pt", new_params)

        summary = {
            "round": round_idx,
            "task_kind": task_kind,
            "num_samples": len(train_dataset),
            "loss": round_loss,
            "eval_num_samples": len(eval_dataset) if eval_dataset is not None else 0,
            "eval_loss": eval_loss,
            "eval_acc": eval_acc,
            "eval_ppl": eval_ppl,
            "eval_clipscore": eval_clipscore,
            "eval_lpips": eval_lpips,
            "optimizer_steps": total_optimizer_steps,
            "global_optimizer_step": global_optimizer_step,
            "lr_start": round_lr_start,
            "lr_end": round_lr_end,
            "grad_l2": float(_flatten_param_dict(round_avg_grads).norm().item()),
            "delta_l2": float(_flatten_param_dict(round_delta).norm().item()),
            "metric_errors": metric_errors,
        }
        (round_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        eval_loss_str = f"{summary['eval_loss']:.6f}" if summary["eval_loss"] is not None else "n/a"
        print(
            f"round={round_idx} samples={len(train_dataset)} "
            f"train_loss={summary['loss']:.6f} "
            f"eval_loss={eval_loss_str} "
            f"eval_acc={summary['eval_acc'] if summary['eval_acc'] is not None else 'n/a'} "
            f"eval_ppl={summary['eval_ppl'] if summary['eval_ppl'] is not None else 'n/a'} "
            f"eval_clipscore={summary['eval_clipscore'] if summary['eval_clipscore'] is not None else 'n/a'} "
            f"eval_lpips={summary['eval_lpips'] if summary['eval_lpips'] is not None else 'n/a'} "
            f"lr_start={summary['lr_start'] if summary['lr_start'] is not None else 'n/a'} "
            f"lr_end={summary['lr_end'] if summary['lr_end'] is not None else 'n/a'} "
            f"grad_l2={summary['grad_l2']:.6f} delta_l2={summary['delta_l2']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
