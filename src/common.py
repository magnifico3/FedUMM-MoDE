# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers: seeding, dataset loading, param exchange, generic training loop."""

from contextlib import nullcontext
import json
import os
import random
import string
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import Dataset, DownloadConfig, load_dataset


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATASETS_ROOT = (_REPO_ROOT.parent / "Datasets").resolve()
_LEGACY_LOCAL_VQAV2_ROOT = Path("/home/Datasets/VQAv2")
_DEFAULT_LOCAL_VQAV2_ROOT = str(_DEFAULT_DATASETS_ROOT)
_LOCAL_VQAV2_ARROW_ROOT = os.path.join(
    _DEFAULT_LOCAL_VQAV2_ROOT,
    "HuggingFaceM4___vq_av2",
    "default",
    "1.0.0",
    "e4d008385143be7a6bd81e99483e671d5096942bcb987542217121a5ac2cb420",
)


def get_default_datasets_root() -> str:
    """Return the repo-adjacent datasets directory used by default."""
    return str(_DEFAULT_DATASETS_ROOT)


def get_default_cc3m_dir() -> str:
    """Return the default CC3M download directory under the datasets root."""
    return str(_DEFAULT_DATASETS_ROOT / "cc3m")


def create_accelerator():
    """Create an optional Hugging Face Accelerator with sensible defaults."""
    try:
        from accelerate import Accelerator
    except ImportError:
        return None

    mixed_precision = "no"
    if torch.cuda.is_available():
        mixed_precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    return Accelerator(mixed_precision=mixed_precision)


def accelerator_autocast(accelerator):
    if accelerator is None:
        return nullcontext()
    return accelerator.autocast()


def backward_loss(loss: torch.Tensor, accelerator) -> None:
    if accelerator is None:
        loss.backward()
        return
    accelerator.backward(loss)


def unwrap_model(model, accelerator):
    if accelerator is None:
        return model
    return accelerator.unwrap_model(model)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    """Pick CPU or the CUDA device with the most free memory.

    If `CUDA_VISIBLE_DEVICES` is already set, respect that selection and use
    the default `cuda` device exposed by the environment.
    """
    if not torch.cuda.is_available():
        return "cpu"

    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda"

    best_idx = 0
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(idx)
        if free_bytes > best_free:
            best_idx = idx
            best_free = free_bytes
    return f"cuda:{best_idx}"


def vqa_soft_score(pred: str, gt_answers: List[str]) -> float:
    """VQA v2 official soft accuracy: min(#matches / 3, 1)."""
    p = pred.strip().lower()
    return min(1.0, sum(1 for a in gt_answers if p == a.strip().lower()) / 3.0)


def clean_generation_text(text: str) -> str:
    """Remove tokenizer artifacts without changing answer semantics."""
    cleaned = (text or "").replace("Ġ", " ").replace("▁", " ")
    cleaned = cleaned.translate(str.maketrans("", "", string.punctuation))
    return " ".join(cleaned.split()).strip()


def resolve_site_output_dir(site_name: str, start_paths: Optional[List[str]] = None) -> str:
    """Find the simulator site directory for writing debugging artifacts."""
    checked = []
    for start in (start_paths or []) + [os.getcwd()]:
        if not start:
            continue
        path = Path(start).resolve()
        for candidate in [path] + list(path.parents):
            if candidate.name == site_name:
                candidate.mkdir(parents=True, exist_ok=True)
                return str(candidate)
            checked.append(candidate)
            workspace_candidate = candidate / "workspace_simulator" / site_name
            if workspace_candidate.exists():
                workspace_candidate.mkdir(parents=True, exist_ok=True)
                return str(workspace_candidate)

    fallback = Path(os.getcwd()).resolve() / "workspace_simulator" / site_name
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


def write_vqa_prediction_report(output_dir: str, site_name: str, phase: str,
                                round_idx: Optional[int], records: List[Dict[str, Any]]):
    """Write VQA prediction details to JSONL and a human-readable TXT file."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    round_label = f"{int(round_idx):03d}" if round_idx is not None else "na"
    stem = f"vqa_predictions_{phase}_round_{round_label}"
    jsonl_path = out_dir / f"{stem}.jsonl"
    txt_path = out_dir / f"{stem}.txt"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    header = [
        f"site: {site_name}",
        f"phase: {phase}",
        f"round: {round_idx}",
        f"num_examples: {len(records)}",
        "",
    ]
    lines = header
    for idx, record in enumerate(records, 1):
        gt_answers = ", ".join(record.get("gt_answers", []))
        lines.extend([
            f"[{idx}] question_id={record.get('question_id')} image_id={record.get('image_id')}",
            f"question: {record.get('question', '')}",
            f"prediction_raw: {record.get('prediction_raw', '')}",
            f"prediction_clean: {record.get('prediction_clean', '')}",
            f"multiple_choice_answer: {record.get('multiple_choice_answer', '')}",
            f"gt_answers: {gt_answers}",
            f"soft_score: {record.get('soft_score', 0.0):.4f}",
            f"match_count: {record.get('match_count', 0)} / {record.get('num_gt_answers', 0)}",
            f"question_type: {record.get('question_type', '')}",
            f"answer_type: {record.get('answer_type', '')}",
            "",
        ])

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {
        "jsonl": str(jsonl_path),
        "txt": str(txt_path),
    }


def maybe_subsample(ds, max_samples: Optional[int], seed: int):
    if max_samples is None or max_samples < 0 or max_samples >= len(ds):
        return ds
    return ds.shuffle(seed=seed).select(range(max_samples))


def _local_vqav2_arrow_path(split: str, data_path: str = "") -> str:
    roots = []
    if data_path:
        roots.append(data_path)
    roots.extend(
        [
            _DEFAULT_LOCAL_VQAV2_ROOT,
            os.path.join(_DEFAULT_LOCAL_VQAV2_ROOT, "VQAv2"),
            str(_LEGACY_LOCAL_VQAV2_ROOT),
        ]
    )

    split_name = split.split("[", 1)[0]
    filename = f"vq_av2-{split_name}.arrow"
    seen = set()
    for root in roots:
        if not root:
            continue
        normalized_root = os.path.abspath(root)
        if normalized_root in seen:
            continue
        seen.add(normalized_root)
        candidates = [
            os.path.join(normalized_root, filename),
            os.path.join(
                normalized_root,
                "HuggingFaceM4___vq_av2",
                "default",
                "1.0.0",
                "e4d008385143be7a6bd81e99483e671d5096942bcb987542217121a5ac2cb420",
                filename,
            ),
            os.path.join(
                normalized_root,
                "default",
                "1.0.0",
                "e4d008385143be7a6bd81e99483e671d5096942bcb987542217121a5ac2cb420",
                filename,
            ),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    if os.path.exists(os.path.join(_LOCAL_VQAV2_ARROW_ROOT, filename)):
        return os.path.join(_LOCAL_VQAV2_ARROW_ROOT, filename)
    return ""


def _apply_percent_slice(ds, split: str):
    if "[" not in split or "]" not in split:
        return ds

    slice_expr = split[split.index("[") + 1: split.index("]")]
    if ":" not in slice_expr:
        return ds

    begin, finish = slice_expr.split(":", 1)
    if not begin.endswith("%") and begin != "":
        return ds
    if not finish.endswith("%") and finish != "":
        return ds

    begin_pct = int(begin.rstrip("%")) if begin else 0
    finish_pct = int(finish.rstrip("%")) if finish else 100
    n = len(ds)
    begin_idx = n * begin_pct // 100
    finish_idx = n * finish_pct // 100
    return ds.select(range(begin_idx, finish_idx))


def configure_hf_transfer_timeouts(timeout_seconds: int) -> Dict[str, Any]:
    """Configure Hugging Face/fsspec timeouts for large downloads on slow links."""
    timeout_seconds = max(1, int(timeout_seconds))
    metadata_timeout = min(timeout_seconds, 120)

    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(timeout_seconds)
    os.environ["HF_HUB_ETAG_TIMEOUT"] = str(metadata_timeout)

    try:
        import aiohttp
    except ImportError:
        return {}

    return {
        "client_kwargs": {
            # Keep the overall transfer unbounded so multi-GB downloads can finish,
            # while still failing if the connection goes idle for too long.
            "timeout": aiohttp.ClientTimeout(
                total=None,
                connect=metadata_timeout,
                sock_connect=metadata_timeout,
                sock_read=timeout_seconds,
            )
        }
    }


def build_hf_download_config(cache_dir: str = "", max_retries: int = 8,
                             timeout_seconds: int = 900) -> DownloadConfig:
    storage_options = configure_hf_transfer_timeouts(timeout_seconds)
    return DownloadConfig(
        cache_dir=cache_dir or None,
        resume_download=True,
        max_retries=max(1, int(max_retries)),
        storage_options=storage_options,
    )


def load_dataset_with_local_fallback(dataset_name: str, split: str, data_path: str = "",
                                     cache_dir: str = "", hf_max_retries: int = 8,
                                     hf_timeout: int = 900):
    """Prefer a local VQAv2 Arrow export before falling back to Hugging Face."""
    if dataset_name == "HuggingFaceM4/VQAv2":
        local_arrow = _local_vqav2_arrow_path(split, data_path)
        if local_arrow:
            return _apply_percent_slice(Dataset.from_file(local_arrow), split)

    require_legacy_datasets_for_vqav2(dataset_name)
    download_config = build_hf_download_config(
        cache_dir=cache_dir,
        max_retries=hf_max_retries,
        timeout_seconds=hf_timeout,
    )
    return load_dataset(
        dataset_name,
        split=split,
        cache_dir=cache_dir,
        download_config=download_config,
        storage_options=download_config.storage_options,
    )


def prefer_local_model_path(model_name_or_path: str, default_model_id: str,
                            local_candidates: List[str]) -> str:
    """Prefer an explicit path first, then known local model directories."""
    if model_name_or_path:
        return model_name_or_path

    for candidate in local_candidates:
        if os.path.exists(candidate):
            if os.path.basename(candidate) == "snapshots" and os.path.isdir(candidate):
                children = sorted(
                    os.path.join(candidate, name)
                    for name in os.listdir(candidate)
                    if os.path.isdir(os.path.join(candidate, name))
                )
                if children:
                    return children[-1]
            return candidate
    return default_model_id


def require_legacy_datasets_for_vqav2(dataset_name: str) -> None:
    """Fail fast for legacy script-based datasets on unsupported `datasets` versions.

    `HuggingFaceM4/VQAv2` is distributed as a dataset loading script (`VQAv2.py`).
    Hugging Face removed support for dataset scripts in `datasets` 4.x, so without
    this guard users only see a deep stack trace from inside the library.
    """
    if dataset_name != "HuggingFaceM4/VQAv2":
        return

    cur = pkg_version("datasets")
    major = int(cur.split(".", 1)[0])
    if major >= 4:
        raise RuntimeError(
            "The default dataset 'HuggingFaceM4/VQAv2' uses a legacy dataset script, "
            "but Hugging Face 'datasets' "
            f"{cur} no longer supports dataset scripts. "
            "Please install a 2.x/3.x release, for example:\n"
            '  pip install "datasets>=2.14,<4"\n'
            "Then rerun the command."
        )


def shard_dataset(ds, num_clients: int, site_id: int,
                  alpha: float = 0.0, seed: int = 42,
                  label_key: str = "multiple_choice_answer"):
    """Partition a HuggingFace dataset across clients.

    Args:
        ds: HuggingFace Dataset.
        num_clients: total number of FL clients.
        site_id: this client's index (0-based).
        alpha: Dirichlet concentration.
            alpha <= 0  -> deterministic round-robin (IID baseline).
            alpha > 0   -> Dirichlet non-IID (lower = more skewed).
        seed: random seed for reproducibility.
        label_key: column name used as the label for Dirichlet grouping.
            For VQA datasets this is typically "multiple_choice_answer".
    """
    if alpha <= 0.0:
        # IID round-robin fallback
        return ds.select([i for i in range(len(ds)) if i % num_clients == site_id])

    rng = np.random.default_rng(seed)
    n = len(ds)

    # Build label -> indices mapping
    if label_key in ds.column_names:
        # Map answer strings to integer class ids
        answers = ds[label_key]
        unique = sorted(set(answers))
        label_to_id = {a: i for i, a in enumerate(unique)}
        labels = np.array([label_to_id[a] for a in answers])
    else:
        # Fallback: use index mod 100 as pseudo-label
        labels = np.array([i % 100 for i in range(n)])

    num_classes = int(labels.max()) + 1

    # Dirichlet: for each class, sample a proportion vector over clients
    # proportions[c] = [p_client0, p_client1, ..., p_clientK]
    # sum(proportions[c]) = 1
    proportions = rng.dirichlet([alpha] * num_clients, size=num_classes)

    client_indices = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)

        # Split class_idx according to proportions
        cumsum = np.cumsum(proportions[c])
        # Convert proportions to index boundaries
        splits = (cumsum[:-1] * len(class_idx)).astype(int)
        chunks = np.split(class_idx, splits)

        for k in range(num_clients):
            client_indices[k].extend(chunks[k].tolist())

    # Sort for deterministic ordering, then return this client's subset
    indices = sorted(client_indices[site_id])
    return ds.select(indices)


def count_trainable_params(model) -> str:
    t = sum(p.numel() for p in model.parameters() if p.requires_grad)
    a = sum(p.numel() for p in model.parameters())
    return f"trainable: {t:,} / {a:,} ({100*t/a:.4f}%)"


def get_trainable_params(model) -> Dict[str, torch.Tensor]:
    params = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        tensor = p.detach().cpu()
        if tensor.is_floating_point():
            # NVFlare's default numpy transport does not support bfloat16.
            tensor = tensor.to(torch.float32)
        params[n] = tensor
    return params


def load_trainable_params(model, params: Dict[str, Any], device: str) -> None:
    tmap = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, v in (params or {}).items():
        if n not in tmap:
            continue
        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v)
        tmap[n].data.copy_(v.to(device=device, dtype=tmap[n].dtype))


def train_one_epoch(model, dataloader, optimizer, device, grad_accum, backend,
                    accelerator=None) -> float:
    """Generic training loop - delegates per-batch loss to backend.train_step."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss, num_steps = 0.0, len(dataloader)

    for step, batch in enumerate(dataloader, 1):
        with accelerator_autocast(accelerator):
            loss = backend.train_step(model, batch, device)
        total_loss += loss.item()
        backward_loss(loss / grad_accum, accelerator)
        if step % grad_accum == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if num_steps % grad_accum != 0:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return total_loss / max(num_steps, 1)
