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

"""Centralized (non-FL) training baseline for any registered VLM."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import src  # noqa: F401  - triggers backend registration
from src.common import (
    accelerator_autocast,
    configure_hf_transfer_timeouts,
    count_trainable_params,
    create_accelerator,
    get_default_datasets_root,
    load_dataset_with_local_fallback,
    maybe_subsample,
    pick_device,
    set_seed,
    train_one_epoch,
    unwrap_model,
)
from src.model_registry import get_backend


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_backend", type=str, required=True)
    p.add_argument("--model_name_or_path", type=str, default="")
    p.add_argument("--output_dir", type=str, default="./workspace_centralized")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_q_len", type=int, default=64)
    p.add_argument("--max_a_len", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--data_path",
        type=str,
        default=get_default_datasets_root(),
        help="Dataset/cache root. Defaults to the ../Datasets directory next to the repo.",
    )
    p.add_argument("--hf_max_retries", type=int, default=8)
    p.add_argument("--hf_timeout", type=int, default=900)
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    cache_dir = args.data_path or os.environ.get("HF_HOME", "/tmp/hf_cache")
    configure_hf_transfer_timeouts(args.hf_timeout)
    accelerator = create_accelerator()
    backend = get_backend(args.model_backend)

    train_hf = load_dataset_with_local_fallback(
        backend.hf_dataset_name(),
        backend.hf_train_split(),
        data_path=args.data_path,
        cache_dir=cache_dir,
        hf_max_retries=args.hf_max_retries,
        hf_timeout=args.hf_timeout,
    )
    eval_hf = load_dataset_with_local_fallback(
        backend.hf_dataset_name(),
        backend.hf_eval_split(),
        data_path=args.data_path,
        cache_dir=cache_dir,
        hf_max_retries=args.hf_max_retries,
        hf_timeout=args.hf_timeout,
    )
    keep = set(backend.keep_columns())
    train_hf = train_hf.remove_columns([c for c in train_hf.column_names if c not in keep])
    eval_hf = eval_hf.remove_columns([c for c in eval_hf.column_names if c not in keep])
    train_hf = maybe_subsample(train_hf, args.max_train_samples, args.seed)
    eval_hf = maybe_subsample(eval_hf, args.max_eval_samples, args.seed)
    print(f"Train: {len(train_hf)}, Eval: {len(eval_hf)}")

    device = str(accelerator.device) if accelerator is not None else pick_device()
    print(f"Using device: {device}")
    model, processor = backend.build_model_and_processor(
        args.model_name_or_path, args.lora_r, args.lora_alpha,
        args.lora_dropout, device)
    print(count_trainable_params(model))

    train_ds = backend.build_dataset(train_hf, processor, args.max_q_len, args.max_a_len)
    eval_ds = backend.build_dataset(eval_hf, processor, args.max_q_len, args.max_a_len)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers,
                          collate_fn=backend.collate_fn, pin_memory=True)
    eval_ld = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers,
                         collate_fn=backend.collate_fn, pin_memory=True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr)
    if accelerator is not None:
        model, optimizer, train_ld, eval_ld = accelerator.prepare(
            model, optimizer, train_ld, eval_ld
        )
        print(f"Accelerate enabled (mixed_precision={accelerator.mixed_precision})")

    for epoch in range(args.num_epochs):
        loss = train_one_epoch(
            model, train_ld, optimizer, device, args.grad_accum, backend,
            accelerator=accelerator,
        )
        with accelerator_autocast(accelerator):
            acc = backend.evaluate(unwrap_model(model, accelerator), eval_ld, processor, device)
        print(f"Epoch {epoch+1}/{args.num_epochs}  loss={loss:.4f}  acc={acc:.4f}")

    unwrap_model(model, accelerator).save_pretrained(args.output_dir)
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
