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

"""Unified NVFlare FL client for any registered VLM backend.

Select the model at launch time::

    python fl_client.py --model_backend blip_vqa ...
    python fl_client.py --model_backend januspro ...

The SubprocessLauncher in the NVFlare job config passes ``--model_backend``
via ``script_args``, so different sites can even run *different* models
(though typically all sites use the same one for FedAvg to make sense).
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

import nvflare.client as flare

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# This import triggers backend registration
import src  # noqa: F401
from src.common import (
    accelerator_autocast,
    configure_hf_transfer_timeouts,
    count_trainable_params,
    create_accelerator,
    get_default_datasets_root,
    get_trainable_params,
    load_dataset_with_local_fallback,
    load_trainable_params,
    maybe_subsample,
    pick_device,
    resolve_site_output_dir,
    set_seed,
    shard_dataset,
    train_one_epoch,
    unwrap_model,
    write_vqa_prediction_report,
)
from src.model_registry import get_backend


def parse_site_id(site_name: str) -> int:
    try:
        return int(site_name.split("-")[-1]) - 1
    except (ValueError, IndexError):
        return 0


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_backend", type=str, required=True,
                    help="Registry key: blip_vqa | januspro")
    p.add_argument("--model_name_or_path", type=str, default="",
                    help="HF model id (uses backend default if empty).")
    p.add_argument("--num_clients", type=int, default=2)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_q_len", type=int, default=64)
    p.add_argument("--max_a_len", type=int, default=16)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.1)
    p.add_argument(
        "--data_path",
        type=str,
        default=get_default_datasets_root(),
        help="Dataset/cache root. Defaults to the ../Datasets directory next to the repo.",
    )
    p.add_argument("--hf_max_retries", type=int, default=8,
                    help="Retries for Hugging Face dataset downloads.")
    p.add_argument("--hf_timeout", type=int, default=900,
                    help="Idle/read timeout in seconds for Hugging Face downloads.")
    p.add_argument("--dirichlet_alpha", type=float, default=0.0,
                    help="Dirichlet concentration for non-IID partition. "
                         "0 = IID round-robin, 0.1 = extreme non-IID, "
                         "0.5 = moderate, 1.0 = mild. (Paper: 0.1/0.5/1.0)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def run_eval_with_optional_logging(backend, model, eval_loader, processor, device,
                                   site: str, cur_round, phase: str,
                                   accelerator=None) -> float:
    eval_model = unwrap_model(model, accelerator)
    if not hasattr(backend, "evaluate_with_details"):
        with accelerator_autocast(accelerator):
            return backend.evaluate(eval_model, eval_loader, processor, device)

    with accelerator_autocast(accelerator):
        acc, records = backend.evaluate_with_details(eval_model, eval_loader, processor, device)
    output_dir = resolve_site_output_dir(site, start_paths=[ROOT_DIR, os.getcwd()])
    report_paths = write_vqa_prediction_report(output_dir, site, phase, cur_round, records)
    print(
        f"[{site}] wrote {len(records)} predictions to "
        f"{report_paths['txt']} and {report_paths['jsonl']}",
        flush=True,
    )
    return acc


def main() -> None:
    args = _parse_args()
    set_seed(args.seed)
    cache_dir = args.data_path or os.environ.get("HF_HOME", "/tmp/hf_cache")
    configure_hf_transfer_timeouts(args.hf_timeout)
    accelerator = create_accelerator()

    backend = get_backend(args.model_backend)
    print(f">>> Backend: {backend.name}", flush=True)

    # ---- Data ----
    print(">>> Loading dataset ...", flush=True)
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

    # ---- NVFlare init ----
    flare.init()
    site = flare.get_site_name()
    site_id = parse_site_id(site)
    train_hf = shard_dataset(train_hf, args.num_clients, site_id,
                             alpha=args.dirichlet_alpha, seed=args.seed)
    eval_hf = shard_dataset(eval_hf, args.num_clients, site_id,
                            alpha=0.0, seed=args.seed)  # eval always IID
    print(f"[{site}] train={len(train_hf)}, eval={len(eval_hf)}", flush=True)

    # ---- Model ----
    device = str(accelerator.device) if accelerator is not None else pick_device()
    print(f"[{site}] using device={device}", flush=True)
    model, processor = backend.build_model_and_processor(
        args.model_name_or_path, args.lora_r, args.lora_alpha,
        args.lora_dropout, device,
    )
    print(f"[{site}] {count_trainable_params(model)}", flush=True)

    # ---- Dataloaders ----
    train_ds = backend.build_dataset(train_hf, processor, args.max_q_len, args.max_a_len)
    eval_ds = backend.build_dataset(eval_hf, processor, args.max_q_len, args.max_a_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=backend.collate_fn, pin_memory=True)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers,
                             collate_fn=backend.collate_fn, pin_memory=True)
    if accelerator is not None:
        model, train_loader, eval_loader = accelerator.prepare(model, train_loader, eval_loader)
        print(
            f"[{site}] accelerate enabled (mixed_precision={accelerator.mixed_precision})",
            flush=True,
        )

    # ---- FL loop ----
    while flare.is_running():
        input_model = flare.receive()
        cur_round = getattr(input_model, "current_round", None)

        if input_model and getattr(input_model, "params", None):
            load_trainable_params(unwrap_model(model, accelerator), input_model.params, device)

        # -- validate --
        if flare.is_evaluate():
            acc = run_eval_with_optional_logging(
                backend, model, eval_loader, processor, device, site, cur_round, "validate",
                accelerator=accelerator,
            )
            print(f"[{site}] validate round={cur_round} acc={acc:.4f}", flush=True)
            flare.send(flare.FLModel(
                params=None,
                metrics={"global_acc": float(acc), "n_eval": len(eval_ds)},
                meta={"n_eval": len(eval_ds)},
            ))
            continue

        # -- train --
        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr)
        if accelerator is not None:
            optimizer = accelerator.prepare(optimizer)
        loss = 0.0
        for _ in range(args.local_epochs):
            loss = train_one_epoch(
                model, train_loader, optimizer, device, args.grad_accum, backend,
                accelerator=accelerator,
            )
        acc = run_eval_with_optional_logging(
            backend, model, eval_loader, processor, device, site, cur_round, "train",
            accelerator=accelerator,
        )
        steps = args.local_epochs * len(train_loader)
        print(f"[{site}] train round={cur_round} loss={loss:.4f} acc={acc:.4f}",
              flush=True)

        flare.send(flare.FLModel(
            params=get_trainable_params(unwrap_model(model, accelerator)),
            metrics={"train_loss": float(loss), "local_acc": float(acc),
                     "global_acc": float(acc),
                     "n_train": len(train_ds), "n_eval": len(eval_ds)},
            meta={"NUM_STEPS_CURRENT_ROUND": steps,
                  "n_train": len(train_ds), "n_eval": len(eval_ds)},
        ))


if __name__ == "__main__":
    main()
