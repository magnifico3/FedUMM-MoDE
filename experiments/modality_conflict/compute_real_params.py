#!/usr/bin/env python3
"""Compute real LoRA weight changes (B @ A) and save as flattened vectors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch


def extract_lora_deltas(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract all LoRA weight deltas (B @ A) from a parameter dictionary.
    
    For each LoRA layer, finds pairs of lora_A and lora_B parameters,
    computes their matrix product to get the effective weight change.
    """
    # Collect all LoRA parameters by layer
    lora_params: Dict[str, Dict[str, torch.Tensor]] = {}
    
    for param_name, param_tensor in params.items():
        if "lora_A" in param_name or "lora_B" in param_name:
            # Extract layer identifier, removing .default.weight suffix
            if ".lora_A.default.weight" in param_name:
                layer_key = param_name.replace(".lora_A.default.weight", "")
                param_type = "lora_A"
            elif ".lora_B.default.weight" in param_name:
                layer_key = param_name.replace(".lora_B.default.weight", "")
                param_type = "lora_B"
            else:
                continue
            
            if layer_key not in lora_params:
                lora_params[layer_key] = {}
            lora_params[layer_key][param_type] = param_tensor
    
    # Compute B @ A for each layer
    lora_deltas = {}
    for layer_key, layer_params in lora_params.items():
        if "lora_B" in layer_params and "lora_A" in layer_params:
            lora_b = layer_params["lora_B"]  # shape: [output_dim, r]
            lora_a = layer_params["lora_A"]  # shape: [r, input_dim]
            
            # Compute deltaW = B @ A
            delta_w = torch.matmul(lora_b, lora_a)  # shape: [output_dim, input_dim]
            lora_deltas[layer_key] = delta_w.float()
    
    return lora_deltas


def flatten_lora_deltas(lora_deltas: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Flatten all LoRA deltas into a single vector."""
    flat_list = []
    for key in sorted(lora_deltas.keys()):
        flat_list.append(lora_deltas[key].reshape(-1))
    return torch.cat(flat_list, dim=0) if flat_list else torch.tensor([])


def process_experiment(exp_dir: Path, output_suffix: str = "real") -> None:
    """Process initial and all rounds in an experiment directory."""
    exp_dir = Path(exp_dir).resolve()
    processed_count = 0
    failed_count = 0
    total_params = 0

    # First process initial params as round 0
    initial_path = exp_dir / "initial_params.pt"
    if initial_path.exists():
        initial_output_path = exp_dir / f"initial_{output_suffix}_flat.pt"
        try:
            print(f"Processing initial_params...", end=" ", flush=True)
            params = torch.load(initial_path, map_location="cpu")
            lora_deltas = extract_lora_deltas(params)
            flat_delta = flatten_lora_deltas(lora_deltas)
            torch.save(flat_delta, initial_output_path)
            processed_count += 1
            total_params += len(flat_delta)
            print(f"✓ ({len(flat_delta):,} params)")
            del params, lora_deltas, flat_delta
            torch.cuda.empty_cache()
        except Exception as e:
            failed_count += 1
            print(f"✗ Error initial_params: {e}")

    for round_dir in sorted(exp_dir.glob("round_*")):
        try:
            round_idx = int(round_dir.name.split("_")[-1])
        except ValueError:
            continue

        params_path = round_dir / "round_params.pt"
        output_path = round_dir / f"round_{output_suffix}_flat.pt"

        if not params_path.exists():
            continue

        try:
            print(f"Processing {round_dir.name}...", end=" ", flush=True)

            # Load parameters
            params = torch.load(params_path, map_location="cpu")

            # Extract and compute LoRA deltas
            lora_deltas = extract_lora_deltas(params)
            flat_delta = flatten_lora_deltas(lora_deltas).to(torch.float16)
            
            # Save flattened delta in fp16 to节约内存与磁盘
            processed_count += 1
            total_params += len(flat_delta)

            print(f"✓ ({len(flat_delta):,} params)")

            # Clean up
            del params, lora_deltas, flat_delta
            torch.cuda.empty_cache()

        except Exception as e:
            failed_count += 1
            print(f"✗ Error: {e}")

    print(f"\n✓ Processed {processed_count} items (including initial)")
    if failed_count > 0:
        print(f"✗ Failed: {failed_count} items")
    print(f"Total parameters per item: {total_params:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute real LoRA weight changes (B @ A) from round_params."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Experiment directories containing round_* subdirectories.",
    )
    parser.add_argument(
        "--output_suffix",
        type=str,
        default="real",
        help="Suffix for output file (default: 'real' -> round_real_flat.pt).",
    )
    args = parser.parse_args()
    
    for exp_dir in args.experiment_dirs:
        exp_path = Path(exp_dir).resolve()
        print(f"\n{'='*60}")
        print(f"Processing: {exp_path.name}")
        print(f"{'='*60}")
        process_experiment(exp_path, args.output_suffix)


if __name__ == "__main__":
    main()
