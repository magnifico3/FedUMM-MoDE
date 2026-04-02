#!/usr/bin/env python3
"""Verify whether initial LoRA params produce zero effective weight change."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch


def extract_lora_deltas(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    lora_params: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, tensor in params.items():
        if ".lora_A.default.weight" in name:
            layer = name.replace(".lora_A.default.weight", "")
            lora_params.setdefault(layer, {})["A"] = tensor
        elif ".lora_B.default.weight" in name:
            layer = name.replace(".lora_B.default.weight", "")
            lora_params.setdefault(layer, {})["B"] = tensor

    lora_deltas: Dict[str, torch.Tensor] = {}
    for layer, p in lora_params.items():
        if "A" in p and "B" in p:
            lo = p["B"] @ p["A"]
            lora_deltas[layer] = lo
    return lora_deltas


def main() -> None:
    base = Path("/root/ad/outputs/modality_conflict_sched")
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        pls = sub / "initial_params.pt"
        if not pls.exists():
            print(f"skip {sub.name}: no initial_params.pt")
            continue

        params = torch.load(pls, map_location="cpu")
        lora_deltas = extract_lora_deltas(params)
       
        total_norm = 0.0
        total_elems = 0
        total_nonzero = 0
        for layer, d in lora_deltas.items():
            if d is None:
                continue
            flat = d.flatten()
            norm = float(flat.norm().item())
            nonzero = int((flat != 0).sum().item())
            total_norm += norm
            total_elems += flat.numel()
            total_nonzero += nonzero
        
        print(f"{sub.name}: layers={len(lora_deltas)}, total_norm={total_norm:.6f}, "
              f"total_elems={total_elems}, nonzero={total_nonzero}")
        if total_nonzero == 0:
            print("  => initial B@A is all zero (as expected if A,B init zero)")
        else:
            print("  => initial B@A has non-zero values")


if __name__ == '__main__':
    main()
