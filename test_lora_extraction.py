#!/usr/bin/env python3
"""Test script to verify LoRA parameter extraction."""

import torch
from pathlib import Path

# Test with CC3M round 1
params_path = Path("/root/ad/outputs/modality_conflict_sched/cc3m_s1000_e250_lr3e4/round_001/round_params.pt")

def extract_lora_deltas(params):
    """Extract all LoRA weight deltas (B @ A) from a parameter dictionary."""
    # Collect all LoRA parameters by layer
    lora_params = {}
    
    for param_name, param_tensor in params.items():
        if "lora_A" in param_name or "lora_B" in param_name:
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

if params_path.exists():
    print(f"Loading {params_path}...")
    params = torch.load(params_path, map_location="cpu")
    
    print(f"\nTotal parameters: {len(params)}")
    
    lora_deltas = extract_lora_deltas(params)
    print(f"\nFound {len(lora_deltas)} LoRA layer deltas")
    
    print("\nLoRA weight deltas (B @ A):")
    total_norm = 0
    for layer_key in sorted(lora_deltas.keys()):
        delta_w = lora_deltas[layer_key]
        flat_delta = delta_w.reshape(-1)
        norm_val = flat_delta.norm().item()
        total_norm += norm_val ** 2
        print(f"  {layer_key}:")
        print(f"    Shape: {delta_w.shape}, Norm: {norm_val:.6f}")
    
    total_norm = (total_norm ** 0.5)
    print(f"\nTotal norm of all deltas: {total_norm:.6f}")
else:
    print(f"File not found: {params_path}")

