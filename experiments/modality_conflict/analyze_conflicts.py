#!/usr/bin/env python3
"""Analyze pairwise LoRA update conflicts across saved Janus trace runs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise dot products / cosine similarities for saved LoRA updates."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Directories produced by run_janus_round_trace.py.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Where to save the per-round pairwise conflict table.",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default="lora_delta",
        choices=["grad", "delta", "lora_delta"],
        help="Whether to compare saved round gradients, parameter deltas, or LoRA weight change directions (B@A).",
    )
    return parser.parse_args()


def load_metadata(exp_dir: Path) -> Dict[str, str]:
    metadata_path = exp_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"dataset_profile": exp_dir.name}


def load_round_vectors(exp_dir: Path, artifact: str) -> Dict[int, torch.Tensor]:
    """Load round vectors from either grad/delta flat files or compute from LoRA params."""
    if artifact == "lora_delta":
        return load_round_lora_deltas(exp_dir)
    
    vectors = {}
    for round_dir in sorted(exp_dir.glob("round_*")):
        try:
            round_idx = int(round_dir.name.split("_")[-1])
        except ValueError:
            continue
        vector_path = round_dir / f"round_{artifact}_flat.pt"
        if vector_path.exists():
            vectors[round_idx] = torch.load(vector_path, map_location="cpu")
    return vectors


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
            # e.g., language_model.base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight
            # -> language_model.base_model.model.model.layers.0.mlp.gate_proj
            
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


def load_round_lora_deltas(exp_dir: Path) -> Dict[int, str]:
    """Return paths to params files instead of loading into memory."""
    vectors = {}
    for round_dir in sorted(exp_dir.glob("round_*")):
        try:
            round_idx = int(round_dir.name.split("_")[-1])
        except ValueError:
            continue
        
        params_path = round_dir / "round_params.pt"
        if params_path.exists():
            vectors[round_idx] = str(params_path)
    
    return vectors


def compute_lora_delta_vector(params_path: Path) -> torch.Tensor:
    """Compute LoRA delta vector for a single round params file."""
    params = torch.load(params_path, map_location="cpu")
    lora_deltas = extract_lora_deltas(params)
    flat_delta = flatten_lora_deltas(lora_deltas)
    return flat_delta


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Compute various similarity metrics between two vectors."""
    a_norm = float(a.norm().item())
    b_norm = float(b.norm().item())
    dot_prod = float(torch.dot(a, b).item())
    
    cosine = cosine_similarity(a, b)
    
    # Compute Pearson correlation coefficient
    a_mean = a.mean().item()
    b_mean = b.mean().item()
    a_centered = a - a_mean
    b_centered = b - b_mean
    
    numerator = float(torch.dot(a_centered, b_centered).item())
    denominator = (a_centered.norm().item() * b_centered.norm().item())
    pearson = numerator / denominator if denominator > 0 else 0.0
    
    return {
        "dot": dot_prod,
        "cosine": cosine,
        "pearson": pearson,
        "l2_a": a_norm,
        "l2_b": b_norm,
        "conflict": dot_prod < 0,
    }


def main() -> None:
    args = parse_args()
    experiment_dirs = [Path(x).resolve() for x in args.experiment_dirs]
    loaded = []
    
    # Load each experiment's metadata and vector references
    for exp_dir in experiment_dirs:
        print(f"Loading {exp_dir.name}...")
        metadata = load_metadata(exp_dir)
        vectors = load_round_vectors(exp_dir, args.artifact)
        loaded.append(
            {
                "path": str(exp_dir),
                "name": metadata.get("dataset_profile", exp_dir.name),
                "vectors": vectors,
            }
        )
        print(f"  Found {len(vectors)} rounds")

    rows: List[Dict[str, object]] = []
    
    # Process pairwise comparisons
    total_pairs = len(loaded) * (len(loaded) - 1) // 2
    pair_count = 0
    
    for left, right in itertools.combinations(loaded, 2):
        pair_count += 1
        shared_rounds = sorted(set(left["vectors"].keys()) & set(right["vectors"].keys()))
        print(f"Comparing {left['name']} <-> {right['name']}: {len(shared_rounds)} shared rounds ({pair_count}/{total_pairs})")
        
        for round_idx in shared_rounds:
            # For lora_delta, vectors are paths; load and compute on-the-fly
            if args.artifact == "lora_delta":
                a = compute_lora_delta_vector(Path(left["vectors"][round_idx]))
                b = compute_lora_delta_vector(Path(right["vectors"][round_idx]))
            else:
                a = left["vectors"][round_idx]
                b = right["vectors"][round_idx]
            
            metrics = compute_metrics(a, b)
            
            rows.append(
                {
                    "round": round_idx,
                    "exp_a": left["name"],
                    "exp_b": right["name"],
                    "artifact": args.artifact,
                    "dot": metrics["dot"],
                    "cosine": metrics["cosine"],
                    "pearson": metrics["pearson"],
                    "l2_a": metrics["l2_a"],
                    "l2_b": metrics["l2_b"],
                    "conflict": metrics["conflict"],
                    "path_a": left["path"],
                    "path_b": right["path"],
                }
            )
            
            # Clean up memory
            if args.artifact == "lora_delta":
                del a, b
                torch.cuda.empty_cache()

    if not rows:
        artifact_type = "LoRA weight deltas" if args.artifact == "lora_delta" else f"'{args.artifact}' vectors"
        raise RuntimeError(
            f"No shared rounds with saved {artifact_type} found across the provided experiment directories."
        )

    if args.output_csv:
        out_path = Path(args.output_csv).resolve()
    else:
        out_path = experiment_dirs[0].parent / f"conflict_report_{args.artifact}.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Wrote {len(rows)} pairwise comparisons to {out_path}")
    
    # Print summary statistics
    cosines = [row["cosine"] for row in rows]
    conflicts = sum(1 for row in rows if row["conflict"])
    
    print(f"\nSummary ({args.artifact}):")
    print(f"  Total comparisons: {len(rows)}")
    print(f"  Conflicts (dot < 0): {conflicts}")
    print(f"  Mean cosine similarity: {sum(cosines) / len(cosines):.4f}")
    print(f"  Min cosine similarity: {min(cosines):.4f}")
    print(f"  Max cosine similarity: {max(cosines):.4f}")


if __name__ == "__main__":
    main()
