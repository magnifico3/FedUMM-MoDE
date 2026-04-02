#!/usr/bin/env python3
"""Analyze similarity of LoRA weight deltas across tasks using pre-computed real params."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pairwise similarity metrics for LoRA weight deltas across tasks."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        help="Directories containing pre-computed round_real_flat.pt files.",
    )
    parser.add_argument(
        "--input_suffix",
        type=str,
        default="real",
        help="Suffix of pre-computed files (default: 'real' -> round_real_flat.pt).",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="",
        help="Where to save the similarity report. Default: conflict_report_real.csv in parent dir.",
    )
    return parser.parse_args()


def load_metadata(exp_dir: Path) -> Dict[str, str]:
    """Load metadata if available."""
    metadata_path = exp_dir / "metadata.json"
    if metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"dataset_profile": exp_dir.name}


def load_round_vectors(exp_dir: Path, suffix: str) -> Dict[int, str]:
    """Return paths to pre-computed vectors including initial and rounds."""
    vectors = {}
    # Try load initial vector as round 0
    initial_path = exp_dir / f"initial_{suffix}_flat.pt"
    if initial_path.exists():
        vectors[0] = str(initial_path)

    for round_dir in sorted(exp_dir.glob("round_*")):
        try:
            round_idx = int(round_dir.name.split("_")[-1])
        except ValueError:
            continue

        vector_path = round_dir / f"round_{suffix}_flat.pt"
        if vector_path.exists():
            vectors[round_idx] = str(vector_path)

    return vectors


def _dot_chunked(a: torch.Tensor, b: torch.Tensor, chunk_size: int = 20_000_000) -> float:
    """Compute dot product in chunks to lower peak memory pressure."""
    total = 0.0
    for i in range(0, a.numel(), chunk_size):
        ai = a[i:i + chunk_size].float()
        bi = b[i:i + chunk_size].float()
        total += float(torch.dot(ai, bi).item())
    return total


def _norm_chunked(a: torch.Tensor, chunk_size: int = 20_000_000) -> float:
    total = 0.0
    for i in range(0, a.numel(), chunk_size):
        ai = a[i:i + chunk_size].float()
        total += float((ai * ai).sum().item())
    return float(total ** 0.5)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    a_norm = _norm_chunked(a)
    b_norm = _norm_chunked(b)

    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0

    dot = _dot_chunked(a, b)
    return float(dot / (a_norm * b_norm))


def pearson_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient."""
    a_mean = a.mean().item()
    b_mean = b.mean().item()
    
    a_centered = a - a_mean
    b_centered = b - b_mean
    
    numerator = float(torch.dot(a_centered, b_centered).item())
    denominator = (a_centered.norm().item() * b_centered.norm().item())
    
    if denominator == 0.0:
        return 0.0
    
    return numerator / denominator


def compute_metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    """Compute comprehensive similarity metrics."""
    dot_prod = _dot_chunked(a, b)
    a_norm = _norm_chunked(a)
    b_norm = _norm_chunked(b)

    return {
        "dot_product": dot_prod,
        "cosine_similarity": cosine_similarity(a, b),
        "pearson_correlation": pearson_correlation(a, b),
        "l2_norm_a": a_norm,
        "l2_norm_b": b_norm,
        "is_conflict": dot_prod < 0,
    }


def main() -> None:
    args = parse_args()
    experiment_dirs = [Path(x).resolve() for x in args.experiment_dirs]
    
    # Load all experiments
    print("Loading experiments...")
    loaded_exps = []
    
    for exp_dir in experiment_dirs:
        metadata = load_metadata(exp_dir)
        task_name = metadata.get("dataset_profile", exp_dir.name)
        vectors = load_round_vectors(exp_dir, args.input_suffix)
        
        print(f"  {task_name}: {len(vectors)} rounds")
        
        if vectors:
            loaded_exps.append({
                "path": str(exp_dir),
                "name": task_name,
                "vectors": vectors,
            })
    
    if len(loaded_exps) < 2:
        raise RuntimeError(f"Need at least 2 experiments, got {len(loaded_exps)}")
    
    print(f"\nTotal experiments: {len(loaded_exps)}")
    print(f"Task names: {', '.join(exp['name'] for exp in loaded_exps)}")
    
    # Compute pairwise similarities
    rows: List[Dict[str, object]] = []
    total_pairs = len(loaded_exps) * (len(loaded_exps) - 1) // 2
    pair_count = 0

    # compute total steps for progress
    total_steps = 0
    for left, right in itertools.combinations(loaded_exps, 2):
        shared_rounds = set(left["vectors"].keys()) & set(right["vectors"].keys())
        total_steps += len(shared_rounds)

    current_step = 0
    for left, right in itertools.combinations(loaded_exps, 2):
        pair_count += 1
        shared_rounds = sorted(set(left["vectors"].keys()) & set(right["vectors"].keys()))

        print(f"[{pair_count}/{total_pairs}] Comparing {left['name']} ↔ {right['name']}: "
              f"{len(shared_rounds)} shared rounds")

        for round_idx in shared_rounds:
            # Load vectors on-the-fly to save memory
            a = torch.load(left["vectors"][round_idx], map_location="cpu")
            b = torch.load(right["vectors"][round_idx], map_location="cpu")

            metrics = compute_metrics(a, b)

            current_step += 1
            pct = (current_step / total_steps) * 100 if total_steps > 0 else 100
            print(f"  Step {current_step}/{total_steps} ({pct:.1f}%), round {round_idx}", end="\r", flush=True)

            rows.append({
                "round": round_idx,
                "task_a": left["name"],
                "task_b": right["name"],
                "dot_product": metrics["dot_product"],
                "cosine_similarity": metrics["cosine_similarity"],
                "pearson_correlation": metrics["pearson_correlation"],
                "l2_norm_a": metrics["l2_norm_a"],
                "l2_norm_b": metrics["l2_norm_b"],
                "conflict": metrics["is_conflict"],
                "path_a": left["path"],
                "path_b": right["path"],
            })

            # Clean up
            del a, b
            torch.cuda.empty_cache()

        # keep line after pair done
        print()
    
    if not rows:
        raise RuntimeError("No shared rounds found across experiments!")
    
    # Determine output path
    if args.output_csv:
        out_path = Path(args.output_csv).resolve()
    else:
        out_path = experiment_dirs[0].parent / f"similarity_report_{args.input_suffix}.csv"
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write CSV
    with out_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Wrote {len(rows)} comparisons to: {out_path}")
    
    # Print summary statistics
    cosines = [row["cosine_similarity"] for row in rows]
    dots = [row["dot_product"] for row in rows]
    conflicts = sum(1 for row in rows if row["conflict"])
    
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"Total comparisons: {len(rows)}")
    print(f"Conflicts (dot < 0): {conflicts} ({100*conflicts/len(rows):.1f}%)")
    
    print(f"\nCosine Similarity:")
    print(f"  Mean:      {sum(cosines) / len(cosines):.4f}")
    print(f"  Min:       {min(cosines):.4f}")
    print(f"  Max:       {max(cosines):.4f}")
    print(f"  Median:    {sorted(cosines)[len(cosines)//2]:.4f}")
    
    print(f"\nDot Product:")
    print(f"  Mean:      {sum(dots) / len(dots):.2f}")
    print(f"  Min:       {min(dots):.2f}")
    print(f"  Max:       {max(dots):.2f}")
    
    # Per-task pair analysis
    print(f"\n{'='*60}")
    print("Per-Task Pair Analysis")
    print(f"{'='*60}")
    
    task_pairs = {}
    for row in rows:
        pair_key = tuple(sorted([row["task_a"], row["task_b"]]))
        if pair_key not in task_pairs:
            task_pairs[pair_key] = []
        task_pairs[pair_key].append(row)
    
    for (task_a, task_b), pair_rows in sorted(task_pairs.items()):
        pair_cosines = [r["cosine_similarity"] for r in pair_rows]
        pair_conflicts = sum(1 for r in pair_rows if r["conflict"])
        
        print(f"\n{task_a} ↔ {task_b}:")
        print(f"  Rounds: {len(pair_rows)}")
        print(f"  Mean cosine: {sum(pair_cosines) / len(pair_cosines):.4f}")
        print(f"  Min cosine:  {min(pair_cosines):.4f}")
        print(f"  Max cosine:  {max(pair_cosines):.4f}")
        print(f"  Conflicts:   {pair_conflicts}")


if __name__ == "__main__":
    main()
