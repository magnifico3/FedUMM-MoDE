#!/usr/bin/env python3
"""Visualize summary metrics and similarity reports from LoRA analysis."""

import json
import csv
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict


def collect_summary_data(base_dir: Path) -> Dict[str, List[Dict]]:
    """Collect summary data from all experiments."""
    experiments = {}
    
    for exp_dir in base_dir.glob("*_s*_e*_lr*"):
        if not exp_dir.is_dir():
            continue
            
        task_name = exp_dir.name.split('_')[0]  # e.g., 'cc3m' from 'cc3m_s1000_e250_lr3e4'
        rounds_data = []
        
        # Load initial summary if exists
        initial_summary = exp_dir / "initial_summary.json"
        if initial_summary.exists():
            with open(initial_summary, 'r') as f:
                data = json.load(f)
                data['round'] = 0
                rounds_data.append(data)
        
        # Load round summaries
        for round_dir in sorted(exp_dir.glob("round_*")):
            summary_path = round_dir / "summary.json"
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    data = json.load(f)
                    rounds_data.append(data)
        
        if rounds_data:
            experiments[task_name] = sorted(rounds_data, key=lambda x: x['round'])
    
    return experiments


def load_similarity_report(csv_path: Path) -> pd.DataFrame:
    """Load similarity report CSV."""
    return pd.read_csv(csv_path)


def plot_summary_metrics(experiments: Dict[str, List[Dict]], output_dir: Path):
    """Plot summary metrics for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Metrics Across Tasks', fontsize=16)
    
    metrics = ['loss', 'eval_loss', 'eval_acc', 'grad_l2']
    metric_names = ['Training Loss', 'Eval Loss', 'Eval Accuracy', 'Grad L2 Norm']
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i//2, i%2]
        
        for j, (task, rounds) in enumerate(experiments.items()):
            rounds_nums = [r['round'] for r in rounds]
            values = [r.get(metric, None) for r in rounds]
            
            # Filter out None values
            valid_data = [(rn, v) for rn, v in zip(rounds_nums, values) if v is not None]
            if valid_data:
                rns, vs = zip(*valid_data)
                ax.plot(rns, vs, label=task, color=colors[j % len(colors)], marker='o', markersize=3)
        
        ax.set_xlabel('Round')
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_similarity_metrics(df: pd.DataFrame, output_dir: Path):
    """Plot similarity metrics across rounds."""
    # Get unique task pairs
    task_pairs = df[['task_a', 'task_b']].drop_duplicates()
    task_pairs = [tuple(sorted([row['task_a'], row['task_b']])) for _, row in task_pairs.iterrows()]
    task_pairs = list(set(task_pairs))  # unique pairs
    
    metrics = ['cosine_similarity', 'dot_product', 'pearson_correlation']
    metric_names = ['Cosine Similarity', 'Dot Product', 'Pearson Correlation']
    
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]
    
    colors = plt.cm.tab10.colors
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        for j, (task_a, task_b) in enumerate(sorted(task_pairs)):
            pair_data = df[(df['task_a'] == task_a) & (df['task_b'] == task_b) | 
                          (df['task_a'] == task_b) & (df['task_b'] == task_a)]
            
            if not pair_data.empty:
                pair_data = pair_data.sort_values('round')
                ax.plot(pair_data['round'], pair_data[metric], 
                       label=f'{task_a} ↔ {task_b}', 
                       color=colors[j % len(colors)], marker='o', markersize=3)
        
        ax.set_xlabel('Round')
        ax.set_ylabel(name)
        ax.set_title(f'{name} Between Task Pairs')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'similarity_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_conflicts(df: pd.DataFrame, output_dir: Path):
    """Plot conflict analysis."""
    # Count conflicts per round per pair
    conflict_summary = df.groupby(['round', 'task_a', 'task_b'])['conflict'].sum().reset_index()
    
    task_pairs = df[['task_a', 'task_b']].drop_duplicates()
    task_pairs = [tuple(sorted([row['task_a'], row['task_b']])) for _, row in task_pairs.iterrows()]
    task_pairs = list(set(task_pairs))
    
    plt.figure(figsize=(12, 6))
    
    colors = plt.cm.tab10.colors
    for j, (task_a, task_b) in enumerate(sorted(task_pairs)):
        pair_data = conflict_summary[(conflict_summary['task_a'] == task_a) & (conflict_summary['task_b'] == task_b) | 
                                    (conflict_summary['task_a'] == task_b) & (conflict_summary['task_b'] == task_a)]
        
        if not pair_data.empty:
            pair_data = pair_data.sort_values('round')
            plt.plot(pair_data['round'], pair_data['conflict'], 
                    label=f'{task_a} ↔ {task_b}', 
                    color=colors[j % len(colors)], marker='s', markersize=4)
    
    plt.xlabel('Round')
    plt.ylabel('Number of Conflicts')
    plt.title('Conflicts (Dot Product < 0) Across Rounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / 'conflicts.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    base_dir = Path('/root/ad/outputs/modality_conflict_sched')
    similarity_csv = base_dir / 'similarity_report_real.csv'
    output_dir = Path('/root/ad/experiments/modality_conflict')
    
    print("Collecting summary data...")
    experiments = collect_summary_data(base_dir)
    print(f"Found {len(experiments)} experiments: {list(experiments.keys())}")
    
    print("Loading similarity report...")
    if similarity_csv.exists():
        df = load_similarity_report(similarity_csv)
        print(f"Loaded {len(df)} similarity records")
    else:
        print(f"Similarity report not found at {similarity_csv}")
        df = None
    
    print("Generating plots...")
    
    # Plot summary metrics
    plot_summary_metrics(experiments, output_dir)
    print("✓ Saved summary_metrics.png")
    
    if df is not None:
        # Plot similarity metrics
        plot_similarity_metrics(df, output_dir)
        print("✓ Saved similarity_metrics.png")
        
        # Plot conflicts
        plot_conflicts(df, output_dir)
        print("✓ Saved conflicts.png")
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()