#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize per-round LoRA B@A direction dynamics.")
    parser.add_argument(
        "--dataset",
        default="cc3m_s1000_e250_lr3e4",
        help="Experiment directory name under outputs/modality_conflict_sched.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        help="Transformer layer index.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="Multiple transformer layer indices. If set, overrides --layer.",
    )
    parser.add_argument(
        "--proj",
        default="up_proj",
        choices=["up_proj", "down_proj", "gate_proj", "up", "down", "gate"],
        help="MLP projection name.",
    )
    parser.add_argument(
        "--projs",
        nargs="+",
        choices=["up_proj", "down_proj", "gate_proj", "up", "down", "gate"],
        help="Multiple projection names. If set, overrides --proj.",
    )
    args = parser.parse_args()

    proj_alias = {
        "up": "up_proj",
        "down": "down_proj",
        "gate": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
        "gate_proj": "gate_proj",
    }
    layers = args.layers if args.layers else [args.layer]
    proj_names = [proj_alias[name] for name in (args.projs if args.projs else [args.proj])]

    base_dir = Path(__file__).resolve().parents[2] / "outputs" / "modality_conflict_sched" / args.dataset
    round_dirs = sorted(base_dir.glob("round_*"))
    if not round_dirs:
        raise SystemExit(f"no rounds found in {base_dir}")

    for layer in layers:
        for proj_name in proj_names:
            round_ids = []
            vectors = []
            norms = []
            cos_to_first = []
            cos_to_prev = []
            first_vec = None
            prev_vec = None

            key_a = f"language_model.base_model.model.model.layers.{layer}.mlp.{proj_name}.lora_A.default.weight"
            key_b = f"language_model.base_model.model.model.layers.{layer}.mlp.{proj_name}.lora_B.default.weight"

            for round_dir in round_dirs:
                delta_path = round_dir / "round_delta.pt"
                if not delta_path.exists():
                    continue

                delta = torch.load(delta_path, map_location="cpu")
                if key_a not in delta or key_b not in delta:
                    continue

                vec = (delta[key_b].float() @ delta[key_a].float()).reshape(-1)
                norm = vec.norm().item()

                round_ids.append(int(round_dir.name.split("_")[-1]))
                vectors.append(vec)
                norms.append(norm)

                if first_vec is None:
                    first_vec = vec
                    cos_to_first.append(1.0)
                else:
                    denom = max(first_vec.norm().item() * norm, 1e-12)
                    cos_to_first.append(torch.dot(vec, first_vec).item() / denom)

                if prev_vec is None:
                    cos_to_prev.append(1.0)
                else:
                    denom = max(prev_vec.norm().item() * norm, 1e-12)
                    cos_to_prev.append(torch.dot(vec, prev_vec).item() / denom)
                prev_vec = vec

            if not vectors:
                print(f"skip layer={layer}, proj={proj_name}: no valid vectors")
                continue

            matrix = torch.stack(vectors, dim=0)
            normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
            centered = normalized - normalized.mean(dim=0, keepdim=True)
            _, _, v = torch.pca_lowrank(centered, q=2)
            coords = centered @ v[:, :2]

            fig, axes = plt.subplots(2, 2, figsize=(16, 10))

            axes[0, 0].plot(round_ids, norms, marker="o")
            axes[0, 0].set_title(f"{args.dataset} layer {layer} {proj_name} B@A norm")
            axes[0, 0].set_xlabel("round")
            axes[0, 0].set_ylabel("l2 norm")

            axes[0, 1].plot(round_ids, cos_to_first, marker="o", label="cos to round1")
            axes[0, 1].plot(round_ids, cos_to_prev, marker="s", label="cos to prev round")
            axes[0, 1].axhline(0.0, color="black", linewidth=1)
            axes[0, 1].set_title("direction similarity")
            axes[0, 1].set_xlabel("round")
            axes[0, 1].set_ylabel("cosine")
            axes[0, 1].legend()

            axes[1, 0].bar(round_ids, [1 if x < 0 else 0 for x in cos_to_prev], color="tomato")
            axes[1, 0].set_title("sign flip vs previous round")
            axes[1, 0].set_xlabel("round")
            axes[1, 0].set_ylabel("flip")
            axes[1, 0].set_yticks([0, 1])

            axes[1, 1].plot(coords[:, 0].numpy(), coords[:, 1].numpy(), marker="o")
            for i, round_id in enumerate(round_ids):
                axes[1, 1].text(coords[i, 0].item(), coords[i, 1].item(), str(round_id), fontsize=8)
            axes[1, 1].set_title("direction trajectory (PCA of normalized B@A, unstable when norm is tiny)")
            axes[1, 1].set_xlabel("pc1")
            axes[1, 1].set_ylabel("pc2")

            fig.tight_layout()

            output_path = (
                Path(__file__).resolve().parent
                / f"{args.dataset}_layer{layer}_{proj_name}_direction.png"
            )
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"dataset={args.dataset}")
            print(f"layer={layer}, proj={proj_name}")
            print(f"rounds={round_ids}")
            print(f"norms={norms}")
            print(f"cos_to_first={cos_to_first}")
            print(f"cos_to_prev={cos_to_prev}")
            print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
