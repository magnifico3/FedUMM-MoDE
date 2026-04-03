from pathlib import Path
import re

import matplotlib.pyplot as plt
import torch


def plot_conflict(results):
    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)
    proj_names = ["up_proj", "down_proj", "gate_proj"]
    titles = ["up_proj", "down_proj", "gate_proj"]
    round_name = results["round_name"]
    task_a = results["task_a"]
    task_b = results["task_b"]

    for ax, proj_name, title in zip(axes, proj_names, titles):
        layers = [item["layer"] for item in results[proj_name]]
        dots = [item["dot"] for item in results[proj_name]]
        colors = ["red" if item["conflict"] else "steelblue" for item in results[proj_name]]

        ax.bar(layers, dots, color=colors)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_title(f"{task_a} vs {task_b} {round_name} {title}")
        ax.set_ylabel("dot")
        ax.set_xticks(layers)

        for layer, dot, item in zip(layers, dots, results[proj_name]):
            if item["conflict"]:
                ax.text(layer, dot, "conflict", color="red", fontsize=8, ha="center", va="top")

    axes[-1].set_xlabel("layer")
    fig.tight_layout()
    output_path = results["output_path"]
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved to: {output_path}")


def cul_confict_each_matrix(cc3m_path, vqav2_path):
    cc3m = torch.load(cc3m_path, map_location="cpu")
    vqav2 = torch.load(vqav2_path, map_location="cpu")

    results = {
        "up_proj": [],
        "down_proj": [],
        "gate_proj": [],
        "round_name": cc3m_path.parent.name,
        "task_a": cc3m_path.parent.parent.name,
        "task_b": vqav2_path.parent.parent.name,
        "output_path": (
            Path(__file__).resolve().parent
            / f"{cc3m_path.parent.parent.name}_vs_{vqav2_path.parent.parent.name}_{cc3m_path.parent.name}_conflicts.png"
        ),
    }
    grouped = {"up_proj": {}, "down_proj": {}, "gate_proj": {}}
    pattern = re.compile(r"layers\.(\d+)\.mlp\.(up_proj|down_proj|gate_proj)\.lora_([AB])\.default\.weight")

    for key in sorted(set(cc3m.keys()) & set(vqav2.keys())):
        match = pattern.search(key)
        if match is None:
            continue

        layer = int(match.group(1))
        proj_name = match.group(2)
        matrix_name = match.group(3)
        if layer not in grouped[proj_name]:
            grouped[proj_name][layer] = {
                "cc3m": {"A": None, "B": None},
                "vqav2": {"A": None, "B": None},
            }

        grouped[proj_name][layer]["cc3m"][matrix_name] = cc3m[key].float()
        grouped[proj_name][layer]["vqav2"][matrix_name] = vqav2[key].float()

    for proj_name in ["up_proj", "down_proj", "gate_proj"]:
        for layer in sorted(grouped[proj_name].keys()):
            cc3m_a = grouped[proj_name][layer]["cc3m"]["A"]
            cc3m_b = grouped[proj_name][layer]["cc3m"]["B"]
            vqav2_a = grouped[proj_name][layer]["vqav2"]["A"]
            vqav2_b = grouped[proj_name][layer]["vqav2"]["B"]
            if cc3m_a is None or cc3m_b is None or vqav2_a is None or vqav2_b is None:
                continue

            cc3m_ba = torch.matmul(cc3m_b, cc3m_a).reshape(-1)
            vqav2_ba = torch.matmul(vqav2_b, vqav2_a).reshape(-1)
            dot = torch.dot(cc3m_ba, vqav2_ba).item()
            results[proj_name].append({"layer": layer, "dot": dot, "conflict": dot < 0})

    return results


def main():
    base_dir = Path(__file__).resolve().parents[2] / "outputs" / "modality_conflict_sched"
    task_a_dir = base_dir / "cc3m_s1000_e250_lr3e4"
    task_b_dir = base_dir / "vqav2_s2000_e500_lr3e5"

    for round_dir in sorted(task_a_dir.glob("round_*")):
        round_name = round_dir.name
        cc3m_path = task_a_dir / round_name / "round_delta.pt"
        vqav2_path = task_b_dir / round_name / "round_delta.pt"
        if not cc3m_path.exists() or not vqav2_path.exists():
            continue

        results = cul_confict_each_matrix(cc3m_path, vqav2_path)
        print(results)
        plot_conflict(results)


if __name__ == "__main__":
    main()
