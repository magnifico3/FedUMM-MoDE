#!/usr/bin/env python3

from pathlib import Path

import torch


def main():
    round_dir = (
        Path(__file__).resolve().parents[2]
        / "outputs"
        / "modality_conflict_sched"
        / "cc3m_s1000_e250_lr3e4"
        / "round_001"
    )

    pt_files = [
        "round_params.pt",
        "round_grad.pt",
        "round_delta.pt",
        "round_params_flat.pt",
        "round_grad_flat.pt",
        "round_delta_flat.pt",
    ]

    for name in pt_files:
        path = round_dir / name
        print(f"\n=== {name} ===")
        obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict):
            print(f"type: dict, len: {len(obj)}")
            for key, value in obj.items():
                if hasattr(value, "shape"):
                    print(f"{key}: shape={tuple(value.shape)}, dtype={value.dtype}")
                else:
                    print(f"{key}: {type(value)}")
        elif hasattr(obj, "shape"):
            print(f"type: tensor, shape={tuple(obj.shape)}, dtype={obj.dtype}")
            print(obj)
        else:
            print(type(obj))
            print(obj)


if __name__ == "__main__":
    main()
