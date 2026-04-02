#!/usr/bin/env python3
"""Download WikiText and InstructPix2Pix datasets into /root/Datasets."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict


DEFAULT_ROOT = Path("/root/Datasets")
DATASET_SPECS: Dict[str, Dict[str, str]] = {
    "wikitext": {
        "dataset_name": "wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "output_subdir": "wikitext/wikitext-2-raw-v1",
    },
    "instructpix2pix-10k": {
        "dataset_name": "imthanhlv/instructpix2pix-clip-filtered-10k",
        "dataset_config": "",
        "output_subdir": "instructpix2pix/instructpix2pix-clip-filtered-10k",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download WikiText and InstructPix2Pix datasets into a local directory."
    )
    parser.add_argument(
        "--root_dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Local dataset root directory. Defaults to /root/Datasets.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=DEFAULT_ROOT,
        help="Hugging Face cache directory. Defaults to /root/Datasets.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=sorted(DATASET_SPECS.keys()),
        default=sorted(DATASET_SPECS.keys()),
        help="Only download selected datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload and overwrite existing saved datasets.",
    )
    return parser.parse_args()


def save_dataset(
    dataset_key: str,
    root_dir: Path,
    cache_dir: Path,
    force: bool,
) -> None:
    spec = DATASET_SPECS[dataset_key]
    output_dir = root_dir / spec["output_subdir"]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        if not force:
            print(f"[skip] {dataset_key}: {output_dir} already exists")
            return
        if output_dir.is_dir():
            shutil.rmtree(output_dir)
        else:
            output_dir.unlink()

    print(
        f"[download] {dataset_key}: {spec['dataset_name']}"
        + (f" ({spec['dataset_config']})" if spec["dataset_config"] else "")
    )
    from datasets import DatasetDict, load_dataset

    dataset = load_dataset(
        spec["dataset_name"],
        spec["dataset_config"] or None,
        cache_dir=str(cache_dir),
    )

    if isinstance(dataset, DatasetDict):
        split_summary = {split: len(split_dataset) for split, split_dataset in dataset.items()}
    else:
        split_summary = {"train": len(dataset)}

    dataset.save_to_disk(str(output_dir))
    metadata = {
        "dataset_key": dataset_key,
        "dataset_name": spec["dataset_name"],
        "dataset_config": spec["dataset_config"],
        "output_dir": str(output_dir),
        "cache_dir": str(cache_dir),
        "splits": split_summary,
    }
    (output_dir / "download_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[saved] {dataset_key}: {output_dir}")
    print(f"[splits] {dataset_key}: {split_summary}")


def main() -> None:
    args = parse_args()
    args.root_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key in args.only:
        save_dataset(
            dataset_key=dataset_key,
            root_dir=args.root_dir,
            cache_dir=args.cache_dir,
            force=args.force,
        )


if __name__ == "__main__":
    main()
