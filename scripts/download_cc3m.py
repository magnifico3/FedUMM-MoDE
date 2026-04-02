#!/usr/bin/env python3
"""Download Conceptual Captions 3M metadata and images."""

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from datasets import load_dataset
from tqdm.auto import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.common import build_hf_download_config, get_default_cc3m_dir  # noqa: E402


VALID_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download CC3M metadata and images.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="conceptual_captions",
        help="Hugging Face dataset id for CC3M metadata.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "all"],
        help="Which split to download.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=get_default_cc3m_dir(),
        help="Where to save metadata and downloaded images.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Hugging Face cache directory.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Limit the number of samples per split. -1 means all.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Concurrent image download workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-request read timeout in seconds for image downloads.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Retry count for failed image downloads.",
    )
    parser.add_argument(
        "--hf_timeout",
        type=int,
        default=900,
        help="Timeout in seconds for Hugging Face metadata downloads.",
    )
    parser.add_argument(
        "--hf_max_retries",
        type=int,
        default=8,
        help="Retry count for Hugging Face metadata downloads.",
    )
    parser.add_argument(
        "--metadata_only",
        action="store_true",
        help="Only download/save CC3M metadata without fetching image files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload images even if the local file already exists.",
    )
    return parser.parse_args()


def iter_splits(split: str) -> Iterable[str]:
    if split == "all":
        return ("train", "validation")
    return (split,)


def resolve_extension(url: str, content_type: str = "") -> str:
    path = urlparse(url).path.lower()
    ext = Path(path).suffix
    if ext in VALID_EXTENSIONS:
        return ext

    content_type = (content_type or "").lower()
    if "png" in content_type:
        return ".png"
    if "webp" in content_type:
        return ".webp"
    if "gif" in content_type:
        return ".gif"
    if "bmp" in content_type:
        return ".bmp"
    return ".jpg"


def local_name(index: int, url: str) -> str:
    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{index:08d}_{url_hash}"


def load_split(dataset_name: str, split_name: str, cache_dir: str,
               hf_max_retries: int, hf_timeout: int):
    download_config = build_hf_download_config(
        cache_dir=cache_dir,
        max_retries=hf_max_retries,
        timeout_seconds=hf_timeout,
    )
    return load_dataset(
        dataset_name,
        split=split_name,
        cache_dir=cache_dir or None,
        download_config=download_config,
        storage_options=download_config.storage_options,
    )


def save_metadata(records, output_path: Path, max_samples: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx, row in enumerate(records):
            if max_samples >= 0 and idx >= max_samples:
                break
            payload = {
                "id": idx,
                "image_url": row["image_url"],
                "caption": row["caption"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def download_one(index: int, url: str, image_dir: Path, timeout: int,
                 retries: int, overwrite: bool) -> Dict[str, object]:
    base_name = local_name(index, url)

    for ext in sorted(VALID_EXTENSIONS):
        existing = image_dir / f"{base_name}{ext}"
        if existing.exists() and not overwrite:
            return {
                "id": index,
                "image_url": url,
                "status": "exists",
                "path": str(existing),
                "bytes": existing.stat().st_size,
            }

    last_error = "unknown error"
    for attempt in range(retries + 1):
        try:
            req = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; fedumm-cc3m-downloader/1.0)"},
            )
            with urlopen(req, timeout=timeout) as resp:
                content_type = resp.headers.get("Content-Type", "")
                ext = resolve_extension(url, content_type)
                final_path = image_dir / f"{base_name}{ext}"
                tmp_path = final_path.with_suffix(final_path.suffix + ".part")
                data = resp.read()

            if not overwrite and final_path.exists():
                return {
                    "id": index,
                    "image_url": url,
                    "status": "exists",
                    "path": str(final_path),
                    "bytes": final_path.stat().st_size,
                }

            with tmp_path.open("wb") as f:
                f.write(data)
            tmp_path.replace(final_path)
            return {
                "id": index,
                "image_url": url,
                "status": "ok",
                "path": str(final_path),
                "bytes": len(data),
            }
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2 ** attempt, 5))

    return {
        "id": index,
        "image_url": url,
        "status": "error",
        "path": "",
        "bytes": 0,
        "error": last_error,
    }


def download_split(records, split_dir: Path, max_samples: int, num_workers: int,
                   timeout: int, retries: int, overwrite: bool) -> Tuple[int, int]:
    image_dir = split_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = split_dir / "downloads.jsonl"

    total = len(records) if max_samples < 0 else min(len(records), max_samples)
    success = 0
    failed = 0

    with manifest_path.open("a", encoding="utf-8") as manifest, ThreadPoolExecutor(
        max_workers=max(1, num_workers)
    ) as pool:
        pending = {}
        next_index = 0
        progress = tqdm(total=total, desc=f"Downloading {split_dir.name}", unit="img")

        while next_index < total or pending:
            while next_index < total and len(pending) < max(1, num_workers * 2):
                row = records[next_index]
                future = pool.submit(
                    download_one,
                    next_index,
                    row["image_url"],
                    image_dir,
                    timeout,
                    retries,
                    overwrite,
                )
                pending[future] = {
                    "id": next_index,
                    "caption": row["caption"],
                    "image_url": row["image_url"],
                }
                next_index += 1

            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                meta = pending.pop(future)
                result = future.result()
                payload = {
                    **meta,
                    "status": result["status"],
                    "path": result.get("path", ""),
                    "relative_path": (
                        os.path.relpath(result["path"], start=split_dir)
                        if result.get("path")
                        else ""
                    ),
                    "bytes": result.get("bytes", 0),
                }
                if "error" in result:
                    payload["error"] = result["error"]
                manifest.write(json.dumps(payload, ensure_ascii=False) + "\n")
                manifest.flush()
                if result["status"] in {"ok", "exists"}:
                    success += 1
                else:
                    failed += 1
                progress.update(1)

        progress.close()

    return success, failed


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    for split_name in iter_splits(args.split):
        print(f"Loading CC3M split: {split_name}", flush=True)
        records = load_split(
            dataset_name=args.dataset,
            split_name=split_name,
            cache_dir=args.cache_dir,
            hf_max_retries=args.hf_max_retries,
            hf_timeout=args.hf_timeout,
        )

        split_dir = output_root / split_name
        metadata_path = split_dir / "metadata.jsonl"
        count = save_metadata(records, metadata_path, args.max_samples)
        print(f"Saved {count} metadata rows to {metadata_path}", flush=True)

        if args.metadata_only:
            continue

        success, failed = download_split(
            records=records,
            split_dir=split_dir,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            timeout=args.timeout,
            retries=args.retries,
            overwrite=args.overwrite,
        )
        print(
            f"Finished {split_name}: success={success} failed={failed} "
            f"manifest={split_dir / 'downloads.jsonl'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
