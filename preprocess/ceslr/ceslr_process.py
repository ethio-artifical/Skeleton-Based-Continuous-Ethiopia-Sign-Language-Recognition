#!/usr/bin/env python3
"""
Build CESLR gloss dict / split info / STM files in the same style as
preprocess/mslr2025/mslr_process.py, so SkeletonFeeder can consume CESLR.

Example:
    python preprocess/ceslr/ceslr_process.py \\
        --dataset-root ./datasets/CESLR-multisigner \\
        --save-root ./datasets/ceslr
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Preprocess CESLR annotations")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "datasets/CESLR-multisigner",
    )
    parser.add_argument(
        "--save-root",
        type=Path,
        default=repo_root / "datasets/ceslr",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
    )
    return parser.parse_args()


def load_split(dataset_root: Path, split: str) -> list[dict]:
    csv_path = dataset_root / "annotations" / "manual" / f"{split}.corpus.csv"
    info_list = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            video_id = row["id"].strip()
            gloss_seq = row["annotation"].strip()
            signer = row["signer"].strip()
            # Keep a pipe-delimited original_info string compatible with feeder logging.
            original_info = f"{video_id}|{gloss_seq}|{signer}"
            info_list.append(
                {
                    "signer": signer,
                    "video_id": video_id,
                    "gloss_sequence": gloss_seq,
                    "sentence_id": video_id,
                    "folder": row["folder"].strip(),
                    "original_info": original_info,
                }
            )
    return info_list


def sign_dict_update(total_dict: dict, info: list[dict]) -> dict:
    for item in info:
        for gloss in item["gloss_sequence"].split():
            if not gloss:
                continue
            total_dict[gloss] = total_dict.get(gloss, 0) + 1
    return total_dict


def generate_gt_stm(info: list[dict], save_path: Path) -> None:
    with open(save_path, "w", encoding="utf-8") as f:
        for item in info:
            f.write(
                f"{item['video_id']} 1 {item['signer']} 0.0 1.79769e+308 {item['gloss_sequence']}\n"
            )


def main():
    args = parse_args()
    args.save_root.mkdir(parents=True, exist_ok=True)

    sign_dict = {}
    for split in args.splits:
        split_info = load_split(args.dataset_root, split)
        out_json = args.save_root / f"{split}_info.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(split_info, f, indent=4, ensure_ascii=False)
        generate_gt_stm(split_info, args.save_root / f"ceslr-groundtruth-{split}.stm")
        if split in {"train", "dev"}:
            sign_dict_update(sign_dict, split_info)
        print(f"{split}: {len(split_info)} samples → {out_json}")

    # Gloss dict is built from train+dev by default (same idea as mslr_process).
    items = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {"id2gloss": {}, "gloss2id": {}}
    for idx, (key, value) in enumerate(items):
        save_dict["gloss2id"][key] = {"index": idx + 1, "frequency": value}
        save_dict["id2gloss"][idx + 1] = {"gloss": key, "frequency": value}

    gloss_path = args.save_root / "gloss_dict.json"
    with open(gloss_path, "w", encoding="utf-8") as f:
        json.dump(save_dict, f, indent=4, ensure_ascii=False)
    print(f"vocab size: {len(items)} → {gloss_path}")


if __name__ == "__main__":
    main()
