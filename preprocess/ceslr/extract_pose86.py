#!/usr/bin/env python3
"""
Extract Isharah / MSLR Pose86 keypoints from CESLR-multisigner RGB frames.

Output matches the competition pickle format used by this repo
(datasets/pose_data_isharah1000_hands_lips_body_May12.pkl):

    {
        "<video_id>": {"keypoints": np.ndarray[T, 86, 2]},
        ...
    }

Keypoint layout (MediaPipe Holistic), matching skeleton_feeder.py:
    0–20   right hand  (21)
    21–41  left hand   (21)
    42–60  lips        (19, face-mesh outer contour)
    61–85  upper body  (25, pose landmarks 0–24)

Coordinates are MediaPipe-normalized [0, 1] scaled by COORD_SCALE (10240),
which matches SkeletonFeeder.norm_div = (10240 - 1) / 2.

Example:
    python preprocess/ceslr/extract_pose86.py \\
        --dataset-root ./datasets/CESLR-multisigner \\
        --output ./datasets/pose_data_ceslr_hands_lips_body.pkl
"""

from __future__ import annotations

import argparse
import csv
import pickle
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

# Same canvas scale as SkeletonFeeder for Isharah pose pkls.
COORD_SCALE = 10240.0
NUM_KEYPOINTS = 86

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/latest/"
    "holistic_landmarker.task"
)

# MediaPipe Face Mesh outer-lip contour (19 pts, ring order).
# Derived from the standard 20-point outer lip ring by dropping 409.
LIP_LANDMARKS = [
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
]

# MediaPipe Pose upper-body / face-torso subset (25 pts): landmarks 0–24.
BODY_LANDMARKS = list(range(25))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract Pose86 MediaPipe features from CESLR-multisigner"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./datasets/CESLR-multisigner"),
        help="CESLR-multisigner root (annotations/ + features/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./datasets/pose_data_ceslr_hands_lips_body.pkl"),
        help="Output pickle path (Isharah-compatible)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("./preprocess/ceslr/models/holistic_landmarker.task"),
        help="Path to holistic_landmarker.task (downloaded if missing)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        help="Corpus splits to process",
    )
    parser.add_argument(
        "--frame-subdir",
        type=str,
        default="fullFrame-210x260px",
        help="Folder under features/ that holds split/video frames",
    )
    parser.add_argument(
        "--min-side",
        type=int,
        default=256,
        help="Upsample so min(H, W) >= this (helps MediaPipe on small frames)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip video_ids already present in --output",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Checkpoint pickle every N newly processed videos",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Optional cap for a dry-run / smoke test",
    )
    return parser.parse_args()


def ensure_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path
    print(f"Downloading Holistic model → {model_path}")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def load_corpus_rows(dataset_root: Path, splits: list[str]) -> list[dict]:
    rows = []
    for split in splits:
        csv_path = dataset_root / "annotations" / "manual" / f"{split}.corpus.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing annotation file: {csv_path}")
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="|")
            for row in reader:
                row = dict(row)
                row["split"] = split
                rows.append(row)
    return rows


def resolve_frame_dir(dataset_root: Path, frame_subdir: str, split: str, folder: str) -> Path:
    # folder looks like: 01day_2016_sentence_0/1/*.png
    rel = folder.replace("/*.png", "").replace("/*.jpg", "").strip("/")
    return dataset_root / "features" / frame_subdir / split / rel


def list_frames(frame_dir: Path) -> list[Path]:
    frames = list(frame_dir.glob("*.png")) + list(frame_dir.glob("*.jpg"))

    def sort_key(p: Path):
        name = p.stem
        if "_frame_" in name:
            try:
                return int(name.split("_frame_")[-1])
            except ValueError:
                return name
        return name

    return sorted(frames, key=sort_key)


def write_selected(landmarks, indices: list[int], out: np.ndarray) -> None:
    """Write selected landmarks into out[len(indices), 2]. Missing → zeros."""
    if not landmarks:
        out[:] = 0.0
        return
    n = len(landmarks)
    for i, idx in enumerate(indices):
        if idx >= n:
            out[i] = 0.0
            continue
        lm = landmarks[idx]
        out[i, 0] = lm.x * COORD_SCALE
        out[i, 1] = lm.y * COORD_SCALE


def write_hand(landmarks, out: np.ndarray) -> None:
    if not landmarks:
        out[:] = 0.0
        return
    count = min(len(landmarks), out.shape[0])
    for i in range(count):
        lm = landmarks[i]
        out[i, 0] = lm.x * COORD_SCALE
        out[i, 1] = lm.y * COORD_SCALE
    if count < out.shape[0]:
        out[count:] = 0.0


def maybe_upsample(image_bgr: np.ndarray, min_side: int) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    short = min(h, w)
    if short >= min_side:
        return image_bgr
    scale = min_side / float(short)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def bgr_to_mp_image(image_bgr: np.ndarray) -> mp.Image:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)


def extract_video_keypoints(
    frame_paths: list[Path],
    landmarker: vision.HolisticLandmarker,
    min_side: int,
    timestamp_ms: list[int],
) -> np.ndarray:
    """Return keypoints array of shape (T, 86, 2).

    timestamp_ms is a length-1 list used as a mutable global clock so VIDEO-mode
    timestamps stay strictly increasing across videos.
    """
    T = len(frame_paths)
    kps = np.zeros((T, NUM_KEYPOINTS, 2), dtype=np.float32)

    for t, path in enumerate(frame_paths):
        image_bgr = cv2.imread(str(path))
        if image_bgr is None:
            continue
        image_bgr = maybe_upsample(image_bgr, min_side)
        mp_image = bgr_to_mp_image(image_bgr)
        timestamp_ms[0] += 33
        result = landmarker.detect_for_video(mp_image, timestamp_ms[0])

        write_hand(result.right_hand_landmarks, kps[t, 0:21])
        write_hand(result.left_hand_landmarks, kps[t, 21:42])
        write_selected(result.face_landmarks, LIP_LANDMARKS, kps[t, 42:61])
        write_selected(result.pose_landmarks, BODY_LANDMARKS, kps[t, 61:86])

    return kps


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def create_landmarker(model_path: Path) -> vision.HolisticLandmarker:
    options = vision.HolisticLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.VIDEO,
        min_face_detection_confidence=0.5,
        min_face_landmarks_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_pose_landmarks_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    return vision.HolisticLandmarker.create_from_options(options)


def main():
    args = parse_args()
    model_path = ensure_model(args.model_path)
    rows = load_corpus_rows(args.dataset_root, args.splits)
    if args.max_videos is not None:
        rows = rows[: args.max_videos]

    pose_db = {}
    if args.resume and args.output.exists():
        with open(args.output, "rb") as f:
            pose_db = pickle.load(f)
        print(f"Resumed with {len(pose_db)} existing videos from {args.output}")

    processed_since_save = 0
    skipped = 0
    failed = []
    timestamp_ms = [0]
    landmarker = create_landmarker(model_path)

    try:
        for row in tqdm(rows, desc="Extracting Pose86"):
            video_id = row["id"].strip()
            if video_id in pose_db:
                skipped += 1
                continue

            frame_dir = resolve_frame_dir(
                args.dataset_root, args.frame_subdir, row["split"], row["folder"]
            )
            if not frame_dir.exists():
                failed.append((video_id, f"missing frame dir: {frame_dir}"))
                continue

            frames = list_frames(frame_dir)
            if not frames:
                failed.append((video_id, f"no frames in {frame_dir}"))
                continue

            try:
                keypoints = extract_video_keypoints(
                    frames, landmarker, args.min_side, timestamp_ms
                )
            except Exception as exc:  # noqa: BLE001 - keep batch job running
                failed.append((video_id, str(exc)))
                # Recreate landmarker if internal state breaks after an error.
                landmarker.close()
                landmarker = create_landmarker(model_path)
                timestamp_ms[0] = 0
                continue

            pose_db[video_id] = {"keypoints": keypoints}
            processed_since_save += 1

            if processed_since_save >= args.save_every:
                save_pickle(pose_db, args.output)
                processed_since_save = 0
    finally:
        landmarker.close()

    save_pickle(pose_db, args.output)

    if pose_db:
        sample_id = next(iter(pose_db))
        sample = pose_db[sample_id]["keypoints"]
        print(
            f"Saved {len(pose_db)} videos → {args.output}\n"
            f"Example: {sample_id} shape={sample.shape} "
            f"(expect T×{NUM_KEYPOINTS}×2), "
            f"coord range=[{sample.min():.1f}, {sample.max():.1f}]"
        )
    else:
        print("No videos were written.")

    print(f"Skipped (already present): {skipped}")
    if failed:
        print(f"Failed: {len(failed)}")
        for vid, reason in failed[:20]:
            print(f"  - {vid}: {reason}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")


if __name__ == "__main__":
    main()
