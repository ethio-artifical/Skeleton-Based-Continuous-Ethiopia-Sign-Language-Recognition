#!/usr/bin/env python3
"""
Overlay Pose86 keypoints on CESLR frames for visual inspection.

Example:
    python preprocess/ceslr/visualize_pose86.py \\
        --pose-pkl ./datasets/pose_data_ceslr_smoke.pkl \\
        --video-id 01day_2016_sentence_0 \\
        --out-dir ./datasets/pose_vis_smoke
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import cv2
import numpy as np

COORD_SCALE = 10240.0

# BGR colors
COLOR_RHAND = (0, 0, 255)      # red
COLOR_LHAND = (0, 165, 255)    # orange
COLOR_LIPS = (255, 0, 255)     # magenta
COLOR_BODY = (0, 255, 0)       # green

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# MediaPipe pose 0–24 subset edges (upper body)
BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
]

LIP_EDGES = [(i, i + 1) for i in range(18)] + [(18, 0)]


def parse_args():
    p = argparse.ArgumentParser(description="Visualize Pose86 keypoints on CESLR frames")
    p.add_argument("--pose-pkl", type=Path, default=Path("./datasets/pose_data_ceslr_smoke.pkl"))
    p.add_argument("--dataset-root", type=Path, default=Path("./datasets/CESLR-multisigner"))
    p.add_argument("--video-id", type=str, default=None, help="Video id; default = first in pkl")
    p.add_argument("--out-dir", type=Path, default=Path("./datasets/pose_vis_smoke"))
    p.add_argument("--frame-subdir", type=str, default="fullFrame-210x260px")
    p.add_argument("--save-video", action="store_true", help="Also write overlay.mp4")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--stride", type=int, default=1, help="Save every Nth frame as PNG")
    p.add_argument("--num-preview", type=int, default=8, help="How many PNG previews to keep")
    return p.parse_args()


def find_split_and_folder(dataset_root: Path, video_id: str) -> tuple[str, str]:
    for split in ("train", "dev", "test"):
        csv_path = dataset_root / "annotations" / "manual" / f"{split}.corpus.csv"
        with open(csv_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f, delimiter="|"):
                if row["id"].strip() == video_id:
                    return split, row["folder"].strip()
    raise KeyError(f"video_id not found in CESLR annotations: {video_id}")


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


def to_pixel(xy: np.ndarray, w: int, h: int) -> tuple[int, int] | None:
    if abs(xy[0]) < 1e-6 and abs(xy[1]) < 1e-6:
        return None
    x = int(round((xy[0] / COORD_SCALE) * w))
    y = int(round((xy[1] / COORD_SCALE) * h))
    return x, y


def draw_points(img, pts, color, radius=2):
    h, w = img.shape[:2]
    for p in pts:
        px = to_pixel(p, w, h)
        if px is not None:
            cv2.circle(img, px, radius, color, -1, lineType=cv2.LINE_AA)


def draw_edges(img, pts, edges, color, thickness=1):
    h, w = img.shape[:2]
    for i, j in edges:
        if i >= len(pts) or j >= len(pts):
            continue
        a = to_pixel(pts[i], w, h)
        b = to_pixel(pts[j], w, h)
        if a is not None and b is not None:
            cv2.line(img, a, b, color, thickness, lineType=cv2.LINE_AA)


def overlay_frame(image_bgr: np.ndarray, kps_86: np.ndarray) -> np.ndarray:
    out = image_bgr.copy()
    # Upscale for clearer viewing if frames are small
    h, w = out.shape[:2]
    if min(h, w) < 320:
        scale = 320 / float(min(h, w))
        out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    rhand, lhand = kps_86[0:21], kps_86[21:42]
    lips, body = kps_86[42:61], kps_86[61:86]

    draw_edges(out, body, BODY_EDGES, COLOR_BODY, 2)
    draw_points(out, body, COLOR_BODY, 3)

    draw_edges(out, lips, LIP_EDGES, COLOR_LIPS, 1)
    draw_points(out, lips, COLOR_LIPS, 2)

    draw_edges(out, rhand, HAND_EDGES, COLOR_RHAND, 1)
    draw_points(out, rhand, COLOR_RHAND, 2)

    draw_edges(out, lhand, HAND_EDGES, COLOR_LHAND, 1)
    draw_points(out, lhand, COLOR_LHAND, 2)

    # Legend
    y0 = 18
    for label, color in [
        ("body", COLOR_BODY),
        ("lips", COLOR_LIPS),
        ("R-hand", COLOR_RHAND),
        ("L-hand", COLOR_LHAND),
    ]:
        cv2.putText(out, label, (8, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        y0 += 16
    return out


def main():
    args = parse_args()
    with open(args.pose_pkl, "rb") as f:
        pose_db = pickle.load(f)

    video_id = args.video_id or next(iter(pose_db))
    if video_id not in pose_db:
        raise KeyError(f"{video_id} not in {args.pose_pkl}. Available: {list(pose_db)[:5]}...")

    keypoints = pose_db[video_id]["keypoints"]  # T x 86 x 2
    split, folder = find_split_and_folder(args.dataset_root, video_id)
    rel = folder.replace("/*.png", "").replace("/*.jpg", "").strip("/")
    frame_dir = args.dataset_root / "features" / args.frame_subdir / split / rel
    frames = list_frames(frame_dir)
    if not frames:
        raise FileNotFoundError(f"No frames in {frame_dir}")

    T = min(len(frames), keypoints.shape[0])
    if args.max_frames is not None:
        T = min(T, args.max_frames)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = args.out_dir / "frames"
    preview_dir.mkdir(parents=True, exist_ok=True)

    writer = None
    preview_idxs = np.linspace(0, T - 1, num=min(args.num_preview, T), dtype=int)

    for t in range(T):
        img = cv2.imread(str(frames[t]))
        if img is None:
            continue
        vis = overlay_frame(img, keypoints[t])

        if writer is None and args.save_video:
            h, w = vis.shape[:2]
            video_path = args.out_dir / f"{video_id}_pose86.mp4"
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                15,
                (w, h),
            )
            print(f"Writing video → {video_path}")

        if writer is not None:
            writer.write(vis)

        if t in set(preview_idxs.tolist()) or (args.stride > 1 and t % args.stride == 0):
            out_png = preview_dir / f"{video_id}_frame_{t:04d}.png"
            cv2.imwrite(str(out_png), vis)

    if writer is not None:
        writer.release()

    # Always save a contact sheet of previews
    sheet_paths = sorted(preview_dir.glob(f"{video_id}_frame_*.png"))[: args.num_preview]
    if sheet_paths:
        tiles = [cv2.imread(str(p)) for p in sheet_paths]
        tiles = [t for t in tiles if t is not None]
        if tiles:
            h = max(t.shape[0] for t in tiles)
            w = max(t.shape[1] for t in tiles)
            resized = [cv2.resize(t, (w, h)) for t in tiles]
            cols = min(4, len(resized))
            rows = int(np.ceil(len(resized) / cols))
            canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
            for i, tile in enumerate(resized):
                r, c = divmod(i, cols)
                canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
            sheet_path = args.out_dir / f"{video_id}_grid.png"
            cv2.imwrite(str(sheet_path), canvas)
            print(f"Grid preview → {sheet_path}")

    print(
        f"Video: {video_id}\n"
        f"Keypoints: {keypoints.shape}\n"
        f"Frames used: {T}\n"
        f"PNG previews → {preview_dir}\n"
        f"Legend: green=body, magenta=lips, red=right hand, orange=left hand"
    )


if __name__ == "__main__":
    main()
