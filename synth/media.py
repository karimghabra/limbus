"""Animated previews: real and synthetic clips side by side, as small looping WebP files."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def to_gray(frames, low_pct=0.5, high_pct=99.5):
    """8-bit frames stretched between percentiles of the whole stack (so brightness changes stay visible)."""
    frames = np.asarray(frames, np.float32)
    low, high = np.nanpercentile(frames[:: max(1, len(frames) // 16)], [low_pct, high_pct])
    return np.clip((np.nan_to_num(frames, nan=low) - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)


def side_by_side(left, right, labels=("real", "synthetic"), height=None, gap=6):
    """Two (T, H, W) stacks, each stretched on its own, scaled to one height and placed side by side; the shorter
    clip repeats. Labels are written in the top-left corner of each half."""
    left, right = to_gray(left), to_gray(right)
    height = height or min(left.shape[1], right.shape[1])
    n = max(len(left), len(right))

    def scaled(stack, k):
        frame = stack[k % len(stack)]
        width = int(round(frame.shape[1] * height / frame.shape[0]))
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    out = []
    for k in range(n):
        a, b = scaled(left, k), scaled(right, k)
        frame = np.hstack([a, np.full((height, gap), 255, np.uint8), b])
        for text, x in ((labels[0], 6), (labels[1], a.shape[1] + gap + 6)):
            cv2.putText(frame, text, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 255, 3, cv2.LINE_AA)
            cv2.putText(frame, text, (x, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
        out.append(frame)
    return np.stack(out)


def webp(frames, path, fps, width=960, max_mb=2.5, max_frames=150):
    """Write a looping animated WebP no larger than max_mb (narrower and fewer frames until it fits)."""
    frames = np.asarray(frames)[:max_frames]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        w = min(width, frames.shape[2])
        h = int(round(frames.shape[1] * w / frames.shape[2]))
        images = [Image.fromarray(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)) for f in frames]
        buffer = io.BytesIO()
        images[0].save(buffer, format="WEBP", save_all=True, append_images=images[1:], duration=int(1000 / fps),
                       loop=0, quality=70, method=4)
        if buffer.tell() <= max_mb * 1e6 or width < 240:
            path.write_bytes(buffer.getvalue())
            return path
        width = int(width * 0.8)
