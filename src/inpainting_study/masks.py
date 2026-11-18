"""Irregular damage masks of about 5 %, 20 % and 40 % (Section 5 of the notebook).

Masks are built from uneven brush strokes (scratch/tear-like) and irregular blotches (stain-like) at random
positions and sizes, added until coverage falls inside target +- 1 percentage point.
Each image x level gets its own reproducible seed. 1 = damaged pixel.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import IMG_SIZE, LEVELS, MASK_TOL, SEED


def _stroke(canvas, rng, scale):
    n = rng.integers(3, 10)
    x, y = rng.integers(0, IMG_SIZE, 2)
    thick = int(max(2, rng.uniform(0.012, 0.05) * IMG_SIZE * scale))
    pts = [(int(x), int(y))]
    angle = rng.uniform(0, 2 * np.pi)
    for _ in range(n):
        angle += rng.normal(0, 0.6)                       # uneven, wandering path
        length = rng.uniform(0.04, 0.18) * IMG_SIZE * scale
        x = int(np.clip(x + length * np.cos(angle), 0, IMG_SIZE - 1))
        y = int(np.clip(y + length * np.sin(angle), 0, IMG_SIZE - 1))
        pts.append((x, y))
    for a, b in zip(pts[:-1], pts[1:]):
        t = max(1, int(thick * rng.uniform(0.6, 1.3)))    # varying width along the stroke
        cv2.line(canvas, a, b, 1, t, lineType=cv2.LINE_AA)
        cv2.circle(canvas, b, t // 2, 1, -1)


def _blotch(canvas, rng, scale):
    cx, cy = rng.integers(0, IMG_SIZE, 2)
    r = rng.uniform(0.03, 0.12) * IMG_SIZE * scale
    k = rng.integers(8, 16)
    ang = np.sort(rng.uniform(0, 2 * np.pi, k))
    rad = r * rng.uniform(0.5, 1.3, k)                    # jagged outline
    poly = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
    cv2.fillPoly(canvas, [poly], 1, lineType=cv2.LINE_AA)


def make_mask(target, seed, tol=MASK_TOL, max_tries=5000):
    rng = np.random.default_rng(seed)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
    scale = 0.6 + 2.0 * target                            # bigger elements for bigger targets
    for _ in range(max_tries):
        cov = mask.mean()
        if cov >= target - tol:
            break
        cand = mask.copy()
        (_stroke if rng.random() < 0.6 else _blotch)(cand, rng, scale)
        if cand.mean() <= target + tol:
            mask = cand
        else:
            scale = max(0.15, scale * 0.8)                # too big -> try smaller pieces
    return (mask > 0).astype(np.uint8)


def make_all_masks(images, out=None):
    """One mask per photograph and damage level. Seed = SEED + 1000 * image index + level index.

    Returns (masks, mask_table): masks = {(image_id, level): uint8 array}.
    """
    masks, rows = {}, []
    for i, img_id in enumerate(images):
        for j, (lvl, tgt) in enumerate(LEVELS.items()):
            m = make_mask(tgt, seed=SEED + 1000 * i + j)
            masks[(img_id, lvl)] = m
            if out is not None:
                out = Path(out)
                cv2.imwrite(str(out / 'masks' / f'{img_id}__{lvl}.png'), m * 255)
                dmg = images[img_id].copy()
                dmg[m.astype(bool)] = 255                 # white damage, for display only
                cv2.imwrite(str(out / 'damaged' / f'{img_id}__{lvl}.png'), cv2.cvtColor(dmg, cv2.COLOR_RGB2BGR))
            rows.append({'image_id': img_id, 'level': lvl, 'target': tgt, 'actual_coverage': m.mean()})
    table = pd.DataFrame(rows)
    if out is not None:
        table.to_csv(Path(out) / 'tables' / 'mask_coverage.csv', index=False)
    return masks, table
