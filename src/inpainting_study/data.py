"""Loading the 20 photographs (Section 4 of the notebook)."""
import hashlib
import urllib.request
from pathlib import Path

import cv2
import pandas as pd

from .config import BSDS_PHOTOS, BSDS_URL, CATEGORIES, IMG_SIZE


def to_square(img, size=IMG_SIZE):
    """Centre-crop to a square and resize to size x size px."""
    h, w = img.shape[:2]
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    img = img[y0:y0 + s, x0:x0 + s]
    interp = cv2.INTER_AREA if s > size else cv2.INTER_CUBIC
    return cv2.resize(img, (size, size), interpolation=interp)


def read_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Cannot read {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class _IdMaker:
    """Simple study ID per category: portrait_1, portrait_2, ..., landscape_1, ..., group_1, ..."""
    def __init__(self):
        self.count = {c: 0 for c in CATEGORIES}

    def __call__(self, cat):
        self.count[cat] += 1
        return f'{cat}_{self.count[cat]}'


def load_bsds500(out):
    """Download the 20 BSDS500 test photographs, verify their MD5 checksums and return them as 256x256 RGB arrays.

    Returns (images, cat_of, dataset_table): images = {image_id: array}, cat_of = {image_id: category}.
    """
    out = Path(out)
    raw_dir = out / 'source_photos'
    raw_dir.mkdir(parents=True, exist_ok=True)
    next_id = _IdMaker()
    records, ds_rows = [], []
    for bid, (cat, md5) in BSDS_PHOTOS.items():
        f = raw_dir / f'{bid}.jpg'
        if not f.exists() or hashlib.md5(f.read_bytes()).hexdigest() != md5:
            urllib.request.urlretrieve(BSDS_URL.format(bid), f)
        got = hashlib.md5(f.read_bytes()).hexdigest()
        assert got == md5, f'Checksum mismatch for BSDS {bid}: file differs from the one used in the study'
        img = read_rgb(f)
        image_id = next_id(cat)
        records.append({'image_id': image_id, 'category': cat, 'img': to_square(img)})
        ds_rows.append({'image_id': image_id, 'category': cat, 'collection': 'BSDS500 (test split)', 'bsds_id': bid,
                        'original_px': f'{img.shape[1]}x{img.shape[0]}', 'md5': md5, 'url': BSDS_URL.format(bid)})
    return _finish(records, ds_rows, out)


def load_folder(folder, out):
    """Use your own photographs instead. File names must start with portrait_, landscape_ or group_."""
    out = Path(out)
    files = sorted(p for p in Path(folder).iterdir()
                   if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'))
    assert files, f'No images found in {folder}'
    next_id = _IdMaker()
    records, ds_rows = [], []
    for p in files:
        cat = p.stem.split('_')[0].lower()
        assert cat in CATEGORIES, f'{p.name}: file name must start with one of {CATEGORIES}'
        image_id = next_id(cat)
        records.append({'image_id': image_id, 'category': cat, 'img': to_square(read_rgb(p))})
        ds_rows.append({'image_id': image_id, 'category': cat, 'collection': 'own folder', 'file': p.name})
    return _finish(records, ds_rows, out)


def _finish(records, ds_rows, out):
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    (out / 'clean').mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(ds_rows)
    table.to_csv(out / 'tables' / 'dataset_photos.csv', index=False)
    for r in records:
        cv2.imwrite(str(out / 'clean' / f"{r['image_id']}.png"), cv2.cvtColor(r['img'], cv2.COLOR_RGB2BGR))
    images = {r['image_id']: r['img'] for r in records}
    cat_of = {r['image_id']: r['category'] for r in records}
    return images, cat_of, table
