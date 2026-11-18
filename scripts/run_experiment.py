"""Run the whole pilot study from the command line (same steps as the notebook).

20 photos x 3 damage levels x 3 methods = 180 restored outputs, then metrics, statistics and figures.

Usage (from the repository root):

    git clone https://github.com/zsyOAOA/ResShift.git
    python scripts/run_experiment.py --resshift-dir ResShift --out outputs

Quick test without a GPU (classical methods only, 3 photos):

    python scripts/run_experiment.py --classical-only --limit 3 --out outputs_test
"""
import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from inpainting_study import analysis, plotting                          # noqa: E402
from inpainting_study.config import LEVELS, METHODS_ORDER, OUT_SUBDIRS, SEED   # noqa: E402
from inpainting_study.data import load_bsds500, load_folder              # noqa: E402
from inpainting_study.masks import make_all_masks                        # noqa: E402
from inpainting_study.methods import get_methods                         # noqa: E402
from inpainting_study.metrics import LpipsScorer, score                  # noqa: E402


def set_all_seeds(s):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def environment():
    env = {'python': platform.python_version(), 'opencv': cv2.__version__, 'numpy': np.__version__,
           'cpu': platform.processor() or platform.machine(), 'cpu_threads': os.cpu_count()}
    try:
        import skimage
        import torch
        env.update({'torch': torch.__version__, 'scikit-image': skimage.__version__,
                    'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none',
                    'gpu_memory_GB': round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                    if torch.cuda.is_available() else 0})
    except ImportError:
        env.update({'torch': 'not installed', 'gpu': 'none', 'gpu_memory_GB': 0})
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='outputs', help='output folder')
    ap.add_argument('--resshift-dir', default='ResShift', help='clone of github.com/zsyOAOA/ResShift')
    ap.add_argument('--photos', default=None, help='use your own photos from this folder instead of BSDS500')
    ap.add_argument('--classical-only', action='store_true', help='skip the diffusion method (no GPU needed)')
    ap.add_argument('--limit', type=int, default=None, help='use only the first N photos (for a quick test)')
    args = ap.parse_args()

    out = Path(args.out)
    for sub in OUT_SUBDIRS:
        (out / sub).mkdir(parents=True, exist_ok=True)
    set_all_seeds(SEED)
    plotting.set_theme()
    env = environment()
    json.dump(env, open(out / 'tables' / 'environment.json', 'w'), indent=2)

    # 1. data
    images, cat_of, _ = load_folder(args.photos, out) if args.photos else load_bsds500(out)
    if args.limit:
        keep = list(images)[:args.limit]
        images = {k: images[k] for k in keep}
        cat_of = {k: cat_of[k] for k in keep}
    print(f'{len(images)} images loaded:', pd.Series(cat_of).value_counts().to_dict())

    # 2. masks
    masks, mask_df = make_all_masks(images, out)
    print(mask_df.groupby('level')['actual_coverage'].agg(['mean', 'std', 'min', 'max']).round(4))

    # 3. methods
    methods, load_time = get_methods(None if args.classical_only else args.resshift_dir)
    if load_time is not None:
        print(f'ResShift loaded in {load_time:.1f}s (one-off, NOT counted in per-image time)')
    use_cuda = 'Diffusion' in methods

    # 4. main experiment: all runs in a randomised (but fixed) order
    runs = [(i, l, m) for i in images for l in LEVELS for m in methods]
    random.Random(SEED).shuffle(runs)
    pd.DataFrame(runs, columns=['image_id', 'level', 'method']).to_csv(out / 'tables' / 'run_order.csv',
                                                                       index_label='order')
    if use_cuda:                                     # warm-up, excluded from timing (as in the notebook)
        first = list(images)[0]
        methods['Diffusion'](images[first], masks[(first, 'medium')])
    timing = []
    for k, (img_id, lvl, meth) in enumerate(runs):
        img, m = images[img_id], masks[(img_id, lvl)]
        if meth == 'Diffusion':
            import torch
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        res = methods[meth](img, m)
        if meth == 'Diffusion':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        cv2.imwrite(str(out / 'restored' / f'{img_id}__{lvl}__{meth}.png'), cv2.cvtColor(res, cv2.COLOR_RGB2BGR))
        timing.append({'order': k, 'image_id': img_id, 'level': lvl, 'method': meth, 'time_s': dt})
        if k % 20 == 0 or k == len(runs) - 1:
            print(f'[{k + 1:3d}/{len(runs)}] {img_id:<22} {lvl:<7} {meth:<9} {dt:7.3f}s')

    # 5. metrics
    lpips_fn = LpipsScorer() if use_cuda else None
    rows = []
    for r in timing:
        gt = images[r['image_id']]
        m = masks[(r['image_id'], r['level'])]
        res = cv2.cvtColor(cv2.imread(str(out / 'restored' / f"{r['image_id']}__{r['level']}__{r['method']}.png")),
                           cv2.COLOR_BGR2RGB)
        rows.append({**r, 'category': cat_of[r['image_id']], 'coverage': m.mean(), **score(gt, res, m, lpips_fn)})
    df = pd.DataFrame(rows)
    df['level'] = pd.Categorical(df['level'], list(LEVELS), ordered=True)
    df['method'] = pd.Categorical(df['method'], [x for x in METHODS_ORDER if x in methods], ordered=True)
    df = df.sort_values(['image_id', 'level', 'method']).reset_index(drop=True)
    df.to_csv(out / 'tables' / 'results_per_image.csv', index=False)

    # 6. tables and statistics
    print(analysis.main_table(df, out))
    analysis.by_category(df, out)
    stats_df = analysis.statistics(df, out)
    if use_cuda:
        baseline_df = analysis.diffusion_vs_baselines(df, stats_df, out)
        print(analysis.key_results(df, stats_df, baseline_df, mask_df, env, load_time, out))

    # 7. figures
    all_types = set(cat_of.values()) == {'portrait', 'landscape', 'group'}
    if all_types:
        plotting.fig_dataset(images, cat_of, out)
        plotting.fig_masks(images, cat_of, masks, out)
    if use_cuda and all_types:
        plotting.fig_quality(df, out)
        plotting.fig_time(df, out)
        plotting.fig_qualitative(images, cat_of, masks, out)
        plotting.fig_by_type_large(df, out)
    print('All outputs written to', out.resolve())


if __name__ == '__main__':
    main()
