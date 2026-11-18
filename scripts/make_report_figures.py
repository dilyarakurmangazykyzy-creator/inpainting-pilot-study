"""Rebuild the extra figures used in the paper, the poster and the README from the saved result tables.

No GPU and no re-run of the experiment is needed: the numbers come from results/tables/.

    python scripts/make_report_figures.py --out outputs/report_figures

Creates
* paper_fig2_quality.png   mean PSNR, SSIM, LPIPS per damage level (SSIM and LPIPS on a common 0-1 axis), Fig. 2 of the paper
* fig_by_type_large.png    PDE-based vs diffusion-based per photo type at 40 % damage (README)
* poster_fig2_quality.png  quality chart in the poster colours
* poster_fig3_visual.png   restorations at 40 % damage, first photo of each type (cropped from the paper figures)
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
import pandas as pd                                 # noqa: E402
from matplotlib.patches import Patch                # noqa: E402
from PIL import Image                               # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from inpainting_study.config import LEVEL_LABEL, METHODS_ORDER   # noqa: E402
from inpainting_study.plotting import COL, set_theme             # noqa: E402

LEVELS = ['small', 'medium', 'large']


def load_means():
    t = pd.read_csv(ROOT / 'results' / 'tables' / 'table_main_mean_sd.csv')
    t = t.rename(columns={'PSNR_dB_mean': 'PSNR', 'SSIM_mean': 'SSIM', 'LPIPS_mean': 'LPIPS'})
    return t.set_index(['level', 'method'])


def paper_quality(means, out):
    """Fig. 2 of the paper: grouped bars, SSIM and LPIPS share a fixed 0-1 axis, PSNR stays in dB."""
    set_theme()
    quality = [('PSNR', 'PSNR (dB)', 'higher is better'), ('SSIM', 'SSIM', 'higher is better'),
               ('LPIPS', 'LPIPS', 'lower is better')]
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), gridspec_kw={'wspace': 0.28})
    w = 0.26
    for ax, (met, title, better) in zip(axs, quality):
        top = max(means.loc[(l, m), met] for l in LEVELS for m in METHODS_ORDER)
        for j, m in enumerate(METHODS_ORDER):
            x = np.arange(3) + (j - 1) * (w + 0.02)
            ax.bar(x, [means.loc[(l, m), met] for l in LEVELS], w, color=COL[m], zorder=2)
        ax.set_xticks(np.arange(3), [LEVEL_LABEL[l] for l in LEVELS])
        ax.set_xlabel('Damaged area')
        ax.set_ylim(0, 1.0) if met in ('SSIM', 'LPIPS') else ax.set_ylim(0, top * 1.14)
        ax.grid(True, axis='y')
        ax.tick_params(length=0)
        ax.set_title(f'{title}\n({better})', loc='left')
    fig.legend(handles=[Patch(fc=COL[m], label=m) for m in METHODS_ORDER], loc='upper center',
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7.5)
    fig.savefig(out / 'paper_fig2_quality.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def by_type(out):
    c = pd.read_csv(ROOT / 'results' / 'tables' / 'summary_by_category.csv')
    c = c[c.level == 'large'].set_index('category')
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans']})
    types = ['portrait', 'landscape', 'group']
    label = {'portrait': 'Portraits\n(7)', 'landscape': 'Landscapes\n(7)', 'group': 'Groups\n(6)'}
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    x, w = np.arange(3), 0.36
    for ax, (met, title, better, fmt) in zip(axs, [('PSNR', 'PSNR (dB)', 'higher is better', '{:.1f}'),
                                                    ('SSIM', 'SSIM', 'higher is better', '{:.2f}'),
                                                    ('LPIPS', 'LPIPS', 'lower is better', '{:.3f}')]):
        mx = 0
        for j, m in enumerate(['PDE', 'Diffusion']):
            v = [c.loc[t, f'{met}_{m}'] for t in types]
            mx = max(mx, max(v))
            bars = ax.bar(x + (j - 0.5) * (w + 0.02), v, w, color=COL[m], label=f'{m}-based', zorder=2)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                        ha='center', va='bottom', fontsize=9)
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=24)
        ax.text(0, 1.03, better, transform=ax.transAxes, fontsize=9.5, color='#6b7280')
        ax.set_xticks(x, [label[t] for t in types])
        ax.set_ylim(0, mx * 1.15)
        for s in ['top', 'right', 'left']:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', labelsize=9, length=0, colors='#6b7280')
        ax.tick_params(axis='x', length=0)
        ax.yaxis.grid(True, color='#e2dfd6', lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=3)
    fig.savefig(out / 'fig_by_type_large.png', dpi=200)
    plt.close(fig)


POSTER_COL = {'PDE': '#A99BD3', 'Exemplar': '#F2A65A', 'Diffusion': '#4B2A7B'}   # purple poster theme


def poster_quality(means, out):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans'],
                         'font.size': 17, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#666', 'axes.labelcolor': '#333',
                         'xtick.color': '#333', 'ytick.color': '#333'})
    x, w = np.arange(3), 0.27
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.3), gridspec_kw={'wspace': 0.32})
    for ax, (met, t) in zip(axs, [('PSNR', 'PSNR (dB)  ↑'), ('SSIM', 'SSIM  ↑'), ('LPIPS', 'LPIPS  ↓')]):
        for i, m in enumerate(METHODS_ORDER):
            ax.bar(x + (i - 1) * w, [means.loc[(l, m), met] for l in LEVELS], w, color=POSTER_COL[m])
        ax.set_xticks(x, [LEVEL_LABEL[l] for l in LEVELS])
        ax.set_title(t, loc='left', fontsize=18, fontweight='bold', color='#2E1A4F')
        ax.set_ylim(0, 1) if met != 'PSNR' else ax.set_ylim(0, 40)
        ax.yaxis.grid(True, color='#ECE8F3')
        ax.set_axisbelow(True)
    axs[2].text(x[2] + w + 0.08, means.loc[('large', 'Diffusion'), 'LPIPS'] + 0.04,
                f"{means.loc[('large', 'Diffusion'), 'LPIPS'] + 1e-6:.3f}", ha='center', fontsize=15,
                color='#4B2A7B', fontweight='bold')
    axs[2].text(x[2] - w - 0.08, means.loc[('large', 'PDE'), 'LPIPS'] + 0.04,
                f"{means.loc[('large', 'PDE'), 'LPIPS'] + 1e-6:.3f}", ha='center', fontsize=15, color='#6E5FA8')
    fig.legend(handles=[Patch(color=POSTER_COL[m], label=f'{m}-based') for m in METHODS_ORDER],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=16)
    fig.savefig(out / 'poster_fig2_quality.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def poster_visual(out):
    """Crop the 40 % rows from the paper's restoration figures (3 rows x 5 panels each)."""
    src = {'portrait_1': 'fig3_portrait1.png', 'landscape_1': 'fig4_landscape1.png', 'group_1': 'fig5_group1.png'}
    cols = ['Original', 'Damaged', 'PDE-\nbased', 'Exemplar-\nbased', 'Diffusion-\nbased']
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'DejaVu Sans']})
    fig, axs = plt.subplots(3, 5, figsize=(10, 6.6), gridspec_kw={'wspace': 0.04, 'hspace': 0.06})
    for r, (name, f) in enumerate(src.items()):
        im = Image.open(ROOT / 'paper' / 'figures' / f).convert('RGB')
        for c in range(5):
            x0 = int(round(26 + 145.5 * c))               # panel positions in the 751 x 526 px paper figures
            p = im.crop((x0, 383, x0 + 139, 522)).resize((420, 420), Image.LANCZOS)
            a = axs[r, c]
            a.imshow(p)
            a.set_xticks([])
            a.set_yticks([])
            for s in a.spines.values():
                s.set_visible(c == 4)
                s.set_color('#4B2A7B')
                s.set_linewidth(3)
            if r == 0:
                a.set_title(cols[c], fontsize=16, fontweight='bold', color='#4B2A7B' if c == 4 else '#333')
            if c == 0:
                a.set_ylabel(name, fontsize=15, fontweight='bold')
    fig.savefig(out / 'poster_fig3_visual.png', dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='outputs/report_figures')
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    means = load_means()
    paper_quality(means, out)
    by_type(out)
    poster_quality(means, out)
    poster_visual(out)
    print('Figures written to', out.resolve())


if __name__ == '__main__':
    main()
