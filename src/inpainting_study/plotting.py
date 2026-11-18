"""Figures (Sections 3, 4, 5, 11, 12 of the notebook). One look for every figure."""
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from .config import CATEGORIES, LEVEL_LABEL, LEVELS, METHODS_ORDER

INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'            # text: primary / secondary / muted
GRID, AXIS, SURFACE = '#e1e0d9', '#c3c2b7', '#ffffff'           # hairline grid, axis line, background
COL = {'PDE': '#2a78d6', 'Exemplar': '#eb6834', 'Diffusion': '#1baf7a'}   # colour-blind-safe categorical slots
HW = {'PDE': 'CPU', 'Exemplar': 'CPU', 'Diffusion': 'GPU'}
DAMAGE_EDGE = '#e34948'                                         # outline of damaged areas in image figures


def set_theme():
    mpl.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
        'font.size': 8, 'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
        'axes.titlecolor': INK, 'axes.titlepad': 6, 'axes.labelsize': 7.5, 'axes.labelcolor': INK2,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': False, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.axisbelow': True,
        'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'xtick.major.size': 0, 'ytick.major.size': 0,
        'xtick.minor.size': 0, 'ytick.minor.size': 0, 'xtick.major.pad': 4, 'ytick.major.pad': 4,
        'legend.frameon': False, 'legend.fontsize': 7.5, 'legend.handlelength': 1.8, 'legend.columnspacing': 1.6,
        'lines.linewidth': 1.6, 'lines.markersize': 5, 'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
        'savefig.facecolor': SURFACE, 'figure.dpi': 130, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def image_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


def show_damage(ax, img, mask):
    """Damaged pixels shown white (what the methods receive) with a thin red outline."""
    d = img.copy()
    d[mask.astype(bool)] = 255
    ax.imshow(d)
    ax.contour(mask.astype(float), levels=[0.5], colors=[DAMAGE_EDGE], linewidths=0.35)


def save_fig(fig, out, name, pdf=True):
    folder = Path(out) / 'figures'
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f'{name}.png')
    if pdf:
        fig.savefig(folder / f'{name}.pdf')
    plt.close(fig)


# ---------- Section 4: the 20 photographs ----------
def fig_dataset(images, cat_of, out):
    ncol = max(sum(c == k for c in cat_of.values()) for k in CATEGORIES)
    fig, axs = plt.subplots(3, ncol, figsize=(7.16, 3.35), gridspec_kw={'wspace': 0.04, 'hspace': 0.22})
    for r, cat in enumerate(CATEGORIES):
        ids = [i for i in images if cat_of[i] == cat]
        for c in range(ncol):
            ax = axs[r, c]
            image_ax(ax)
            if c < len(ids):
                ax.imshow(images[ids[c]])
                ax.set_xlabel(ids[c], fontsize=6.5, color=INK2, labelpad=2)
            else:
                ax.set_visible(False)
        axs[r, 0].set_ylabel(f'{cat.capitalize()}s\n(n = {len(ids)})', fontsize=7.5, color=INK, labelpad=6)
    save_fig(fig, out, 'fig_dataset_20_photos')


# ---------- Section 5: the three damage levels on one photograph of each category ----------
def fig_masks(images, cat_of, masks, out):
    ex_ids = [next(i for i in images if cat_of[i] == c) for c in CATEGORIES]
    fig, axs = plt.subplots(len(ex_ids), 4, figsize=(4.6, 3.55), gridspec_kw={'wspace': 0.04, 'hspace': 0.06})
    for r, img_id in enumerate(ex_ids):
        axs[r, 0].imshow(images[img_id])
        for c, lvl in enumerate(LEVELS, start=1):
            m = masks[(img_id, lvl)]
            show_damage(axs[r, c], images[img_id], m)
            axs[r, c].text(0.03, 0.04, f'{m.mean():.1%}', transform=axs[r, c].transAxes, fontsize=6, color='white',
                           bbox=dict(boxstyle='round,pad=0.25', fc='#0b0b0bb0', ec='none'))
        for a in axs[r]:
            image_ax(a)
        axs[r, 0].set_ylabel(img_id, fontsize=7.5, color=INK, labelpad=4)
    for c, t in enumerate(['Original', 'Small (≈5%)', 'Medium (≈20%)', 'Large (≈40%)']):
        axs[0, c].set_title(t, fontsize=7.5, loc='center', weight='bold')
    save_fig(fig, out, 'fig_masks')


# ---------- Section 11: quality and time charts ----------
LEVEL_NAME = {'small': 'Small\n(5% damaged)', 'medium': 'Medium\n(20% damaged)', 'large': 'Large\n(40% damaged)'}
QUALITY = [('PSNR', 'PSNR (dB)', 'higher is better', '{:.1f}'),
           ('SSIM', 'SSIM', 'higher is better', '{:.2f}'),
           ('LPIPS', 'LPIPS', 'lower is better', '{:.3f}')]
BAR_W = 0.26


def _quality_bars(ax, df, met, title, better, fmt, values=True, short=False):
    lv = list(LEVELS)
    means = df.groupby(['level', 'method'], observed=True)[met].mean().unstack('method').reindex(lv)[METHODS_ORDER]
    top = means.values.max()
    for j, meth in enumerate(METHODS_ORDER):
        x = np.arange(len(lv)) + (j - 1) * (BAR_W + 0.02)
        ax.bar(x, means[meth].values, BAR_W, color=COL[meth], label=meth, zorder=2)
        if values:
            for xi, v in zip(x, means[meth].values):
                ax.text(xi, v + top * 0.015, fmt.format(v), ha='center', va='bottom', fontsize=6.2, color=INK2)
    if short:
        ax.set_xticks(np.arange(len(lv)), [LEVEL_LABEL[l] for l in lv])
        ax.set_xlabel('Damaged area')
    else:
        ax.set_xticks(np.arange(len(lv)), [LEVEL_NAME[l] for l in lv])
    ax.set_ylim(0, top * 1.14)
    ax.grid(True, axis='y')
    ax.tick_params(length=0)
    ax.set_title(f'{title}  ({better})', loc='left')


def fig_quality(df, out):
    for met, title, better, fmt in QUALITY:                     # one figure per quality measure
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        _quality_bars(ax, df, met, title, better, fmt)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=3, fontsize=7)
        save_fig(fig, out, f'fig_{met.lower()}')
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.4), gridspec_kw={'wspace': 0.28})   # all three in one figure
    for ax, (met, title, better, fmt) in zip(axs, QUALITY):
        _quality_bars(ax, df, met, title, better, fmt, values=False, short=True)
        ax.set_title(f'{title}\n({better})', loc='left')
    fig.legend(handles=[Patch(fc=COL[m], label=m) for m in METHODS_ORDER], loc='upper center',
               bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=7.5)
    save_fig(fig, out, 'fig_quality_all')


def fig_time(df, out):
    t_mean = df.groupby('method', observed=True).time_s.mean().reindex(METHODS_ORDER)
    fig, ax = plt.subplots(figsize=(3.5, 1.7))
    y = np.arange(len(METHODS_ORDER))
    ax.barh(y, t_mean.values, 0.55, color=[COL[m] for m in METHODS_ORDER], zorder=2)
    for yi, v in zip(y, t_mean.values):
        ax.text(v + t_mean.max() * 0.02, yi, f'{v:.2f} s' if v >= 0.01 else f'{v:.3f} s', va='center',
                fontsize=7, color=INK)
    ax.set_yticks(y, [f'{m} ({HW[m]})' for m in METHODS_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, t_mean.max() * 1.22)
    ax.set_xlabel('Seconds per image (average)')
    ax.grid(True, axis='x')
    ax.tick_params(length=0)
    ax.set_title('Processing time per image  (lower is better)', loc='left')
    save_fig(fig, out, 'fig_time')


# ---------- Section 12: qualitative comparison ----------
def _load_out(out, i, l, m):
    return cv2.cvtColor(cv2.imread(str(Path(out) / 'restored' / f'{i}__{l}__{m}.png')), cv2.COLOR_BGR2RGB)


def _restoration_row(axs_row, out, images, masks, img_id, lvl, first_row):
    m = masks[(img_id, lvl)]
    axs_row[0].imshow(images[img_id])
    show_damage(axs_row[1], images[img_id], m)
    for c_, meth in enumerate(METHODS_ORDER, start=2):
        axs_row[c_].imshow(_load_out(out, img_id, lvl, meth))
    for a_ in axs_row:
        image_ax(a_)
    if first_row:
        for a_, t in zip(axs_row, ['Original', 'Damaged (white)'] + METHODS_ORDER):
            a_.set_title(t, fontsize=7.5, loc='center', weight='bold')


def fig_qualitative(images, cat_of, masks, out):
    examples = [next(i for i in images if cat_of[i] == c) for c in CATEGORIES]   # first photo of each type
    for img_id in examples:
        fig, axs = plt.subplots(len(LEVELS), 5, figsize=(7.16, 4.75), gridspec_kw={'wspace': 0.04, 'hspace': 0.22})
        for r, lvl in enumerate(LEVELS):
            _restoration_row(axs[r], out, images, masks, img_id, lvl, r == 0)
            axs[r, 0].set_ylabel(f'{LEVEL_LABEL[lvl]} damaged', fontsize=7.5, color=INK, labelpad=4)
        fig.suptitle(img_id, x=0.125, ha='left', fontsize=8.5, weight='bold', y=0.945)
        save_fig(fig, out, f'fig_qualitative_{img_id}')
    fig, axs = plt.subplots(len(examples), 5, figsize=(7.16, 4.75), gridspec_kw={'wspace': 0.04, 'hspace': 0.22})
    for r, img_id in enumerate(examples):                       # overview: one row per category, largest damage
        _restoration_row(axs[r], out, images, masks, img_id, 'large', r == 0)
        axs[r, 0].set_ylabel(img_id, fontsize=7.5, color=INK, labelpad=4)
    save_fig(fig, out, 'fig_qualitative_overview')


# ---------- photo-type graph: PDE-based vs diffusion-based at large damage ----------
def fig_by_type_large(df, out):
    types = ['portrait', 'landscape', 'group']
    type_label = {'portrait': 'Portraits', 'landscape': 'Landscapes', 'group': 'Groups'}
    show = ['PDE', 'Diffusion']
    show_label = {'PDE': 'PDE-based', 'Diffusion': 'Diffusion-based'}
    panels = [('PSNR', 'PSNR (dB)', 'higher is better', '{:.1f}'),
              ('SSIM', 'SSIM', 'higher is better', '{:.2f}'),
              ('LPIPS', 'LPIPS', 'lower is better', '{:.3f}')]
    large = df[df.level == 'large']
    means = large.groupby(['category', 'method'], observed=True)[['PSNR', 'SSIM', 'LPIPS']].mean()
    n_per_type = large.groupby('category').image_id.nunique()
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    x, w = np.arange(len(types)), 0.36
    for ax, (met, title, better, fmt) in zip(axs, panels):
        for j, meth in enumerate(show):
            vals = [means.loc[(t, meth), met] for t in types]
            bars = ax.bar(x + (j - 0.5) * (w + 0.02), vals, w, color=COL[meth], label=show_label[meth], zorder=2)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(b.get_height()),
                        ha='center', va='bottom', fontsize=9)
        ax.set_title(title, fontsize=13, fontweight='bold', loc='left', pad=24)
        ax.text(0, 1.03, better, transform=ax.transAxes, fontsize=9.5, color='#6b7280', ha='left', va='bottom')
        ax.set_xticks(x, [f'{type_label[t]}\n({n_per_type[t]})' for t in types])
        ax.set_ylim(0, max(means.loc[(t, m), met] for t in types for m in show) * 1.15)
        for s in ['top', 'right', 'left']:
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', labelsize=9, length=0, colors='#6b7280')
        ax.tick_params(axis='x', length=0)
        ax.yaxis.grid(True, color='#e2dfd6', lw=0.8, zorder=0)
        ax.set_axisbelow(True)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=3)
    save_fig(fig, out, 'fig_by_type_large')
