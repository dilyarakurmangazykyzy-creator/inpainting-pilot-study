"""Summary tables and statistical tests (Sections 9, 10 and 13 of the notebook).

Every method is applied to the same image-mask pairs, so the data are paired. With a pilot sample
(n = 20 per cell) and no normality assumption the analysis uses
* a Friedman test across the three methods at each damage level, and
* Wilcoxon signed-rank tests for the pairwise comparisons, Holm-corrected within each measure and level.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import BETTER, LEVELS, METRICS, METHODS_ORDER

PAIRS = [('Diffusion', 'PDE'), ('Diffusion', 'Exemplar'),   # diffusion vs each classical baseline
         ('PDE', 'Exemplar')]                                # the two baselines against each other


def _fmt(mean, sd, metric):
    d = {'PSNR': 2, 'PSNR_hole': 2, 'SSIM': 4, 'LPIPS': 4, 'time_s': 3}[metric]
    return f'{mean:.{d}f} ± {sd:.{d}f}'


def main_table(df, out=None):
    """Mean +- SD for every method x damage level (Table I). Also writes a LaTeX version."""
    summary = df.groupby(['level', 'method'], observed=True)[METRICS + ['PSNR_hole']].agg(['mean', 'std', 'median'])
    table = pd.DataFrame({met: df.groupby(['level', 'method'], observed=True)[met]
                          .agg(lambda s, met=met: _fmt(s.mean(), s.std(), met)) for met in METRICS})
    table.columns = ['PSNR (dB) ↑', 'SSIM ↑', 'LPIPS ↓', 'Time (s) ↓']
    if out is not None:
        out = Path(out)
        summary.to_csv(out / 'tables' / 'summary_method_x_level.csv')
        table.to_csv(out / 'tables' / 'table_main.csv')
        (out / 'tables' / 'table_main.tex').write_text(_latex_table(df))
    return table


def _latex_table(df):
    lat = df.groupby(['level', 'method'], observed=True)[METRICS].agg(['mean', 'std'])
    lines = [r'\begin{table}[t]', r'\centering',
             r'\caption{Restoration quality and processing time (mean $\pm$ SD, $n=%d$ images per cell).}'
             % df.image_id.nunique(),
             r'\label{tab:main}', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{llcccc}', r'\hline',
             r'Damage & Method & PSNR (dB)$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & Time (s)$\downarrow$ \\',
             r'\hline']
    methods = [m for m in METHODS_ORDER if m in set(df.method)]
    for lvl in LEVELS:
        sub = lat.loc[lvl]
        best = {m: (None if sub[(m, 'mean')].isna().all()          # e.g. LPIPS when run without a GPU
                    else sub[(m, 'mean')].idxmax() if BETTER[m] == 'higher' else sub[(m, 'mean')].idxmin())
                for m in METRICS}
        for i, meth in enumerate(methods):
            cells = []
            for met in METRICS:
                s = _fmt(sub.loc[meth, (met, 'mean')], sub.loc[meth, (met, 'std')], met).replace('±', r'$\pm$')
                cells.append(r'\textbf{' + s + '}' if best[met] == meth else s)
            label = f'{lvl.capitalize()} ({int(LEVELS[lvl] * 100)}\\%)' if i == 0 else ''
            lines.append(f'{label} & {meth} & ' + ' & '.join(cells) + r' \\')
        lines.append(r'\hline')
    lines += [r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def by_category(df, out=None):
    """Blocking factor: mean scores per photo type (Table III)."""
    t = df.groupby(['category', 'level', 'method'], observed=True)[['PSNR', 'SSIM', 'LPIPS']].mean().round(4)
    if out is not None:
        t.to_csv(Path(out) / 'tables' / 'summary_by_category.csv')
    return t.unstack('method')


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for rank, idx in enumerate(order):
        run = max(run, (len(p) - rank) * p[idx])
        adj[idx] = min(1, run)
    return adj


def statistics(df, out=None):
    """Friedman test per measure and level, pairwise Wilcoxon signed-rank tests with Holm correction."""
    methods = [m for m in METHODS_ORDER if m in set(df.method)]
    rows = []
    for met in METRICS:
        for lvl in LEVELS:
            wide = df[df.level == lvl].pivot(index='image_id', columns='method', values=met)[methods]
            if len(wide) >= 3 and len(methods) >= 3:
                fr = stats.friedmanchisquare(*[wide[m] for m in methods])
                fr_stat, fr_p = fr.statistic, fr.pvalue
            else:
                fr_stat = fr_p = np.nan
            pairs = [(a, b) for a, b in PAIRS if a in methods and b in methods]
            ps, rs, meds = [], [], []
            for a, b in pairs:
                d = wide[a] - wide[b]
                try:
                    ps.append(stats.wilcoxon(wide[a], wide[b]).pvalue)
                except ValueError:
                    ps.append(np.nan)
                nz = d[d != 0]
                rk = stats.rankdata(np.abs(nz))
                rs.append((rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum() if len(nz) else 0.0)   # rank-biserial r
                meds.append(d.median())
            adj = holm(np.nan_to_num(ps, nan=1.0))
            for (a, b), p, pa, r, md_ in zip(pairs, ps, adj, rs, meds):
                rows.append({'metric': met, 'level': lvl, 'friedman_chi2': fr_stat, 'friedman_p': fr_p,
                             'comparison': f'{a} vs {b}', 'median_diff(a-b)': md_,
                             'wilcoxon_p': p, 'p_holm': pa, 'effect_r': r})
    stats_df = pd.DataFrame(rows).round(5)
    if out is not None:
        stats_df.to_csv(Path(out) / 'tables' / 'statistics.csv', index=False)
    return stats_df


def diffusion_vs_baselines(df, stats_df, out=None):
    """Diffusion minus each baseline: mean difference, photos where diffusion is better, Holm p (Table II)."""
    rows = []
    for lvl in LEVELS:
        sub = df[df.level == lvl]
        for base in ['PDE', 'Exemplar']:
            for met in METRICS:
                w = sub.pivot(index='image_id', columns='method', values=met)
                d = w['Diffusion'] - w[base]
                better = (d > 0) if BETTER[met] == 'higher' else (d < 0)
                st_ = stats_df[(stats_df.metric == met) & (stats_df.level == lvl)
                               & (stats_df.comparison == f'Diffusion vs {base}')].iloc[0]
                rows.append({'level': lvl, 'baseline': base, 'metric': met,
                             'diffusion_mean': w['Diffusion'].mean(), 'baseline_mean': w[base].mean(),
                             'mean_diff (diff - base)': d.mean(), 'diffusion_better_in': f'{better.sum()}/{len(d)}',
                             'p_holm': st_.p_holm, 'effect_r': st_.effect_r})
    baseline_df = pd.DataFrame(rows)
    if out is not None:
        baseline_df.to_csv(Path(out) / 'tables' / 'diffusion_vs_baselines.csv', index=False)
    return baseline_df


def key_results(df, stats_df, baseline_df, mask_df, env=None, model_load_time=None, out=None):
    """Key numbers in sentences, for writing the Results section (Section 13 of the notebook)."""
    methods = [m for m in METHODS_ORDER if m in set(df.method)]
    lines = [f'# Key results (n = {df.image_id.nunique()} photographs from BSDS500, {len(df)} restored outputs)\n']
    if env:
        lines.append(f"Environment: Python {env['python']}, PyTorch {env['torch']}, OpenCV {env['opencv']}, "
                     f"GPU {env['gpu']} ({env['gpu_memory_GB']} GB).")
    if model_load_time is not None:
        lines.append(f'ResShift one-off model loading time: {model_load_time:.1f} s (excluded from per-image times).\n')
    cov = mask_df.groupby('level')['actual_coverage'].mean()
    lines.append('Actual mean mask coverage: ' + ', '.join(f'{l} {cov[l] * 100:.1f}%' for l in LEVELS) + '.\n')
    mean = df.groupby(['level', 'method'], observed=True)[METRICS + ['PSNR_hole']].mean()
    for lvl in LEVELS:
        s = mean.loc[lvl]
        lines.append(f'## {lvl.capitalize()} damage ({int(LEVELS[lvl] * 100)}%)')
        for met in ['PSNR', 'SSIM', 'LPIPS', 'PSNR_hole']:
            best = s[met].idxmax() if BETTER[met] == 'higher' else s[met].idxmin()
            lines.append(f'- {met}: best = {best} ({s.loc[best, met]:.4g}); '
                         + ', '.join(f'{m} {s.loc[m, met]:.4g}' for m in methods))
        lines.append('- Time (s): ' + ', '.join(f"{m} {s.loc[m, 'time_s']:.3f}" for m in methods))
        sig = stats_df[(stats_df.level == lvl) & (stats_df.metric.isin(['PSNR', 'SSIM', 'LPIPS']))
                       & (stats_df.p_holm < 0.05)]
        for _, r in sig.iterrows():
            lines.append(f'  - {r.metric}: {r.comparison} significant (Holm p = {r.p_holm:.4f}, r = {r.effect_r:.2f})')
        lines.append('')
    lines.append('## Diffusion vs the classical baselines (mean difference, photos where diffusion is better)')
    for lvl in LEVELS:
        for base in ['PDE', 'Exemplar']:
            b = baseline_df[(baseline_df.level == lvl) & (baseline_df.baseline == base)].set_index('metric')
            lines.append(f'- {lvl} vs {base}: ' + '; '.join(
                f"{m} {b.loc[m, 'mean_diff (diff - base)']:+.4g} ({b.loc[m, 'diffusion_better_in']}, "
                f"Holm p = {b.loc[m, 'p_holm']:.3g})" for m in METRICS))
    lines.append('')
    t = df.groupby('method', observed=True)['time_s'].mean()
    line = 'Overall mean time per image: ' + ', '.join(f'{m} {t[m]:.3f}s' for m in methods) + '.'
    if {'PDE', 'Exemplar', 'Diffusion'} <= set(methods):
        line += (f" Diffusion is {t['Diffusion'] / t['PDE']:.1f}x PDE and {t['Diffusion'] / t['Exemplar']:.2f}x Exemplar "
                 '(note: classical methods run on CPU, diffusion on GPU).')
    lines.append(line + '\n')
    txt = '\n'.join(lines)
    if out is not None:
        (Path(out) / 'results_summary.md').write_text(txt)
    return txt
