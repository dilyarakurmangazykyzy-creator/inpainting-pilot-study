"""Check that this code reproduces the published numbers (CPU only, about 10 seconds).

Downloads the 20 BSDS500 photos, rebuilds all 60 masks and restores group_1 with the PDE-based and
exemplar-based methods, then compares the results with the values saved by the original notebook run.

    python scripts/verify_reproducibility.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from inpainting_study.data import load_bsds500             # noqa: E402
from inpainting_study.masks import make_all_masks          # noqa: E402
from inpainting_study.methods import inpaint_exemplar, inpaint_pde   # noqa: E402
from inpainting_study.metrics import score                 # noqa: E402

# values from results/tables/per_image_example_group_1.csv and results/mask_coverage_summary.txt
EXPECTED_COVERAGE = {'small': 0.0460, 'medium': 0.1968, 'large': 0.3969}
EXPECTED = {('small', 'PDE'): (33.163817, 0.974469), ('medium', 'PDE'): (24.680557, 0.900542),
            ('large', 'PDE'): (20.099303, 0.764017), ('small', 'Exemplar'): (31.502272, 0.967906)}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        images, _, _ = load_bsds500(tmp)
        masks, mask_df = make_all_masks(images)
    ok = True
    cov = mask_df.groupby('level')['actual_coverage'].mean().round(4)
    for lvl, exp in EXPECTED_COVERAGE.items():
        good = abs(cov[lvl] - exp) < 1e-4
        ok &= good
        print(f"mask coverage {lvl:<6} {cov[lvl]:.4f} (expected {exp:.4f})  {'OK' if good else 'DIFFERENT'}")
    g = images['group_1']
    for (lvl, meth), (psnr, ssim) in EXPECTED.items():
        m = masks[('group_1', lvl)]
        res = (inpaint_pde if meth == 'PDE' else inpaint_exemplar)(g, m)
        s = score(g, res, m)
        good = np.isclose(s['PSNR'], psnr, atol=1e-3) and np.isclose(s['SSIM'], ssim, atol=1e-4)
        ok &= good
        print(f"group_1 {lvl:<6} {meth:<8} PSNR {s['PSNR']:.4f} SSIM {s['SSIM']:.4f} "
              f"(expected {psnr:.4f} / {ssim:.4f})  {'OK' if good else 'DIFFERENT'}")
    print('\nAll checks passed.' if ok else '\nSome values differ (check the OpenCV / NumPy versions).')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
