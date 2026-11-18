"""PSNR (Eq. 1), SSIM (Eq. 2) and LPIPS (Eq. 3), Section 8 of the notebook.

Computed between each restored image and its clean original on the whole image (as in the paper).
As an extra, PSNR is also computed inside the damaged region only (PSNR_hole).
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


class LpipsScorer:
    """Standard LPIPS (AlexNet, v0.1)."""

    def __init__(self, device='cuda'):
        import lpips
        import torch
        self.torch, self.device = torch, device
        self.fn = lpips.LPIPS(net='alex', verbose=False).to(device).eval()

    def _t(self, a):
        return self.torch.from_numpy(a).float().div(127.5).sub(1).permute(2, 0, 1)[None].to(self.device)

    def __call__(self, gt, out):
        with self.torch.no_grad():
            return self.fn(self._t(gt), self._t(out)).item()


def psnr_hole(gt, out, mask):
    m = mask.astype(bool)
    mse = np.mean((gt[m].astype(np.float64) - out[m].astype(np.float64)) ** 2)
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def score(gt, out, mask, lpips_fn=None):
    return {'PSNR': peak_signal_noise_ratio(gt, out, data_range=255),
            'SSIM': structural_similarity(gt, out, channel_axis=2, data_range=255),
            'LPIPS': lpips_fn(gt, out) if lpips_fn is not None else np.nan,
            'PSNR_hole': psnr_hole(gt, out, mask)}
