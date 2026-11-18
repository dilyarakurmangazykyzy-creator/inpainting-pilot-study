"""The three inpainting methods (Section 6 of the notebook).

All three receive exactly the same input: the damaged RGB image (uint8, 256x256) and the binary mask
(1 = damaged). Known pixels are never changed, only the masked area is filled.
"""
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .config import IMG_SIZE, RESSHIFT_CKPT_URL, RESSHIFT_CONFIG, RESSHIFT_VQ_URL, SEED


# ---------- Method 1: PDE-based (Navier-Stokes, OpenCV) ----------
def inpaint_pde(img, mask):
    return cv2.inpaint(img, mask * 255, 3, cv2.INPAINT_NS)


# ---------- Method 2: Exemplar-based (Criminisi et al., 2004), own implementation ----------
def inpaint_exemplar(img, mask, patch=9):
    h = patch // 2
    H, W = mask.shape
    pad = lambda a, v=0: cv2.copyMakeBorder(a, h, h, h, h, cv2.BORDER_CONSTANT, value=v)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = cv2.copyMakeBorder(lab, h, h, h, h, cv2.BORDER_REFLECT)
    out = cv2.copyMakeBorder(img, h, h, h, h, cv2.BORDER_REFLECT).copy()
    hole = pad(mask.astype(np.uint8), 0).astype(bool)
    inside = np.zeros_like(hole)
    inside[h:h + H, h:h + W] = True                      # real image area
    hole &= inside
    conf = (~hole).astype(np.float32)
    lab[hole] = 0
    box = np.ones((patch, patch), np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    while hole.any():
        # 1. fill front
        front = hole & cv2.dilate((~hole).astype(np.uint8), k).astype(bool)
        ys, xs = np.nonzero(front)
        # 2. confidence term
        C = cv2.filter2D(conf, -1, box, borderType=cv2.BORDER_CONSTANT) / (patch * patch)
        # 3. data term: isophote (from known pixels) . front normal
        gray = lab[..., 0] / 255.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        valid = cv2.erode((~hole).astype(np.uint8), k).astype(bool)
        gx[~valid] = 0
        gy[~valid] = 0
        mag = gx ** 2 + gy ** 2
        nx = cv2.Sobel(hole.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        ny = cv2.Sobel(hole.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        nrm = np.sqrt(nx ** 2 + ny ** 2) + 1e-8
        # strongest known isophote inside each front pixel's patch (vectorised)
        off = np.arange(-h, h + 1)
        wy = np.clip(ys[:, None, None] + off[None, :, None], 0, mag.shape[0] - 1)
        wx = np.clip(xs[:, None, None] + off[None, None, :], 0, mag.shape[1] - 1)
        wy, wx = np.broadcast_to(wy, (len(ys), patch, patch)), np.broadcast_to(wx, (len(ys), patch, patch))
        idx = mag[wy, wx].reshape(len(ys), -1).argmax(1)
        py = wy.reshape(len(ys), -1)[np.arange(len(ys)), idx]
        px = wx.reshape(len(ys), -1)[np.arange(len(ys)), idx]
        D = np.abs(-gy[py, px] * nx[ys, xs] + gx[py, px] * ny[ys, xs]) / nrm[ys, xs]
        P = C[ys, xs] * (D + 1e-3)
        t = int(np.argmax(P))
        y, x = ys[t], xs[t]
        # 4. best exemplar among fully-known source patches
        tgt = lab[y - h:y + h + 1, x - h:x + h + 1]
        known = (~hole[y - h:y + h + 1, x - h:x + h + 1]).astype(np.float32)
        tmask = np.repeat(known[..., None], 3, axis=2)
        ssd = cv2.matchTemplate(lab, tgt, cv2.TM_SQDIFF, mask=tmask)
        src_ok = cv2.filter2D(hole.astype(np.float32), -1, box, anchor=(0, 0),
                              borderType=cv2.BORDER_CONSTANT)[:ssd.shape[0], :ssd.shape[1]] < 0.5
        src_ok &= cv2.filter2D((~inside).astype(np.float32), -1, box, anchor=(0, 0),
                               borderType=cv2.BORDER_CONSTANT)[:ssd.shape[0], :ssd.shape[1]] < 0.5
        ssd = np.where(src_ok & np.isfinite(ssd), ssd, np.inf)
        sy, sx = np.unravel_index(np.argmin(ssd), ssd.shape)
        # 5. copy into the unknown pixels of the target patch, update confidence
        fill = ~known.astype(bool)
        sl_t = (slice(y - h, y + h + 1), slice(x - h, x + h + 1))
        sl_s = (slice(sy, sy + patch), slice(sx, sx + patch))
        out[sl_t][fill] = out[sl_s][fill]
        lab[sl_t][fill] = lab[sl_s][fill]
        conf[sl_t][fill] = C[y, x]
        hole[sl_t] = False
        hole &= inside
    return out[h:h + H, h:h + W]


# ---------- Method 3: Diffusion-based (ResShift, Yue et al. 2025) ----------
class ResShiftInpainter:
    """Wrapper around the official ResShift inpainting sampler (4 sampling steps, pretrained on ImageNet).

    ``resshift_dir`` is a clone of https://github.com/zsyOAOA/ResShift. The checkpoints are downloaded
    into ``<resshift_dir>/weights`` on first use. Loading time is measured once and is NOT part of the
    per-image processing time.
    """

    def __init__(self, resshift_dir, seed=SEED, img_size=IMG_SIZE):
        import torch
        from omegaconf import OmegaConf

        self.torch = torch
        self.seed = seed
        resshift_dir = Path(resshift_dir).resolve()
        sys.path.insert(0, str(resshift_dir))
        cwd = os.getcwd()
        os.chdir(resshift_dir)                           # the official code uses paths relative to its folder
        try:
            from basicsr.utils.download_util import load_file_from_url
            _orig_load = torch.load                      # newer PyTorch defaults to weights_only=True;

            def _load(*a, **k):                          # the official checkpoints need the old behaviour
                k.setdefault('weights_only', False)
                return _orig_load(*a, **k)
            torch.load = _load
            from sampler import ResShiftSampler

            t0 = time.perf_counter()
            Path('weights').mkdir(exist_ok=True)
            ck = load_file_from_url(RESSHIFT_CKPT_URL, model_dir='weights', progress=True,
                                    file_name='resshift_inpaint_imagenet_s4.pth')
            vq = load_file_from_url(RESSHIFT_VQ_URL, model_dir='weights', progress=True,
                                    file_name='autoencoder_vq_f4.pth')
            cfg = OmegaConf.load(RESSHIFT_CONFIG)
            cfg.model.ckpt_path = ck
            cfg.autoencoder.ckpt_path = vq
            cfg.diffusion.params.sf = 1
            self.sampler = ResShiftSampler(cfg, sf=1, chop_size=img_size * 4, chop_stride=img_size * 4, chop_bs=1,
                                           use_amp=True, seed=seed,
                                           padding_offset=cfg.model.params.get('lq_size', 64))
            self.load_time = time.perf_counter() - t0
        finally:
            os.chdir(cwd)

    def __call__(self, img, mask, seed=None):
        torch = self.torch
        with torch.no_grad():
            self.sampler.setup_seed(self.seed if seed is None else seed)
            H, W = mask.shape
            x = torch.from_numpy(img).float().div(255).permute(2, 0, 1)[None].cuda()
            m = torch.from_numpy(mask.astype(np.float32))[None, None].cuda()
            x = x * (1 - m)                              # damaged area = 0, as in ResShift training
            # ResShift works on sizes that are multiples of 256 (our images are exactly 256x256); if another
            # size is used, pad image AND mask the same way (the official sampler pads only the image).
            off = self.sampler.padding_offset
            ph, pw = (-H) % off, (-W) % off
            if ph or pw:
                mode = 'reflect' if (ph < H and pw < W) else 'replicate'
                xp = torch.nn.functional.pad(x, (0, pw, 0, ph), mode=mode)
                mp = torch.nn.functional.pad(m, (0, pw, 0, ph), mode=mode)
            else:
                xp, mp = x, m
            with torch.autocast('cuda'):
                y = self.sampler.sample_func(xp * 2 - 1, noise_repeat=False, mask=mp * 2 - 1)  # [-1, 1]; 1 = damaged
            y = y.float()[:, :, :H, :W] * 0.5 + 0.5
            y = y * m + x * (1 - m)                      # keep known pixels (official mask_back)
            return (y[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)


def get_methods(resshift_dir=None):
    """Return {'PDE': fn, 'Exemplar': fn, 'Diffusion': fn}. Without resshift_dir only the classical methods."""
    methods = {'PDE': inpaint_pde, 'Exemplar': inpaint_exemplar}
    load_time = None
    if resshift_dir is not None:
        diff = ResShiftInpainter(resshift_dir)
        methods['Diffusion'] = diff
        load_time = diff.load_time
    return methods, load_time
