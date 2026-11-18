# Key results (n = 20 photographs from BSDS500, 180 restored outputs)

Environment: Python 3.13.16, PyTorch 2.11.0+cu130, OpenCV 5.0.0, GPU Tesla T4 (15.6 GB).
ResShift one-off model loading time: 11.9 s (excluded from per-image times).

Actual mean mask coverage: small 4.6%, medium 19.7%, large 39.7%.

## Small damage (5%)
- PSNR: best = Diffusion (35.43); PDE 35.33, Exemplar 32.34, Diffusion 35.43
- SSIM: best = PDE (0.9813); PDE 0.9813, Exemplar 0.9709, Diffusion 0.9804
- LPIPS: best = Diffusion (0.009504); PDE 0.01568, Exemplar 0.0206, Diffusion 0.009504
- PSNR_hole: best = Diffusion (22.04); PDE 21.94, Exemplar 18.94, Diffusion 22.04
- Time (s): PDE 0.005, Exemplar 3.260, Diffusion 0.388
  - PSNR: Diffusion vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - PSNR: PDE vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - SSIM: Diffusion vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - SSIM: PDE vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - LPIPS: Diffusion vs PDE significant (Holm p = 0.0000, r = -1.00)
  - LPIPS: Diffusion vs Exemplar significant (Holm p = 0.0000, r = -1.00)
  - LPIPS: PDE vs Exemplar significant (Holm p = 0.0362, r = -0.53)

## Medium damage (20%)
- PSNR: best = Diffusion (26.86); PDE 26.6, Exemplar 23.88, Diffusion 26.86
- SSIM: best = Diffusion (0.9084); PDE 0.9072, Exemplar 0.8643, Diffusion 0.9084
- LPIPS: best = Diffusion (0.04302); PDE 0.08909, Exemplar 0.09728, Diffusion 0.04302
- PSNR_hole: best = Diffusion (19.8); PDE 19.54, Exemplar 16.81, Diffusion 19.8
- Time (s): PDE 0.020, Exemplar 13.318, Diffusion 0.384
  - PSNR: Diffusion vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - PSNR: PDE vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - SSIM: Diffusion vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - SSIM: PDE vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - LPIPS: Diffusion vs PDE significant (Holm p = 0.0000, r = -1.00)
  - LPIPS: Diffusion vs Exemplar significant (Holm p = 0.0000, r = -1.00)

## Large damage (40%)
- PSNR: best = Diffusion (22.54); PDE 21.94, Exemplar 19.7, Diffusion 22.54
- SSIM: best = Diffusion (0.79); PDE 0.7867, Exemplar 0.707, Diffusion 0.79
- LPIPS: best = Diffusion (0.1025); PDE 0.2156, Exemplar 0.228, Diffusion 0.1025
- PSNR_hole: best = Diffusion (18.53); PDE 17.92, Exemplar 15.68, Diffusion 18.53
- Time (s): PDE 0.035, Exemplar 25.606, Diffusion 0.383
  - PSNR: Diffusion vs PDE significant (Holm p = 0.0296, r = 0.55)
  - PSNR: Diffusion vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - PSNR: PDE vs Exemplar significant (Holm p = 0.0000, r = 0.95)
  - SSIM: Diffusion vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - SSIM: PDE vs Exemplar significant (Holm p = 0.0000, r = 1.00)
  - LPIPS: Diffusion vs PDE significant (Holm p = 0.0000, r = -1.00)
  - LPIPS: Diffusion vs Exemplar significant (Holm p = 0.0000, r = -1.00)

## Diffusion vs the classical baselines (mean difference, photos where diffusion is better)
- small vs PDE: PSNR +0.09894 (11/20, Holm p = 0.729); SSIM -0.0008737 (6/20, Holm p = 0.0973); LPIPS -0.006177 (20/20, Holm p = 1e-05); time_s +0.3832 (0/20, Holm p = 1e-05)
- small vs Exemplar: PSNR +3.093 (20/20, Holm p = 1e-05); SSIM +0.009518 (20/20, Holm p = 1e-05); LPIPS -0.0111 (20/20, Holm p = 1e-05); time_s -2.872 (20/20, Holm p = 1e-05)
- medium vs PDE: PSNR +0.2618 (10/20, Holm p = 0.648); SSIM +0.001189 (11/20, Holm p = 0.674); LPIPS -0.04607 (20/20, Holm p = 1e-05); time_s +0.3643 (0/20, Holm p = 1e-05)
- medium vs Exemplar: PSNR +2.986 (20/20, Holm p = 1e-05); SSIM +0.04403 (20/20, Holm p = 1e-05); LPIPS -0.05426 (20/20, Holm p = 1e-05); time_s -12.93 (20/20, Holm p = 1e-05)
- large vs PDE: PSNR +0.6046 (13/20, Holm p = 0.0296); SSIM +0.003245 (10/20, Holm p = 0.409); LPIPS -0.1131 (20/20, Holm p = 1e-05); time_s +0.3475 (0/20, Holm p = 1e-05)
- large vs Exemplar: PSNR +2.844 (20/20, Holm p = 1e-05); SSIM +0.08303 (20/20, Holm p = 1e-05); LPIPS -0.1256 (20/20, Holm p = 1e-05); time_s -25.22 (20/20, Holm p = 1e-05)

Overall mean time per image: PDE 0.020s, Exemplar 14.061s, Diffusion 0.385s. Diffusion is 19.2× PDE and 0.03× Exemplar (note: classical methods run on CPU, diffusion on GPU).

