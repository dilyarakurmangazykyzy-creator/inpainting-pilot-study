"""Experiment settings (Section 2 of the notebook)."""

IMG_SIZE = 256                                      # common resolution (ResShift was trained at 256x256)
LEVELS = {'small': 0.05, 'medium': 0.20, 'large': 0.40}   # target share of damaged pixels
MASK_TOL = 0.01                                     # accepted coverage error (+-1 percentage point)
SEED = 2026
CATEGORIES = ['portrait', 'landscape', 'group']     # blocking factor (image content)

METHODS_ORDER = ['PDE', 'Exemplar', 'Diffusion']
LEVEL_LABEL = {'small': '5%', 'medium': '20%', 'large': '40%'}
BETTER = {'PSNR': 'higher', 'SSIM': 'higher', 'LPIPS': 'lower', 'time_s': 'lower', 'PSNR_hole': 'higher'}
METRICS = ['PSNR', 'SSIM', 'LPIPS', 'time_s']

# The 20 photographs: test split of BSDS500 (Arbelaez et al., IEEE TPAMI 2011), pinned to one commit
BSDS_COMMIT = 'a04b7c6c3a9f0ace74bf205c72a43d32e1c72722'
BSDS_URL = 'https://raw.githubusercontent.com/BIDS/BSDS500/' + BSDS_COMMIT + '/BSDS500/data/images/test/{}.jpg'
BSDS_PHOTOS = {   # BSDS id: (category, MD5 checksum)
    '14092': ('portrait', '9da8165fb621f46c1adf13332498a82a'),  '15062': ('portrait', '464e66695dcc3abef682075d2dd4640d'),
    '198087': ('portrait', '0a87ef1cff46ce897cf70ae9ea9e8da4'), '246009': ('portrait', '1cefa4bb78c58fc10f28ab24e9b53a92'),
    '302022': ('portrait', 'f803373b0f1ec401c83b626ff9b76580'), '388006': ('portrait', 'f685f9f211ec720c600ccb3438fcfbb8'),
    '258089': ('portrait', '02305cce4cdd3b80d012138b5ee83965'),
    '28083': ('landscape', 'e7c19bcc58de49a116bd2f3bb73be333'),  '36046': ('landscape', '1780bde47418d1dcd7c5771a897647e6'),
    '117025': ('landscape', '0fdc429100329affa18c7d2e49f6b382'), '228076': ('landscape', '4d67c261a7c3de938ac831ce3c227405'),
    '145079': ('landscape', 'cda37c9175c9f11d2fc2e493e8a1ce26'), '207049': ('landscape', '2361dd52fd00be7a5cb95b313d28cfe4'),
    '79073': ('landscape', '3b915f5e99de3bfaccd5635c9f55653e'),
    '157087': ('group', 'a456b5f14a396c50f9bf03b41770541b'), '290035': ('group', 'f6e742d799a08632849139dcae44f13d'),
    '368037': ('group', 'da6bbc4403f0480770ff3d631ec44f89'), '15011': ('group', '4f83ac60ccfbcd38f21ca969b4bba756'),
    '226043': ('group', 'bd88a6620adde00cadb84d2d30098204'), '20069': ('group', '8b2655a2e71c21fa73d0ba53c2ac292c'),
}

# Official ResShift inpainting checkpoints (Yue et al., IEEE TPAMI 2025)
RESSHIFT_CKPT_URL = 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_inpainting_imagenet_s4.pth'
RESSHIFT_VQ_URL = 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth'
RESSHIFT_CONFIG = 'configs/inpaint_lama256_imagenet.yaml'

OUT_SUBDIRS = ['clean', 'masks', 'damaged', 'restored', 'figures', 'tables']
