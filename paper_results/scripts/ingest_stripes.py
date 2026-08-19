"""Ingest real bacterial-colony stripe PNGs into a registry dataset.

Builds data/datasets/stripes_colony_2ch/payload.h5 from the 8 frame pairs in
~/projects/personal/rngrn/3N-stripes-images/preview_png (frame_NN_C1/C2, 300x300
grayscale). These are interior crops of ONE colony engineered with a 3-node circuit;
only 2 of the 3 nodes are imaged (C1, C2) — partial observation, m=2 of N=3.

Preprocessing per channel:
  1. float in [0,1]
  2. subtract a least-squares 2nd-order polynomial illumination background
  3. downsample 300 -> 96 px (LANCZOS; low-passes the cellular speckle)
  4. rescale to [0,1]  (concentrations are positive; scale is arbitrary anyway —
     only frame.mean() and the FFT peak reach the training objective)
No mask: the crops contain no colony edge.

ANSWER-KEY DISCLOSURE (D-REAL-1): this is real data — there is NO generating model,
so the `k_star` attribute (hard-required by data/gate.py::from_registry, defined there
as the LINEAR answer-key wavenumber of the generator) does not exist for these samples.
We write the FFT-measured peak of channel 0 into BOTH `k_star` and `k_star_fft` so the
loader accepts the samples. Consequence: every kstar_rel_err-style score on this
dataset is CIRCULAR (model k* vs the same FFT anchor training used) and must never be
reported as accuracy against truth. L=1.0 is an arbitrary unit (no scale bar).

This script opens payload.h5 and is therefore listed in tests/test_firewall.py::FORBIDDEN.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
from PIL import Image

SRC_DEFAULT = os.path.expanduser(
    "~/projects/personal/rngrn/3N-stripes-images/preview_png")
DATASET_ID = "stripes_colony_2ch"
N_FRAMES = 8
OUT_SIZE = 96
L = 1.0  # arbitrary units — no physical scale bar; only sets units of D and k


def _detrend_poly2(img: np.ndarray) -> np.ndarray:
    """Subtract the least-squares 2nd-order 2D polynomial (illumination background)."""
    ny, nx = img.shape
    y, x = np.mgrid[0:ny, 0:nx]
    x = (x - nx / 2) / nx
    y = (y - ny / 2) / ny
    basis = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)
    A = basis.reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(A, img.ravel(), rcond=None)
    return img - (A @ coef).reshape(ny, nx)


def _load_channel(path: str) -> np.ndarray:
    img = np.asarray(Image.open(path), dtype=np.float64) / 255.0
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError(f"{path}: expected square grayscale, got shape {img.shape}")
    img = _detrend_poly2(img)
    small = Image.fromarray(img.astype(np.float32), mode="F").resize(
        (OUT_SIZE, OUT_SIZE), Image.LANCZOS)
    a = np.asarray(small, dtype=np.float64)
    lo, hi = a.min(), a.max()
    if hi - lo <= 0:
        raise ValueError(f"{path}: degenerate (constant) frame after preprocessing")
    return (a - lo) / (hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--datasets-root", default="data/datasets")
    args = ap.parse_args()

    from rngrn import observables as obs  # FFT peak of the processed frame

    out_dir = os.path.join(args.datasets_root, DATASET_ID)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "payload.h5")

    with h5py.File(out_path, "w") as f:
        for i in range(1, N_FRAMES + 1):
            chans = []
            for ch in ("C1", "C2"):
                p = os.path.join(args.src, f"frame_{i:02d}_{ch}_preview.png")
                if not os.path.exists(p):
                    raise FileNotFoundError(p)
                chans.append(_load_channel(p))
            frame = np.stack(chans, axis=0)  # (2, 96, 96), C1 = channel 0
            kfft = float(obs.kstar_of(frame[0], L=L))
            if not np.isfinite(kfft) or kfft <= 0:
                raise ValueError(f"frame_{i:02d}: non-finite/zero FFT peak {kfft}")
            g = f.create_group(f"sample_{i:04d}")
            g.create_dataset("final_frame", data=frame)
            g.attrs["L"] = L
            # DISCLOSED PLACEHOLDER: k_star is answer-key-only for generated data; for
            # real data the FFT peak is the only wavenumber we have. See module docstring.
            g.attrs["k_star"] = kfft
            g.attrs["k_star_fft"] = kfft
            g.attrs["split"] = "real"
            g.attrs["system_id"] = "colony_3node_engineered"
            print(f"sample_{i:04d}: k_star_fft={kfft:.4f} "
                  f"(wavelength {2*np.pi/kfft:.4f} L, "
                  f"{L*kfft/(2*np.pi):.2f} periods/frame)")

    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
