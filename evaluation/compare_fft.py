"""
Reproducible comparison of spatial vs frequency-domain methods on SYNTHETIC
degradations of two clean photos bundled with scikit-image, so ground truth
is exact. Run from the repo root:

    python evaluation/compare_fft.py

Caveat: synthetic haze here uses a smooth depth map (a top-to-bottom ramp or
a constant), which is the easiest case for the non-edge-aware FFT transmission
low-pass. Real scenes such as SOTS have sharp depth discontinuities where the
guided filter is expected to hold up better. Confirm on SOTS before claiming
either method is better in general.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from skimage import data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dehazing.dark_channel_prior import dehaze
from dehazing.fft_dehazing import dehaze_fft
from deraining.fft_notch import derain_fft
from deraining.rain_streak_removal import derain


def load(name):
    return cv2.cvtColor(getattr(data, name)(), cv2.COLOR_RGB2BGR)


def add_rain(img, tilt_deg, n=700, seed=1):
    h, w = img.shape[:2]
    rng = np.random.default_rng(seed)
    out = img.astype(np.float32).copy()
    t = np.deg2rad(tilt_deg)
    for _ in range(n):
        x0, y0 = rng.uniform(0, w), rng.uniform(-30, h)
        length, bright = rng.integers(20, 45), rng.integers(50, 110)
        for i in range(length):
            x, y = int(round(x0 + i * np.sin(t))), int(round(y0 + i * np.cos(t)))
            if 0 <= y < h and 0 <= x < w:
                out[y, x] = np.clip(out[y, x] + bright, 0, 255)
                if x + 1 < w:
                    out[y, x + 1] = np.clip(out[y, x + 1] + bright * 0.5, 0, 255)
    return out.astype(np.uint8)


def add_haze(img, beta=2.0, A=0.92, mode="ramp"):
    h, w = img.shape[:2]
    d = (np.linspace(1.0, 0.1, h)[:, None] * np.ones((1, w))
         if mode == "ramp" else np.full((h, w), 0.6))
    t = np.exp(-beta * d)[..., None]
    J = img.astype(np.float32) / 255
    return (np.clip(J * t + A * (1 - t), 0, 1) * 255).astype(np.uint8)


def row(clean, name, res):
    p = psnr(clean, res, data_range=255)
    s = ssim(clean, res, data_range=255, channel_axis=2)
    print(f"  {name:34s} PSNR {p:6.2f} dB   SSIM {s:.4f}")


if __name__ == "__main__":
    print("=== RAIN STREAK REMOVAL (chelsea, 700 streaks) ===")
    clean = load("chelsea")
    for tilt in (0, 10, -10):
        rainy = add_rain(clean, tilt)
        print(f"\ntrue streak tilt {tilt:+d} deg")
        row(clean, "rainy (no processing)", rainy)
        row(clean, "spatial top-hat (Week 3)", derain(rainy))
        res, est = derain_fft(rainy, auto_angle=True)
        row(clean, f"FFT notch (auto angle {est:+.0f})", res)

    print("\n=== DEHAZING: DCP baseline vs FFT homomorphic ===")
    totals = {"hazy": [], "dcp": [], "fft": []}
    for name in ("chelsea", "coffee"):
        clean = load(name)
        for beta in (1.0, 2.0):
            for mode in ("ramp", "uniform"):
                hazy = add_haze(clean, beta=beta, mode=mode)
                outs = {"hazy": hazy, "dcp": dehaze(hazy), "fft": dehaze_fft(hazy)}
                print(f"\n{name}, beta={beta}, {mode} haze")
                row(clean, "hazy (no processing)", outs["hazy"])
                row(clean, "DCP baseline", outs["dcp"])
                row(clean, "FFT homomorphic", outs["fft"])
                for k, v in outs.items():
                    totals[k].append(psnr(clean, v, data_range=255))
    print("\nMean PSNR over all 8 dehazing cases:",
          {k: round(float(np.mean(v)), 2) for k, v in totals.items()})
