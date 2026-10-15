"""
Frequency-domain Rain Streak Removal (directional FFT notch filter)

Why this works
--------------
A rain streak is a thin line, nearly constant along its length. A pattern
that is constant along one direction has all of its Fourier energy on the
line through the origin PERPENDICULAR to that direction. So a population of
near-vertical streaks puts a bright band of energy along (almost) the
horizontal frequency axis, while ordinary scene content is spread over the
whole spectrum. Attenuating that band removes the streaks.

Steps
-----
1. Reflect-pad each channel (avoids false axis energy from the FFT's
   periodic boundary assumption).
2. FFT -> multiply by a smooth directional notch mask -> inverse FFT.
3. Keep min(original, filtered): rain only adds brightness (bright_only).
4. Crop the padding.

The mask is a Gaussian band of half-width `band_width` (cycles/pixel,
roughly 1 / streak length) around the line perpendicular to the streak
direction. A low-frequency gate protects DC and the broad background so
overall brightness and large-scale structure are untouched.

Limitation: genuine near-vertical scene edges (poles, building sides) lie
in the same band and are softened too. This is the trade-off of a purely
global frequency-domain method; the spatial method in
rain_streak_removal.py is local and does not have it.
"""

import cv2
import numpy as np

from fft_utils import apply_mask, frequency_grid, reflect_pad


def build_notch_mask(shape, angle_deg=0.0, band_width=0.10,
                     protect_radius=0.02, strength=1.0):
    """Directional notch mask in unshifted FFT layout.

    angle_deg     : streak tilt from vertical in degrees (0 = straight down;
                    positive leans toward +x as it goes down the image).
    band_width    : Gaussian std of the band in cycles/pixel (~1/streak length).
    protect_radius: radius (cycles/pixel) of the low-frequency region left alone.
    strength      : 0 = no filtering, 1 = full removal at the band centre.
    """
    fx, fy = frequency_grid(shape)
    theta = np.deg2rad(angle_deg)
    # perpendicular distance of each frequency from the line along which
    # streak energy lies
    d_perp = fx * np.sin(theta) + fy * np.cos(theta)
    band = np.exp(-0.5 * (d_perp / band_width) ** 2)
    r2 = fx ** 2 + fy ** 2
    gate = 1.0 - np.exp(-r2 / (2.0 * protect_radius ** 2))
    return 1.0 - strength * band * gate


def estimate_streak_angle(gray, search_deg=30, band_width=0.01, min_radius=0.05):
    """Estimate dominant streak tilt from the spectrum: scan candidate angles
    and pick the one whose thin band carries the most high-frequency energy.
    gray: 2D float array. Returns degrees from vertical."""
    pad = 32
    g = reflect_pad(gray.astype(np.float64), pad)
    mag = np.abs(np.fft.fft2(g - g.mean()))
    fx, fy = frequency_grid(g.shape)
    r = np.sqrt(fx ** 2 + fy ** 2)
    valid = r > min_radius
    best, best_score = 0.0, -1.0
    for a in np.arange(-search_deg, search_deg + 1, 1.0):
        t = np.deg2rad(a)
        d_perp = np.abs(fx * np.sin(t) + fy * np.cos(t))
        sel = valid & (d_perp < band_width)
        score = mag[sel].mean() if sel.any() else 0.0
        if score > best_score:
            best, best_score = a, score
    return float(best)


def derain_fft(image_bgr, angle_deg=0.0, band_width=0.10,
               protect_radius=0.02, strength=1.0, auto_angle=False,
               bright_only=True):
    """Frequency-domain rain streak removal.

    bright_only: rain only ADDS brightness, so the notch result is combined
        as min(original, filtered): a pixel may get darker (streak removed)
        but never brighter. Wherever the filter would brighten a pixel it is
        smearing a scene edge, not removing rain, so that change is rejected.

    image_bgr: HxWx3 uint8 BGR. Returns (derained uint8 BGR, angle used)."""
    img = image_bgr.astype(np.float32) / 255.0
    if auto_angle:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle_deg = estimate_streak_angle(gray)

    pad = 32
    h, w = img.shape[:2]
    padded_shape = (h + 2 * pad, w + 2 * pad)
    mask = build_notch_mask(padded_shape, angle_deg, band_width,
                            protect_radius, strength)
    out = np.zeros_like(img)
    for c in range(3):
        filt = apply_mask(reflect_pad(img[:, :, c], pad), mask)
        out[:, :, c] = filt[pad:pad + h, pad:pad + w]
    if bright_only:
        out = np.minimum(out, img)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8), angle_deg
