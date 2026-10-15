"""
Frequency-domain Dehazing (homomorphic filtering) — no Dark Channel Prior.

Idea
----
An image is formed as illumination x reflectance. Haze (airlight plus the
veil it lays over the scene) varies slowly across the image, so it lives in
the LOW spatial frequencies, while scene texture and edges live in the HIGH
frequencies. Taking the logarithm turns the product into a sum,
    log I = log(illumination/veil) + log(reflectance),
so a linear frequency filter can treat the two separately:

    log -> FFT -> H(u,v) -> inverse FFT -> exp -> contrast stretch

H is a Gaussian high-frequency-emphasis filter
    H(D) = (gH - gL) * (1 - exp(-D^2 / (2 D0^2))) + gL
with gL < 1 (suppress the slowly varying veil) and gH > 1 (boost detail).
D is the distance from the spectrum origin in cycles/pixel, so D0 does not
depend on image resolution.

Honest limitations (measured, see docs/FFT_METHODS.md)
------------------------------------------------------
* It is a contrast/illumination method, not a physical haze model. It cannot
  estimate depth-dependent haze the way the Dark Channel Prior does.
* It does not restore colour saturation lost to haze; it scores well below
  DCP on PSNR/SSIM against ground truth.
* It can leave bright halos beside strong edges.
* A physically motivated frequency-domain alternative (estimating the
  transmission from local contrast measured with FFT filters) was tried and
  rejected: on real photos it cannot distinguish "hazy" from "naturally
  low-contrast", the inversion goes negative, and the result collapses to black.
"""

import cv2
import numpy as np

from fft_utils import apply_mask, frequency_grid, reflect_pad


def build_homomorphic_mask(shape, cutoff=0.03, gamma_low=0.5, gamma_high=1.5):
    """Homomorphic high-frequency-emphasis mask in unshifted FFT layout."""
    fx, fy = frequency_grid(shape)
    d2 = fx ** 2 + fy ** 2
    return (gamma_high - gamma_low) * (1.0 - np.exp(-d2 / (2.0 * cutoff ** 2))) + gamma_low


def _stretch(channel, low_pct, high_pct):
    lo, hi = np.percentile(channel, [low_pct, high_pct])
    return np.clip((channel - lo) / (hi - lo + 1e-8), 0, 1)


def _homomorphic_channel(channel, mask, pad, stretch):
    h, w = channel.shape
    logc = np.log(channel + 1e-3)
    filt = apply_mask(reflect_pad(logc, pad), mask)[pad:pad + h, pad:pad + w]
    return _stretch(np.exp(filt), *stretch)


def dehaze_fft(image_bgr, cutoff=0.03, gamma_low=0.5, gamma_high=1.5,
               color_mode="rgb", stretch=(1.0, 99.0)):
    """Homomorphic-filter dehazing.

    color_mode:
      'rgb'       : filter B, G, R independently. Removes the low-frequency
                    colour veil too, which restores some saturation.
      'luminance' : filter only Y (YCrCb) and keep the original chroma.
                    Safer against colour casts, but leaves haze desaturation.
    stretch: (low, high) percentiles used for the final contrast stretch.

    image_bgr: HxWx3 uint8 BGR. Returns uint8 BGR."""
    img = image_bgr.astype(np.float32) / 255.0
    pad = 32
    h, w = img.shape[:2]
    mask = build_homomorphic_mask((h + 2 * pad, w + 2 * pad),
                                  cutoff, gamma_low, gamma_high)
    if color_mode == "luminance":
        ycc = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        ycc[:, :, 0] = _homomorphic_channel(ycc[:, :, 0], mask, pad, stretch)
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    else:
        out = np.zeros_like(img)
        for c in range(3):
            out[:, :, c] = _homomorphic_channel(img[:, :, c], mask, pad, stretch)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)
