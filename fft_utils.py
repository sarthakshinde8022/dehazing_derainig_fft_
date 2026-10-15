"""
Shared FFT helpers used by the frequency-domain dehazing and deraining
modules and by the FFT Lab page.
"""

import numpy as np


def reflect_pad(channel, pad):
    """Reflect-pad a 2D array. The FFT treats an image as periodic, so
    without padding the mismatch between opposite borders creates a bright
    cross of false energy along both spectrum axes, which would be mistaken
    for (and filtered like) vertical/horizontal structure."""
    return np.pad(channel, pad, mode="reflect")


def frequency_grid(shape):
    """Return (fx, fy) in cycles/pixel, in the UNSHIFTED layout that
    np.fft.fft2 produces (DC at index [0, 0])."""
    h, w = shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return fx, fy


def apply_mask(channel, mask):
    """Multiply the 2D spectrum of `channel` by `mask` (same shape,
    unshifted layout) and return the filtered spatial image."""
    spectrum = np.fft.fft2(channel)
    return np.real(np.fft.ifft2(spectrum * mask))


def log_spectrum(channel):
    """Centered, log-scaled magnitude spectrum scaled to uint8 for display."""
    mag = np.abs(np.fft.fftshift(np.fft.fft2(channel)))
    mag = np.log1p(mag)
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-12)
    return (mag * 255).astype(np.uint8)


def centered_mask_image(mask):
    """Centered view of an unshifted mask, scaled to uint8 for display."""
    return (np.fft.fftshift(mask) * 255).clip(0, 255).astype(np.uint8)
