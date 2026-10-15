"""
Try the Demo — frequency-domain (FFT) dehazing and rain streak removal,
with the spectra and filter masks shown so the method is visible, not just
its result.
"""

import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from dehazing.fft_dehazing import build_homomorphic_mask, dehaze_fft
from deraining.fft_notch import build_notch_mask, derain_fft
from fft_utils import centered_mask_image, log_spectrum, reflect_pad
from theme import inject_custom_css

st.set_page_config(page_title="Try the Demo — Clarity", page_icon="📡", layout="wide")
inject_custom_css()

MAX_SIDE = 1600  # keep FFTs responsive on very large uploads


def load_bgr(uploaded):
    arr = np.array(Image.open(uploaded).convert("RGB"))
    h, w = arr.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        arr = cv2.resize(arr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def png_bytes(rgb):
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def metrics_row(gt_file, result_rgb):
    gt = np.array(Image.open(gt_file).convert("RGB"))
    if gt.shape[:2] != result_rgb.shape[:2]:
        gt = cv2.resize(gt, (result_rgb.shape[1], result_rgb.shape[0]))
    st.markdown('<hr class="clarity-divider">', unsafe_allow_html=True)
    st.markdown("### Metrics vs. ground truth")
    m1, m2 = st.columns(2)
    m1.metric("PSNR", f"{psnr(gt, result_rgb, data_range=255):.2f} dB")
    m2.metric("SSIM", f"{ssim(gt, result_rgb, data_range=255, channel_axis=2):.4f}")


st.page_link("app.py", label="← Home", icon="🏠")
st.markdown('<div class="eyebrow">LIVE DEMO · FREQUENCY DOMAIN</div>', unsafe_allow_html=True)
st.markdown("# Try It")

with st.expander("📘 Beginner's Guide: Terms and Controls", expanded=False):
    st.markdown("""
    ### Essential image-processing terms

    **FFT (Fast Fourier Transform):** A fast mathematical method
    for representing image variations as spatial frequencies.

    **Spatial domain:** The normal image made of pixels.

    **Frequency domain:** A representation showing how quickly
    brightness changes across an image.

    **Spectrum:** A visualization of the distribution and strength
    of spatial frequencies in an image.

    **Filter mask:** A map specifying which frequencies should be
    preserved, reduced, or amplified.

    **Low frequencies:** Slowly changing brightness and broad image
    structures. These are not always haze; they can also represent
    legitimate image content.

    **High frequencies:** Rapid brightness changes, such as edges,
    fine textures, and some noise.

    ### Dehazing controls

    **Cutoff D0 (cycles/px):** Controls the frequency scale at which
    the homomorphic filter changes its response. The effect depends
    on the filter implementation; compare results rather than
    assuming one value is always best.

    **Gamma low (γ low):** The filter response associated with
    low frequencies. Lower values generally suppress those
    components more strongly.

    **Gamma high (γ high):** The filter response associated with
    high frequencies. Higher values generally emphasize fine
    variations, but may also amplify noise.

    **Colour mode — RGB:** Processes the colour channels separately.

    **Colour mode — luminance:** Applies the filtering to image
    brightness while aiming to preserve the colour information.

    ### Rain streak removal controls

    **Streak angle:** The estimated or manually selected orientation
    of the rain streaks. The meaning of the angle depends on the
    convention used by the algorithm.

    **Band width:** Controls the width of the frequency region
    suppressed by the notch filter. A wider band may remove more
    streak-related energy but can also erase genuine image details.

    **Protected low-frequency radius:** Controls the central frequency
    region that the notch mask is designed to protect.

    **Auto-detect streak angle:** Lets the algorithm estimate the
    streak direction from the image instead of relying on a manual
    setting.

    **Bright-only:** Uses a constraint intended to prevent the
    restoration step from making pixels brighter than their input
    values. This may also limit recovery of legitimate details.

    ### Evaluation terms

    **Ground truth:** A clean reference image showing the same scene.
    It should be correctly aligned with the restored image.

    **PSNR (Peak Signal-to-Noise Ratio):** A numerical measure based
    on pixel-level error. When images are comparable, a higher value
    generally indicates lower error. It is measured in decibels (dB).

    **SSIM (Structural Similarity Index):** A metric comparing image
    structure, contrast, and luminance. Values closer to 1 generally
    indicate greater similarity.

    **Important:** PSNR and SSIM require a suitable reference image.
    They do not independently prove that a restoration looks natural.

    ### Practical advice

    1. Start with the default settings.
    2. Change one parameter at a time.
    3. Compare the original and restored images at the same scale.
    4. Watch for blurred edges, artificial halos, and excessive noise.
    """)

st.markdown(
    '<p style="color:#93A5B1; max-width:700px;">Both problems are solved in the '
    'Fourier domain. Rain streaks are near-vertical lines, so their energy sits '
    'on one line through the spectrum and can be notched out. Haze is a slowly '
    'varying veil, so it lives in the low frequencies and can be suppressed with '
    'a homomorphic filter.</p>',
    unsafe_allow_html=True,
)
st.markdown('<hr class="clarity-divider">', unsafe_allow_html=True)

tab_haze, tab_rain = st.tabs(["Dehazing", "Rain Streak Removal"])

# ------------------------------------------------------------------ HAZE
with tab_haze:
    up_col, gt_col = st.columns(2)
    hazy_file = up_col.file_uploader("Upload a hazy image", type=["jpg", "jpeg", "png"], key="fft_hazy")
    gt_haze = gt_col.file_uploader("Optional: ground-truth clean image (for PSNR/SSIM)", type=["jpg", "jpeg", "png"], key="fft_gt_haze")

    p1, p2, p3, p4 = st.columns(4)
    cutoff_h = p1.slider("Cutoff D0 (cycles/px)", 0.005, 0.15, 0.03, 0.005, key="fft_d0", format="%.3f")
    g_low = p2.slider("γ low (suppress haze veil)", 0.1, 1.0, 0.5, 0.05, key="fft_gl")
    g_high = p3.slider("γ high (boost detail)", 1.0, 3.0, 1.5, 0.1, key="fft_gh")
    cmode = p4.selectbox("Colour mode", ["rgb", "luminance"], key="fft_cmode")

    if hazy_file is not None:
        bgr = load_bgr(hazy_file)
        with st.spinner("Filtering in the frequency domain..."):
            out_bgr = dehaze_fft(bgr, cutoff=cutoff_h, gamma_low=g_low,
                                 gamma_high=g_high, color_mode=cmode)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        in_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        a, b = st.columns(2)
        a.markdown('<div class="eyebrow">BEFORE</div>', unsafe_allow_html=True)
        a.image(in_rgb, width="stretch")
        b.markdown('<div class="eyebrow">AFTER</div>', unsafe_allow_html=True)
        b.image(out_rgb, width="stretch")
        st.download_button("Download dehazed image", png_bytes(out_rgb),
                           "dehazed_fft.png", "image/png", key="dl_fft_haze")

        with st.expander("See the homomorphic filter"):
            pad = 32
            shape = (bgr.shape[0] + 2 * pad, bgr.shape[1] + 2 * pad)
            m = build_homomorphic_mask(shape, cutoff_h, g_low, g_high)
            m_img = (np.fft.fftshift(m) - m.min()) / (m.max() - m.min() + 1e-9)
            st.image((m_img * 255).astype(np.uint8),
                     caption="Filter H(u,v): dark centre = low frequencies (haze veil) attenuated, bright outside = detail boosted",
                     width=360)
        st.caption("Homomorphic filtering improves local contrast but has no model of how haze "
                   "forms, so it does not restore colour or depth-dependent haze the way a "
                   "physical-model method does. See docs/FFT_METHODS.md for measured limitations.")
        if gt_haze is not None:
            metrics_row(gt_haze, out_rgb)
    else:
        st.info("Upload a hazy image to see the frequency-domain result.")

# ------------------------------------------------------------------ RAIN
with tab_rain:
    up_col, gt_col = st.columns(2)
    rainy_file = up_col.file_uploader("Upload a rainy image", type=["jpg", "jpeg", "png"], key="fft_rainy")
    gt_rain = gt_col.file_uploader("Optional: ground-truth clean image (for PSNR/SSIM)", type=["jpg", "jpeg", "png"], key="fft_gt_rain")

    c1, c2, c3, c4 = st.columns(4)
    auto = c1.checkbox("Auto-detect streak angle", value=True, key="fft_auto")
    angle = c2.slider("Streak tilt from vertical (°)", -30, 30, 0, 1, key="fft_angle", disabled=auto)
    band = c3.slider("Band width (cycles/px)", 0.02, 0.25, 0.10, 0.01, key="fft_band")
    protect = c4.slider("Protected low-freq radius", 0.005, 0.08, 0.02, 0.005, key="fft_protect")
    bright_only = st.checkbox(
        "Bright-only (rain only adds brightness, so never brighten a pixel)",
        value=True, key="fft_bright")

    if rainy_file is not None:
        bgr = load_bgr(rainy_file)
        with st.spinner("Filtering in the frequency domain..."):
            out_bgr, used_angle = derain_fft(
                bgr, angle_deg=float(angle), band_width=band,
                protect_radius=protect, auto_angle=auto, bright_only=bright_only)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        in_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if auto:
            st.caption(f"Estimated streak tilt from the spectrum: **{used_angle:+.0f}°** from vertical.")

        a, b = st.columns(2)
        a.markdown('<div class="eyebrow">BEFORE</div>', unsafe_allow_html=True)
        a.image(in_rgb, width="stretch")
        b.markdown('<div class="eyebrow">AFTER</div>', unsafe_allow_html=True)
        b.image(out_rgb, width="stretch")
        st.download_button("Download derained image", png_bytes(out_rgb),
                           "derained_fft.png", "image/png", key="dl_fft_rain")

        with st.expander("See what the filter did in the frequency domain"):
            pad = 32
            gray_in = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
            gray_out = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
            shape = (gray_in.shape[0] + 2 * pad, gray_in.shape[1] + 2 * pad)
            mask = build_notch_mask(shape, used_angle, band, protect, 1.0)
            s1, s2, s3 = st.columns(3)
            s1.image(log_spectrum(reflect_pad(gray_in, pad)), caption="Spectrum of the rainy image — streaks form the bright band", width="stretch")
            s2.image(centered_mask_image(mask), caption="Notch mask (dark = removed)", width="stretch")
            s3.image(log_spectrum(reflect_pad(gray_out, pad)), caption="Spectrum after filtering", width="stretch")
            st.caption("Trade-off: real near-vertical edges (poles, building sides) lie in the same band, "
                       "so a global notch softens them too.")
        if gt_rain is not None:
            metrics_row(gt_rain, out_rgb)
    else:
        st.info("Upload a rainy image to see the FFT notch filter at work.")
