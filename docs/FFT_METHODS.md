# Frequency-Domain (FFT) Pipeline

Both tasks solved in the Fourier domain, with no Dark Channel Prior.
Code: `dehazing/fft_dehazing.py`, `deraining/fft_notch.py`, `fft_utils.py`.
Demo: the **Try the Demo** page. Reproduce the comparison numbers with
`python evaluation/compare_fft.py`; run the SOTS benchmark with
`python evaluation/evaluate_sots.py --root <SOTS> --method fft` (or `--method dcp`).

The Dark Channel Prior (`dehazing/dark_channel_prior.py`) and the spatial
top-hat rain method (`deraining/rain_streak_removal.py`) are kept in the repo
**as baselines**, so every claim about the FFT pipeline is a measured
comparison, not an assertion.

## 1. Dehazing: homomorphic filtering

Image = illumination x reflectance. Haze (airlight + veil) varies slowly, so
it sits in the low spatial frequencies; texture and edges sit in the high
frequencies. The log turns the product into a sum, so one linear frequency
filter can attenuate the veil and boost detail:

    log -> FFT -> H(u,v) -> inverse FFT -> exp -> contrast stretch
    H(D) = (gH - gL)(1 - exp(-D^2 / 2*D0^2)) + gL,   gL < 1 < gH

D is measured in cycles/pixel, so the cutoff D0 does not depend on image
resolution. Defaults: D0 = 0.03, gL = 0.5, gH = 1.5, RGB mode.

## 2. Rain streaks: directional FFT notch

A pattern constant along one direction has all its Fourier energy on the line
through the origin *perpendicular* to it, so near-vertical streaks form a
bright band along (almost) the horizontal frequency axis.

1. Reflect-pad each channel (the DFT assumes periodicity; border mismatch
   would add false energy along both axes).
2. FFT, multiply by a smooth Gaussian notch mask, inverse FFT.
3. *Bright-only constraint:* rain only adds brightness, so keep
   `min(original, filtered)`; a pixel the filter would brighten is a smeared
   edge, not rain.

With tilt theta from vertical, `d = fx*sin(theta) + fy*cos(theta)` and
`M = 1 - exp(-d^2 / 2*sigma^2) * (1 - exp(-r^2 / 2*r0^2))`. sigma (about
1 / streak length) sets the band; r0 protects DC and the low frequencies.
The tilt can be auto-detected by scanning candidate angles for the strongest
spectral band; it recovered 0, +10 and -10 degrees exactly in testing.

## 3. Measured results (synthetic degradations, exact ground truth)

Two clean photos bundled with scikit-image, degraded synthetically.

**Dehazing, 8 cases** (chelsea and coffee; beta 1 and 2; ramp and uniform
depth). Mean PSNR: hazy input **10.6 dB**, FFT homomorphic **14.2 dB**,
DCP baseline **22.9 dB**.

| Case | Hazy | DCP | FFT homomorphic |
|---|---|---|---|
| chelsea, beta 1, ramp | 13.18 / 0.817 | 19.48 / 0.922 | 12.12 / 0.689 |
| chelsea, beta 1, uniform | 13.07 / 0.809 | 22.30 / 0.946 | 14.30 / 0.699 |
| chelsea, beta 2, ramp | 9.65 / 0.657 | 18.23 / 0.887 | 9.70 / 0.680 |
| chelsea, beta 2, uniform | 9.24 / 0.623 | 25.68 / 0.952 | 14.26 / 0.683 |
| coffee, beta 1, ramp | 12.33 / 0.691 | 24.81 / 0.928 | 14.20 / 0.717 |
| coffee, beta 1, uniform | 11.30 / 0.655 | 24.27 / 0.931 | 19.33 / 0.760 |
| coffee, beta 2, ramp | 8.65 / 0.543 | 23.83 / 0.855 | 10.44 / 0.637 |
| coffee, beta 2, uniform | 7.48 / 0.491 | 24.34 / 0.886 | 19.06 / 0.741 |

(PSNR dB / SSIM.)

**Rain, 700 bright streaks:** unprocessed 16.7 dB / 0.41; spatial top-hat
27.8-28.0 dB / 0.80-0.81; FFT notch 23.2-25.0 dB / 0.58-0.69.

## 4. Honest conclusions

- **The FFT pipeline is weaker than the spatial/physical baselines.** FFT
  dehazing improves PSNR in 7 of 8 cases but trails DCP by about 9 dB on
  average, and it lowers SSIM relative to the hazy input when haze is mild
  (beta 1). The FFT rain notch recovers about 6 to 8 dB but trails the spatial
  top-hat by about 3 to 5 dB.
- **Why the rain notch trails.** Random finite-length streaks spread their
  energy over a wide band that overlaps scene content, and real near-vertical
  edges lie in the same band and get softened. A wider band removes more rain
  and more detail. Notch filtering is the right tool for *periodic* patterns
  (scan lines, moire); sparse random streaks are a harder fit.
- **Why FFT dehazing trails.** Homomorphic filtering is a contrast method: it
  has no model of how haze forms, cannot recover depth-dependent haze or
  colour, and can create bright halos beside strong edges. Tuning its
  parameters purely for PSNR on held-out images drove them toward almost no
  frequency filtering at all (effectively a global contrast stretch), so PSNR
  alone should not be used to choose them.

## 5. A negative result worth reporting

A physically motivated frequency-domain alternative was built and rejected.
Haze scales local contrast by the transmission t, so local contrast measured
with FFT high-pass and low-pass filters can estimate t, after which
`J = (I - A(1-t)) / t`. With a fixed clear-image contrast prior calibrated on
held-out images it improved on the hazy input on synthetic data (mean 17.2 dB
vs 10.6 dB hazy, still below DCP's 22.9 dB), but on a real hazy photograph it
**collapsed to a black image**: the estimator cannot tell "hazy" from
"naturally low-contrast" (smooth sky, a sun-lit region), the transmission
sinks to its floor, and the inversion goes negative. Clear-photo local
contrast also varied about 2.5x between images, so a fixed prior is too weak.
This is the frequency-domain analogue of why DCP needs its dark-channel prior:
without a reliable prior on the clear image, single-image dehazing is
ill-posed.

## 6. Likely viva questions

**Why do rain streaks form a line in the spectrum?** A vertical line is
constant in y, so its transform is concentrated at fy = 0, along the fx axis;
finite length spreads it slightly in fy (about 1/length).

**Why pad before the FFT?** The DFT assumes periodicity; the mismatch between
opposite borders creates a false bright cross along both axes that would be
filtered like real structure.

**Why a smooth mask instead of a hard cut?** A hard cut in frequency is a
sinc in space and causes ringing; a Gaussian transition avoids it.

**Why take the logarithm in homomorphic filtering?** Illumination and
reflectance multiply; the log makes them add, so a linear filter can attenuate
one (low-frequency veil) and boost the other (detail).

**Why is the FFT pipeline worse than DCP, and why keep DCP?** Haze removal
needs a physical model plus a prior; the DCP supplies both, while a purely
linear frequency filter has neither. DCP is kept as the baseline so the
comparison is measured, which is the point of the project.
