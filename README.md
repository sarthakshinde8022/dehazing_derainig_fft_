# Clarity: Image Dehazing and Rain Streak Removal Using Digital Image Processing

A beginner-friendly mini-project demonstrating **frequency-domain image restoration** using Digital Image Processing (DIP) techniques and the Fast Fourier Transform (FFT).

Clarity explores two image restoration tasks:

- **Image Dehazing:** Improve local contrast in images affected by haze using homomorphic filtering.
- **Rain Streak Removal:** Reduce directional rain streaks using frequency-domain notch filtering.

The project includes an interactive Streamlit application where users can upload images, adjust filter parameters, visualize frequency spectra and filter masks, compare input and output images, and evaluate results using PSNR and SSIM when a suitable clean reference image is available.

> **Project scope:** The current implementation focuses on classical FFT-based image processing. It does not claim to completely reconstruct the original scene or implement deep-learning models such as AOD-Net.

---

## Table of Contents

1. [Project Objectives](#1-project-objectives)
2. [Features](#2-features)
3. [Repository Structure](#3-repository-structure)
4. [Understanding the Basic Concepts](#4-understanding-the-basic-concepts)
5. [How the Algorithms Work](#5-how-the-algorithms-work)
6. [Installation and Setup](#6-installation-and-setup)
7. [Running the Application](#7-running-the-application)
8. [Using the Interactive Demo](#8-using-the-interactive-demo)
9. [Understanding the Filter Parameters](#9-understanding-the-filter-parameters)
10. [Evaluation Metrics](#10-evaluation-metrics)
11. [Datasets and Evaluation](#11-datasets-and-evaluation)
12. [Limitations](#12-limitations)
13. [Future Improvements](#13-future-improvements)
14. [Technologies Used](#14-technologies-used)
15. [Deployment](#15-deployment)
16. [Author and Academic Use](#16-author-and-academic-use)

---

## 1. Project Objectives

The main objective is to demonstrate how frequency-domain filtering can be used to improve the visual quality of degraded images.

The project aims to:

- Understand the difference between spatial-domain and frequency-domain image processing.
- Apply homomorphic filtering for haze-related contrast enhancement.
- Apply FFT notch filtering to suppress directional rain streak patterns.
- Visualize frequency spectra and filtering masks to understand the underlying operations.
- Compare original images with restored results.
- Evaluate restoration quality using reference-based image quality metrics.
- Build an interactive application suitable for demonstrations, learning, and academic evaluation.

---

## 2. Features

### Image Dehazing

- Upload hazy images in JPG, JPEG, or PNG format.
- Adjust the homomorphic filter's cutoff frequency.
- Control low-frequency attenuation and high-frequency enhancement.
- Choose RGB or luminance processing.
- Compare the original and processed images.
- View the homomorphic filter mask.
- Download the restored image.

### Rain Streak Removal

- Upload rain-affected images in JPG, JPEG, or PNG format.
- Automatically estimate the rain streak orientation or select an angle manually.
- Adjust the notch-filter bandwidth.
- Control the protected low-frequency region.
- Compare the original and processed images.
- Visualize the input spectrum, notch mask, and filtered spectrum.
- Download the processed image.

### Image Quality Evaluation

When a suitable clean reference image is available, the application calculates:

- **PSNR:** Peak Signal-to-Noise Ratio.
- **SSIM:** Structural Similarity Index.

The application also includes evaluation scripts for additional experiments.

---

## 3. Repository Structure

```text
dehazing_deraining_fft_/
├── app.py
├── requirements.txt
├── theme.py
├── fft_utils.py
├── dehazing/
│   └── fft_dehazing.py
├── deraining/
│   └── fft_notch.py
├── pages/
│   └── 2_Try_the_Demo.py
├── evaluation/
│   ├── compare_fft.py
│   └── evaluate_sots.py
└── docs/
    └── FFT_METHODS.md
```

### Purpose of Each File

| File | Description |
|---|---|
| `app.py` | Main Streamlit application and project introduction. |
| `theme.py` | Custom CSS styling used by the interface. |
| `fft_utils.py` | Shared FFT utilities for frequency grids, filtering, padding, and spectrum visualization. |
| `dehazing/fft_dehazing.py` | Implements FFT-based homomorphic filtering for haze-related contrast enhancement. |
| `deraining/fft_notch.py` | Implements FFT-based notch filtering for directional rain streak suppression. |
| `pages/2_Try_the_Demo.py` | Interactive demo containing the dehazing and rain-removal tabs. |
| `evaluation/compare_fft.py` | Script for comparing FFT-based methods; see the script for its supported arguments. |
| `evaluation/evaluate_sots.py` | Evaluation script for the SOTS dataset; see the script for its supported arguments. |
| `docs/FFT_METHODS.md` | Technical explanation of the implemented frequency-domain methods. |

---

## 4. Understanding the Basic Concepts

This section introduces the terminology needed to understand the project.

### 4.1 Digital Image Processing (DIP)

Digital Image Processing is the use of computational techniques to analyze, modify, enhance, and interpret digital images.

Examples include removing noise, enhancing contrast, sharpening edges, restoring degraded photographs, and detecting objects.

### 4.2 Pixel

A pixel is the smallest individual element of a digital image. Each pixel stores a brightness value in a grayscale image or color values in a color image.

An image is represented as a two-dimensional arrangement of pixels, with additional color channels when required.

### 4.3 Spatial Domain

The spatial domain is the conventional representation of an image as pixels arranged in rows and columns.

Operations performed directly on pixel values, such as brightness adjustment, convolution, and local filtering, are spatial-domain operations.

### 4.4 Frequency Domain

The frequency domain represents an image in terms of how rapidly its intensity changes across space.

Instead of working directly with the pixel arrangement, frequency-domain processing analyzes different spatial frequencies and modifies selected components.

### 4.5 Fast Fourier Transform (FFT)

The Fast Fourier Transform is an efficient algorithm for computing the Discrete Fourier Transform (DFT).

In this project, the FFT converts an image or image channel into a frequency-domain representation so that selected frequency components can be filtered.

The inverse FFT converts the filtered frequency representation back into an image.

### 4.6 Spatial Frequency

Spatial frequency describes how quickly image intensity changes over distance.

- **Low spatial frequencies:** Slowly varying brightness and broad image structures, such as smooth gradients and large areas of illumination.
- **High spatial frequencies:** Rapid intensity variations commonly associated with edges, fine textures, and some noise.
- **Directional frequency patterns:** Frequency components arranged in particular orientations. They can reveal repeating or directionally structured features.

These categories are useful approximations, not exclusive classifications. Real image content can occupy many frequencies.

### 4.7 Fourier Spectrum

A Fourier spectrum visualizes the strength of frequency components in an image.

For display purposes, the spectrum is commonly shifted so that the zero-frequency component appears at the center. A logarithmic magnitude scale helps make weaker components visible.

The spectrum can reveal directional patterns that may be difficult to identify directly in the image.

### 4.8 Filter Mask

A filter mask is a two-dimensional array that modifies the image's frequency components.

- Values near **1** generally preserve the corresponding frequencies.
- Values near **0** strongly suppress them.
- Intermediate values attenuate them partially.
- Values greater than **1** amplify the corresponding components.

The exact effect depends on the particular filter and its implementation.

---

## 5. How the Algorithms Work

### 5.1 Homomorphic Filtering for Image Dehazing

Homomorphic filtering is a frequency-domain technique that adjusts image illumination and contrast.

An image can be approximated as the product of illumination and reflectance components:

\[
f(x,y)=i(x,y)\,r(x,y)
\]

Here:

- `f(x,y)` represents the observed image.
- `i(x,y)` represents the slowly varying illumination component.
- `r(x,y)` represents the reflectance or finer image-detail component.

Taking the logarithm converts the multiplicative relationship into an additive one:

\[
\ln f(x,y)=\ln i(x,y)+\ln r(x,y)
\]

The image can then be transformed into the frequency domain, filtered, and transformed back.

A typical homomorphic filter attenuates selected low-frequency components and enhances selected higher-frequency components.

#### Processing pipeline

1. Load the hazy image.
2. Prepare the image channels for filtering.
3. Apply a logarithmic transformation.
4. Compute the FFT.
5. Apply the homomorphic filter mask.
6. Compute the inverse FFT.
7. Convert the result into an output image.
8. Display and download the result.

#### Why does it help?

Haze often reduces contrast and makes distant objects appear pale. Adjusting illumination-related variations and image contrast can make some structures more visible.

**Important limitation:** Homomorphic filtering is a general illumination and contrast adjustment technique. It is not, by itself, a physical atmospheric-scattering model. Therefore, it cannot reliably recover all lost colors, depth information, or scene details.

### 5.2 Notch Filtering for Rain Streak Removal

Notch filtering suppresses selected frequency regions while attempting to preserve the rest of the spectrum.

Directional rain streaks can produce recognizable patterns in the Fourier spectrum. When their energy is sufficiently concentrated around a directional frequency band, a notch filter can attenuate that band.

#### Processing pipeline

1. Load the rain-affected image.
2. Prepare an image channel for frequency analysis.
3. Compute the FFT and analyze the spectrum.
4. Estimate or select the streak orientation.
5. Construct a notch-filter mask.
6. Apply the mask to the frequency spectrum.
7. Compute the inverse FFT.
8. Reconstruct and display the processed image.

#### Why does it help?

When unwanted rain streaks produce a recognizable directional frequency pattern, suppressing the corresponding frequencies can reduce the visibility of some streaks.

**Important limitation:** Rain streaks do not always produce perfectly isolated frequency patterns. Genuine image edges, such as building sides, poles, wires, and other elongated structures, may occupy similar frequencies. Removing a broad frequency band can therefore damage legitimate scene details.

Results depend on streak orientation, the amount of rain, the image content, and the selected filter parameters.

---

## 6. Installation and Setup

### Prerequisites

Install Python and Git on your computer.

The project uses Python packages for numerical computation, image processing, image evaluation, and the Streamlit interface.

### Step 1: Clone the repository

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

Replace the placeholders with your actual GitHub username and repository name.

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### Step 3: Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The core dependencies used by the current demo are:

```text
streamlit
numpy
opencv-python-headless
Pillow
scikit-image
```

- **Streamlit:** Builds the interactive web interface.
- **NumPy:** Provides numerical arrays and FFT operations.
- **OpenCV:** Provides image-processing utilities and the `cv2` module.
- **Pillow:** Loads, converts, and saves images.
- **scikit-image:** Provides image-quality metrics such as PSNR and SSIM.

If your evaluation or algorithm scripts use additional packages, include those dependencies in `requirements.txt` as well.

---

## 7. Running the Application

Run the application from the repository root:

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

The home page introduces the project and links to the interactive demo.

If an import or dependency error occurs, check that all required files are committed to the repository and that the required packages are installed in the active Python environment.

---

## 8. Using the Interactive Demo

### Step 1: Choose a restoration task

Open the interactive demo and select one of the two tabs:

- **Dehazing**
- **Rain Streak Removal**

### Step 2: Upload an image

Upload a JPG, JPEG, or PNG image containing the degradation you want to reduce.

For meaningful experimentation, select an image that actually contains haze or visible rain streaks.

### Step 3: Adjust the parameters

Start with the default settings. Change one parameter at a time so you can identify how it affects the result.

### Step 4: Compare the results

Compare the original and restored images at the same scale.

Look for changes in:

- Visibility and contrast.
- Edge sharpness.
- Fine texture.
- Color balance.
- Artifacts and unwanted detail loss.

### Step 5: Inspect the frequency-domain visualization

Use the available expanders to inspect the filter mask and Fourier spectrum.

These visualizations help explain which frequency regions the algorithm is designed to modify.

### Step 6: Download the output

Use the download button to save the processed image as a PNG file.

### Step 7: Evaluate the restoration

If a suitable clean reference image of the same scene is available, upload it to calculate PSNR and SSIM.

The reference should be correctly aligned with the processed image and should represent the same scene without the degradation being evaluated.

---

## 9. Understanding the Filter Parameters

### 9.1 Dehazing Parameters

| Parameter | Meaning | General effect |
|---|---|---|
| Cutoff frequency `D0` | Sets the frequency scale of the filter transition. | Changes which spatial frequencies fall within the lower-frequency response. |
| Low-frequency gain `γL` | Controls the response applied to lower frequencies. | Lower values generally attenuate these components more strongly. |
| High-frequency gain `γH` | Controls the response applied to higher frequencies. | Higher values generally emphasize details, but can also amplify noise. |
| Colour mode | Selects RGB or luminance-based processing. | Changes how the filtering is applied to color information. |

The ideal parameter values depend on the image. Stronger filtering does not necessarily produce a better result.

### 9.2 Rain Streak Removal Parameters

| Parameter | Meaning | General effect |
|---|---|---|
| Auto-detect streak angle | Estimates the streak direction from the image spectrum. | Reduces the need for manual angle selection. |
| Streak tilt | Manually specifies the streak orientation relative to vertical. | Changes the orientation of the suppressed frequency band. |
| Band width | Controls the width of the frequency region suppressed by the notch mask. | Wider bands can suppress more streak-related energy but may remove more image detail. |
| Protected low-frequency radius | Controls the central frequency area the mask is designed to protect. | Helps preserve low-frequency image structure. |
| Bright-only | Applies a constraint intended to prevent output pixels from becoming brighter than their input values. | May reduce bright streaks, but can also limit legitimate restoration changes. |

The effects described above are qualitative. The precise response depends on the implementation in `dehazing/fft_dehazing.py` and `deraining/fft_notch.py`.

---

## 10. Evaluation Metrics

The application supports two reference-based image-quality metrics.

### 10.1 PSNR — Peak Signal-to-Noise Ratio

PSNR measures image error using the Mean Squared Error (MSE).

For images with an 8-bit intensity range:

\[
\operatorname{PSNR}
=10\log_{10}\left(\frac{255^2}{\operatorname{MSE}}\right)
\]

Here, MSE is the average squared difference between corresponding pixel values in the restored image and the reference image.

**Interpretation:**

- Higher PSNR generally means lower pixel-level reconstruction error.
- PSNR is measured in decibels (dB).
- PSNR is meaningful when the images represent the same scene and are aligned.
- PSNR does not necessarily correlate perfectly with perceived visual quality.

### 10.2 SSIM — Structural Similarity Index

SSIM compares image structure, luminance, and contrast.

Its commonly used formulation is:

\[
\operatorname{SSIM}(x,y)=
\frac{(2\mu_x\mu_y+C_1)(2\sigma_{xy}+C_2)}
{(\mu_x^2+\mu_y^2+C_1)(\sigma_x^2+\sigma_y^2+C_2)}
\]

Here:

- `μx`, `μy` are local mean intensities.
- `σx²`, `σy²` are local variances.
- `σxy` is the local covariance.
- `C1`, `C2` are stabilizing constants.

**Interpretation:** SSIM values closer to 1 generally indicate greater structural similarity to the reference.

### What is a ground-truth image?

A ground-truth image is the clean reference used to assess restoration quality. It should depict the same scene without the degradation being evaluated.

For example, a rainy street photograph should be compared with an appropriately aligned clean image of that same scene—not with an unrelated clean street photograph.

**Important:** PSNR and SSIM are not meaningful as reference-based restoration measurements without a suitable reference image. Higher values also do not guarantee that an image looks natural or contains fewer artifacts.

---

## 11. Datasets and Evaluation

The repository includes an evaluation script named `evaluation/evaluate_sots.py` for SOTS-related experiments.

SOTS is part of the RESIDE benchmark, which is widely used for research on single-image dehazing.

Dataset information: https://sites.google.com/view/reside-dehaze-datasets

Before running the evaluation:

1. Obtain the appropriate SOTS dataset.
2. Read the expected folder structure in `evaluation/evaluate_sots.py`.
3. Place the dataset outside the repository if it is too large to commit to Git.
4. Run the evaluation script with the arguments it supports.
5. Save the resulting metrics and sample images for your project report.

To inspect the available command-line arguments, try:

```bash
python evaluation/evaluate_sots.py --help
```

For rain streak removal, use a suitable paired deraining dataset if you want to calculate PSNR and SSIM over a benchmark. Confirm the expected dataset structure and supported arguments of your evaluation scripts before starting a full evaluation.

Do not report benchmark numbers as measured results unless you have run the implementation on the stated dataset and verified the results.

---

## 12. Limitations

### Homomorphic dehazing

- Improves image contrast but does not explicitly model atmospheric scattering.
- May not recover the original colors or the information lost to dense haze.
- Excessive high-frequency enhancement can amplify noise.
- Strong filtering may produce halos, unnatural contrast, or other artifacts.

### FFT-based rain streak removal

- Works best when the unwanted streak patterns are sufficiently directional in the frequency domain.
- May struggle with dense rain, irregular streaks, or streaks with multiple orientations.
- Can weaken legitimate edges that occupy the same frequency regions as rain.
- May produce residual streaks or loss of texture.

### Reference-based evaluation

- Requires a clean reference image of the same scene.
- Is sensitive to differences in image alignment, dimensions, and intensity representation.
- Does not fully capture visual naturalness or semantic image quality.

These limitations are important to demonstrate during the project presentation because they explain the trade-offs of classical frequency-domain filtering.

---

## 13. Future Improvements

Possible extensions for future development include:

- Implement Dark Channel Prior (DCP) as an additional classical dehazing baseline.
- Add more advanced rain removal techniques for irregular streak patterns.
- Implement a combined pipeline for images affected by both haze and rain.
- Add larger-scale benchmark evaluation for dehazing and deraining.
- Investigate lightweight deep-learning models such as AOD-Net for dehazing.
- Compare classical image processing with trained neural networks.
- Add more quantitative evaluation and visual comparison tools.
- Improve handling of large images and processing time.

These are potential extensions, not claims about features already implemented in the current application.

---

## 14. Technologies Used

- **Python** — Core programming language.
- **Streamlit** — Interactive web application.
- **NumPy** — Numerical computation and FFT operations.
- **OpenCV** — Image-processing utilities.
- **Pillow** — Image loading and conversion.
- **scikit-image** — PSNR and SSIM calculations.
- **Git and GitHub** — Version control and source-code hosting.

---

## 15. Deployment

The application can be deployed using Streamlit Community Cloud.

1. Push the repository to GitHub.
2. Visit https://share.streamlit.io/.
3. Sign in and choose to create a new app.
4. Select the GitHub repository and deployment branch.
5. Set the main file path to `app.py`.
6. Deploy the application.

Ensure that `requirements.txt`, `theme.py`, `fft_utils.py`, the algorithm modules, and the `pages/` directory are committed to GitHub.

For the current demo, the main file path should be:

```text
app.py
```

If deployment fails, inspect the complete error traceback and verify that all required Python packages and modules are available.

---

## 16. Author and Academic Use

This repository was developed as an academic mini-project exploring image restoration using Digital Image Processing techniques.

The project is intended for educational use, experimentation, and demonstration. Performance claims should be supported by reproducible experiments, measured results, and appropriate references.

For a project submission or viva, explain the underlying frequency-domain concepts, the reasons for selecting each filter, the effect of the adjustable parameters, the measured evaluation results, and the limitations observed during testing.

**Project summary:** Clarity demonstrates how FFT-based homomorphic filtering and notch filtering can be applied to two image restoration problems through an interactive, beginner-friendly Streamlit application.
