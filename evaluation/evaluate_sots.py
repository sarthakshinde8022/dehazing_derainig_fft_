"""
Evaluate a dehazing method on the SOTS dataset (RESIDE benchmark).

Methods:
    fft  (default) frequency-domain homomorphic filtering  dehazing/fft_dehazing.py
    dcp            Dark Channel Prior baseline             dehazing/dark_channel_prior.py

Expected folder structure (standard SOTS layout):
    SOTS/
        indoor/   hazy/  gt/
        outdoor/  hazy/  gt/

Usage:
    python evaluation/evaluate_sots.py --root /path/to/SOTS --subset indoor --limit 20
    python evaluation/evaluate_sots.py --root /path/to/SOTS --subset indoor --method dcp --csv_out indoor_dcp.csv
"""

import argparse
import glob
import os
import sys

# make the repo root importable whether run directly or as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def get_method(name):
    if name == "dcp":
        from dehazing.dark_channel_prior import dehaze
        return dehaze
    from dehazing.fft_dehazing import dehaze_fft
    return dehaze_fft


def match_gt_filename(hazy_filename):
    """SOTS hazy filenames carry the ground-truth id before the first
    underscore: '1400_1.png' -> '1400', '0001_0.8_0.2.jpg' -> '0001'."""
    base = os.path.splitext(os.path.basename(hazy_filename))[0]
    return base.split("_")[0]


def find_gt_path(gt_dir, gt_id):
    for ext in (".png", ".jpg", ".jpeg", ".bmp"):
        candidate = os.path.join(gt_dir, gt_id + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def evaluate(root, subset, method="fft", limit=None, save_samples_dir=None):
    run = get_method(method)
    hazy_dir = os.path.join(root, subset, "hazy")
    gt_dir = os.path.join(root, subset, "gt")

    hazy_paths = sorted(glob.glob(os.path.join(hazy_dir, "*")))
    if not hazy_paths:
        raise FileNotFoundError(f"No images found in {hazy_dir}. Check --root and folder names.")
    if limit:
        hazy_paths = hazy_paths[:limit]
    if save_samples_dir:
        os.makedirs(save_samples_dir, exist_ok=True)

    records = []
    for hazy_path in hazy_paths:
        gt_path = find_gt_path(gt_dir, match_gt_filename(hazy_path))
        if gt_path is None:
            print(f"[skip] no ground truth found for {hazy_path}")
            continue
        hazy_img, gt_img = cv2.imread(hazy_path), cv2.imread(gt_path)
        if hazy_img is None or gt_img is None:
            print(f"[skip] could not read {hazy_path} or {gt_path}")
            continue
        if hazy_img.shape != gt_img.shape:
            gt_img = cv2.resize(gt_img, (hazy_img.shape[1], hazy_img.shape[0]))

        out = run(hazy_img)
        p = psnr(gt_img, out, data_range=255)
        s = ssim(gt_img, out, data_range=255, channel_axis=2)
        records.append({"hazy_file": os.path.basename(hazy_path),
                        "gt_file": os.path.basename(gt_path),
                        "method": method, "psnr": p, "ssim": s})

        if save_samples_dir:
            name = os.path.splitext(os.path.basename(hazy_path))[0]
            cv2.imwrite(os.path.join(save_samples_dir, f"{name}_{method}_compare.png"),
                        np.hstack([hazy_img, out, gt_img]))
        print(f"{os.path.basename(hazy_path):30s} PSNR={p:6.2f}  SSIM={s:.4f}")

    df = pd.DataFrame(records)
    if len(df):
        print(f"\n=== Summary ({method}, {subset}) ===")
        print(f"Images evaluated: {len(df)}")
        print(f"Mean PSNR: {df['psnr'].mean():.2f}")
        print(f"Mean SSIM: {df['ssim'].mean():.4f}")
    else:
        print("No image pairs were evaluated - check your dataset paths.")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to SOTS root folder")
    parser.add_argument("--subset", choices=["indoor", "outdoor"], default="indoor")
    parser.add_argument("--method", choices=["fft", "dcp"], default="fft")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_samples_dir", default=None)
    parser.add_argument("--csv_out", default="sots_results.csv")
    args = parser.parse_args()

    df = evaluate(args.root, args.subset, args.method, args.limit, args.save_samples_dir)
    if len(df):
        df.to_csv(args.csv_out, index=False)
        print(f"\nSaved detailed results to {args.csv_out}")
