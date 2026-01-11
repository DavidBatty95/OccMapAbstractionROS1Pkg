#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GPU-accelerated radiation field precomputation (leaded-concrete walls: near-perfect attenuation).

Design intent:
- Walls behave like **leaded concrete shielding**: rays that cross walls die hard (near-zero).
- Air/source attenuation is **reduced** so strong sources can “throw beams” through openings/doorways.
- Scatter is **kept low** so you still see beam-like structure rather than everything filling in.

Outputs:
  - radiation_field.npy
  - radiation_field.meta.yaml

Requires:
  - CuPy (CUDA 12.x) + cupyx.scipy.ndimage for gaussian_filter on GPU
  - CPU fallback exists but will be slow
"""

import time
import yaml
import rospkg
from tqdm import tqdm

# ----------------------------
# Backend selection
# ----------------------------
try:
    import cupy as xp
    GPU_ENABLED = True
except Exception:
    import numpy as xp
    GPU_ENABLED = False

import numpy as np  # saving + CPU masks

# ----------------------------
# Map loading
# ----------------------------
def load_map(pkg, map_name):
    base = rospkg.RosPack().get_path(pkg)
    mdir = f"{base}/maps/{map_name}"

    with open(f"{mdir}/{map_name}.yaml") as f:
        cfg = yaml.safe_load(f)

    import cv2
    img = cv2.imread(f"{mdir}/{cfg['image']}", cv2.IMREAD_GRAYSCALE)
    img = np.flipud(img)

    gt = np.full(img.shape, -1, np.int8)
    gt[img < 50] = 100     # occupied
    gt[img > 200] = 0      # free
    # unknown stays -1

    return gt, cfg, mdir

# ----------------------------
# Gaussian blur (GPU/CPU)
# ----------------------------
def gaussian_blur(field, sigma):
    if sigma <= 0:
        return field
    if GPU_ENABLED:
        from cupyx.scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma)
    else:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(field, sigma)

# ----------------------------
# GPU wall-count via batched ray sampling
# ----------------------------
def wall_count_gpu_sampled(
    wall_gpu_flat, w, h,
    sxg, syg,
    X_chunk, Y_chunk,
    n_samples=96
):
    """
    Approximate "wall thickness along ray" by sampling occupancy along the ray at fixed steps.
    Returns wall_count for the chunk as float32 in [0..n_samples].
    """
    t = xp.linspace(0.0, 1.0, int(n_samples), dtype=xp.float32)
    t = xp.maximum(t, xp.float32(1.0 / max(int(n_samples), 1)))  # avoid t=0

    dx = (X_chunk - xp.float32(sxg))
    dy = (Y_chunk - xp.float32(syg))

    xs = xp.rint(xp.float32(sxg) + t[:, None, None] * dx[None, :, :]).astype(xp.int32)
    ys = xp.rint(xp.float32(syg) + t[:, None, None] * dy[None, :, :]).astype(xp.int32)

    xs = xp.clip(xs, 0, w - 1)
    ys = xp.clip(ys, 0, h - 1)

    idx = ys * w + xs
    occ = wall_gpu_flat[idx]  # 0/1
    wc = occ.sum(axis=0, dtype=xp.int32).astype(xp.float32)
    return wc

# ----------------------------
# Main computation
# ----------------------------
def compute_radiation(
    gt, cfg, sources,
    # --- AIR / SOURCE ATTENUATION (reduced) ---
    mu_air=0.02,                 # 1/m : lower -> beams travel further in free space
    r_soft=0.15,                 # m   : smaller -> hotter near source, more “beam punch”

    # --- WALL ATTENUATION (very strong: leaded concrete proxy) ---
    mu_wall=40.0,                # 1/m : big -> near-perfect blocking once walls are crossed
    wall_cell_thickness_scale=2.2,  # scales (resolution) -> makes walls effectively thicker
    n_ray_samples=512,           # more samples -> fewer “leaks” through diagonal/voxelization
    row_chunk=512,               # adjust for VRAM; 128–256 typical

    # --- VISUAL REALISM / BEAMS ---
    scatter_fraction=0.1,       # keep low -> less “fill in” behind shielding (beam-y)
    scatter_sigma=3.0,           # px  : wide-ish scatter blur but tiny fraction
    final_smoothing_sigma=0.9,   # px  : light smoothing (preserves beams)

    # Optional cutoff (leave None for max beams)
    max_range_m=None
):
    h, w = gt.shape
    res = float(cfg["resolution"])
    origin = cfg["origin"]

    # Define free space strictly as gt==0; unknown treated as blocked (more conservative)
    free_mask_cpu = (gt == 0)
    wall_mask_cpu = (gt == 100)

    rad = xp.zeros((h, w), dtype=xp.float32)

    if GPU_ENABLED:
        wall_gpu_flat = xp.asarray(wall_mask_cpu.astype(np.int8)).ravel()  # 0/1
        free_gpu = xp.asarray(free_mask_cpu.astype(np.float32))

        # Precompute X grid (avoid meshgrid allocations)
        X_full = xp.arange(w, dtype=xp.float32)[None, :].repeat(h, axis=0)
    else:
        raise RuntimeError("This script is tuned for CuPy GPU mode; install CuPy or reduce settings for CPU.")

    # Progress weights (walls dominate)
    W_SRC = 10
    W_SCAT = 2
    W_SMO  = 1
    total_steps = len(sources) * W_SRC + W_SCAT + W_SMO

    pbar = tqdm(
        total=total_steps,
        ncols=112,
        smoothing=0.1,
        bar_format="{desc:<32} {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    )

    try:
        for i, s in enumerate(sources):
            sx, sy = s["position"]
            strength = float(s["strength"])

            sxg = int((sx - origin[0]) / res)
            syg = int((sy - origin[1]) / res)

            pbar.set_description(f"Src {i+1}/{len(sources)}: rays+shield")

            if not (0 <= sxg < w and 0 <= syg < h):
                pbar.update(W_SRC)
                continue

            for y0 in range(0, h, int(row_chunk)):
                y1 = min(h, y0 + int(row_chunk))

                Y_chunk = xp.arange(y0, y1, dtype=xp.float32)[:, None].repeat(w, axis=1)
                X_chunk = X_full[y0:y1, :]

                dx = (X_chunk - xp.float32(sxg)) * res
                dy = (Y_chunk - xp.float32(syg)) * res
                r2 = dx*dx + dy*dy
                r  = xp.sqrt(r2 + 1e-12)

                # Base inverse-square with small softening -> higher contrast beams
                base = strength / (r2 + (r_soft * r_soft))

                # Lower air attenuation -> beams persist in corridors / openings
                contrib = base * xp.exp(-mu_air * r)

                # Optional range cutoff
                if max_range_m is not None:
                    contrib *= (r <= float(max_range_m)).astype(xp.float32)

                # Wall “thickness” along ray via sampled occupancy
                wc = wall_count_gpu_sampled(
                    wall_gpu_flat, w, h,
                    sxg, syg,
                    X_chunk, Y_chunk,
                    n_samples=int(n_ray_samples)
                )

                # Convert sampled hits to effective thickness in metres
                thickness_m = wc * (res * float(wall_cell_thickness_scale))

                # Strong shielding (leaded concrete proxy)
                shield = xp.exp(-mu_wall * thickness_m)

                contrib *= shield

                # Only keep in free space
                contrib *= free_gpu[y0:y1, :]

                rad[y0:y1, :] += contrib.astype(xp.float32)

            pbar.update(W_SRC)

        # Scatter (tiny) to avoid totally “dead” visuals without ruining shielding
        pbar.set_description("Scatter build-up (tiny)")
        if scatter_fraction > 0:
            scattered = gaussian_blur(rad, float(scatter_sigma))
            rad = (1.0 - float(scatter_fraction)) * rad + float(scatter_fraction) * scattered
        pbar.update(W_SCAT)

        # Light final smoothing (preserve beams)
        pbar.set_description("Final smoothing (light)")
        rad = gaussian_blur(rad, float(final_smoothing_sigma))
        pbar.update(W_SMO)

        # Hard zero outside free
        rad *= free_gpu

        pbar.set_description("Done")
        return rad.astype(xp.float32)

    finally:
        pbar.close()

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    PKG = "exploration_pkg"
    MAP = "hut_expanded"

    gt, cfg, mdir = load_map(PKG, MAP)

    with open(f"{mdir}/radiation_sources.yaml") as f:
        sources = yaml.safe_load(f)["sources"]

    print(f"[INFO] Backend: {'GPU (CuPy)' if GPU_ENABLED else 'CPU (NumPy)'}")
    print(f"[INFO] Sources: {len(sources)}")

    # Tuned for: near-perfect shielding + long-ish free-space beams
    params = dict(
        mu_air=0.03,
        r_soft=0.20,

        mu_wall=55.0,
        wall_cell_thickness_scale=2.2,
        n_ray_samples=128,
        row_chunk=160,

        scatter_fraction=0.03,
        scatter_sigma=3.0,
        final_smoothing_sigma=0.8,

        max_range_m=None,
    )

    t0 = time.time()
    rad = compute_radiation(gt, cfg, sources, **params)
    dt = time.time() - t0

    rad_cpu = xp.asnumpy(rad)  # GPU mode by design

    out_npy = f"{mdir}/radiation_field.npy"
    np.save(out_npy, rad_cpu)

    meta = {
        "units": "relative_dose_rate",
        "intent": "leaded_concrete_near_perfect_walls_low_air_atten_beams",
        "model": "soft_inv_square * exp(-mu_air*r) * exp(-mu_wall*thickness_samples) + tiny_scatter",
        "max_value": float(rad_cpu.max()),
        "min_value": float(rad_cpu.min()),
        "source_count": int(len(sources)),
        "gpu_used": True,
        "runtime_sec": float(dt),
        **{
            "mu_air": float(params["mu_air"]),
            "r_soft": float(params["r_soft"]),
            "mu_wall": float(params["mu_wall"]),
            "wall_cell_thickness_scale": float(params["wall_cell_thickness_scale"]),
            "n_ray_samples": int(params["n_ray_samples"]),
            "row_chunk": int(params["row_chunk"]),
            "scatter_fraction": float(params["scatter_fraction"]),
            "scatter_sigma": float(params["scatter_sigma"]),
            "final_smoothing_sigma": float(params["final_smoothing_sigma"]),
            "max_range_m": params["max_range_m"],
        }
    }

    with open(f"{mdir}/radiation_field.meta.yaml", "w") as f:
        yaml.dump(meta, f)

    print(f"[OK] Saved {out_npy}")
    print(f"[INFO] Runtime: {dt:.2f}s")