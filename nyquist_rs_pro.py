#!/usr/bin/env python3
"""
nyquist_rs_pro.py (fixed weights)
---------------------------------
Pick an EIS CSV/TSV via a file dialog, plot a Nyquist diagram (Z' vs -Z''),
and estimate Rs robustly.

Model (CNLS):
  Z(ω) = Rs + jωL + (Rct || CPE(Q, α)) + Zw(σ)

Fix in this version:
- Weight vector is always expanded to length 2N to match [Re, Im, Re, Im, ...] residuals,
  preventing broadcasting errors like shapes (2N,) vs (N,).

Requires: numpy, pandas, matplotlib, scipy, tkinter
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.optimize import least_squares

# -------------------- Column name heuristics --------------------
REAL_CANDIDATES = ["Z Real", "Z'", "Zre", "Re(Z)", "Zreal", "Real", "Z_real", "ReZ", "Z_Re"]
IMAG_CANDIDATES = ["Z Imaginary", "Z''", "Zim", "Im(Z)", "Zimag", "Imag", "Z_imag", "ImZ", "Z_Im"]
FREQ_CANDIDATES = ["Frequency", "Freq", "f", "frequency", "FREQ", "omega", "Ω"]  # Hz preferred

def find_column(df: pd.DataFrame, candidates) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for c in df.columns:
            if c.strip().lower() == cand.strip().lower():
                return c
        key = cand.strip().lower().replace(" ", "").replace("_", "")
        for low, orig in cols_lower.items():
            low2 = low.replace(" ", "").replace("_", "")
            if key in low2:
                return orig
    return None

# -------------------- Quick estimators --------------------
def estimate_rs_minreal(zre: np.ndarray, zim: np.ndarray) -> float:
    y = -zim
    m = np.isfinite(zre) & np.isfinite(y)
    zre, y = zre[m], y[m]
    cap = y >= 0
    if not np.any(cap):
        cap = np.isfinite(y)
    zre_cap = zre[cap]
    return float(np.nanmin(zre_cap)) if zre_cap.size else np.nan

def kasa_circle_fit_rs(zre: np.ndarray, zim: np.ndarray) -> float:
    x = np.asarray(zre, float)
    y = -np.asarray(zim, float)
    m = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    x, y = x[m], y[m]
    if x.size < 5:
        return np.nan
    X = np.column_stack([x, y, np.ones_like(x)])
    b = -(x**2 + y**2)
    try:
        A, B, C = np.linalg.lstsq(X, b, rcond=None)[0]
        xc = -A / 2.0
        R = np.sqrt(max((A*A + B*B)/4.0 - C, 0.0))
        return float(xc - R)
    except Exception:
        return np.nan

def hf_median_rs(zre: np.ndarray, freq: Optional[np.ndarray]) -> float:
    if freq is None:
        return np.nan
    f = np.asarray(freq, float)
    zre = np.asarray(zre, float)
    m = np.isfinite(f) & np.isfinite(zre)
    f, zre = f[m], zre[m]
    if f.size < 5:
        return np.nan
    order = np.argsort(f)[::-1]
    n = max(5, int(0.05 * f.size))
    return float(np.nanmedian(zre[order[:n]]))

# -------------------- Elements & Randles-like model --------------------
def Z_cpe(omega: np.ndarray, Q: float, alpha: float) -> np.ndarray:
    jw = 1j * omega
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_abs = np.log(np.maximum(np.abs(jw), 1e-300))
        ang = np.angle(jw)  # ≈ +π/2 for ω>0
        jw_alpha = np.exp(alpha * (ln_abs + 1j * ang))
        Y = Q * jw_alpha
        Z = np.empty_like(jw, dtype=complex)
        nz = np.abs(Y) > 0
        Z[nz] = 1.0 / Y[nz]
        Z[~nz] = np.inf + 0j
    return Z

def Z_warburg_si(omega: np.ndarray, sigma: float) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        mag = np.sqrt(np.maximum(omega, 0.0))
        factor = (1.0 - 1.0j) / np.sqrt(2.0)
        Zw = sigma * factor / mag
        Zw[~np.isfinite(Zw)] = np.inf + 0j
    return Zw

def Z_randles(omega: np.ndarray, Rs: float, Rct: float, Q: float, alpha: float, sigma: float, L: float) -> np.ndarray:
    jwL = 1j * omega * L
    Zcpe = Z_cpe(omega, Q, alpha)
    with np.errstate(divide='ignore', invalid='ignore'):
        Ypar = (1.0 / max(Rct, 1e-30)) + 1.0 / Zcpe
        Zpar = 1.0 / Ypar
    Zw = Z_warburg_si(omega, sigma) if sigma > 0 else 0.0
    return float(Rs) + jwL + Zpar + Zw

# -------------------- Robust CNLS fitting --------------------
@dataclass
class FitResult:
    params: Dict[str, float]
    success: bool
    cost: float
    message: str

def pack_params(pdict: Dict[str, float]) -> np.ndarray:
    return np.array([pdict[k] for k in ["Rs", "Rct", "Q", "alpha", "sigma", "L"]], dtype=float)

def unpack_params(p: np.ndarray) -> Dict[str, float]:
    keys = ["Rs", "Rct", "Q", "alpha", "sigma", "L"]
    return {k: float(v) for k, v in zip(keys, p)}

def bounds():
    lower = [0.0, 0.0, 1e-10, 0.2, 0.0, 0.0]   # Rs, Rct, Q, alpha, sigma, L
    upper = [1e9, 1e12, 1.0,   1.0, 1e9, 1e3]
    return (np.array(lower, float), np.array(upper, float))

def huber_weights(res_complex: np.ndarray, delta: float = 1.5) -> np.ndarray:
    """
    res_complex: complex residuals per frequency point (length N).
    Returns a weight vector of length 2N (for Re/Im stacking).
    """
    r = np.abs(res_complex)              # length N
    w = np.ones_like(r)
    mask = r > delta
    w[mask] = delta / r[mask]
    return np.repeat(w, 2)               # expand to length 2N

def residuals(p: np.ndarray, omega: np.ndarray, Z_meas: np.ndarray, w2: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Returns stacked residuals [Re, Im, Re, Im, ...] of length 2N.
    If w2 is provided, it MUST be length 2N (weights for Re and Im separately).
    """
    d = unpack_params(p)
    Zm = Z_randles(omega, **d)
    diff = Z_meas - Zm
    res = np.empty(Z_meas.size * 2, dtype=float)
    res[0::2] = diff.real
    res[1::2] = diff.imag
    if w2 is not None:
        if w2.shape[0] != res.shape[0]:
            raise ValueError(f"Weight length mismatch: residuals {res.shape[0]} vs weights {w2.shape[0]}")
        res *= w2
    return res

def fit_cnls(omega: np.ndarray, Z_meas: np.ndarray, p0: Dict[str, float]) -> FitResult:
    N = Z_meas.size
    p_init = pack_params(p0)
    lb, ub = bounds()

    # Base magnitude weights 1/|Z|, expanded to 2N
    mag = np.abs(Z_meas)
    mag[~np.isfinite(mag)] = 1.0
    base_w = 1.0 / np.maximum(mag, 1e-9)        # length N
    w2 = np.repeat(base_w, 2)                    # length 2N (Re/Im)

    p = p_init.copy()
    info_msg = ""
    for _ in range(4):  # a few IRLS passes
        fun = lambda prm: residuals(prm, omega, Z_meas, w2=w2)
        res_lsq = least_squares(fun, p, bounds=(lb, ub), method="trf", max_nfev=5000)
        p = res_lsq.x

        # Build complex residuals per point to compute robust weights
        d = unpack_params(p)
        Zm = Z_randles(omega, **d)
        res_c = (Z_meas - Zm)                    # length N, complex
        W_huber = huber_weights(res_c, delta=1.5)  # length 2N
        w2 = w2 * W_huber                        # elementwise; stays length 2N
        info_msg = res_lsq.message
        if not res_lsq.success:
            break

    return FitResult(params=unpack_params(p), success=res_lsq.success, cost=res_lsq.cost, message=info_msg)

# -------------------- Preprocessing & masking --------------------
def preprocess(zre: np.ndarray, zim: np.ndarray, freq: Optional[np.ndarray] = None):
    zre = np.asarray(zre, float)
    zim = np.asarray(zim, float)
    mask = np.isfinite(zre) & np.isfinite(zim)
    zre, zim = zre[mask], zim[mask]

    if freq is not None:
        freq = np.asarray(freq, float)[mask]
        y = -zim
        order = np.argsort(freq)[::-1]
        n = max(5, int(0.05 * freq.size))
        top = order[:n]
        inductive = (y[top] < 0)
        if np.any(inductive):
            keep = np.ones_like(zre, dtype=bool)
            keep[top[inductive]] = False
            zre, zim, freq = zre[keep], zim[keep], freq[keep]
        return zre, zim, freq
    return zre, zim, None

# -------------------- Bootstrap CI for Rs --------------------
def bootstrap_ci(omega: np.ndarray, Z: np.ndarray, p_hat: Dict[str, float],
                 n_boot: int = 150, seed: int = 42):
    rng = np.random.default_rng(seed)
    N = omega.size
    Rs_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        om_b = omega[idx]
        Z_b = Z[idx]
        try:
            fr = fit_cnls(om_b, Z_b, p_hat)
            Rs_samples.append(fr.params["Rs"] if fr.success else np.nan)
        except Exception:
            Rs_samples.append(np.nan)
    arr = np.array(Rs_samples, float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return np.nan, np.nan
    lo = np.percentile(arr, 2.5)
    hi = np.percentile(arr, 97.5)
    return lo, hi

# -------------------- Main GUI flow --------------------
def main():
    # File dialog
    root = tk.Tk()
    root.withdraw()
    fpath = filedialog.askopenfilename(
        title="Select EIS data file (CSV/TSV)",
        filetypes=[("CSV/TSV/TXT", "*.csv *.tsv *.txt"), ("All files", "*.*")]
    )
    if not fpath:
        print("No file selected.")
        return

    # Read
    try:
        df = pd.read_csv(fpath)
        if df.shape[1] == 1:
            df = pd.read_csv(fpath, sep="\t")
    except Exception as e:
        messagebox.showerror("Read Error", f"Could not read file:\n{e}")
        return

    # Identify columns
    col_re = find_column(df, REAL_CANDIDATES)
    col_im = find_column(df, IMAG_CANDIDATES)
    col_f  = find_column(df, FREQ_CANDIDATES)

    if col_re is None or col_im is None:
        messagebox.showerror("Column Error", f"Could not find impedance columns.\nAvailable: {df.columns.tolist()}")
        return

    zre = pd.to_numeric(df[col_re], errors="coerce").to_numpy()
    zim = pd.to_numeric(df[col_im], errors="coerce").to_numpy()
    freq = pd.to_numeric(df[col_f], errors="coerce").to_numpy() if col_f is not None else None

    zre, zim, freq = preprocess(zre, zim, freq)

    if zre.size < 8:
        messagebox.showerror("Data Error", "Not enough valid impedance points after cleaning/masking.")
        return

    # Quick estimates (guardrails)
    rs_min = estimate_rs_minreal(zre, zim)
    rs_kasa = kasa_circle_fit_rs(zre, zim)
    rs_hf = hf_median_rs(zre, freq) if freq is not None else np.nan

    # Frequencies (rad/s) for model fit
    if freq is None:
        order = np.argsort(np.abs(zim))[::-1]
        omega = np.linspace(1.0, 1e3, zre.size)
        omega = omega[np.argsort(order)]
    else:
        omega = 2 * np.pi * np.maximum(freq, 1e-12)

    Z_meas = zre + 1j * zim

    # Initial guess
    candidates = [v for v in (rs_hf, rs_min, rs_kasa) if np.isfinite(v)]
    Rs0 = float(np.nanmedian(candidates)) if candidates else max(1.0, np.nanmin(zre[np.isfinite(zre)]))
    if not np.isfinite(Rs0) or Rs0 <= 0:
        Rs0 = max(1.0, np.nanmin(zre[np.isfinite(zre)]))
    Rct0 = max(10.0, float(np.nanmax(zre)) - Rs0)
    Q0, alpha0, sigma0, L0 = 1e-4, 0.85, 0.0, 0.0

    p0 = {"Rs": Rs0, "Rct": Rct0, "Q": Q0, "alpha": alpha0, "sigma": sigma0, "L": L0}

    # CNLS fit
    fit = fit_cnls(omega, Z_meas, p0)

    # Second pass enabling Warburg if first pass suppressed it
    if fit.success and fit.params["sigma"] < 1e-8:
        p1 = fit.params.copy()
        p1["sigma"] = max(1e-3, p1["Rct"] / 1000.0)
        fit2 = fit_cnls(omega, Z_meas, p1)
        if fit2.success and fit2.cost <= fit.cost * 0.98:
            fit = fit2

    # Bootstrap CI
    ci_lo, ci_hi = bootstrap_ci(omega, Z_meas, fit.params, n_boot=150, seed=42) if fit.success else (np.nan, np.nan)

    # Final Rs choice with guardrails
    Rs_fit = fit.params["Rs"] if fit.success else np.nan
    Rs_fallback = rs_min if np.isfinite(rs_min) else np.nan
    if not np.isfinite(Rs_fit) or Rs_fit <= 0:
        Rs_final = Rs_fallback
    else:
        if np.isfinite(Rs_fallback) and Rs_fallback > 0 and (Rs_fit > 5.0 * Rs_fallback):
            Rs_final = Rs_fallback
        else:
            Rs_final = Rs_fit

    # -------------------- Plot --------------------
    plt.figure(figsize=(7, 5), dpi=140)
    plt.plot(zre, -zim, '.', ms=3, label="Data")
    if np.isfinite(Rs_final):
        plt.axvline(Rs_final, linestyle='--', linewidth=1.2, label=f"Rs ≈ {Rs_final:.3g} Ω")

    if fit.success:
        pos = omega[omega > 0]
        wmin = float(np.nanmax([np.nanmin(pos) if pos.size else 1e-2, 1e-3]))
        wmax = float(np.nanmax(pos) if pos.size else 1e6)
        if wmax <= wmin:
            wmax = wmin * 10
        w_plot = np.logspace(np.log10(wmin), np.log10(wmax), 500)
        Z_fit = Z_randles(w_plot, **fit.params)
        plt.plot(Z_fit.real, -Z_fit.imag, '-', linewidth=1.3, label="CNLS fit")

    plt.xlabel("Z' (Ω)")
    plt.ylabel("-Z'' (Ω)")
    plt.title("Nyquist Plot (robust CNLS fit)")
    plt.legend()
    plt.tight_layout()

    outplot = Path(fpath).with_suffix(".nyquist_pro.png")
    plt.savefig(outplot, dpi=160)
    try:
        plt.show()
    except Exception:
        pass  # headless environments

    # -------------------- Report --------------------
    print("\n=== Quick estimates ===")
    print(f"Leftmost-real Rs: {rs_min if np.isfinite(rs_min) else np.nan:.6g} Ω")
    print(f"Kasa circle-fit Rs: {rs_kasa if np.isfinite(rs_kasa) else np.nan:.6g} Ω")
    print(f"HF median Rs: {rs_hf if np.isfinite(rs_hf) else np.nan:.6g} Ω")

    print("\n=== CNLS fit (Randles + L + Warburg) ===")
    if fit.success:
        for k, v in fit.params.items():
            print(f"{k:>6s} = {v:.6g}")
        print(f"Fit cost: {fit.cost:.6g}")
        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            print(f"Rs 95% CI (bootstrap): [{ci_lo:.6g}, {ci_hi:.6g}] Ω")
    else:
        print("Fit failed:", fit.message)

    print(f"\nChosen Rs ≈ {Rs_final:.6g} Ω")
    print(f"Saved Nyquist plot: {outplot}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Unhandled error:", e, file=sys.stderr)
        sys.exit(1)
