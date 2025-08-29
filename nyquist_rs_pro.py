#!/usr/bin/env python3
"""
nyquist_rs_pro_v2.py  —  robust Rs from EIS (auto sign, multi-estimator, circle fit)

What it does
------------
• Lets you pick an EIS CSV/TSV/TXT via a file dialog (tkinter).
• Detects columns for Z' (real), Z'' (imag), Frequency (Hz), Phase (deg) flexibly.
• Auto-detects the sign/orientation so the Nyquist shows capacitive arc in +Y.
• Removes obvious HF inductive hook points.
• Estimates Rs with several independent methods:
    1) Leftmost capacitive Z'
    2) High-frequency (top-quantile) median Z'
    3) Small-phase median Z'  (|θ| ≤ 2→5→10→15°)
    4) Small-phase linear x-intercept (robust Huber regression of -Z'' vs Z')
    5) Taubin circle fit on the HF arc → left x-intercept
• Combines them via robust median (+ MAD) and reports a partial-comp value (default 80%).
• Saves a Nyquist plot with all candidates marked and the final Rs.

Dependencies: numpy, pandas, matplotlib, tkinter (std), no external EIS libs.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional, Tuple

# ---------------- User-tunable knobs ----------------
PARTIAL_FRACTION = 0.80        # 0.7–0.9 typical for partial iR comp
HF_TOP_FRACTION  = 0.25        # use top 25% highest frequencies for HF median / circle fit
PHASE_WINDOWS    = (2, 5, 10, 15)   # widen if too few points at very small |θ|
SMALL_PHASE_FOR_LINEAR = 7     # degrees for robust linear x-intercept
IQR_TRIM = 0.05                # trim 5% tails when taking medians
# ----------------------------------------------------

# Common column name variants seen across vendors
REAL_CANDS  = ["Z Real","Z'","Zre","Re(Z)","Z_real","ReZ","Z_Re","Zreal","Real(Z)"]
IMAG_CANDS  = ["Z Imaginary","Z''","Zim","Im(Z)","Z_imag","ImZ","Z_Im","Zimag","Imag(Z)","-Z''","-Zimag"]
FREQ_CANDS  = ["Frequency","Freq","f","omega","Ω","FREQ","Frequency (Hz)","Hz"]
PHASE_CANDS = ["Phase","Phase (deg)","Z (theta)","θ","Theta","Phase_Z","Phase(Z) (deg)","Z theta","Z_theta"]

# --------- helpers: column finding & cleaning ---------
def find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    low = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace(" ","").replace("_","")
        for k, orig in low.items():
            kk = k.replace(" ","").replace("_","")
            if key in kk:
                return orig
    return None

def iqr_mask(x: np.ndarray, lo=IQR_TRIM, hi=1.0-IQR_TRIM) -> np.ndarray:
    """Keep central quantile region; drop extreme tails."""
    x = np.asarray(x, float)
    qlo, qhi = np.nanquantile(x, [lo, hi])
    return (x >= qlo) & (x <= qhi)

# --------- sign/orientation & preprocess --------------
def orient_capacitive(zre, zim, freq=None):
    """
    Return (x, y) where y is the capacitive Nyquist ordinate (should be >=0 on the main arc).
    Many files store Z'' negative already; we'll choose y = -Zim or y = +Zim based on the bulk sign.
    """
    zre = np.asarray(zre, float); zim = np.asarray(zim, float)
    if freq is not None and np.isfinite(freq).sum() >= 10:
        f = np.asarray(freq, float)
        # mid-frequency band (20–80%) to avoid ends
        m = np.isfinite(f) & np.isfinite(zim)
        if m.sum() >= 10:
            order = np.argsort(f)
            lo = order[int(0.2*m.sum())] if m.sum() else 0
            hi = order[int(0.8*m.sum())] if m.sum() else -1
    # decide based on median sign of zim (capacitive arc should give +y after transform)
    if np.nanmedian(zim) < 0:
        y = -zim
    else:
        y = +zim
    return zre, y

def remove_inductive_hook(zre, ycap, freq):
    """Drop obvious inductive points among the highest frequencies (ycap < 0)."""
    if freq is None:
        # still drop any ycap<0 (rare after orientation) as clear inductive artefacts
        keep = ycap >= 0
        return zre[keep], ycap[keep], None
    f = np.asarray(freq, float)
    m = np.isfinite(f) & np.isfinite(zre) & np.isfinite(ycap)
    if m.sum() < 5:
        return zre[m], ycap[m], f[m] if freq is not None else None
    order = np.argsort(f)[::-1]
    top = order[:max(5, int(0.05 * m.sum()))]
    drop = (ycap[top] < 0)
    keep = np.ones_like(m, bool)
    keep[top[drop]] = False
    keep &= m
    return zre[keep], ycap[keep], f[keep]

# -------------- estimators for Rs ---------------------
def rs_leftmost_capacitive(x, y):
    m = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    return float(np.nanmin(x[m])) if m.sum() else np.nan

def rs_hf_median(x, f):
    if f is None: return np.nan
    x = np.asarray(x, float); f = np.asarray(f, float)
    m = np.isfinite(x) & np.isfinite(f)
    if m.sum() < 5: return np.nan
    order = np.argsort(f)[::-1]
    k = max(5, int(HF_TOP_FRACTION * m.sum()))
    vals = x[order[:k]]
    sel = iqr_mask(vals)
    return float(np.nanmedian(vals[sel])) if sel.sum() else float(np.nanmedian(vals))

def rs_small_phase_median(x, y, theta_deg=None):
    if theta_deg is None:
        # infer "phase" as atan2(y, x) in degrees (OK; equals Z phase on Nyquist)
        theta_deg = np.degrees(np.arctan2(y, x))
    x = np.asarray(x, float); th = np.asarray(theta_deg, float)
    m = np.isfinite(x) & np.isfinite(th)
    x, th = x[m], th[m]
    for deg in PHASE_WINDOWS:
        sel = np.abs(th) <= deg
        if sel.sum() >= 5:
            vals = x[sel]
            s = iqr_mask(vals)
            vals = vals[s] if s.sum() else vals
            return float(np.nanmedian(vals)), deg
    return np.nan, 0

def rs_small_phase_linear_xint(x, y, theta_deg=None, max_deg=SMALL_PHASE_FOR_LINEAR):
    if theta_deg is None:
        theta_deg = np.degrees(np.arctan2(y, x))
    x = np.asarray(x, float); y = np.asarray(y, float); th = np.asarray(theta_deg, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(th) & (np.abs(th) <= max_deg)
    if m.sum() < 6: return np.nan
    xr, yr = x[m], y[m]
    # robust Huber reweighting
    w = np.ones_like(xr)
    for _ in range(5):
        A = np.vstack([xr, np.ones_like(xr)]).T
        sol, *_ = np.linalg.lstsq(A * w[:,None], yr * w, rcond=None)
        a, b = sol
        resid = yr - (a*xr + b)
        scale = 1.4826 * np.median(np.abs(resid)) + 1e-12
        t = resid / (1.345*scale)
        w = 1.0 / np.maximum(1.0, np.abs(t))
    if abs(a) < 1e-12: return np.nan
    return float(-b / a)

def rs_circle_fit_taubin(x, y, f=None):
    """
    Taubin circle fit on (x,y) points likely from HF arc, then return left x-intercept.
    If freq is present, use top HF_TOP_FRACTION frequencies; else pick small-phase & y>0.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    if f is not None and np.isfinite(f).sum() >= 10:
        order = np.argsort(f)[::-1]
        k = max(10, int(HF_TOP_FRACTION * np.isfinite(f).sum()))
        idx = order[:k]
    else:
        # fallback: take points with small phase (<45°) and y>0 as "HF side" of arc
        th = np.degrees(np.arctan2(y, x))
        idx = np.where((np.abs(th) <= 45) & (y >= 0))[0]
        if idx.size < 10:
            return np.nan
    xc, yc, R = _taubin_circle(x[idx], y[idx])
    if not np.isfinite(xc) or not np.isfinite(yc) or not np.isfinite(R):
        return np.nan
    disc = R*R - yc*yc
    if disc <= 0: return np.nan
    x_left  = xc - np.sqrt(disc)
    x_right = xc + np.sqrt(disc)
    # Rs is the left intercept of the arc
    return float(x_left)

def _taubin_circle(x, y):
    """Algebraic circle fit (Taubin). Returns xc, yc, R."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    x = x - np.mean(x); y = y - np.mean(y)
    z = x*x + y*y
    Z = np.vstack([z, x, y, np.ones_like(x)]).T
    # Build scatter matrix
    M = Z.T @ Z
    # Constraint matrix (Taubin)
    C = np.zeros((4,4))
    C[0,0] = 0; C[0,1] = C[1,0] = 0; C[0,2] = C[2,0] = 0; C[0,3] = 2
    C[1,1] = 1; C[2,2] = 1; C[3,3] = 0
    # Solve generalized eig problem M v = lambda C v
    try:
        w, V = np.linalg.eig(np.linalg.pinv(M) @ C)
        v = V[:, np.argmax(np.real(w))]
    except Exception:
        return np.nan, np.nan, np.nan
    A, B, Cc, D = v  # z, x, y, 1 coefficients
    if abs(A) < 1e-12:
        return np.nan, np.nan, np.nan
    xc = -B / (2*A)
    yc = -Cc / (2*A)
    R  = np.sqrt((B*B + Cc*Cc)/(4*A*A) - D/A)
    # restore mean shift
    xc += np.mean(x)
    yc += np.mean(y)
    return float(xc), float(yc), float(R)

# -------------------------- main --------------------------
def main():
    root = tk.Tk(); root.withdraw()
    fpath = filedialog.askopenfilename(
        title="Select EIS CSV/TSV/TXT",
        filetypes=[("CSV/TSV/TXT","*.csv *.tsv *.txt"), ("All files","*.*")]
    )
    if not fpath:
        print("No file selected."); return
    fpath = Path(fpath)

    # read (try CSV then TSV)
    try:
        df = pd.read_csv(fpath)
        if df.shape[1] == 1:
            df = pd.read_csv(fpath, sep="\t")
    except Exception as e:
        messagebox.showerror("Read Error", f"Could not read file:\n{e}")
        return

    col_re = find_col(df, REAL_CANDS)
    col_im = find_col(df, IMAG_CANDS)
    col_f  = find_col(df, FREQ_CANDS)
    col_th = find_col(df, PHASE_CANDS)

    if not col_re or not col_im:
        messagebox.showerror("Column Error", f"Need real/imag columns. Found: {list(df.columns)}")
        return

    zre   = pd.to_numeric(df[col_re], errors="coerce").to_numpy()
    zim   = pd.to_numeric(df[col_im], errors="coerce").to_numpy()
    freq  = pd.to_numeric(df[col_f],  errors="coerce").to_numpy() if col_f else None
    theta = pd.to_numeric(df[col_th], errors="coerce").to_numpy() if col_th else None

    # keep aligned, drop NaNs
    m = np.isfinite(zre) & np.isfinite(zim)
    if freq is not None: m &= np.isfinite(freq)
    if theta is not None: pass  # theta NaNs allowed
    zre, zim = zre[m], zim[m]
    freq = (freq[m] if freq is not None else None)
    theta = (theta[m] if theta is not None else None)

    # orient so capacitive arc is y >= 0
    x, y = orient_capacitive(zre, zim, freq=freq)

    # remove inductive hook (HF)
    x, y, freq = remove_inductive_hook(x, y, freq)

    # candidate Rs estimates
    cand = {}

    cand["leftmost"] = rs_leftmost_capacitive(x, y)
    cand["hf_median"] = rs_hf_median(x, freq)

    rs_phase_med, used_deg = rs_small_phase_median(x, y, theta_deg=theta)
    cand["phase_median"] = rs_phase_med

    cand["phase_linear"] = rs_small_phase_linear_xint(x, y, theta_deg=theta)

    cand["circle_fit"] = rs_circle_fit_taubin(x, y, f=freq)

    # combine robustly
    arr = np.array([v for v in cand.values() if np.isfinite(v)], float)
    if arr.size == 0:
        messagebox.showerror("Estimation Error", "Could not compute any Rs candidates.")
        return

    # trim extremes, take robust median and MAD
    keep = iqr_mask(arr, 0.1, 0.9) if arr.size >= 6 else np.ones_like(arr, bool)
    core = arr[keep]
    Rs_final = float(np.nanmedian(core))
    MAD = 1.4826 * np.nanmedian(np.abs(core - Rs_final)) if core.size else np.nan

    Rs_partial = PARTIAL_FRACTION * Rs_final

    # disagreement warning
    spread = np.nanmax(arr) - np.nanmin(arr) if arr.size else np.nan
    flag = (spread > max(0.05*Rs_final, 3*MAD)) if np.isfinite(MAD) else (spread > 0.1*Rs_final)

    # ----- plot -----
    plt.figure(figsize=(8,5.5), dpi=140)
    plt.plot(x, y, '.', ms=2.5, label="EIS data")
    colors = {"leftmost":"#6666cc","hf_median":"#66aa66","phase_median":"#cc6666",
              "phase_linear":"#aa66cc","circle_fit":"#ccaa66"}
    for k, v in cand.items():
        if np.isfinite(v):
            plt.axvline(v, ls='--', lw=1.0, color=colors.get(k,"0.5"), label=f"{k.replace('_',' ')}: {v:.3g} Ω")
    plt.axvline(Rs_final,  ls='-',  lw=1.8, color="k", label=f"Final Rs (robust): {Rs_final:.3g} Ω")
    plt.axvline(Rs_partial, ls=':',  lw=1.8, color="k", label=f"{int(PARTIAL_FRACTION*100)}% Rs: {Rs_partial:.3g} Ω")
    plt.xlabel("Z' (Ω)")
    plt.ylabel("-Z'' (Ω)  (capacitive)")
    plt.title("Nyquist with multi-method Rs estimates")
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    out_png = fpath.with_suffix(".nyquist_rs_v2.png")
    plt.savefig(out_png, dpi=180)
    try: plt.show()
    except Exception: pass

    # ----- report -----
    print("\n=== Rs candidates (Ω) ===")
    for k,v in cand.items():
        print(f"{k:>13}: {v:.6g}")
    print(f"\nFinal Rs (robust median) : {Rs_final:.6g} Ω")
    if np.isfinite(MAD):
        print(f"Uncertainty (±MAD)       : ±{MAD:.6g} Ω")
    print(f"Recommended {int(PARTIAL_FRACTION*100)}% Rs  : {Rs_partial:.6g} Ω")
    if flag:
        print("\n[!] Note: Candidate methods disagree noticeably. This often means:")
        print("    • diffusion/Warburg tail encroaching on HF arc")
        print("    • film/adsorption (two-time-constant behavior)")
        print("    • limited HF points or heavy analog filtering")
        print("  Use the HF-median / small-phase results and consider shortening the frequency span.")
    print(f"(Saved plot: {out_png.name})")

if __name__ == "__main__":
    main()
