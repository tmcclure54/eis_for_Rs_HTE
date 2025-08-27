#!/usr/bin/env python3
"""
nyquist_rs_pro.py  (EIS-only, phase-plateau Rs + partial-comp suggestion)
------------------------------------------------------------------------
Pick an EIS CSV/TSV via a file dialog. The script:
  • Reads Z' (real), Z'' (imag), and, if present, Frequency (Hz) and Phase (deg)
  • Removes very-high-frequency inductive “hook” points
  • Estimates Rs from the small-phase resistive plateau (with cross-checks)
  • Saves a Nyquist plot marking BOTH:
      - Full Rs
      - Recommended partial-comp value (default 80% of Rs)

Notes:
  – This script does NOT read/IR-correct CVs; it only outputs Rs numbers & Nyquist.
  – Literature practice commonly uses 70–90% compensation to avoid over-correction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import Optional, Tuple

# ---------- user-tunable ----------
PARTIAL_FRACTION = 0.80  # 0.7–0.9 typical; set your lab’s standard here
HF_FRACTION      = 0.10  # top 10% of highest frequencies for HF median
PHASE_WINDOWS    = (2, 5, 10, 15)  # try |θ|<=2°, then widen if needed
# ----------------------------------

REAL_CANDS  = ["Z Real","Z'","Zre","Re(Z)","Z_real","ReZ","Z_Re"]
IMAG_CANDS  = ["Z Imaginary","Z''","Zim","Im(Z)","Z_imag","ImZ","Z_Im"]
FREQ_CANDS  = ["Frequency","Freq","f","omega","Ω","FREQ"]
PHASE_CANDS = ["Z (theta)","Phase","Phase (deg)","θ","Theta","Z theta","Z_theta","Z theta (deg)"]

def find_col(df: pd.DataFrame, candidates) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        # exact CI match
        for c in df.columns:
            if c.strip().lower() == cand.strip().lower():
                return c
        # relaxed contains (ignore spaces/underscores)
        key = cand.strip().lower().replace(" ","").replace("_","")
        for low, orig in lower.items():
            if key in low.replace(" ","").replace("_",""):
                return orig
    return None

def preprocess(zre, zim, freq=None, theta=None) -> Tuple[np.ndarray,np.ndarray,Optional[np.ndarray],Optional[np.ndarray]]:
    """Keep arrays aligned; drop NaNs; remove inductive points in top 5% highest f."""
    zre = np.asarray(zre, float); zim = np.asarray(zim, float)
    base = np.isfinite(zre) & np.isfinite(zim)
    f = np.asarray(freq, float) if freq is not None else None
    th = np.asarray(theta, float) if theta is not None else None
    if f is not None:  base &= np.isfinite(f)
    # theta NaNs allowed; we’ll mask later where needed
    zre, zim = zre[base], zim[base]
    f   = (f[base] if f is not None else None)
    th  = (th[base] if th is not None else None)

    # remove inductive “hook” among very highest f (first quadrant: -Z''<0)
    if f is not None and zre.size:
        y = -zim
        order = np.argsort(f)[::-1]
        k = max(5, int(0.05 * f.size))
        top = order[:k]
        inductive = (y[top] < 0)
        if np.any(inductive):
            keep = np.ones_like(zre, dtype=bool)
            keep[top[inductive]] = False
            zre, zim, f = zre[keep], zim[keep], f[keep]
            if th is not None: th = th[keep]
    return zre, zim, f, th

def leftmost_capacitive(zre, zim) -> float:
    x = np.asarray(zre, float); y = -np.asarray(zim, float)
    cap = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    if not np.any(cap): cap = np.isfinite(x) & np.isfinite(y)
    return float(np.nanmin(x[cap])) if np.any(cap) else np.nan

def hf_median(zre, freq, frac=HF_FRACTION) -> float:
    if freq is None: return np.nan
    f = np.asarray(freq, float); x = np.asarray(zre, float)
    m = np.isfinite(f) & np.isfinite(x)
    f, x = f[m], x[m]
    if f.size < 5: return np.nan
    order = np.argsort(f)[::-1]
    k = max(5, int(frac * f.size))
    return float(np.nanmedian(x[order[:k]]))

def small_phase_median(zre, theta, levels=PHASE_WINDOWS) -> Tuple[float,int]:
    if theta is None: return np.nan, 0
    x  = np.asarray(zre, float)
    th = np.asarray(theta, float)
    m = np.isfinite(x) & np.isfinite(th)
    x, th = x[m], th[m]
    for deg in levels:
        sel = np.abs(th) <= deg
        if np.sum(sel) >= 5:
            return float(np.nanmedian(x[sel])), int(deg)
    return np.nan, 0

def small_phase_linear_xint(zre, zim, theta, max_deg=7) -> float:
    if theta is None: return np.nan
    x  = np.asarray(zre, float)
    y  = -np.asarray(zim, float)
    th = np.asarray(theta, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(th)
    x, y, th = x[m], y[m], th[m]
    sel = np.abs(th) <= max_deg
    if np.sum(sel) < 3: return np.nan
    xr, yi = x[sel], y[sel]
    A = np.vstack([xr, np.ones_like(xr)]).T
    try:
        a, b = np.linalg.lstsq(A, yi, rcond=None)[0]
        if abs(a) < 1e-9: return np.nan
        return float(-b / a)
    except Exception:
        return np.nan

def main():
    # pick file
    root = tk.Tk(); root.withdraw()
    fpath = filedialog.askopenfilename(
        title="Select EIS CSV/TSV",
        filetypes=[("CSV/TSV/TXT","*.csv *.tsv *.txt"), ("All files","*.*")]
    )
    if not fpath:
        print("No file selected."); return
    fpath = Path(fpath)

    # read
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

    # keep arrays aligned
    zre, zim, freq, theta = preprocess(zre, zim, freq, theta)

    # estimates
    rs_left        = leftmost_capacitive(zre, zim)
    rs_hf          = hf_median(zre, freq)
    rs_phase, win  = small_phase_median(zre, theta)
    rs_lin         = small_phase_linear_xint(zre, zim, theta)

    # combine robustly
    cands = [v for v in [rs_phase, rs_hf, rs_lin] if np.isfinite(v)]
    if not cands:
        cands = [v for v in [rs_left, rs_hf] if np.isfinite(v)]
    rs_raw = float(np.nanmedian(cands)) if cands else float(rs_left)

    # plausibility band from data
    lo = rs_left if np.isfinite(rs_left) else float(np.nanmin(zre))
    if theta is not None and theta.size:
        sel15 = np.isfinite(theta) & (np.abs(theta) <= 15)
        hi = float(np.nanpercentile(zre[sel15], 95)) if np.any(sel15) else float(np.nanpercentile(zre, 95))
    else:
        hi = float(np.nanpercentile(zre, 95))
    rs_final = float(np.clip(rs_raw, lo, hi))

    # partial-comp suggestion
    rs_partial = float(PARTIAL_FRACTION * rs_final)

    # plot
    plt.figure(figsize=(7,5), dpi=140)
    plt.plot(zre, -zim, '.', ms=3, label="EIS data")
    if np.isfinite(rs_final):
        plt.axvline(rs_final,  ls='--', lw=1.2, label=f"Full Rs ≈ {rs_final:.3g} Ω")
        plt.axvline(rs_partial, ls=':',  lw=1.4, label=f"{int(PARTIAL_FRACTION*100)}% Rs ≈ {rs_partial:.3g} Ω")
    plt.xlabel("Z' (Ω)")
    plt.ylabel("-Z'' (Ω)")
    plt.title("Nyquist with Rs (full) and partial-comp suggestion")
    plt.legend()
    plt.tight_layout()
    out = fpath.with_suffix(".nyquist_rs.png")
    plt.savefig(out, dpi=160)
    try: plt.show()
    except Exception: pass

    # report
    print("\n=== Rs estimates (Ω) ===")
    print(f"Leftmost capacitive Z'   : {rs_left:.6g}")
    print(f"HF-median (top {int(HF_FRACTION*100)}% f): {rs_hf if np.isfinite(rs_hf) else np.nan:.6g}")
    print(f"Small-phase median       : {rs_phase if np.isfinite(rs_phase) else np.nan:.6g}"
          + (f"  (|θ|≤{win}°)" if win else ""))
    print(f"Small-phase linear x-int : {rs_lin if np.isfinite(rs_lin) else np.nan:.6g}")
    print(f"\nFull Rs                  : {rs_final:.6g} Ω")
    print(f"Recommended {int(PARTIAL_FRACTION*100)}% Rs : {rs_partial:.6g} Ω")
    print(f"(Saved plot: {out.name})")

if __name__ == "__main__":
    main()
