"""Calibration math: Beer's-law concentration curves and software wavelength recalibration.

Nothing here talks to the spectrometer. A wavelength correction is applied in software on top of
the factory calibration stored in the spectrometer, which is never changed.
"""
from __future__ import annotations

import math

import numpy as np

# Mercury-argon calibration lamp lines (nm, in air): Hg first, then Ar.
HG_AR_LINES = [253.652, 296.728, 313.155, 365.015, 404.656, 435.833, 546.074, 576.960, 579.066,
               696.543, 706.722, 738.398, 750.387, 763.511, 772.376, 811.531, 826.452, 842.465, 912.297]
# Deuterium lamp: the two strong Balmer lines D-beta and D-alpha (nm, in air).
DEUTERIUM_LINES = [486.000, 656.100]

# name -> (reference lines, search half-width in nm)
LAMP_PRESETS = {
    "Hg-Ar lamp": (HG_AR_LINES, 1.5),
    "Deuterium (D2) lamp": (DEUTERIUM_LINES, 2.0),
}


# ---------------------------------------------------------------- concentration curve

def fit_line(conc, absorb, through_zero: bool = False) -> dict:
    """Least-squares A = slope * c + intercept (intercept fixed at 0 if through_zero).

    Returns slope, intercept, r2, n, s_y (standard error of the regression), and the
    detection / quantitation limits LOD = 3.3 s_y / slope and LOQ = 10 s_y / slope.
    """
    c = np.asarray(conc, dtype=float)
    a = np.asarray(absorb, dtype=float)
    n = c.size
    need = 1 if through_zero else 2
    if n < need or (not through_zero and np.ptp(c) == 0):
        raise ValueError(f"Need at least {need} standards with different concentrations.")
    if through_zero:
        denom = float(np.dot(c, c))
        if denom == 0:
            raise ValueError("Standards must not all be zero.")
        slope, intercept = float(np.dot(c, a) / denom), 0.0
        dof = n - 1
    else:
        slope, intercept = (float(v) for v in np.polyfit(c, a, 1))
        dof = n - 2
    resid = a - (slope * c + intercept)
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    s_y = math.sqrt(ss_res / dof) if dof > 0 else float("nan")
    lod = 3.3 * s_y / abs(slope) if slope and math.isfinite(s_y) else float("nan")
    loq = 10.0 * s_y / abs(slope) if slope and math.isfinite(s_y) else float("nan")
    return {"slope": slope, "intercept": intercept, "r2": r2, "n": n, "s_y": s_y, "lod": lod, "loq": loq,
            "through_zero": through_zero, "residuals": resid.tolist(),
            "c_min": float(c.min()), "c_max": float(c.max())}


def predict_conc(fit: dict, absorbance: float) -> float:
    if not fit or fit["slope"] == 0:
        return float("nan")
    return (absorbance - fit["intercept"]) / fit["slope"]


# ---------------------------------------------------------------- wavelength recalibration

def noise_level(y: np.ndarray) -> float:
    """Robust per-pixel noise estimate from pixel-to-pixel differences (MAD)."""
    d = np.diff(np.asarray(y, dtype=float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / math.sqrt(2))


def subpixel_peak(y: np.ndarray, i: int, baseline: float = 0.0) -> float:
    """Fractional pixel index of the peak at i, from pixels i-1, i, i+1.

    Uses a parabola through the logarithms of the counts above `baseline` (exact for a Gaussian
    line, which a spectrometer's line shape closely follows); falls back to a plain parabola when
    a neighbour is at or below the baseline.
    """
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    a, b, c = (float(v) - baseline for v in (y[i - 1], y[i], y[i + 1]))
    if a > 0 and b > 0 and c > 0:
        la, lb, lc = math.log(a), math.log(b), math.log(c)
        den = la - 2 * lb + lc
        if den < 0:
            return i + 0.5 * (la - lc) / den
    den = a - 2 * b + c
    if den >= 0:  # not a maximum
        return float(i)
    return i + 0.5 * (a - c) / den


def estimate_shift(wl, y, ref_lines, max_shift: float = 15.0, step: float = 0.05) -> float:
    """Overall offset (nm) between where the reference lines appear and where they belong.

    Slides the whole line pattern across the spectrum (from -max_shift to +max_shift) and returns the
    shift where the sum of the line heights above the local background is largest. Moving all lines
    together keeps close neighbours (e.g. the Hg 577/579 nm doublet) from being confused.
    """
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)
    step_px = float(np.mean(np.diff(wl)))
    # subtract a broad running minimum so continuum (e.g. a deuterium UV hump) does not score
    half = max(3, int(round(3.0 / step_px)))
    pad = np.pad(y, half, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1)
    background = windows.min(axis=1)
    peaks = np.clip(y - background, 0, None)
    refs = [r for r in ref_lines if wl[0] + max_shift <= r <= wl[-1] - max_shift] or list(ref_lines)
    shifts = np.arange(-max_shift, max_shift + step / 2, step)
    score = np.array([np.sum(np.interp(np.asarray(refs) + s, wl, peaks, left=0, right=0)) for s in shifts])
    if not np.any(score > 0):
        return 0.0
    return float(shifts[int(np.argmax(score))])


def find_lines(wl, y, ref_lines, window: float = 1.5, min_snr: float = 15.0,
               max_shift: float = 0.0) -> list[dict]:
    """Locate each reference line in a measured spectrum.

    For every reference wavelength inside the spectrum, take the highest pixel within +/- window nm
    and accept it if it stands min_snr noise levels above the window's lowest point and is not on
    the window's edge. Its position is refined to a fraction of a pixel and converted to nm with wl.
    With max_shift > 0 the overall offset is estimated first (estimate_shift) and each line is then
    searched for around ref + offset, so a calibration that is several nm off is still found.
    Returns dicts: ref, found, error (found - ref), height, snr.
    """
    wl = np.asarray(wl, dtype=float)
    y = np.asarray(y, dtype=float)
    if max_shift > 0:
        shift = estimate_shift(wl, y, ref_lines, max_shift)
        if shift != 0.0:
            found = find_lines(wl - shift, y, ref_lines, window, min_snr)
            for d in found:
                d["found"] += shift
                d["error"] += shift
            return found
    noise = max(noise_level(y), 1e-12)
    idx = np.arange(wl.size, dtype=float)
    out = []
    for ref in ref_lines:
        if not wl[0] + window <= ref <= wl[-1] - window:
            continue
        sel = np.where((wl >= ref - window) & (wl <= ref + window))[0]
        if sel.size < 5:
            continue
        i = int(sel[np.argmax(y[sel])])
        if i in (sel[0], sel[-1]):
            continue  # still climbing at the edge: not this line (or the calibration is way off)
        height = float(y[i] - np.min(y[sel]))
        snr = height / noise
        if snr < min_snr:
            continue
        f = subpixel_peak(y, i, baseline=float(np.min(y[sel])))
        found = float(np.interp(f, idx, wl))
        out.append({"ref": float(ref), "found": found, "error": found - float(ref), "height": height,
                    "snr": snr})
    # a found peak can only belong to one reference line (keep the closer one)
    out.sort(key=lambda d: abs(d["error"]))
    taken, unique = [], []
    for d in out:
        if all(abs(d["found"] - t) > 0.05 for t in taken):
            taken.append(d["found"])
            unique.append(d)
    return sorted(unique, key=lambda d: d["ref"])


def fit_wavelength_correction(found, ref, order: int = 1) -> dict:
    """Fit delta(lambda) = ref - found as a polynomial in the factory wavelength.

    order 0 = constant shift, 1 = shift + stretch, 2 = adds curvature. The order is lowered
    automatically when there are too few lines. Returns coeffs (highest power first, for
    numpy.polyval), the order used, rms residual (nm) and the residual per line.
    """
    f = np.asarray(found, dtype=float)
    r = np.asarray(ref, dtype=float)
    if f.size == 0:
        raise ValueError("No lines to fit.")
    order = int(max(0, min(order, f.size - 1, 2)))
    if order > 0 and np.ptp(f) < 1.0:
        order = 0
    coeffs = np.polyfit(f, r - f, order) if order > 0 else np.array([float(np.mean(r - f))])
    resid = (f + np.polyval(coeffs, f)) - r
    rms = float(math.sqrt(np.mean(resid ** 2)))
    return {"coeffs": [float(v) for v in coeffs], "order": order, "rms": rms, "residuals": resid.tolist(),
            "range": [float(f.min()), float(f.max())]}


def apply_wavelength_correction(factory_wl, coeffs) -> np.ndarray:
    wl = np.asarray(factory_wl, dtype=float)
    if not coeffs:
        return wl
    return wl + np.polyval(np.asarray(coeffs, dtype=float), wl)
