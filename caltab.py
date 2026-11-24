"""The Calibration tab: a Beer's-law concentration curve and software wavelength recalibration.

Both are stored in calibration.json next to settings.json (per computer, never shipped or touched
by updates). Wavelength corrections are kept per spectrometer serial number and applied in
software; the calibration stored inside the spectrometer is never changed.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from datetime import datetime

import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import calibration as cal
import spectro

ORDERS = ["0 (shift only)", "1 (shift + stretch)", "2 (+ curvature)"]


class CalibrationTab:
    def __init__(self, app, parent: ttk.Frame, store_path: str):
        self.app = app
        self.path = store_path
        self._wl_touched: set[str] = set()  # serials whose correction this computer changed
        self.v_cwl = tk.StringVar()
        self.v_cband = tk.StringVar()
        self.v_unit = tk.StringVar()
        self.v_zero = tk.BooleanVar()
        self.v_conc = tk.StringVar(value="")
        self.standards: list[dict] = []
        self.unknowns: list[dict] = []
        self._take(self._load())
        self.fit: dict | None = None
        self.pending: tuple | None = None  # ("standard", conc, t_click) or ("unknown", name, t_click)

        self.v_lines = tk.StringVar(value=", ".join(f"{x:g}" for x in cal.HG_AR_LINES))
        self.v_window = tk.StringVar(value="1.5")
        self.v_order = tk.StringVar(value=ORDERS[1])
        self.v_demo_conc = tk.StringVar(value="")
        self.found: list[dict] = []
        self.preview: dict | None = None
        self.dirty = False
        self._build(parent)
        self._refit()

    # ------------------------------------------------------------ storage
    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _take(self, data: dict) -> None:
        """Use calibration data read from disk (start-up, or a newly chosen data folder)."""
        self.data = data
        cc = self.data.setdefault("concentration", {})
        if not isinstance(self.data.get("wavelength"), dict):
            self.data["wavelength"] = {}
        self.v_cwl.set(str(cc.get("wavelength", "520")))
        self.v_cband.set(str(cc.get("band", "1.0")))
        self.v_unit.set(cc.get("unit", "mM"))
        self.v_zero.set(bool(cc.get("through_zero", False)))
        self.standards = list(cc.get("standards", []))
        self.unknowns = list(cc.get("unknowns", []))

    def save(self) -> None:
        """Write calibration.json. In a shared folder other computers write it too, so wavelength
        corrections are merged per spectrometer: only the serials changed here overwrite the file's."""
        cc = self.data["concentration"]
        cc.update({"wavelength": self.v_cwl.get(), "band": self.v_cband.get(), "unit": self.v_unit.get(),
                   "through_zero": bool(self.v_zero.get()), "standards": self.standards,
                   "unknowns": self.unknowns})
        disk = self._load().get("wavelength")
        merged = dict(disk) if isinstance(disk, dict) else {}
        for serial in self._wl_touched:
            if serial in self.data["wavelength"]:
                merged[serial] = self.data["wavelength"][serial]
            else:
                merged.pop(serial, None)
        self.data["wavelength"] = merged
        tmp = self.path + ".tmp"
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, indent=2)
            os.replace(tmp, self.path)  # never leave a half-written file for a sync client to copy
        except OSError as exc:
            self.app.set_status(f"Could not save calibration.json: {exc}", "bad")

    def reload_wavelengths(self) -> None:
        """Pick up wavelength corrections other computers saved to a shared calibration.json."""
        disk = self._load().get("wavelength")
        if isinstance(disk, dict):
            self.data["wavelength"] = disk

    def set_path(self, path: str) -> str:
        """Switch to the calibration.json in another data folder. If that folder has none yet, the
        calibrations in use here are copied into it. Returns "loaded" or "copied"."""
        self.path = path
        if os.path.exists(path):
            self._take(self._load())
            self._wl_touched = set()
            how = "loaded"
        else:
            self._wl_touched = set(self.data["wavelength"])
            how = "copied"
        self._refit()  # also saves
        if self.app.dev is not None and self.app.factory_wl is not None:
            self.app.set_wavelengths(self.corrected(self.app.dev.serial, self.app.factory_wl))
        self._update_wl_state()
        self.redraw()
        return how

    # ------------------------------------------------------------ layout
    def _build(self, parent) -> None:
        top = ttk.Frame(parent, padding=(4, 4))
        top.pack(side="top", fill="x")
        top.columnconfigure(0, weight=1, uniform="c")
        top.columnconfigure(1, weight=1, uniform="c")

        # ---- concentration
        f = ttk.LabelFrame(top, text="Concentration calibration (Beer's law)", padding=(6, 3))
        f.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Wavelength").pack(side="left")
        self.e_cwl = ttk.Entry(row, textvariable=self.v_cwl, width=7)
        self.e_cwl.pack(side="left", padx=2)
        ttk.Label(row, text="nm  +/-").pack(side="left")
        self.e_cband = ttk.Entry(row, textvariable=self.v_cband, width=4)
        self.e_cband.pack(side="left", padx=2)
        ttk.Label(row, text="nm   Units").pack(side="left")
        ttk.Entry(row, textvariable=self.v_unit, width=7).pack(side="left", padx=2)
        self.b_peak = ttk.Button(row, text="Use peak in view", command=self.use_peak)
        self.b_peak.pack(side="left", padx=(8, 0))

        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Standard's concentration").pack(side="left")
        e = ttk.Entry(row, textvariable=self.v_conc, width=9)
        e.pack(side="left", padx=4)
        e.bind("<Return>", lambda _e: self.record_standard())
        ttk.Button(row, text="Record standard", command=self.record_standard).pack(side="left")
        ttk.Checkbutton(row, text="Line through zero", variable=self.v_zero,
                        command=self._refit).pack(side="left", padx=(12, 0))

        self.t_std = ttk.Treeview(f, columns=("c", "a", "res"), show="headings", height=5, selectmode="extended")
        for col, text, w in (("c", "Concentration", 130), ("a", "Absorbance", 110), ("res", "Fit residual (A)", 130)):
            self.t_std.heading(col, text=text)
            self.t_std.column(col, width=w, anchor="e")
        self.t_std.pack(fill="x", pady=(4, 0))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(2, 0))
        ttk.Button(row, text="Remove selected", command=self.remove_standards).pack(side="left")
        ttk.Button(row, text="Clear standards", command=self.clear_standards).pack(side="left", padx=4)
        ttk.Button(row, text="Save calibration (CSV)", command=self.save_csv).pack(side="right")
        self.l_fit = ttk.Label(f, text="", wraplength=640, justify="left")
        self.l_fit.pack(fill="x", pady=(4, 0))
        self.l_live = ttk.Label(f, text="", style="Big.TLabel")
        self.l_live.pack(fill="x", pady=(2, 0))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Record unknown", command=self.record_unknown).pack(side="left")
        ttk.Label(row, text="(uses the Sample name in the Save panel)", foreground="#666").pack(side="left", padx=6)
        ttk.Button(row, text="Clear unknowns", command=self.clear_unknowns).pack(side="right")
        self.t_unk = ttk.Treeview(f, columns=("s", "a", "c"), show="headings", height=3)
        for col, text, w in (("s", "Sample", 200), ("a", "Absorbance", 110), ("c", "Concentration", 150)):
            self.t_unk.heading(col, text=text)
            self.t_unk.column(col, width=w, anchor="w" if col == "s" else "e")
        self.t_unk.pack(fill="x", pady=(2, 0))
        if self.app.demo:
            row = ttk.Frame(f)
            row.pack(fill="x", pady=(4, 0))
            ttk.Label(row, text="Demo: sample concentration (mM, blank = fading sample)").pack(side="left")
            e = ttk.Entry(row, textvariable=self.v_demo_conc, width=7)
            e.pack(side="left", padx=4)
            e.bind("<Return>", lambda _e: self.set_demo_conc())
            ttk.Button(row, text="Set", command=self.set_demo_conc).pack(side="left")

        # ---- wavelength
        g = ttk.LabelFrame(top, text="Wavelength calibration (calibration lamp, e.g. Hg-Ar)", padding=(6, 3))
        g.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        self.l_wlstate = ttk.Label(g, text="", wraplength=640, justify="left", style="Big.TLabel")
        self.l_wlstate.pack(fill="x")
        row = ttk.Frame(g)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Reference lines (nm)").pack(side="left")
        ttk.Entry(row, textvariable=self.v_lines).pack(side="left", fill="x", expand=True, padx=4)
        row = ttk.Frame(g)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="Search +/-").pack(side="left")
        ttk.Entry(row, textvariable=self.v_window, width=4).pack(side="left", padx=2)
        ttk.Label(row, text="nm   Fit").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.v_order, values=ORDERS, state="readonly", width=17)
        cb.pack(side="left", padx=4)
        cb.bind("<<ComboboxSelected>>", lambda _e: self._preview())
        ttk.Button(row, text="Find lines", command=self.find_lines).pack(side="left", padx=(8, 0))
        self.t_wl = ttk.Treeview(g, columns=("use", "ref", "found", "err", "res"), show="headings", height=7)
        for col, text, w in (("use", "Use", 50), ("ref", "Reference (nm)", 120), ("found", "Found at (nm)", 120),
                             ("err", "Factory error (nm)", 140), ("res", "After fit (nm)", 120)):
            self.t_wl.heading(col, text=text)
            self.t_wl.column(col, width=w, anchor="center" if col == "use" else "e")
        self.t_wl.pack(fill="x", pady=(4, 0))
        self.t_wl.bind("<Double-1>", self._toggle_line)
        row = ttk.Frame(g)
        row.pack(fill="x", pady=(2, 0))
        self.b_apply = ttk.Button(row, text="Apply correction", command=self.apply_correction, state="disabled")
        self.b_apply.pack(side="left")
        ttk.Button(row, text="Reset to factory", command=self.reset_correction).pack(side="left", padx=4)
        self.l_wlfit = ttk.Label(g, text="", wraplength=640, justify="left")
        self.l_wlfit.pack(fill="x", pady=(4, 0))
        ttk.Label(g, foreground="#666", wraplength=640, justify="left",
                  text="Point the fiber at the calibration lamp in Scope mode (Boxcar 0, no saturation), then "
                       "Find lines. Double-click a row to leave a line out. The correction is stored on this "
                       "computer for this spectrometer's serial number and applied in software; the "
                       "calibration inside the spectrometer is never changed.").pack(fill="x", pady=(4, 0))

        # ---- plots
        self.fig = Figure(figsize=(8, 3), dpi=100, layout="constrained")
        self.ax_c = self.fig.add_subplot(121)
        self.ax_w = self.fig.add_subplot(122)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # ------------------------------------------------------------ helpers
    def _num(self, var: tk.StringVar, what: str) -> float | None:
        try:
            v = float(var.get())
            if not math.isfinite(v):
                raise ValueError
            return v
        except ValueError:
            self.app.set_status(f"{what} must be a number.", "bad")
            return None

    def _abs_ready(self) -> bool:
        a = self.app
        if a.dev is None:
            a.set_status("Connect a spectrometer first.", "warn")
            return False
        if a.v_mode.get() != "Absorbance" or a.dark is None or a.ref is None:
            a.set_status("Concentration calibration needs Absorbance mode: store a dark and a reference "
                         "(blank) first, then pick Absorbance.", "warn")
            return False
        return True

    def _absorbance_now(self) -> float | None:
        a = self.app
        if a.y is None or not a.y_ok or a.v_mode.get() != "Absorbance":
            return None
        lam = self._num(self.v_cwl, "Wavelength")
        band = self._num(self.v_cband, "Band")
        if lam is None or band is None:
            return None
        v = spectro.band_value(a.wl, a.y, lam, max(0.0, band))
        return v if math.isfinite(v) else None

    def _request(self, kind: str, value) -> None:
        self.pending = (kind, value, time.time())
        if not self.app.acq.running:
            self.app.acq.single = True

    def _fmt_c(self, c: float) -> str:
        return f"{c:.4g} {self.v_unit.get()}".strip()

    # ------------------------------------------------------------ concentration
    def use_peak(self) -> None:
        a = self.app
        if a.y is None or a.wl is None:
            return
        lo, hi = a.ax.get_xlim()
        pk = spectro.peak_in_window(a.wl, a.y, lo, hi)
        if pk:
            self.v_cwl.set(f"{pk[0]:.1f}")
            self.save()
            self.on_spectrum()

    def record_standard(self) -> None:
        c = self._num(self.v_conc, "The standard's concentration")
        if c is None or not self._abs_ready():
            return
        if self._num(self.v_cwl, "Wavelength") is None or self._num(self.v_cband, "Band") is None:
            return
        self._request("standard", c)
        self.app.set_status(f"Recording standard {self._fmt_c(c)}: waiting for a fresh spectrum...")

    def record_unknown(self) -> None:
        if not self.fit:
            self.app.set_status("Record at least two standards first (one with 'Line through zero').", "warn")
            return
        if not self._abs_ready():
            return
        self._request("unknown", self.app.v_sample.get().strip() or f"Unknown {len(self.unknowns) + 1}")
        self.app.set_status("Recording unknown: waiting for a fresh spectrum...")

    def _complete_pending(self) -> None:
        kind, value, _t = self.pending
        self.pending = None
        a_now = self._absorbance_now()
        if a_now is None:
            self.app.set_status("No absorbance at that wavelength (is there light there?). Nothing recorded.", "bad")
            return
        stamp = datetime.now().isoformat(timespec="seconds")
        if kind == "standard":
            self.standards.append({"c": value, "A": a_now, "time": stamp,
                                   "wavelength": self.v_cwl.get(), "band": self.v_cband.get()})
            self.v_conc.set("")
            self.app.set_status(f"Recorded standard {self._fmt_c(value)}: A = {a_now:.4f}", "good")
        else:
            c = cal.predict_conc(self.fit, a_now)
            self.unknowns.append({"sample": value, "A": a_now, "c": c, "time": stamp})
            self.app.set_status(f"{value}: A = {a_now:.4f} -> {self._fmt_c(c)}", "good")
        self._refit()

    def remove_standards(self) -> None:
        rows = sorted((self.t_std.index(i) for i in self.t_std.selection()), reverse=True)
        for r in rows:
            del self.standards[r]
        self._refit()

    def clear_standards(self) -> None:
        if self.standards and not messagebox.askyesno("Clear standards", "Remove all recorded standards?"):
            return
        self.standards = []
        self._refit()

    def clear_unknowns(self) -> None:
        self.unknowns = []
        self._refit()

    def _refit(self) -> None:
        cs = [s["c"] for s in self.standards]
        As = [s["A"] for s in self.standards]
        try:
            self.fit = cal.fit_line(cs, As, through_zero=bool(self.v_zero.get())) if self.standards else None
        except ValueError:
            self.fit = None
        # standards fix the analysis wavelength: unlock only when there are none
        state = "disabled" if self.standards else "normal"
        self.e_cwl.configure(state=state)
        self.e_cband.configure(state=state)
        self.b_peak.configure(state=state)

        self.t_std.delete(*self.t_std.get_children())
        for k, s in enumerate(self.standards):
            res = f"{self.fit['residuals'][k]:+.4f}" if self.fit else ""
            self.t_std.insert("", "end", values=(self._fmt_c(s["c"]), f"{s['A']:.4f}", res))
        self.t_unk.delete(*self.t_unk.get_children())
        for u in self.unknowns:
            self.t_unk.insert("", "end", values=(u["sample"], f"{u['A']:.4f}", self._fmt_c(u["c"])))
        unit = self.v_unit.get()
        if self.fit:
            f = self.fit
            eq = (f"A = {f['slope']:.5g} x c" if f["through_zero"]
                  else f"A = {f['slope']:.5g} x c {'+' if f['intercept'] >= 0 else '-'} {abs(f['intercept']):.4g}")
            lim = (f"   LOD {f['lod']:.3g} {unit}, LOQ {f['loq']:.3g} {unit}" if math.isfinite(f["lod"]) else "")
            self.l_fit.configure(text=f"{eq}   (A at {self.v_cwl.get()} nm, c in {unit})   "
                                      f"R² = {f['r2']:.5f}   n = {f['n']}{lim}",
                                 style="Warn.TLabel" if f["r2"] < 0.99 else "TLabel")
        else:
            need = "one standard" if self.v_zero.get() else "two standards with different concentrations"
            self.l_fit.configure(text=f"No calibration yet: record at least {need}.", style="TLabel")
        self.save()
        self.on_spectrum()
        self.redraw()

    def save_csv(self) -> str | None:
        if not self.standards:
            self.app.set_status("No standards to save.", "warn")
            return None
        a = self.app
        path = os.path.join(a.logs, "calibration", a._stem() + "_calibration.csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        f, unit = self.fit, self.v_unit.get()
        with open(path, "w", newline="", encoding="utf-8") as fh:
            meta = {"Saved": datetime.now().isoformat(timespec="seconds"),
                    "Spectrometer": f"{a.dev.model} serial {a.dev.serial}" if a.dev else "?",
                    "Analysis wavelength (nm)": self.v_cwl.get(), "Band (+/- nm)": self.v_cband.get(),
                    "Units": unit, "Line through zero": bool(self.v_zero.get()),
                    "Wavelength calibration": self.describe(a.dev.serial if a.dev else None)}
            if f:
                meta.update({"Slope (A per unit)": f"{f['slope']:.6g}", "Intercept (A)": f"{f['intercept']:.6g}",
                             "R2": f"{f['r2']:.6f}", "Standard error s_y (A)": f"{f['s_y']:.4g}",
                             f"LOD ({unit})": f"{f['lod']:.4g}", f"LOQ ({unit})": f"{f['loq']:.4g}"})
            for k, v in meta.items():
                fh.write(f"# {k}: {v}\n")
            w = csv.writer(fh)
            w.writerow(["kind", "sample", f"concentration ({unit})", "absorbance", "fit_residual", "time"])
            for k, s in enumerate(self.standards):
                w.writerow(["standard", "", f"{s['c']:.6g}", f"{s['A']:.6g}",
                            f"{f['residuals'][k]:.6g}" if f else "", s["time"]])
            for u in self.unknowns:
                w.writerow(["unknown", u["sample"], f"{u['c']:.6g}", f"{u['A']:.6g}", "", u["time"]])
        a.set_status(f"Saved calibration: {a._short(path)}", "good")
        return path

    def set_demo_conc(self) -> None:
        dev = self.app.dev
        if not isinstance(dev, spectro.DemoSpectrometer):
            return
        text = self.v_demo_conc.get().strip()
        try:
            dev.sample_conc = float(text) if text else None
        except ValueError:
            self.app.set_status("Demo concentration must be a number (or blank).", "bad")
            return
        self.app.set_status("Demo sample: " + (f"{dev.sample_conc:g} mM" if text else "fading sample"))

    # ------------------------------------------------------------ live update (every spectrum)
    def on_spectrum(self) -> None:
        a = self.app
        if self.pending and a.info and a.info["t_start"] >= self.pending[2]:
            self._complete_pending()
            return
        a_now = self._absorbance_now()
        if not self.fit:
            self.l_live.configure(text="", style="Big.TLabel")
        elif a_now is None:
            self.l_live.configure(text="Live result appears in Absorbance mode.", style="TLabel")
        else:
            c = cal.predict_conc(self.fit, a_now)
            span = self.fit["c_max"] - self.fit["c_min"]
            outside = c < self.fit["c_min"] - 0.02 * span or c > self.fit["c_max"] + 0.02 * span
            self.l_live.configure(
                text=f"Now: A = {a_now:.4f}  ->  {self._fmt_c(c)}"
                     + ("   (outside the standards' range: extrapolated)" if outside else ""),
                style="Warn.TLabel" if outside else "Big.TLabel")

    # ------------------------------------------------------------ wavelength
    def corrected(self, serial: str, factory_wl: np.ndarray) -> np.ndarray:
        entry = self.data["wavelength"].get(str(serial))
        return cal.apply_wavelength_correction(factory_wl, entry["coeffs"]) if entry else factory_wl

    def describe(self, serial) -> str:
        entry = self.data["wavelength"].get(str(serial)) if serial is not None else None
        if not entry:
            return "factory (stored in the spectrometer)"
        return (f"factory + software correction (fit order {entry['order']}, {len(entry['lines'])} lines, "
                f"RMS {entry['rms']:.3f} nm, {entry['date']})")

    def on_connect(self) -> None:
        self.found, self.preview = [], None
        self._show_found()
        self._update_wl_state()
        self.redraw()

    def _update_wl_state(self) -> None:
        dev = self.app.dev
        if dev is None:
            self.l_wlstate.configure(text="Connect a spectrometer.", style="TLabel")
            return
        entry = self.data["wavelength"].get(str(dev.serial))
        if entry:
            self.l_wlstate.configure(style="Good.TLabel", text=(
                f"Correction in use for {dev.serial}: {len(entry['lines'])} lines, fit order {entry['order']}, "
                f"RMS {entry['rms']:.3f} nm (set {entry['date']}). Valid between "
                f"{entry['range'][0]:.0f} and {entry['range'][1]:.0f} nm."))
        else:
            self.l_wlstate.configure(style="TLabel",
                                     text=f"Using the factory calibration stored in {dev.serial}.")

    def find_lines(self) -> None:
        a = self.app
        if a.raw is None or a.dev is None:
            a.set_status("Connect and wait for a spectrum first.", "warn")
            return
        try:
            refs = [float(x) for x in self.v_lines.get().replace(";", ",").split(",") if x.strip()]
            window = float(self.v_window.get())
        except ValueError:
            a.set_status("Reference lines must be numbers separated by commas, and the search width a number.", "bad")
            return
        if a.raw.max() >= 0.98 * a.dev.max_intensity:
            a.set_status("The spectrum is saturated: lower the integration time so the lines are not cut off, "
                         "then Find lines again.", "bad")
            return
        found = cal.find_lines(a.factory_wl, a.raw, refs, window=window)
        for d in found:
            d["use"] = True
        self.found = found
        if not found:
            a.set_status("No reference lines found. Is the lamp in view and bright enough? You can also widen "
                         "the search.", "warn")
        else:
            a.set_status(f"Found {len(found)} of {len(refs)} reference lines.", "good")
        self._preview()

    def _order(self) -> int:
        return ORDERS.index(self.v_order.get()) if self.v_order.get() in ORDERS else 1

    def _preview(self) -> None:
        used = [d for d in self.found if d["use"]]
        self.preview = None
        if used:
            self.preview = cal.fit_wavelength_correction([d["found"] for d in used], [d["ref"] for d in used],
                                                         self._order())
            res = iter(self.preview["residuals"])
            for d in self.found:
                d["resid"] = next(res) if d["use"] else None
            mean_err = float(np.mean([d["error"] for d in used]))
            note = (f" (asked for order {self._order()}, used {self.preview['order']}: too few lines)"
                    if self.preview["order"] < self._order() else "")
            self.l_wlfit.configure(
                text=f"Factory calibration is off by {mean_err:+.3f} nm on average. With this fit: RMS "
                     f"{self.preview['rms']:.3f} nm over {len(used)} lines{note}. Apply to use it.",
                style="TLabel")
        else:
            self.l_wlfit.configure(text="")
        self.b_apply.configure(state="normal" if self.preview else "disabled")
        self._show_found()
        self.redraw()

    def _show_found(self) -> None:
        self.t_wl.delete(*self.t_wl.get_children())
        for d in self.found:
            res = d.get("resid")
            self.t_wl.insert("", "end", values=(
                "yes" if d["use"] else "no", f"{d['ref']:.3f}", f"{d['found']:.3f}", f"{d['error']:+.3f}",
                f"{res:+.3f}" if res is not None else ""))

    def _toggle_line(self, event) -> None:
        item = self.t_wl.identify_row(event.y)
        if not item:
            return
        k = self.t_wl.index(item)
        self.found[k]["use"] = not self.found[k]["use"]
        self._preview()

    def apply_correction(self) -> None:
        dev = self.app.dev
        if not self.preview or dev is None:
            return
        used = [d for d in self.found if d["use"]]
        self._wl_touched.add(str(dev.serial))
        self.data["wavelength"][str(dev.serial)] = {
            "coeffs": self.preview["coeffs"], "order": self.preview["order"], "rms": self.preview["rms"],
            "range": self.preview["range"], "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "lines": [{"ref": d["ref"], "found": d["found"]} for d in used]}
        self.save()
        self.app.set_wavelengths(self.corrected(dev.serial, self.app.factory_wl))
        self._update_wl_state()
        self.app.set_status(f"Wavelength correction applied for {dev.serial} (RMS {self.preview['rms']:.3f} nm).",
                            "good")
        self.redraw()

    def reset_correction(self) -> None:
        dev = self.app.dev
        if dev is None:
            return
        self._wl_touched.add(str(dev.serial))
        self.data["wavelength"].pop(str(dev.serial), None)
        self.save()
        self.app.set_wavelengths(self.app.factory_wl)
        self._update_wl_state()
        self.app.set_status(f"Back to the factory wavelength calibration for {dev.serial}.", "good")
        self.redraw()

    # ------------------------------------------------------------ plots
    def redraw(self) -> None:
        unit = self.v_unit.get()
        ax = self.ax_c
        ax.clear()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel(f"Concentration ({unit})")
        ax.set_ylabel(f"A at {self.v_cwl.get()} nm")
        ax.set_title("Concentration calibration", fontsize=10)
        if self.standards:
            cs = np.array([s["c"] for s in self.standards])
            As = np.array([s["A"] for s in self.standards])
            ax.plot(cs, As, "o", color="#1f5fbf", label="Standards")
            if self.fit:
                x = np.linspace(0, max(cs.max(), max((u["c"] for u in self.unknowns), default=0)) * 1.08, 50)
                ax.plot(x, self.fit["slope"] * x + self.fit["intercept"], "-", color="#334155", lw=1,
                        label=f"Fit, R² = {self.fit['r2']:.4f}")
            if self.unknowns:
                ax.plot([u["c"] for u in self.unknowns], [u["A"] for u in self.unknowns], "^", color="#c2410c",
                        label="Unknowns")
            ax.legend(loc="upper left", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Record standards to build the curve", transform=ax.transAxes, ha="center",
                    color="#777")

        ax = self.ax_w
        ax.clear()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Found - reference (nm)")
        ax.set_title("Wavelength calibration", fontsize=10)
        dev = self.app.dev
        entry = self.data["wavelength"].get(str(dev.serial)) if dev else None
        lo, hi = (float(self.app.factory_wl[0]), float(self.app.factory_wl[-1])) \
            if getattr(self.app, "factory_wl", None) is not None else (200.0, 1100.0)
        x = np.linspace(lo, hi, 200)
        if entry:
            ax.plot(x, -np.polyval(entry["coeffs"], x), "--", color="#15803d", lw=1, label="Correction in use")
        if self.found:
            use = [d for d in self.found if d["use"]]
            off = [d for d in self.found if not d["use"]]
            ax.plot([d["ref"] for d in use], [d["error"] for d in use], "o", color="#1f5fbf", label="Lines found")
            if off:
                ax.plot([d["ref"] for d in off], [d["error"] for d in off], "x", color="#999", label="Left out")
            if self.preview:
                ax.plot(x, -np.polyval(self.preview["coeffs"], x), "-", color="#334155", lw=1, label="New fit")
        if entry or self.found:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "Find lines with a calibration lamp in view", transform=ax.transAxes,
                    ha="center", color="#777")
        if self.app.nb.index("current") == 2:
            self.canvas.draw_idle()
        else:
            self.dirty = True
