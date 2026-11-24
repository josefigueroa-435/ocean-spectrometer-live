"""Ocean Spectrometer Live - live spectra from an Ocean Optics spectrometer (made for the HR4000).

Run:  python app.py            (real spectrometer)
      python app.py --demo     (simulated HR4000, no hardware)
      python app.py --backend pyseabreeze   (force the pure-Python USB backend)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import queue
import threading
import time
from collections import deque
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk


def _require_packages() -> None:
    """The launcher uses pythonw (no console), so explain missing packages in a dialog."""
    import importlib.util

    missing = [m for m in ("numpy", "matplotlib", "seabreeze", "usb", "libusb_package")
               if importlib.util.find_spec(m) is None]
    if missing:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Ocean Spectrometer Live",
            "Some Python packages this app needs are not installed:\n\n  " + ", ".join(missing) +
            "\n\nDouble-click 'Install requirements.bat' in the app folder (needs internet), "
            "then start the app again.")
        root.destroy()
        sys.exit(1)


_require_packages()

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

import caltab  # noqa: E402
import spectro  # noqa: E402
import updater  # noqa: E402
from acquire import Acquirer  # noqa: E402
from version import __version__  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
APP_NAME = "Ocean Spectrometer Live"

DEFAULTS = {
    "int_ms": "100",
    "avg": "1",
    "boxcar": "0",
    "edark": True,
    "nonlin": False,
    "mode": "Scope",
    "autoy": True,
    "sample": "",
    "track": "520",
    "band": "1.0",
    "every": "0",
    "data_dir": "",
    "light_source": "Full range (any source)",
    "geometry": "zoomed",
}

TEXT_WIDGETS = ("Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox", "Text")


def load_settings(path: str) -> dict:
    s = dict(DEFAULTS)
    try:
        with open(path, encoding="utf-8") as fh:
            s.update({k: v for k, v in json.load(fh).items() if k in DEFAULTS})
    except (OSError, ValueError):
        pass
    return s


class App:
    def __init__(self, root: tk.Tk, demo: bool = False, backend: str | None = None,
                 settings_path: str | None = None, logs_dir: str | None = None,
                 auto_connect: bool = True, check_updates: bool | None = None):
        self.root = root
        self.demo = demo
        self.backend_pref = backend
        self.settings_path = settings_path or os.path.join(HERE, "settings.json")
        self.s = load_settings(self.settings_path)
        # where saved files and calibration.json live: the app folder, or a shared data folder
        self._default_logs = logs_dir or os.path.join(HERE, "logs")
        self._default_cal = os.path.join(os.path.dirname(os.path.abspath(self.settings_path)), "calibration.json")
        self.data_dir = self.s.get("data_dir") or ""
        self._data_dir_missing = bool(self.data_dir) and not os.path.isdir(self.data_dir)
        if self.data_dir and not self._data_dir_missing:
            self.logs, cal_path = self.data_dir, os.path.join(self.data_dir, "calibration.json")
        else:
            self.logs, cal_path = self._default_logs, self._default_cal
        self._cal_path = cal_path

        self.acq = Acquirer()
        self.dev: spectro.SpectrometerBase | None = None
        self.devices: list = []
        self.wl: np.ndarray | None = None          # wavelengths in use (factory + any software correction)
        self.factory_wl: np.ndarray | None = None  # as stored in the spectrometer
        self.raw = None
        self.info = None
        self.y = None          # displayed (processed) spectrum
        self.y_ok = False      # False when the mode could not be computed (no dark/reference)
        self.dark = self.dark_info = None
        self.ref = self.ref_info = None
        self.pending_store: tuple[str, float] | None = None
        self.held: list = []
        self.trend: dict | None = None
        self.trend_data: dict | None = None
        self._arrivals = deque(maxlen=40)
        self._last_status_error = None
        self._closing = False
        self._update_manifest: dict | None = None
        self._updating = False
        self._ui_calls: queue.SimpleQueue = queue.SimpleQueue()

        root.title(f"{APP_NAME} {__version__}" + (" (demo)" if demo else ""))
        root.minsize(1000, 640)
        if self.s.get("geometry") == "zoomed":
            try:
                root.state("zoomed")
            except tk.TclError:
                root.geometry(DEFAULTS["geometry"])
        else:
            root.geometry(self.s.get("geometry") or DEFAULTS["geometry"])
        icon = os.path.join(HERE, "ocean_live.ico")
        if os.path.exists(icon):
            try:
                root.iconbitmap(icon)
            except tk.TclError:
                pass
        self._build()
        self._bind_keys()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        if auto_connect:
            root.after(150, lambda: self.refresh_devices(auto=True))
        if auto_connect if check_updates is None else check_updates:
            root.after(3000, self.check_updates)
        self._poll_id = root.after(60, self._poll)

    # ================================================================ layout
    def _build(self) -> None:
        r = self.root
        style = ttk.Style(r)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Warn.TLabel", foreground="#b45309")
        style.configure("Bad.TLabel", foreground="#b91c1c")
        style.configure("Good.TLabel", foreground="#15803d")
        style.configure("Big.TLabel", font=("Segoe UI", 10, "bold"))
        # tables: rows tall enough for the font at any display scaling
        from tkinter import font as tkfont
        style.configure("Treeview", rowheight=int(tkfont.nametofont("TkDefaultFont").metrics("linespace") * 1.2))

        self.v_int = tk.StringVar(value=self.s["int_ms"])
        self.v_avg = tk.StringVar(value=self.s["avg"])
        self.v_box = tk.StringVar(value=self.s["boxcar"])
        self.v_edark = tk.BooleanVar(value=bool(self.s["edark"]))
        self.v_nonlin = tk.BooleanVar(value=bool(self.s["nonlin"]))
        self.v_mode = tk.StringVar(value=self.s["mode"] if self.s["mode"] in spectro.MODES else "Scope")
        self.v_autoy = tk.BooleanVar(value=bool(self.s["autoy"]))
        self.v_sample = tk.StringVar(value=self.s["sample"])
        self.v_track = tk.StringVar(value=self.s["track"])
        self.v_band = tk.StringVar(value=self.s["band"])
        self.v_every = tk.StringVar(value=self.s["every"])

        # status bar first, so it keeps its line when the window is short
        sbar = ttk.Frame(r, relief="groove")
        sbar.pack(side="bottom", fill="x")
        self.l_version = ttk.Label(sbar, text=f"v{__version__}  (check for updates)", foreground="#1f5fbf",
                                   cursor="hand2", padding=(8, 2))
        self.l_version.pack(side="right")
        self.l_version.bind("<Button-1>", lambda _e: self.check_updates(manual=True))
        self.status = ttk.Label(sbar, text="", anchor="w", padding=(8, 2))
        self.status.pack(side="left", fill="x", expand=True)

        # "update available" bar, shown only when there is one
        self.update_bar = tk.Frame(r, bg="#fff4d6", highlightbackground="#e0b453", highlightthickness=1)
        self.l_update = tk.Label(self.update_bar, text="", bg="#fff4d6", anchor="w", justify="left")
        self.l_update.pack(side="left", padx=10, pady=4, fill="x", expand=True)
        self.b_update_later = ttk.Button(self.update_bar, text="Later", command=self._hide_update_bar)
        self.b_update_later.pack(side="right", padx=(4, 10), pady=4)
        self.b_update = ttk.Button(self.update_bar, text="Update now", command=self.install_update)
        self.b_update.pack(side="right", pady=4)
        outer = ttk.Frame(r)
        outer.pack(fill="both", expand=True)
        self._outer = outer
        left = ttk.Frame(outer, padding=(8, 6, 4, 2))
        left.pack(side="left", fill="y")
        right = ttk.Frame(outer, padding=(4, 8, 8, 4))
        right.pack(side="left", fill="both", expand=True)

        # ---- device
        f = ttk.LabelFrame(left, text="Spectrometer", padding=(6, 3))
        f.pack(fill="x", pady=(0, 4))
        self.cb_dev = ttk.Combobox(f, state="readonly", width=34)
        self.cb_dev.pack(fill="x")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Refresh", command=self.refresh_devices).pack(side="left")
        self.b_connect = ttk.Button(row, text="Connect", command=self.connect)
        self.b_connect.pack(side="left", padx=4)
        self.b_disconnect = ttk.Button(row, text="Disconnect", command=self.disconnect, state="disabled")
        self.b_disconnect.pack(side="left")
        self.l_devinfo = ttk.Label(f, text="Looking for spectrometers...", wraplength=380, justify="left")
        self.l_devinfo.pack(fill="x", pady=(6, 0))

        # ---- acquisition
        f = ttk.LabelFrame(left, text="Acquisition", padding=(6, 3))
        f.pack(fill="x", pady=(0, 4))
        g = ttk.Frame(f)
        g.pack(fill="x")
        src = self.s.get("light_source")
        self.v_source = tk.StringVar(value=src if src in spectro.LIGHT_SOURCES else DEFAULTS["light_source"])
        ttk.Label(g, text="Light source").grid(row=4, column=0, sticky="w", pady=(2, 0))
        cb = ttk.Combobox(g, textvariable=self.v_source, values=list(spectro.LIGHT_SOURCES),
                          state="readonly", width=21)
        cb.grid(row=4, column=1, columnspan=2, sticky="e", pady=(2, 0))
        cb.bind("<<ComboboxSelected>>", lambda _e: self.set_light_source())
        ttk.Label(g, text="Integration time (ms)").grid(row=0, column=0, sticky="w")
        e = ttk.Entry(g, textvariable=self.v_int, width=8)
        e.grid(row=0, column=1, sticky="e", padx=(4, 0))
        ttk.Button(g, text="Set", width=4, command=self.auto_integration).grid(row=0, column=2, sticky="e",
                                                                               padx=(4, 0))
        e.bind("<Return>", lambda _e: self.apply_params())
        e.bind("<FocusOut>", lambda _e: self.apply_params())
        ttk.Label(g, text="Scans to average").grid(row=1, column=0, sticky="w", pady=(2, 0))
        sp = ttk.Spinbox(g, from_=1, to=1000, textvariable=self.v_avg, width=7, command=self.apply_params)
        sp.grid(row=1, column=1, columnspan=2, sticky="e", pady=(2, 0))
        sp.bind("<Return>", lambda _e: self.apply_params())
        sp.bind("<FocusOut>", lambda _e: self.apply_params())
        ttk.Label(g, text="Boxcar half-width (px)").grid(row=2, column=0, sticky="w", pady=(2, 0))
        sp = ttk.Spinbox(g, from_=0, to=100, textvariable=self.v_box, width=7, command=self._redraw)
        sp.grid(row=2, column=1, columnspan=2, sticky="e", pady=(2, 0))
        sp.bind("<Return>", lambda _e: self._redraw())
        sp.bind("<FocusOut>", lambda _e: self._redraw())
        g.columnconfigure(0, weight=1)
        self.c_edark = ttk.Checkbutton(f, text="Electric dark correction", variable=self.v_edark,
                                       command=self.apply_params)
        self.c_edark.pack(anchor="w", pady=(2, 0))
        self.c_nonlin = ttk.Checkbutton(f, text="Nonlinearity correction", variable=self.v_nonlin,
                                        command=self.apply_params)
        self.c_nonlin.pack(anchor="w")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(3, 0))
        self.b_run = ttk.Button(row, text="Run", command=self.toggle_run, state="disabled")
        self.b_run.pack(side="left")
        self.b_single = ttk.Button(row, text="Single", command=self.single_shot, state="disabled")
        self.b_single.pack(side="left", padx=4)
        self.l_rate = ttk.Label(row, text="")
        self.l_rate.pack(side="right")
        self.l_sat = ttk.Label(f, text="", style="Big.TLabel")
        self.l_sat.pack(anchor="w", pady=(2, 0))

        # ---- mode
        f = ttk.LabelFrame(left, text="Mode", padding=(6, 3))
        f.pack(fill="x", pady=(0, 4))
        g = ttk.Frame(f)
        g.pack(fill="x")
        for i, m in enumerate(spectro.MODES):
            ttk.Radiobutton(g, text=m, value=m, variable=self.v_mode, command=self._on_mode).grid(
                row=i // 2, column=i % 2, sticky="w", padx=(0, 10))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(6, 0))
        ttk.Button(row, text="Store dark", command=lambda: self.request_store("dark")).pack(side="left")
        ttk.Button(row, text="Store reference", command=lambda: self.request_store("reference")).pack(
            side="left", padx=4)
        self.l_dark = ttk.Label(f, text="Dark: none")
        self.l_dark.pack(anchor="w", pady=(4, 0))
        self.l_ref = ttk.Label(f, text="Reference: none")
        self.l_ref.pack(anchor="w")

        # ---- save
        f = ttk.LabelFrame(left, text="Save", padding=(6, 3))
        f.pack(fill="x", pady=(0, 4))
        row = ttk.Frame(f)
        row.pack(fill="x")
        ttk.Label(row, text="Sample name").pack(side="left")
        ttk.Entry(row, textvariable=self.v_sample).pack(side="left", fill="x", expand=True, padx=(4, 0))
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Save CSV", command=self.save_spectrum).pack(side="left")
        ttk.Button(row, text="Save graph", command=self.save_graph).pack(side="left", padx=4)
        mb = ttk.Menubutton(row, text="Data folder")
        menu = tk.Menu(mb, tearoff=False)
        menu.add_command(label="Open data folder", command=self.open_logs)
        menu.add_command(label="Change data folder (e.g. a shared folder)...", command=self.choose_data_dir)
        menu.add_command(label="Use the app folder (default)", command=lambda: self.set_data_dir(""))
        mb["menu"] = menu
        mb.pack(side="left")
        row = ttk.Frame(f)
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="Hold", command=self.hold).pack(side="left")
        ttk.Button(row, text="Overlay file...", command=self.overlay_file).pack(side="left", padx=4)
        ttk.Button(row, text="Clear held", command=self.clear_held).pack(side="left")

        # ---- demo controls
        self.demo_frame = ttk.LabelFrame(left, text="Demo spectrometer", padding=(6, 3))
        self.v_lamp = tk.StringVar(value="continuum")
        self.v_sample_in = tk.BooleanVar(value=True)
        self.v_shutter = tk.BooleanVar(value=False)
        row = ttk.Frame(self.demo_frame)
        row.pack(fill="x")
        ttk.Radiobutton(row, text="Halogen lamp", value="continuum", variable=self.v_lamp,
                        command=self._demo_controls).pack(side="left")
        ttk.Radiobutton(row, text="Hg-Ar lamp", value="lines", variable=self.v_lamp,
                        command=self._demo_controls).pack(side="left", padx=6)
        ttk.Checkbutton(row, text="Shutter closed", variable=self.v_shutter,
                        command=self._demo_controls).pack(side="left", padx=(4, 0))
        ttk.Checkbutton(self.demo_frame, text="Sample in beam (absorbs near 520 nm, fading)",
                        variable=self.v_sample_in, command=self._demo_controls).pack(anchor="w")

        # ---- notebook with spectrum + trend
        self.nb = ttk.Notebook(right)
        self.nb.pack(fill="both", expand=True)
        tab_spec = ttk.Frame(self.nb)
        tab_trend = ttk.Frame(self.nb)
        self.nb.add(tab_spec, text="Spectrum")
        self.nb.add(tab_trend, text="Trend")
        self._spec_dirty = self._trend_dirty = False
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab)

        self.fig = Figure(figsize=(8, 5), dpi=100, layout="constrained")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Wavelength (nm)")
        self.ax.set_ylabel(spectro.MODE_UNITS[self.v_mode.get()])
        self.ax.grid(True, alpha=0.3)
        (self.line,) = self.ax.plot([], [], lw=1.0, color="#1f5fbf", label="Live", animated=True)
        (self.peak_mark,) = self.ax.plot([], [], "v", color="#c2410c", ms=8, animated=True)
        self._bg = None
        self._saving = False
        self.msg = self.ax.text(0.5, 0.5, "", transform=self.ax.transAxes, ha="center", va="center",
                                fontsize=12, color="#555")
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab_spec)
        bar = ttk.Frame(tab_spec)
        bar.pack(side="top", fill="x")
        tb = NavigationToolbar2Tk(self.canvas, bar, pack_toolbar=False)
        tb.update()
        # matplotlib reserves a wide blank label for "(x, y) = ..."; the app has its own cursor
        # readout, so drop it to leave room for the light-source controls on small screens
        for child in tb.winfo_children():
            if isinstance(child, tk.Label):  # the message label and its two-line spacer
                child.pack_forget()
        tb.pack_propagate(True)  # matplotlib fixes the toolbar at the figure's width; size it to its buttons
        tb.pack(side="left")
        ttk.Checkbutton(bar, text="Auto-scale Y", variable=self.v_autoy, command=self._redraw).pack(
            side="left", padx=10)
        ttk.Button(bar, text="Full range", command=self.full_range).pack(side="left")
        readout = ttk.Frame(tab_spec, padding=(6, 0, 6, 2))
        readout.pack(side="top", fill="x")
        self.l_peak = ttk.Label(readout, text="Peak: -", style="Big.TLabel")
        self.l_peak.pack(side="left")
        self.l_cursor = ttk.Label(readout, text="Cursor: -")
        self.l_cursor.pack(side="right")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("draw_event", self._on_draw_event)
        self._show_message("Connect a spectrometer")

        # trend tab
        bar = ttk.Frame(tab_trend, padding=(0, 4))
        bar.pack(side="top", fill="x")
        ttk.Label(bar, text="Wavelengths (nm, comma-separated)").pack(side="left")
        ttk.Entry(bar, textvariable=self.v_track, width=18).pack(side="left", padx=4)
        ttk.Label(bar, text="+/- band (nm)").pack(side="left", padx=(8, 0))
        ttk.Entry(bar, textvariable=self.v_band, width=5).pack(side="left", padx=4)
        ttk.Label(bar, text="Log every (s, 0 = every spectrum)").pack(side="left", padx=(8, 0))
        ttk.Entry(bar, textvariable=self.v_every, width=5).pack(side="left", padx=4)
        self.b_trend = ttk.Button(bar, text="Start trend", command=self.toggle_trend)
        self.b_trend.pack(side="left", padx=(10, 4))
        ttk.Button(bar, text="Clear", command=self.clear_trend).pack(side="left")
        self.l_trend = ttk.Label(tab_trend, text="Tracks the current mode's value at each wavelength "
                                 "over time and logs it to logs/trends.", foreground="#666")
        self.l_trend.pack(side="top", anchor="w")
        self.tfig = Figure(figsize=(8, 4), dpi=100, layout="constrained")
        self.tax = self.tfig.add_subplot(111)
        self.tax.set_xlabel("Time (s)")
        self.tax.grid(True, alpha=0.3)
        self.tcanvas = FigureCanvasTkAgg(self.tfig, master=tab_trend)
        ttb = NavigationToolbar2Tk(self.tcanvas, tab_trend, pack_toolbar=False)
        ttb.update()
        ttb.pack(side="top", anchor="w")
        self.tcanvas.get_tk_widget().pack(fill="both", expand=True)

        # calibration tab
        tab_cal = ttk.Frame(self.nb)
        self.nb.add(tab_cal, text="Calibration")
        self.cal = caltab.CalibrationTab(self, tab_cal, self._cal_path)
        if self._data_dir_missing:
            self.set_status(f"The shared data folder {self.data_dir} is not reachable right now, so files are "
                            "saved in the app folder. Fix it under Save > Data folder.", "warn")
        elif self.data_dir:
            self.set_status(f"Saving to the shared data folder {self.data_dir}.")


    def _bind_keys(self) -> None:
        keys = {
            "<space>": self.toggle_run,
            "d": lambda: self.request_store("dark"),
            "r": lambda: self.request_store("reference"),
            "s": self.save_spectrum,
            "h": self.hold,
        }
        for key, fn in keys.items():
            self.root.bind_all(key, lambda e, fn=fn: self._key(e, fn))

    def _key(self, event, fn):
        try:
            cls = event.widget.winfo_class()
        except (AttributeError, tk.TclError):
            cls = ""
        if cls in TEXT_WIDGETS:
            return None
        if event.keysym == "space" and cls in ("TButton", "TCheckbutton", "TRadiobutton", "Button"):
            return None  # let the focused button handle its own space press
        fn()
        return "break"

    # ================================================================ status
    def set_status(self, text: str, kind: str = "") -> None:
        style = {"warn": "Warn.TLabel", "bad": "Bad.TLabel", "good": "Good.TLabel"}.get(kind, "TLabel")
        self.status.configure(text=text, style=style)

    def _show_message(self, text: str) -> None:
        self.msg.set_text(text)

    # ================================================================ devices
    def refresh_devices(self, auto: bool = False) -> None:
        if self.demo:
            self.devices = ["demo"]
            self.cb_dev.configure(values=["HR4000 (demo)  (serial DEMO0001)"])
            self.cb_dev.current(0)
            self.l_devinfo.configure(text="Demo mode: a simulated HR4000.")
            if auto:
                self.connect()
            return
        self.l_devinfo.configure(text="Looking for spectrometers...", style="TLabel")
        self.root.update_idletasks()
        try:
            spectro.load_backend(self.backend_pref)
            self.devices = spectro.list_devices()
        except Exception as exc:
            self.devices = []
            self.l_devinfo.configure(text=f"Could not load the USB backend: {exc}", style="Bad.TLabel")
            return
        names = [spectro.describe_device(d) for d in self.devices]
        self.cb_dev.configure(values=names)
        if names:
            self.cb_dev.current(0)
            self.l_devinfo.configure(text=f"Found {len(names)} spectrometer(s). Press Connect.", style="TLabel")
            if auto and len(names) == 1 and self.dev is None:
                self.connect()
        else:
            self.cb_dev.set("")
            self.l_devinfo.configure(text="No spectrometer found. Checking Windows...", style="Warn.TLabel")
            threading.Thread(target=self._diagnose, daemon=True).start()

    def _diagnose(self) -> None:
        hint = spectro.no_device_hint(spectro.windows_usb_status())
        if not self._closing:
            self._later(lambda: self.l_devinfo.configure(text=hint, style="Warn.TLabel"))

    def connect(self) -> None:
        if self.dev is not None:
            self.disconnect()
        i = self.cb_dev.current()
        if i < 0 or i >= len(self.devices):
            self.set_status("Pick a spectrometer first (press Refresh).", "warn")
            return
        self.set_status("Connecting...")
        self.root.update_idletasks()
        try:
            if self.devices[i] == "demo":
                dev = spectro.DemoSpectrometer()
            else:
                dev = spectro.OceanSpectrometer(self.devices[i])
        except Exception as exc:
            self.set_status(f"Could not open the spectrometer: {exc}", "bad")
            self.l_devinfo.configure(
                text="It may be open in another program (OceanView, SpectraSuite, another copy "
                     "of this app). Close that, replug the USB cable, and press Refresh.",
                style="Warn.TLabel")
            return
        self.dev = dev
        self.factory_wl = np.asarray(dev.wavelengths(), dtype=float)
        self.cal.reload_wavelengths()  # another laptop may have saved a correction to the shared folder
        self.wl = self.cal.corrected(dev.serial, self.factory_wl)
        self.raw = self.y = self.info = None
        self.dark = self.dark_info = self.ref = self.ref_info = None
        self.clear_held()
        lo, hi = dev.int_limits_us
        self.l_devinfo.configure(
            text=(f"{dev.pixels} px, {self.wl[0]:.0f}-{self.wl[-1]:.0f} nm, max {dev.max_intensity:.0f} counts\n"
                  f"Integration {lo / 1000:g} ms to {hi / 1e6:g} s (USB: {dev.backend})"),
            style="TLabel")
        self.c_edark.configure(state="normal" if dev.supports_dark else "disabled")
        self.c_nonlin.configure(state="normal" if dev.supports_nonlin else "disabled")
        if not dev.supports_dark:
            self.v_edark.set(False)
        if not dev.supports_nonlin:
            self.v_nonlin.set(False)
        if self.demo:
            self.demo_frame.pack(fill="x", pady=(0, 4))
            self._demo_controls()
        self.acq.attach(dev)
        if not self.apply_params():
            self.v_int.set(DEFAULTS["int_ms"])
            self.apply_params()
        self.acq.running = True
        self.b_run.configure(state="normal", text="Pause")
        self.b_single.configure(state="normal")
        self.b_disconnect.configure(state="normal")
        rng = spectro.LIGHT_SOURCES.get(self.v_source.get())
        if rng:  # open on the light source's working range
            self.ax.set_xlim(max(rng[0], float(self.wl[0])), min(rng[1], float(self.wl[-1])))
        else:
            self.ax.set_xlim(float(self.wl[0]), float(self.wl[-1]))
        self._show_message("Waiting for the first spectrum...")
        self._update_store_labels()
        self.cal.on_connect()
        self.canvas.draw_idle()
        self.set_status(f"Connected to {dev.model} ({dev.serial})."
                        + (" Using your wavelength correction." if self.wl is not self.factory_wl else ""),
                        "good")

    def set_wavelengths(self, wl) -> None:
        """Switch the wavelength axis (after a calibration change); spectra are per pixel, so the
        stored dark and reference stay valid."""
        self.wl = np.asarray(wl, dtype=float)
        self._bg = None
        self._redraw()
        self.canvas.draw_idle()

    def disconnect(self) -> None:
        if self.trend:
            self.toggle_trend()
        self.acq.detach()
        self.dev = None
        self.b_run.configure(state="disabled", text="Run")
        self.b_single.configure(state="disabled")
        self.b_disconnect.configure(state="disabled")
        self.demo_frame.pack_forget()
        self.l_sat.configure(text="")
        self.l_rate.configure(text="")
        self._show_message("Disconnected")
        self.canvas.draw_idle()
        self.set_status("Disconnected.")

    def _demo_controls(self) -> None:
        if isinstance(self.dev, spectro.DemoSpectrometer):
            self.dev.lamp = self.v_lamp.get()
            self.dev.sample_in = self.v_sample_in.get()
            self.dev.shutter_closed = self.v_shutter.get()

    # ================================================================ acquisition
    def apply_params(self) -> bool:
        """Validate the acquisition fields and hand them to the worker. Returns success."""
        try:
            int_ms = float(self.v_int.get())
            avg = int(float(self.v_avg.get()))
            if avg < 1 or avg > 1000:
                raise ValueError
        except ValueError:
            self.set_status("Integration time must be a number of ms and averaging 1-1000.", "bad")
            return False
        int_us = int(round(int_ms * 1000))
        if self.dev is not None:
            lo, hi = self.dev.int_limits_us
            if not lo <= int_us <= hi:
                clipped = min(max(int_us, lo), hi)
                self.set_status(f"Integration time limited to {lo / 1000:g}-{hi / 1000:g} ms; "
                                f"using {clipped / 1000:g} ms.", "warn")
                int_us = clipped
                self.v_int.set(f"{int_us / 1000:g}")
        params = {"int_us": int_us, "avg": avg,
                  "edark": bool(self.v_edark.get()), "nonlin": bool(self.v_nonlin.get())}
        if params != self.acq.params:
            self.acq.params = params
        self._update_store_labels()
        return True

    def toggle_run(self) -> None:
        if self.dev is None:
            return
        self.acq.running = not self.acq.running
        self.b_run.configure(text="Pause" if self.acq.running else "Run")
        self.set_status("Running." if self.acq.running else "Paused.")

    def single_shot(self) -> None:
        if self.dev is None:
            return
        self.acq.running = False
        self.b_run.configure(text="Run")
        self.acq.single = True
        self.set_status("Taking one acquisition...")

    def request_store(self, which: str) -> None:
        if self.dev is None:
            self.set_status("Connect a spectrometer first.", "warn")
            return
        self.pending_store = (which, time.time())
        if not self.acq.running:
            self.acq.single = True
        tip = "block the light path" if which == "dark" else "put the blank/reference in the beam"
        self.set_status(f"Storing {which}: waiting for a fresh spectrum ({tip}).")

    def _store(self, which: str) -> None:
        raw, info = self.raw.copy(), dict(self.info)
        if which == "dark":
            self.dark, self.dark_info = raw, info
        else:
            self.ref, self.ref_info = raw, info
        self.set_status(f"Stored {which} at {datetime.fromtimestamp(info['t_end']):%H:%M:%S} "
                        f"({info['int_us'] / 1000:g} ms x {info['avg']}).", "good")
        self._update_store_labels()

    def _mismatch(self, info) -> str:
        p = self.acq.params
        if info is None:
            return ""
        if info["int_us"] != p["int_us"]:
            return f"taken at {info['int_us'] / 1000:g} ms, now {p['int_us'] / 1000:g} ms: store again"
        if info["edark"] != p["edark"]:
            return "electric dark setting changed since: store again"
        if info["nonlin"] != p["nonlin"]:
            return "nonlinearity setting changed since: store again"
        return ""

    def _update_store_labels(self) -> None:
        for label, name, info in ((self.l_dark, "Dark", self.dark_info), (self.l_ref, "Reference", self.ref_info)):
            if info is None:
                label.configure(text=f"{name}: none", style="TLabel")
                continue
            when = datetime.fromtimestamp(info["t_end"]).strftime("%H:%M:%S")
            bad = self._mismatch(info)
            if bad:
                label.configure(text=f"{name}: {when}, {bad}", style="Warn.TLabel")
            else:
                label.configure(text=f"{name}: {when}, {info['int_us'] / 1000:g} ms x {info['avg']}",
                                style="Good.TLabel")

    # ================================================================ main loop
    def _later(self, fn) -> None:
        """Run fn on the GUI thread. Background threads must use this instead of touching Tk."""
        self._ui_calls.put(fn)

    def _poll(self) -> None:
        if self._closing:
            return
        t0 = time.perf_counter()
        while not self._ui_calls.empty():
            try:
                self._ui_calls.get_nowait()()
            except Exception as exc:  # a failing callback must not stop the poll loop
                self.set_status(f"Internal error: {exc}", "bad")
            if self._closing:
                return
        try:
            if self.acq.error and self.acq.error != self._last_status_error:
                self._last_status_error = self.acq.error
                self.b_run.configure(text="Run")
                self.set_status(f"Acquisition stopped: {self.acq.error}. Press Run to retry, or "
                                "replug and Connect again.", "bad")
            item = self.acq.take()
            if item is not None and self.dev is not None:
                self._last_status_error = None
                self.acq.error = None
                self.raw, self.info = item
                self._arrivals.append((time.time(), self.acq.delivered))
                if self.pending_store and self.info["t_start"] >= self.pending_store[1]:
                    which = self.pending_store[0]
                    self.pending_store = None
                    self._store(which)
                self._redraw(now=True)
                self._trend_sample()
                self.cal.on_spectrum()
                self._update_rate()
        finally:
            # Redraw at most as fast as drawing allows: wait at least twice as long as this
            # update took, so the window stays responsive at short integration times.
            spent_ms = (time.perf_counter() - t0) * 1000
            self._poll_id = self.root.after(int(min(500, max(40, 2 * spent_ms))), self._poll)

    def _update_rate(self) -> None:
        a = self._arrivals
        if len(a) >= 2 and a[-1][0] - a[0][0] > 0 and a[-1][1] > a[0][1]:
            rate = (a[-1][1] - a[0][1]) / (a[-1][0] - a[0][0])
            self.l_rate.configure(text=f"{rate:.1f} spectra/s" if rate >= 1 else f"{1 / rate:.1f} s/spectrum")
        p = self.info
        if p and self.dev is not None:
            mask = self._work_mask()
            peak = float(np.max(self.raw[mask]))
            rng = spectro.LIGHT_SOURCES.get(self.v_source.get())
            where = f" in {rng[0]:.0f}-{rng[1]:.0f} nm" if rng else ""
            # how full the detector itself is: corrected counts plus the offset that was subtracted
            detector_peak = peak + (p.get("offset", 0.0) if p.get("edark") else 0.0)
            sat = p.get("saturated")
            saturated = (bool(np.any(sat[mask])) if sat is not None and sat.shape == mask.shape
                         else detector_peak >= spectro.SATURATION * self.dev.max_intensity)
            if saturated:
                self.l_sat.configure(text=f"SATURATED{where}: lower the integration time (or click Set)",
                                     style="Bad.TLabel")
            else:
                pct = 100 * detector_peak / self.dev.max_intensity
                self.l_sat.configure(text=f"Max{where} {peak:.0f} counts ({pct:.0f}% of full scale)",
                                     style="Warn.TLabel" if pct < 10 else "TLabel")

    def _boxcar(self) -> int:
        try:
            return max(0, min(100, int(float(self.v_box.get()))))
        except ValueError:
            return 0

    def _redraw(self, now: bool = False) -> None:
        """Recompute the displayed spectrum. now=True draws immediately (used by the poll loop)."""
        if self.raw is None or self.wl is None:
            return
        mode = self.v_mode.get()
        before = (self.ax.get_ylabel(), self.msg.get_text(), self.ax.get_ylim(), self.ax.get_xlim())
        try:
            self.y = spectro.process(mode, self.raw, self.dark, self.ref, self._boxcar())
            self.y_ok = True
            self._show_message("")
        except ValueError as exc:
            self.y = spectro.boxcar(self.raw, self._boxcar())
            self.y_ok = False
            self._show_message(f"{mode}: {exc}\n(showing raw counts)")
        self.line.set_data(self.wl, self.y)
        self.ax.set_ylabel(spectro.MODE_UNITS[mode] if self.y_ok else "Counts")
        # In absorbance/transmittance, pixels where the reference barely rose above dark are
        # noise; leave them plotted but keep them out of the peak search and the Y auto-scale.
        ok = self._work_mask()
        if self.y_ok and mode in ("Absorbance", "Transmittance"):
            ok = ok & spectro.reliable_pixels(self.dark, self.ref, self._boxcar())
        y_trusted = self.y if ok.all() else np.where(ok, self.y, np.nan)
        lo, hi = self.ax.get_xlim()
        pk = spectro.peak_in_window(self.wl, y_trusted, lo, hi)
        if pk:
            wl_pk, val, fwhm = pk
            self.peak_mark.set_data([wl_pk], [val])
            unit = self.ax.get_ylabel()
            text = f"Peak in view: {wl_pk:.2f} nm, {val:.4g} {unit.split(' (')[0].lower()}"
            if fwhm:
                text += f", FWHM {fwhm:.2f} nm"
            self.l_peak.configure(text=text)
        else:
            self.peak_mark.set_data([], [])
            self.l_peak.configure(text="Peak in view: -")
        if self.v_autoy.get():
            self._autoscale_y(y_trusted, sticky=now)
        after = (self.ax.get_ylabel(), self.msg.get_text(), self.ax.get_ylim(), self.ax.get_xlim())
        if self.nb.index("current") != 0:
            self._spec_dirty = True  # drawn when the Spectrum tab is shown again
        elif not now:
            self.canvas.draw_idle()
        elif self._bg is None or after != before:
            self.canvas.draw()  # axes changed: full draw (re-caches the background)
        else:
            self._blit()

    # Live trace and peak marker are "animated": a full draw caches everything else as a
    # background image, and each new spectrum only redraws those two artists (blitting).
    def _on_draw_event(self, _event) -> None:
        if self._saving:
            return
        self._bg = self.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.peak_mark)

    def _blit(self) -> None:
        self.canvas.restore_region(self._bg)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.peak_mark)
        self.canvas.blit(self.ax.bbox)

    def _autoscale_y(self, live_y=None, sticky: bool = False) -> None:
        """Fit Y to what is in view. sticky=True (live updates) keeps the current limits while
        the data still fits and fills at least 60% of them, so the axes do not jitter."""
        lo, hi = self.ax.get_xlim()
        vals = []
        traces = [(self.wl, self.y if live_y is None else live_y)]
        traces += [(ln.get_xdata(), ln.get_ydata()) for ln in self.held]
        for x, yy in traces:
            x = np.asarray(x, dtype=float)
            yy = np.asarray(yy, dtype=float)
            if x.size != yy.size:
                continue
            sel = (x >= lo) & (x <= hi) & np.isfinite(yy)
            if np.any(sel):
                vals.append(yy[sel])
        if not vals:
            return
        v = np.concatenate(vals)
        ymin, ymax = float(np.min(v)), float(np.max(v))
        if sticky:
            lo_y, hi_y = self.ax.get_ylim()
            if lo_y <= ymin and ymax <= hi_y and (ymax - ymin) >= 0.6 * (hi_y - lo_y):
                return
        pad = (ymax - ymin) * 0.06 or 1.0
        self.ax.set_ylim(ymin - pad, ymax + pad)

    # ---------------------------------------------------------------- light source preset
    def _work_mask(self) -> np.ndarray:
        if self.wl is None:
            return np.ones(0, dtype=bool)
        return spectro.working_mask(self.wl, self.v_source.get())

    def set_light_source(self) -> None:
        """Measurement preset: show and use only the light source's working range."""
        self.s["light_source"] = self.v_source.get()
        rng = spectro.LIGHT_SOURCES.get(self.v_source.get())
        if self.wl is not None:
            lo, hi = (max(rng[0], float(self.wl[0])), min(rng[1], float(self.wl[-1]))) if rng else \
                (float(self.wl[0]), float(self.wl[-1]))
            self.ax.set_xlim(lo, hi)
            self.v_autoy.set(True)
            self._redraw()
            self._update_rate()
        if rng:
            self.set_status(f"{self.v_source.get()}: working range {rng[0]:.0f}-{rng[1]:.0f} nm. Peak, auto-scale and "
                            "the max-counts readout use only this range. 'Set' (next to Integration time) fills it to ~75%.", "good")
        else:
            self.set_status("Light source: full range.")

    def auto_integration(self) -> None:
        """Pick the integration time that puts the brightest point in the working range at ~75%."""
        if self.dev is None or self.raw is None or self.info is None:
            self.set_status("Connect and wait for a spectrum first.", "warn")
            return
        # only the light grows with integration time, not the electronic offset; and saturation is
        # judged on the uncorrected counts (flags from the acquisition)
        new_us, note = spectro.suggest_integration_us(
            self.info["int_us"], self.raw, self._work_mask(), self.dev.max_intensity, self.dev.int_limits_us,
            offset=self.info.get("offset", 0.0), corrected=bool(self.info.get("edark")),
            saturated=self.info.get("saturated"))
        if new_us == self.info["int_us"]:
            self.set_status(f"Integration time unchanged: {note}.", "warn")
            return
        self.v_int.set(f"{new_us / 1000:g}")
        if self.apply_params():
            self.set_status(f"Integration time set to {new_us / 1000:g} ms ({note}). Store dark and reference "
                            "again if you had stored them.", "good")

    def full_range(self) -> None:
        if self.wl is not None:
            self.ax.set_xlim(float(self.wl[0]), float(self.wl[-1]))
            self.v_autoy.set(True)
            self._redraw()

    def _on_mode(self) -> None:
        self.clear_held()
        self.s["mode"] = self.v_mode.get()
        self._redraw()

    def _on_motion(self, event) -> None:
        if event.inaxes is not self.ax or self.wl is None or self.y is None or event.xdata is None:
            return
        i = int(np.argmin(np.abs(self.wl - event.xdata)))
        self.l_cursor.configure(text=f"Cursor: {self.wl[i]:.2f} nm, {self.y[i]:.4g}  (pixel {i})")

    # ================================================================ held / overlays
    def hold(self) -> None:
        if self.y is None:
            return
        when = datetime.now().strftime("%H:%M:%S")
        name = self.v_sample.get().strip()
        (ln,) = self.ax.plot(self.wl, np.array(self.y), lw=0.9, alpha=0.75,
                             label=f"Held {when}" + (f" {name}" if name else ""))
        self.held.append(ln)
        self.ax.legend(loc="upper right", fontsize=8)
        self._redraw()

    def overlay_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Overlay a saved spectrum", initialdir=os.path.join(self.logs, "spectra"),
            filetypes=[("Spectrum CSV", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            meta, wl, cols = spectro.read_spectrum_csv(path)
        except Exception as exc:
            messagebox.showerror(APP_NAME, f"Could not read that file:\n{exc}")
            return
        file_mode = meta.get("Mode", "Scope")
        name = list(cols)[-1]
        y = cols[name]
        if np.all(np.isnan(y)):
            y, file_mode = cols["raw_counts"], "Scope"
        (ln,) = self.ax.plot(wl, y, lw=0.9, alpha=0.75, label=os.path.basename(path))
        self.held.append(ln)
        self.ax.legend(loc="upper right", fontsize=8)
        if file_mode != self.v_mode.get():
            self.set_status(f"Overlay is in {file_mode} units; the live trace is {self.v_mode.get()}.", "warn")
        self._redraw()
        self.canvas.draw_idle()

    def clear_held(self) -> None:
        for ln in self.held:
            ln.remove()
        self.held = []
        leg = self.ax.get_legend()
        if leg:
            leg.remove()
        self.canvas.draw_idle()

    # ================================================================ saving
    @staticmethod
    def _short(path: str) -> str:
        """Paths inside the app folder are shown relative to it, e.g. logs/spectra/x.csv."""
        try:
            rel = os.path.relpath(path, HERE)
        except ValueError:  # other drive
            return path
        return path if rel.startswith("..") else rel

    def _stem(self) -> str:
        name = spectro.safe_name(self.v_sample.get())
        return spectro.timestamp() + (f"_{name}" if name else "")

    def _meta(self) -> dict:
        d, p = self.dev, self.info or {}
        meta = {
            "Saved": datetime.now().isoformat(timespec="seconds"),
            "Program": f"{APP_NAME} {__version__}",
            "Computer": os.environ.get("COMPUTERNAME", ""),
            "Spectrometer": f"{d.model} serial {d.serial}" if d else "?",
            "USB backend": d.backend if d else "?",
            "Spectrum taken": datetime.fromtimestamp(p["t_end"]).isoformat(timespec="seconds") if p else "?",
            "Integration time (ms)": f"{p.get('int_us', 0) / 1000:g}",
            "Scans averaged": p.get("avg", "?"),
            "Boxcar half-width (px)": self._boxcar(),
            "Electric dark correction": p.get("edark", "?"),
            "Nonlinearity correction": p.get("nonlin", "?"),
            "Mode": self.v_mode.get() if self.y_ok else "Scope",
            "Wavelength calibration": self.cal.describe(d.serial if d else None),
            "Sample": self.v_sample.get().strip(),
            "Light source": self.v_source.get(),
        }
        for name, info in (("Dark", self.dark_info), ("Reference", self.ref_info)):
            meta[f"{name} stored"] = (
                f"{datetime.fromtimestamp(info['t_end']).isoformat(timespec='seconds')}, "
                f"{info['int_us'] / 1000:g} ms x {info['avg']}" if info else "none")
        meta["Note"] = "raw/dark/reference columns are unsmoothed; the last column has the boxcar applied"
        return meta

    def save_spectrum(self) -> str | None:
        if self.raw is None:
            self.set_status("No spectrum to save yet.", "warn")
            return None
        path = os.path.join(self.logs, "spectra", self._stem() + ".csv")
        mode = self.v_mode.get() if self.y_ok else "Scope"
        spectro.write_spectrum_csv(path, self.wl, self.raw, self.dark, self.ref, self.y, mode, self._meta())
        self.set_status(f"Saved spectrum: {self._short(path)}", "good")
        return path

    def save_graph(self) -> str | None:
        path = os.path.join(self.logs, "graphs", self._stem() + ".png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig = self.tfig if self.nb.index("current") == 1 else self.fig
        live = (self.line, self.peak_mark)
        self._saving = True
        try:
            for art in live:
                art.set_animated(False)
            fig.savefig(path, dpi=150)
        finally:
            for art in live:
                art.set_animated(True)
            self._saving = False
            self._bg = None
            self.canvas.draw_idle()
        self.set_status(f"Saved graph: {self._short(path)}", "good")
        return path

    def choose_data_dir(self) -> None:
        path = filedialog.askdirectory(
            title="Data folder for saved files and calibrations (pick the same shared folder on every laptop)",
            initialdir=self.data_dir or os.path.expanduser("~"))
        if path:
            self.set_data_dir(os.path.normpath(path))

    def set_data_dir(self, path: str) -> None:
        """Save files and calibrations in path ("" = the app folder). Settings stay on this computer."""
        if self.trend:
            self.toggle_trend()
        if path:
            try:
                os.makedirs(path, exist_ok=True)
                probe = os.path.join(path, ".write_test")
                with open(probe, "w", encoding="utf-8") as fh:
                    fh.write("ok")
                os.remove(probe)
            except OSError as exc:
                self.set_status(f"Cannot use {path} as the data folder: {exc}", "bad")
                return
        self.data_dir = path
        self._data_dir_missing = False
        self.logs = path or self._default_logs
        how = self.cal.set_path(os.path.join(path, "calibration.json") if path else self._default_cal)
        self.save_settings()
        where = path or "the app folder"
        cal_note = ("its calibrations are now in use" if how == "loaded"
                    else "this computer's calibrations were copied there")
        self.set_status(f"Data folder: {where}. New files are saved there, and {cal_note}.", "good")

    def open_logs(self) -> None:
        os.makedirs(self.logs, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(self.logs)

    # ================================================================ trend
    def toggle_trend(self) -> None:
        if self.trend:
            self.trend["fh"].close()
            path = self.trend["path"]
            self.trend = None
            self.b_trend.configure(text="Start trend")
            self.l_trend.configure(text=f"Trend saved to {self._short(path)}")
            self.set_status(f"Trend saved: {self._short(path)}", "good")
            return
        if self.wl is None:
            self.set_status("Connect a spectrometer first.", "warn")
            return
        try:
            wls = [float(x) for x in self.v_track.get().replace(";", ",").split(",") if x.strip()]
            band = float(self.v_band.get())
            every = float(self.v_every.get() or 0)
            if not wls or len(wls) > 6 or band < 0 or every < 0:
                raise ValueError
        except ValueError:
            self.set_status("Trend needs 1-6 wavelengths (e.g. 520, 700), a band >= 0 and an interval >= 0.", "bad")
            return
        rng = spectro.LIGHT_SOURCES.get(self.v_source.get())
        if rng and any(not rng[0] <= w <= rng[1] for w in wls):
            self.set_status(f"Note: some trend wavelengths are outside the {self.v_source.get()} range "
                            f"({rng[0]:.0f}-{rng[1]:.0f} nm), where there is little light.", "warn")
        out = [w for w in wls if not self.wl[0] <= w <= self.wl[-1]]
        if out:
            self.set_status(f"{out} nm is outside {self.wl[0]:.1f}-{self.wl[-1]:.1f} nm.", "bad")
            return
        mode = self.v_mode.get()
        path = os.path.join(self.logs, "trends", "trend_" + self._stem() + ".csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fh = open(path, "w", newline="", encoding="utf-8")
        meta = self._meta()
        meta["Trend band (+/- nm)"] = band
        meta["Trend interval (s)"] = every
        for k, v in meta.items():
            fh.write(f"# {k}: {v}\n")
        writer = csv.writer(fh)
        writer.writerow(["time", "elapsed_s", "mode"] + [f"{w:g}nm" for w in wls])
        fh.flush()
        self.trend = {"wls": wls, "band": band, "every": every, "t0": time.time(), "last": -1e9,
                      "fh": fh, "w": writer, "path": path}
        self.trend_data = {"t": [], "v": [[] for _ in wls], "mode": mode}
        self.tax.clear()
        self.tax.grid(True, alpha=0.3)
        self.tax.set_xlabel("Time (s)")
        self.tax.set_ylabel(spectro.MODE_UNITS[mode])
        self.tlines = [self.tax.plot([], [], ".-", ms=3, lw=1, label=f"{w:g} nm")[0] for w in wls]
        self.tax.legend(loc="best", fontsize=8)
        self.b_trend.configure(text="Stop trend")
        self.l_trend.configure(text=f"Logging to {self._short(path)}")
        self.set_status("Trend started.", "good")

    def _trend_sample(self) -> None:
        tr = self.trend
        if not tr or self.y is None:
            return
        now = time.time()
        if now - tr["last"] < tr["every"]:
            return
        if self.info and self.info["t_start"] < tr["t0"]:
            return  # acquired (partly) before Start was pressed
        if not self.y_ok:
            self.l_trend.configure(text=f"Not logging: {self.v_mode.get()} needs a stored dark/reference.")
            return
        if self.v_mode.get() != self.trend_data["mode"]:
            self.l_trend.configure(text="Mode changed: stop and start a new trend to log the new mode.")
            return
        tr["last"] = now
        t = (self.info["t_end"] if self.info else now) - tr["t0"]
        vals = [spectro.band_value(self.wl, self.y, w, tr["band"]) for w in tr["wls"]]
        tr["w"].writerow([datetime.fromtimestamp(tr["t0"] + t).isoformat(timespec="milliseconds"),
                          f"{t:.3f}", self.trend_data["mode"]] + [f"{v:.6g}" for v in vals])
        tr["fh"].flush()
        self.trend_data["t"].append(t)
        for k, v in enumerate(vals):
            self.trend_data["v"][k].append(v)
            self.tlines[k].set_data(self.trend_data["t"], self.trend_data["v"][k])
        self.l_trend.configure(text=f"Logging to {self._short(tr['path'])}  ({len(self.trend_data['t'])} points)")
        self.tax.relim()
        self.tax.autoscale_view()
        if self.nb.index("current") == 1:
            self.tcanvas.draw_idle()
        else:
            self._trend_dirty = True

    def _on_tab(self, _event=None) -> None:
        if self.nb.index("current") == 0 and self._spec_dirty:
            self._spec_dirty = False
            self.canvas.draw_idle()
        elif self.nb.index("current") == 1 and self._trend_dirty:
            self._trend_dirty = False
            self.tcanvas.draw_idle()
        elif self.nb.index("current") == 2 and self.cal.dirty:
            self.cal.dirty = False
            self.cal.canvas.draw_idle()

    def clear_trend(self) -> None:
        if self.trend:
            self.toggle_trend()
        self.trend_data = None
        self.tax.clear()
        self.tax.grid(True, alpha=0.3)
        self.tax.set_xlabel("Time (s)")
        self.tcanvas.draw_idle()

    # ================================================================ updates
    def check_updates(self, manual: bool = False) -> None:
        """Ask GitHub (in the background) whether a newer release exists."""
        if self._updating or self._closing:
            return
        if updater.is_dev_copy():
            if manual:
                self.set_status("This is the development copy: publish updates from it with "
                                "'python tools/publish.py'.", "warn")
            return
        if manual:
            self.set_status("Checking for updates...")

        def work():
            try:
                result, error = updater.check(), None
            except Exception as exc:  # offline, GitHub unreachable, ...
                result, error = None, f"{type(exc).__name__}: {exc}"
            if not self._closing:
                self._later(lambda: self._update_checked(result, error, manual))

        threading.Thread(target=work, daemon=True).start()

    def _update_checked(self, manifest, error, manual) -> None:
        if manifest:
            self._update_manifest = manifest
            notes = (manifest.get("notes") or "").strip()
            self.l_update.configure(text=f"Version {manifest['version']} is available (you have {__version__})."
                                         + (f"   What's new: {notes}" if notes else ""))
            self.b_update.configure(state="normal", text="Update now")
            self.b_update_later.configure(state="normal")
            self.update_bar.pack(side="top", fill="x", before=self._outer)
            if manual:
                self.set_status(f"Version {manifest['version']} is available.", "good")
        elif manual and error:
            self.set_status(f"Could not check for updates (no internet?): {error}", "warn")
        elif manual:
            self.set_status(f"You have the latest version ({__version__}).", "good")

    def _hide_update_bar(self) -> None:
        self.update_bar.pack_forget()

    def install_update(self, confirm: bool = True) -> None:
        m = self._update_manifest
        if not m or self._updating:
            return
        notes = (m.get("notes") or "").strip()
        if confirm and not messagebox.askyesno(
                APP_NAME, f"Update from {__version__} to {m['version']}?\n\n"
                          + (f"What's new: {notes}\n\n" if notes else "")
                          + "Measuring pauses, the new version is downloaded and checked, and the app "
                            "restarts. Your saved files and settings are kept."):
            return
        self._updating = True
        self._update_confirm = confirm
        self.acq.running = False
        self.b_run.configure(text="Run")
        if self.trend:
            self.toggle_trend()
        self.b_update.configure(state="disabled")
        self.b_update_later.configure(state="disabled")

        def say(text):
            self._later(lambda: self.l_update.configure(text=text))

        def work():
            try:
                changed = updater.apply(m, progress=say)
                if "requirements.txt" in changed:
                    say("Installing updated Python packages (this can take a minute)...")
                    ok, log = updater.install_requirements()
                    if not ok:
                        raise updater.UpdateError("installing the Python packages failed:\n" + log[-600:])
                self._later(lambda: self._update_done(m, changed, None))
            except Exception as exc:
                self._later(lambda e=exc: self._update_done(m, None, e))

        threading.Thread(target=work, daemon=True).start()

    def _update_done(self, manifest, changed, error) -> None:
        self._updating = False
        confirm = getattr(self, "_update_confirm", True)
        if error is not None:
            self.l_update.configure(text=f"Update failed: {error}")
            self.b_update.configure(state="normal", text="Try again")
            self.b_update_later.configure(state="normal")
            if confirm:
                messagebox.showerror(APP_NAME, f"The update did not install:\n\n{error}")
            return
        self.l_update.configure(text=f"Updated to {manifest['version']}. Restarting...")
        self.root.update_idletasks()
        if confirm:
            messagebox.showinfo(APP_NAME, f"Updated to version {manifest['version']} "
                                          f"({len(changed)} file(s) changed). The app will now restart.")
        updater.restart(sys.argv[1:])
        self.on_close()

    # ================================================================ shutdown
    def save_settings(self) -> None:
        s = {
            "int_ms": self.v_int.get(), "avg": self.v_avg.get(), "boxcar": self.v_box.get(),
            "edark": bool(self.v_edark.get()), "nonlin": bool(self.v_nonlin.get()),
            "mode": self.v_mode.get(), "autoy": bool(self.v_autoy.get()),
            "sample": self.v_sample.get(), "track": self.v_track.get(), "band": self.v_band.get(),
            "every": self.v_every.get(), "data_dir": self.data_dir, "light_source": self.v_source.get(),
            "geometry": "zoomed" if self.root.state() == "zoomed" else self.root.winfo_geometry(),
        }
        try:
            with open(self.settings_path, "w", encoding="utf-8") as fh:
                json.dump(s, fh, indent=2)
        except OSError:
            pass

    def on_close(self) -> None:
        self._closing = True
        if self.trend:
            self.toggle_trend()
        self.save_settings()
        self.acq.stop()
        try:
            self.root.after_cancel(self._poll_id)
        except (tk.TclError, AttributeError):
            pass
        self.root.destroy()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=APP_NAME)
    ap.add_argument("--demo", action="store_true", help="simulated HR4000, no hardware needed")
    ap.add_argument("--backend", choices=["cseabreeze", "pyseabreeze"], help="force a seabreeze backend")
    args = ap.parse_args(argv)
    try:  # crisp text on high-DPI screens
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root, demo=args.demo, backend=args.backend)
    root.mainloop()


if __name__ == "__main__":
    main(sys.argv[1:])
