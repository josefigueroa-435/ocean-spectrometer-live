"""Ocean Optics spectrometer access (python-seabreeze), a demo spectrometer, and spectral math.

The app only ever *reads* spectra and sets the integration time. It never writes the
spectrometer's EEPROM (wavelength / nonlinearity calibration) or any other stored setting.
"""
from __future__ import annotations

import csv
import math
import os
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------- backend

_backend_name: str | None = None


def _add_libusb_to_path() -> None:
    """pyseabreeze needs libusb-1.0.dll; the libusb-package wheel ships one."""
    try:
        import libusb_package
    except ImportError:
        return
    path = libusb_package.get_library_path()
    if path:
        folder = os.path.dirname(str(path))
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(folder)
        os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")


def load_backend(prefer: str | None = None) -> str:
    """Pick the seabreeze backend once per process (seabreeze only allows one).

    cseabreeze talks to Ocean's WinUSB driver directly; pyseabreeze (pure Python + libusb)
    is the fallback. Returns the backend's name.
    """
    global _backend_name
    if _backend_name:
        return _backend_name
    import seabreeze

    order = [prefer] if prefer else ["cseabreeze", "pyseabreeze"]
    last_error: Exception | None = None
    for name in order:
        try:
            if name == "pyseabreeze":
                _add_libusb_to_path()
            seabreeze.use(name)
            import seabreeze.spectrometers  # noqa: F401  (forces the backend to load)

            _backend_name = name
            return name
        except Exception as exc:  # backend missing or failed to load
            last_error = exc
    raise RuntimeError(f"No seabreeze backend could be loaded: {last_error}")


def list_devices() -> list:
    load_backend()
    from seabreeze.spectrometers import list_devices as _list

    return list(_list())


def windows_usb_status() -> list[tuple[str, str, str]]:
    """Ocean Optics devices (USB vendor 2457) Windows can see: (name, status, problem code)."""
    if os.name != "nt":
        return []
    import subprocess

    script = (
        "Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -match 'VID_2457' } | "
        "ForEach-Object { '{0}|{1}|{2}' -f $_.FriendlyName, $_.Status, $_.Problem }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and parts[0]:
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def no_device_hint(usb_rows) -> str:
    """Plain-language next step when seabreeze finds no spectrometer."""
    if not usb_rows:
        return ("No Ocean Optics spectrometer is on USB. Check the cable and that the "
                "spectrometer has power, then press Refresh.")
    bad = [r for r in usb_rows if r[1].upper() != "OK"]
    if bad:
        name = bad[0][0]
        if "HR4000" in name:
            return (f"Windows sees the {name} but has no driver for it. Run "
                    "'Install HR4000 driver (run once).bat' in the app folder (click Yes "
                    "on the admin prompt), unplug and replug the spectrometer, then press Refresh.")
        return (f"Windows sees the {name} but has no driver for it. In Command Prompt run "
                "'py -m seabreeze.os_setup' (installs Ocean's drivers for every model; needs "
                "internet and admin), replug, then press Refresh.")
    name = usb_rows[0][0]
    return (f"Windows has a driver for the {name}, but it could not be opened. Close "
            "OceanView/SpectraSuite or any other program using it, replug, and press Refresh.")


def describe_device(dev) -> str:
    try:
        return f"{dev.model}  (serial {dev.serial_number})"
    except Exception:
        return str(dev)


# ---------------------------------------------------------------- devices


class SpectrometerBase:
    """What the GUI needs from a spectrometer."""

    model = "?"
    serial = "?"
    backend = "?"
    pixels = 0
    max_intensity = 65535.0
    int_limits_us = (1000, 10_000_000)
    supports_dark = False
    supports_nonlin = False

    def wavelengths(self) -> np.ndarray:
        raise NotImplementedError

    def intensities(self, dark: bool = False, nonlin: bool = False) -> np.ndarray:
        raise NotImplementedError

    def set_integration_us(self, us: int) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class OceanSpectrometer(SpectrometerBase):
    """A real spectrometer through python-seabreeze."""

    def __init__(self, device=None):
        self.backend = load_backend()
        from seabreeze.spectrometers import Spectrometer

        self._spec = Spectrometer(device) if device is not None else Spectrometer.from_first_available()
        self.model = self._spec.model
        self.serial = self._spec.serial_number
        self.pixels = int(self._spec.pixels)
        self.max_intensity = float(self._spec.max_intensity)
        lo, hi = self._spec.integration_time_micros_limits
        self.int_limits_us = (int(lo), int(hi))
        self._wl = np.asarray(self._spec.wavelengths(), dtype=float)
        try:  # free-running acquisition
            self._spec.trigger_mode(0)
        except Exception:
            pass
        # same tests seabreeze's Spectrometer.intensities() uses before it corrects
        self.supports_dark = self._truthy(getattr(self._spec, "_dp", None))
        self.supports_nonlin = self._truthy(getattr(self._spec, "_nc", None))

    @staticmethod
    def _truthy(value) -> bool:
        try:
            return bool(value)
        except Exception:
            return value is not None

    def wavelengths(self) -> np.ndarray:
        return self._wl

    def intensities(self, dark: bool = False, nonlin: bool = False) -> np.ndarray:
        return np.asarray(
            self._spec.intensities(
                correct_dark_counts=dark and self.supports_dark,
                correct_nonlinearity=nonlin and self.supports_nonlin,
            ),
            dtype=float,
        )

    def set_integration_us(self, us: int) -> None:
        lo, hi = self.int_limits_us
        self._spec.integration_time_micros(int(min(max(us, lo), hi)))

    def close(self) -> None:
        try:
            self._spec.close()
        except Exception:
            pass


# Hg-Ar lamp lines (nm) used by the demo; real HR4000 wavelengths come from its EEPROM.
DEMO_LINES = [253.65, 296.73, 313.16, 365.02, 404.66, 435.83, 546.07, 576.96, 579.07,
              696.54, 706.72, 738.40, 750.39, 763.51, 772.38, 811.53, 826.45, 842.46, 912.30]


class DemoSpectrometer(SpectrometerBase):
    """Simulated HR4000 (3840 px, 14-bit) for trying the app without hardware.

    Shows a halogen continuum, Hg-Ar lines, and a sample absorption band near 520 nm whose
    depth decays over a few minutes, so the Trend tab has something kinetic to follow. Set
    sample_conc (mM) for a steady Beer's-law sample instead. Its "factory" wavelength calibration
    is deliberately off by about +0.3 nm, so the wavelength recalibration has something to fix.
    """
    EPSILON_PER_MM = 0.045  # demo absorbance per mM at the 520 nm band maximum


    def __init__(self, seed: int | None = None, start: float | None = None):
        self.model = "HR4000 (demo)"
        self.serial = "DEMO0001"
        self.backend = "demo"
        self.pixels = 3840
        self.max_intensity = 16383.0
        self.int_limits_us = (10, 655_350_000)
        self.supports_dark = True
        self.supports_nonlin = True
        self._rng = np.random.default_rng(seed)
        self._start = time.monotonic() if start is None else start
        self._int_us = 100_000
        x = np.linspace(0, 1, self.pixels)
        self._wl = 199.6 + 900.0 * x - 18.0 * x * (1 - x)  # slightly nonlinear like a real grating
        wl = self._wl
        # halogen-ish continuum (Planck at 3000 K) times a detector response
        t_k = 3000.0
        lam_m = wl * 1e-9
        planck = 1.0 / (lam_m**5 * (np.exp(1.4388e-2 / (lam_m * t_k)) - 1.0))
        response = np.exp(-(((wl - 600.0) / 260.0) ** 2))
        cont = planck * response
        self._continuum = cont / cont.max()
        lines = np.zeros_like(wl)
        for i, c in enumerate(DEMO_LINES):
            lines += (0.25 + 0.75 * ((i * 7) % 5) / 4) * np.exp(-0.5 * ((wl - c) / 0.35) ** 2)
        self._lines = lines
        self.lamp = "continuum"  # or "lines"
        self.sample_in = True
        self.shutter_closed = False
        self.sample_conc: float | None = None
        self._dark_level = 95.0
        # what the demo "EEPROM" reports: true wavelength + 0.30 nm + 0.06 nm per 100 nm above 600
        self._wl_reported = wl + 0.30 + 0.0006 * (wl - 600.0)

    def wavelengths(self) -> np.ndarray:
        return self._wl_reported

    def set_integration_us(self, us: int) -> None:
        lo, hi = self.int_limits_us
        self._int_us = int(min(max(us, lo), hi))

    def absorbance_now(self, t: float | None = None) -> np.ndarray:
        """True absorbance of the demo sample (decays with a 90 s time constant)."""
        if self.sample_conc is not None:
            depth = self.EPSILON_PER_MM * self.sample_conc
            return depth * np.exp(-0.5 * ((self._wl - 520.0) / 28.0) ** 2)
        if t is None:
            t = time.monotonic() - self._start
        depth = 0.05 + 0.85 * math.exp(-t / 90.0)
        return depth * np.exp(-0.5 * ((self._wl - 520.0) / 28.0) ** 2)

    def intensities(self, dark: bool = False, nonlin: bool = False) -> np.ndarray:
        seconds = self._int_us / 1e6
        if self.lamp == "lines":
            signal = 9000.0 * self._lines
        else:
            signal = 11000.0 * self._continuum
        if self.sample_in:
            signal = signal * 10.0 ** (-self.absorbance_now())
        if self.shutter_closed:
            signal = signal * 0.0
        signal = signal * (seconds / 0.1)
        shot = self._rng.normal(0.0, 1.0, self.pixels) * np.sqrt(np.maximum(signal, 0.0) * 0.6)
        read = self._rng.normal(0.0, 6.0, self.pixels)
        counts = signal + shot + read + self._dark_level + 40.0 * seconds
        counts = np.clip(np.round(counts), 0, self.max_intensity)
        if dark:
            counts = counts - (self._dark_level + 40.0 * seconds)
        # pace like hardware: one integration per read (capped so the demo stays responsive)
        time.sleep(min(seconds, 0.5) + 0.004)
        return counts


# ---------------------------------------------------------------- math

# Measurement presets: the wavelength range where each light source gives usable light (nm).
# None = no limit. Ranges are approximate working ranges, not hard limits of the lamps.
LIGHT_SOURCES = {
    "Full range (any source)": None,
    "Halogen (visible-NIR)": (400.0, 1000.0),
    "Deuterium (UV)": (200.0, 400.0),
    "Deuterium + halogen": (200.0, 1000.0),
}


def working_mask(wl, source: str) -> np.ndarray:
    """True for pixels inside the light source's working range (all True for no limit)."""
    wl = np.asarray(wl, dtype=float)
    rng = LIGHT_SOURCES.get(source)
    if not rng:
        return np.ones(wl.shape, dtype=bool)
    mask = (wl >= rng[0]) & (wl <= rng[1])
    return mask if mask.any() else np.ones(wl.shape, dtype=bool)


def suggest_integration_us(current_us: int, raw, mask, max_intensity: float, limits_us,
                           target: float = 0.75) -> tuple[int, str]:
    """Integration time that brings the brightest pixel inside mask to `target` of full scale.

    Counts scale linearly with integration time (above the electronic offset). If the range is
    saturated the true level is unknown, so the time is cut to a third and the user tries again.
    Returns (new integration time in us, explanation).
    """
    raw = np.asarray(raw, dtype=float)
    sel = raw[np.asarray(mask, dtype=bool)]
    lo, hi = limits_us
    if sel.size == 0:
        return current_us, "no pixels in the working range"
    peak = float(np.nanmax(sel))
    if peak >= 0.98 * max_intensity:
        new = int(max(lo, current_us / 3))
        return new, "saturated in the working range: shortened, click again to fine-tune"
    if peak < 0.01 * max_intensity:
        return current_us, "almost no light in the working range (is the lamp on and the fiber connected?)"
    new = int(round(current_us * target * max_intensity / peak))
    clipped = int(min(max(new, lo), hi))
    note = f"brightest point in range was {100 * peak / max_intensity:.0f}% of full scale"
    if clipped != new:
        note += "; limited by the spectrometer's integration range"
    return clipped, note


MODES = ("Scope", "Scope - dark", "Absorbance", "Transmittance")
MODE_UNITS = {
    "Scope": "Counts",
    "Scope - dark": "Counts (dark subtracted)",
    "Absorbance": "Absorbance (AU)",
    "Transmittance": "Transmittance (%)",
}


def boxcar(y: np.ndarray, half_width: int) -> np.ndarray:
    """Ocean-style boxcar: average of each pixel with `half_width` neighbours on each side."""
    if y is None or half_width <= 0:
        return y
    w = 2 * int(half_width) + 1
    kernel = np.ones(w) / w
    padded = np.pad(y, half_width, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def process(mode: str, sample, dark=None, reference=None, half_width: int = 0) -> np.ndarray:
    """Turn raw counts into the displayed quantity.

    Absorbance A = -log10((S - D) / (R - D)); transmittance %T = 100 (S - D) / (R - D).
    Pixels where the reference is not above dark come out as NaN (not plotted).
    """
    s = boxcar(np.asarray(sample, dtype=float), half_width)
    d = boxcar(np.asarray(dark, dtype=float), half_width) if dark is not None else None
    if mode == "Scope":
        return s
    if d is None:
        raise ValueError("Store a dark spectrum first.")
    if mode == "Scope - dark":
        return s - d
    if reference is None:
        raise ValueError("Store a reference spectrum first.")
    r = boxcar(np.asarray(reference, dtype=float), half_width)
    num = s - d
    den = r - d
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(den > 0, num / den, np.nan)
        if mode == "Transmittance":
            return 100.0 * ratio
        if mode == "Absorbance":
            return np.where(ratio > 0, -np.log10(ratio), np.nan)
    raise ValueError(f"Unknown mode {mode!r}")


def reliable_pixels(dark, reference, half_width: int = 0, fraction: float = 0.02) -> np.ndarray:
    """Pixels where the reference rose at least `fraction` of its peak above dark.

    Outside these (e.g. below ~300 nm with a halogen lamp) absorbance is mostly noise.
    """
    d = boxcar(np.asarray(dark, dtype=float), half_width)
    r = boxcar(np.asarray(reference, dtype=float), half_width)
    signal = r - d
    top = np.nanmax(signal) if signal.size else 0.0
    if not np.isfinite(top) or top <= 0:
        return np.zeros(signal.shape, dtype=bool)
    return signal > fraction * top


def peak_in_window(wl: np.ndarray, y: np.ndarray, lo: float, hi: float):
    """Largest value between lo and hi nm: returns (wavelength, value, fwhm_nm) or None.

    FWHM is measured at half height above the window's minimum, with linear interpolation.
    """
    if y is None:
        return None
    sel = np.where((wl >= min(lo, hi)) & (wl <= max(lo, hi)) & np.isfinite(y))[0]
    if sel.size < 3:
        return None
    i = sel[np.argmax(y[sel])]
    peak = float(y[i])
    base = float(np.nanmin(y[sel]))
    if peak <= base:
        return float(wl[i]), peak, None
    half = base + (peak - base) / 2.0
    first, last = sel[0], sel[-1]
    left = i
    while left > first and y[left] > half:
        left -= 1
    right = i
    while right < last and y[right] > half:
        right += 1
    fwhm = None
    if y[left] <= half < y[left + 1] and y[right] <= half < y[right - 1]:
        wl_left = np.interp(half, [y[left], y[left + 1]], [wl[left], wl[left + 1]])
        wl_right = np.interp(half, [y[right], y[right - 1]], [wl[right], wl[right - 1]])
        fwhm = float(wl_right - wl_left)
    return float(wl[i]), peak, fwhm


def band_value(wl: np.ndarray, y: np.ndarray, center: float, half_band: float) -> float:
    """Mean of y over center +/- half_band nm (at least the nearest pixel)."""
    sel = (wl >= center - half_band) & (wl <= center + half_band) & np.isfinite(y)
    if not np.any(sel):
        i = int(np.argmin(np.abs(wl - center)))
        return float(y[i])
    return float(np.mean(y[sel]))


# ---------------------------------------------------------------- files


def safe_name(text: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_ ." else "_" for c in text.strip())
    return "_".join(keep.split())[:60]


def write_spectrum_csv(path: str, wl, raw, dark, reference, processed, mode: str, meta: dict) -> str:
    """One spectrum per file: '#' metadata lines, then one row per pixel."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    unit = MODE_UNITS.get(mode, mode)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        for key, value in meta.items():
            fh.write(f"# {key}: {value}\n")
        w = csv.writer(fh)
        w.writerow(["wavelength_nm", "raw_counts", "dark_counts", "reference_counts", f"{mode} [{unit}]"])
        n = len(wl)

        def col(a, k):
            if a is None:
                return ""
            v = float(a[k])
            return "" if not math.isfinite(v) else f"{v:.6g}"

        for k in range(n):
            w.writerow([f"{float(wl[k]):.4f}", col(raw, k), col(dark, k), col(reference, k), col(processed, k)])
    return path


def read_spectrum_csv(path: str):
    """Read a file written by write_spectrum_csv: returns (meta, wavelengths, columns dict)."""
    meta: dict[str, str] = {}
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                meta[key.strip()] = value.strip()
            else:
                rows.append(line)
    reader = csv.reader(rows)
    header = next(reader)
    data = list(reader)
    cols = {}
    for j, name in enumerate(header):
        cols[name] = np.array([float(r[j]) if r[j] != "" else np.nan for r in data])
    wl = cols.pop("wavelength_nm")
    return meta, wl, cols


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")
