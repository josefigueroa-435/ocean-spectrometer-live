# Ocean Spectrometer Live

Live spectra from an Ocean Optics spectrometer, made for the **HR4000** (3840 px, 14-bit). It also
works with any other model python-seabreeze supports (USB2000+, USB4000, Flame, QE65000, ...).

The app only **reads** spectra and sets the integration time. It never writes the spectrometer's
EEPROM (wavelength or nonlinearity calibration).

## First-time setup

1. **Driver (once per computer).** Double-click `Install HR4000 driver (run once).bat` and click
   **Yes** on the admin prompt. It installs Ocean Optics' own WinUSB driver package (WHQL-signed by
   Microsoft, from python-seabreeze's `os_support/windows-driver-files.zip`). Then unplug and
   replug the HR4000. Device Manager should show *Ocean Optics HR4000 (WinUSB)* under
   *Ocean Optics USB Devices*. (For other models, run `py -m seabreeze.os_setup`.)
2. **Python packages** (already installed on this laptop):
   `python -m pip install -r requirements.txt`

## Running

- Desktop shortcut **Ocean Spectrometer Live**, or `Ocean Spectrometer Live.bat`.
- `Ocean Spectrometer Live (demo).bat` runs a simulated HR4000 (halogen or Hg-Ar lamp, a sample
  absorbing near 520 nm that fades over a few minutes). No hardware needed.
- `python app.py --backend pyseabreeze` forces the pure-Python USB backend if the default
  (`cseabreeze`) has trouble.

If no spectrometer is found, the panel says what Windows sees and what to do next.

## Measuring absorbance (or transmittance)

1. Set the integration time so the reference peak is roughly 50-85% of full scale (the label under
   Run/Pause shows the max counts and warns when saturated). Add scans to average for less noise.
2. Block the light, then press **Store dark** (or `D`).
3. Put the blank/reference cuvette in, then press **Store reference** (or `R`).
4. Put the sample in and pick **Absorbance** (A = -log10[(S-D)/(R-D)]) or **Transmittance**.

Dark and reference are taken from the *next* spectrum after you click, so nothing stale is stored.
If you change the integration time or the correction settings afterwards, their labels turn orange
with "store again".

Pixels where the reference barely rises above dark (for a halogen lamp, below about 300 nm) are
still plotted but are left out of the peak search and the Y auto-scale.

## Other features

- **Peak in view**: wavelength, height and FWHM of the largest peak in the visible range. Zoom
  with the toolbar to pick a peak.
- **Hold** (`H`) freezes a copy of the trace, **Overlay file...** loads a saved spectrum, and
  **Clear held** removes them.
- **Save CSV** (`S`) writes `logs/spectra/<time>_<sample>.csv`. Lines starting with `#` hold the
  settings, followed by the wavelength, raw, dark, reference and processed columns (only the
  processed column has the boxcar applied). **Save graph** writes a PNG to `logs/graphs`.
- **Trend tab**: tracks the current mode's value at up to 6 wavelengths (± a band) over time and
  logs to `logs/trends/trend_<time>.csv`. Useful for kinetics, for example absorbance at λmax vs time.
- `Space` toggles Run/Pause. **Single** takes one acquisition.
- Electric dark correction subtracts the mean of the HR4000's optically masked pixels.
  Nonlinearity correction uses the coefficients stored in the spectrometer.

## Handing it to someone else

`python tools/build_package.py OUT_DIR` builds `Ocean_Spectrometer_Live_<date>.zip`: the program, the HR4000
driver, the .bat helpers (`Install requirements.bat`, `Make desktop shortcut.bat`, ...), `READ ME FIRST.txt` and
`Ocean Spectrometer Live - Guide.pdf` (printed from `docs/Ocean_Spectrometer_Live_Guide.html` with Edge).
It runs the tests on the packaged copy first and refuses to build if logs, settings or personal paths would go in.
Guide screenshots in `docs/img` come from the demo spectrometer.

## Files

`spectro.py` (device access, demo spectrometer, spectral math, CSV), `acquire.py` (acquisition
thread), `app.py` (window), `test_spectro.py` (`python -m unittest test_spectro`, 16 tests, no
hardware needed), `driver/` (Ocean's HR4000 INF + catalog + co-installers), `tools/make_icon.py`.
