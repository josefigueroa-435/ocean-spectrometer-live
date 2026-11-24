"""Tests: python -m unittest test_calibration"""
import time
import unittest

import numpy as np

import calibration as cal
import spectro


class TestConcentrationCurve(unittest.TestCase):
    def test_exact_line(self):
        c = [0, 1, 2, 4, 8]
        a = [0.01 + 0.1 * x for x in c]
        f = cal.fit_line(c, a)
        self.assertAlmostEqual(f["slope"], 0.1)
        self.assertAlmostEqual(f["intercept"], 0.01)
        self.assertAlmostEqual(f["r2"], 1.0)
        self.assertAlmostEqual(cal.predict_conc(f, 0.51), 5.0)
        self.assertEqual((f["c_min"], f["c_max"]), (0.0, 8.0))

    def test_through_zero(self):
        f = cal.fit_line([1, 2, 3], [0.2, 0.4, 0.6], through_zero=True)
        self.assertAlmostEqual(f["slope"], 0.2)
        self.assertEqual(f["intercept"], 0.0)
        f1 = cal.fit_line([2], [0.3], through_zero=True)  # one standard is enough through zero
        self.assertAlmostEqual(f1["slope"], 0.15)

    def test_noisy_fit_lod(self):
        rng = np.random.default_rng(0)
        c = np.linspace(0, 10, 8)
        a = 0.05 * c + rng.normal(0, 0.002, c.size)
        f = cal.fit_line(c, a)
        self.assertAlmostEqual(f["slope"], 0.05, delta=0.002)
        self.assertGreater(f["r2"], 0.99)
        self.assertGreater(f["lod"], 0)
        self.assertAlmostEqual(f["loq"] / f["lod"], 10 / 3.3)
        self.assertEqual(len(f["residuals"]), 8)

    def test_not_enough_standards(self):
        with self.assertRaises(ValueError):
            cal.fit_line([1], [0.1])
        with self.assertRaises(ValueError):
            cal.fit_line([2, 2], [0.1, 0.2])


class TestWavelength(unittest.TestCase):
    def spectrum(self, offset=0.4, stretch=0.0, lines=None, noise=2.0, seed=1):
        """Lines at their true positions on a grid whose reported wavelengths are off."""
        true_wl = np.linspace(200, 1100, 3840)
        lines = cal.HG_AR_LINES if lines is None else lines
        y = np.zeros_like(true_wl)
        for k, line in enumerate(lines):
            y += (2000 + 300 * k) * np.exp(-0.5 * ((true_wl - line) / 0.3) ** 2)
        y += np.random.default_rng(seed).normal(0, noise, y.size) + 100
        reported = true_wl + offset + stretch * (true_wl - 600)
        return reported, y, true_wl

    def test_subpixel(self):
        x = np.arange(20.0)
        y = np.exp(-0.5 * ((x - 9.3) / 1.5) ** 2)
        self.assertAlmostEqual(cal.subpixel_peak(y, int(np.argmax(y))), 9.3, delta=0.05)

    def test_subpixel_no_pixel_phase_bias(self):
        """Narrow lines (sigma ~1 pixel, like the HR4000's) land on the right fraction of a pixel
        wherever their centre falls between pixels, also on top of a background."""
        x = np.arange(40.0)
        for sigma in (0.8, 1.1, 1.5):
            for frac in np.linspace(0.0, 0.9, 10):
                c = 20 + frac
                y = 100 + 5000 * np.exp(-0.5 * ((x - c) / sigma) ** 2)
                est = cal.subpixel_peak(y, int(np.argmax(y)), baseline=100)
                self.assertAlmostEqual(est, c, delta=0.01, msg=f"sigma {sigma}, centre {c}")

    def test_find_lines_measures_the_shift(self):
        wl, y, _ = self.spectrum(offset=0.4)
        found = cal.find_lines(wl, y, cal.HG_AR_LINES)
        self.assertEqual(len(found), len(cal.HG_AR_LINES))
        for d in found:
            self.assertAlmostEqual(d["error"], 0.4, delta=0.03)

    def test_doublet_resolved(self):
        wl, y, _ = self.spectrum(offset=0.2)
        found = {round(d["ref"], 2): d for d in cal.find_lines(wl, y, cal.HG_AR_LINES)}
        self.assertAlmostEqual(found[576.96]["found"], 577.16, delta=0.03)
        self.assertAlmostEqual(found[579.07]["found"], 579.27, delta=0.03)

    def test_weak_or_missing_lines_skipped(self):
        wl, y, _ = self.spectrum(lines=[546.074, 435.833])
        found = cal.find_lines(wl, y, cal.HG_AR_LINES)
        self.assertEqual(sorted(round(d["ref"], 1) for d in found), [435.8, 546.1])
        self.assertEqual(cal.find_lines(wl, np.full(wl.size, 100.0), cal.HG_AR_LINES), [])

    def test_fit_and_apply_correction(self):
        wl, y, true_wl = self.spectrum(offset=0.35, stretch=0.0006)
        found = cal.find_lines(wl, y, cal.HG_AR_LINES)
        fit = cal.fit_wavelength_correction([d["found"] for d in found], [d["ref"] for d in found], order=1)
        self.assertEqual(fit["order"], 1)
        self.assertLess(fit["rms"], 0.03)
        corrected = cal.apply_wavelength_correction(wl, fit["coeffs"])
        inside = (true_wl > 260) & (true_wl < 900)
        self.assertLess(np.max(np.abs(corrected[inside] - true_wl[inside])), 0.03)
        # after correcting, the lines are found where they belong
        for d in cal.find_lines(corrected, y, cal.HG_AR_LINES):
            self.assertAlmostEqual(d["error"], 0.0, delta=0.03)

    def test_large_offsets_found(self):
        for offset in (5.7, -8.0, 12.0):
            wl, y, _ = self.spectrum(offset=offset)
            self.assertAlmostEqual(cal.estimate_shift(wl, y, cal.HG_AR_LINES), offset, delta=0.1)
            found = cal.find_lines(wl, y, cal.HG_AR_LINES, window=1.5, max_shift=15)
            self.assertEqual(len(found), len(cal.HG_AR_LINES), offset)
            for d in found:
                self.assertAlmostEqual(d["error"], offset, delta=0.03)
            pair = {round(d["ref"], 2): d["found"] for d in found}
            self.assertAlmostEqual(pair[579.07] - pair[576.96], 2.106, delta=0.05)  # doublet not swapped
        # without the shift search a 5.7 nm error is out of reach of a 1.5 nm window
        wl, y, _ = self.spectrum(offset=5.7)
        self.assertEqual(cal.find_lines(wl, y, cal.HG_AR_LINES, window=1.5), [])

    def test_deuterium_lamp(self):
        """UV continuum + molecular band + two Balmer lines, read 5.7 nm high (like the real HR4000)."""
        true_wl = np.linspace(195, 1126, 3648)
        y = 500 * np.exp(-0.5 * ((true_wl - 380) / 90) ** 2)              # continuum
        y += 1200 * np.exp(-0.5 * ((true_wl - 585) / 8) ** 2)             # ~590 nm band
        y += 2600 * np.exp(-0.5 * ((true_wl - 486.0) / 0.27) ** 2)
        y += 11000 * np.exp(-0.5 * ((true_wl - 656.1) / 0.27) ** 2)
        y += np.random.default_rng(3).normal(0, 30, y.size)
        wl = true_wl + 5.7
        lines, window = cal.LAMP_PRESETS["Deuterium (D2) lamp"]
        found = cal.find_lines(wl, y, lines, window=window, max_shift=15)
        self.assertEqual([round(d["ref"], 1) for d in found], [486.0, 656.1])
        for d in found:
            self.assertAlmostEqual(d["error"], 5.7, delta=0.05)

    def test_order_lowered_with_few_lines(self):
        fit = cal.fit_wavelength_correction([546.4], [546.074], order=2)
        self.assertEqual(fit["order"], 0)
        self.assertAlmostEqual(fit["coeffs"][0], -0.326, delta=1e-9)
        self.assertEqual(cal.apply_wavelength_correction([1.0, 2.0], []).tolist(), [1.0, 2.0])


class TestDemoFixtures(unittest.TestCase):
    def test_demo_wavelength_error_is_correctable(self):
        sleep = spectro.time.sleep
        spectro.time.sleep = lambda s: None
        try:
            d = spectro.DemoSpectrometer(seed=5)
            d.lamp, d.sample_in = "lines", False
            d.set_integration_us(100_000)
            y = np.mean([d.intensities() for _ in range(5)], axis=0)
        finally:
            spectro.time.sleep = sleep
        found = cal.find_lines(d.wavelengths(), y, cal.HG_AR_LINES)
        self.assertGreaterEqual(len(found), 12)
        fit = cal.fit_wavelength_correction([f["found"] for f in found], [f["ref"] for f in found], 1)
        fixed = cal.apply_wavelength_correction(d.wavelengths(), fit["coeffs"])
        self.assertLess(np.max(np.abs(fixed - d._wl)[(d._wl > 300) & (d._wl < 900)]), 0.03)

    def test_demo_concentration(self):
        d = spectro.DemoSpectrometer(seed=6, start=time.monotonic())
        d.sample_conc = 10.0
        a = d.absorbance_now()
        self.assertAlmostEqual(a.max(), 0.45, delta=0.001)


def caltab_orders():
    import caltab
    return caltab.ORDERS


class TestCalibrationTab(unittest.TestCase):
    """Drive the real Calibration tab against the demo spectrometer."""

    def test_both_calibrations(self):
        import json
        import os
        import tempfile
        import tkinter as tk

        import app as app_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = tk.Tk()
            root.withdraw()
            a = app_mod.App(root, demo=True, settings_path=os.path.join(tmp, "settings.json"),
                            logs_dir=os.path.join(tmp, "logs"), auto_connect=False)
            c = a.cal

            def pump(seconds, until=None):
                end = time.time() + seconds
                while time.time() < end:
                    root.update()
                    if until and until():
                        return True
                    time.sleep(0.01)
                return False

            def fresh():
                t = time.time()
                self.assertTrue(pump(5, lambda: a.info and a.info["t_start"] > t), "no fresh spectrum")

            try:
                a.v_int.set("40")
                a.refresh_devices(auto=True)
                dev = a.dev

                # ---- lamp presets fill in lines, search width and (for 2 lines) a shift-only fit
                c.v_lamp.set("Deuterium (D2) lamp")
                c.use_preset()
                self.assertEqual(c.v_lines.get(), "486, 656.1")
                self.assertEqual(c.v_window.get(), "2")
                self.assertEqual(c._order(), 0)
                c.v_lamp.set("Hg-Ar lamp")
                c.use_preset()
                c.v_order.set(caltab_orders()[1])
                self.assertIn("546.074", c.v_lines.get())

                # ---- wavelength: Hg-Ar lamp, find lines, apply, survives reconnect, reset
                dev.lamp, dev.sample_in = "lines", False
                fresh()
                c.find_lines()
                self.assertGreaterEqual(len(c.found), 12)
                for d in c.found:  # the demo's built-in error: 0.30 nm + 0.06 nm per 100 nm above 600
                    # faint demo lines (one scan) wander a few hundredths of a nm with the noise
                    tol = 0.04 if d["snr"] > 100 else 0.10
                    self.assertAlmostEqual(d["error"], 0.30 + 0.0006 * (d["ref"] - 600), delta=tol)
                c.apply_correction()
                inside = (dev._wl > 300) & (dev._wl < 900)
                self.assertLess(np.max(np.abs(a.wl - dev._wl)[inside]), 0.03)
                with open(os.path.join(tmp, "calibration.json"), encoding="utf-8") as fh:
                    self.assertIn("DEMO0001", json.load(fh)["wavelength"])
                self.assertIn("correction", a._meta()["Wavelength calibration"])
                a.connect()  # reconnect: correction restored from calibration.json
                self.assertLess(np.max(np.abs(a.wl - a.dev._wl)[inside]), 0.03)
                c.reset_correction()
                np.testing.assert_array_equal(a.wl, a.factory_wl)
                c.apply_correction()  # keep it on for the rest

                # ---- concentration: dark, blank, four standards, one unknown
                dev = a.dev
                dev.lamp, dev.sample_in, dev.shutter_closed = "continuum", False, True
                a.request_store("dark")
                self.assertTrue(pump(5, lambda: a.dark is not None))
                dev.shutter_closed = False
                a.request_store("reference")
                self.assertTrue(pump(5, lambda: a.ref is not None))
                a.v_mode.set("Absorbance")
                a._on_mode()
                dev.sample_in = True
                c.v_cwl.set("520")
                c.v_cband.set("2")
                a.v_avg.set("5")  # average like a real measurement, so R2 is not at the mercy of one noisy scan
                a.apply_params()
                for conc in (2, 4, 6, 8):
                    dev.sample_conc = conc
                    fresh()
                    c.v_conc.set(str(conc))
                    n = len(c.standards)
                    c.record_standard()
                    self.assertTrue(pump(5, lambda: len(c.standards) > n), f"standard {conc} not recorded")
                self.assertEqual(str(c.e_cwl.cget("state")), "disabled")  # wavelength locked by standards
                self.assertGreater(c.fit["r2"], 0.998)
                self.assertAlmostEqual(c.fit["slope"], dev.EPSILON_PER_MM, delta=0.003)
                dev.sample_conc = 5.0
                fresh()
                a.v_sample.set("unknown A")
                c.record_unknown()
                self.assertTrue(pump(5, lambda: c.unknowns))
                self.assertAlmostEqual(c.unknowns[0]["c"], 5.0, delta=0.15)
                self.assertIn("mM", c.l_live.cget("text"))
                path = c.save_csv()
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                self.assertIn("# R2:", text)
                self.assertIn("unknown,unknown A,", text)
            finally:
                a.on_close()


class TestSharedDataFolder(unittest.TestCase):
    """Two 'laptops' (separate settings files) sharing one data folder."""

    def make_app(self, tmp, name):
        import os
        import tkinter as tk

        import app as app_mod

        root = tk.Tk()
        root.withdraw()
        os.makedirs(os.path.join(tmp, name), exist_ok=True)
        a = app_mod.App(root, demo=True, settings_path=os.path.join(tmp, name, "settings.json"),
                        logs_dir=os.path.join(tmp, name, "logs"), auto_connect=False)

        def pump(seconds, until=None):
            end = time.time() + seconds
            while time.time() < end:
                root.update()
                if until and until():
                    return True
                time.sleep(0.01)
            return False

        return a, pump

    def test_shared_folder(self):
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            shared = os.path.join(tmp, "shared")
            # ---- laptop A: wavelength correction made locally, then switch to the (empty) shared folder
            a, pump = self.make_app(tmp, "laptopA")
            try:
                a.v_int.set("40")
                a.refresh_devices(auto=True)
                a.dev.lamp, a.dev.sample_in = "lines", False
                t = time.time()
                pump(5, lambda: a.info and a.info["t_start"] > t)
                a.cal.find_lines()
                a.cal.apply_correction()
                a.set_data_dir(shared)
                self.assertEqual(a.logs, shared)
                with open(os.path.join(shared, "calibration.json"), encoding="utf-8") as fh:
                    self.assertIn("DEMO0001", json.load(fh)["wavelength"])  # copied into the shared folder
                path = a.save_spectrum()
                self.assertTrue(path.startswith(os.path.join(shared, "spectra")))
                self.assertTrue(os.path.exists(path))
            finally:
                a.on_close()
            with open(os.path.join(tmp, "laptopA", "settings.json"), encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["data_dir"], shared)

            # ---- laptop B: points at the same folder and gets A's correction for the same spectrometer
            os.makedirs(os.path.join(tmp, "laptopB"), exist_ok=True)
            with open(os.path.join(tmp, "laptopB", "settings.json"), "w", encoding="utf-8") as fh:
                json.dump({"data_dir": shared}, fh)
            b, pump = self.make_app(tmp, "laptopB")
            try:
                self.assertEqual(b.logs, shared)
                b.refresh_devices(auto=True)
                self.assertFalse(np.array_equal(b.wl, b.factory_wl))  # correction applied
                inside = (b.dev._wl > 300) & (b.dev._wl < 900)
                self.assertLess(np.max(np.abs(b.wl - b.dev._wl)[inside]), 0.03)

                # another laptop saves a correction for a different spectrometer meanwhile...
                p = os.path.join(shared, "calibration.json")
                with open(p, encoding="utf-8") as fh:
                    data = json.load(fh)
                data["wavelength"]["OTHER123"] = {"coeffs": [0.1], "order": 0, "rms": 0.0, "range": [300, 800],
                                                  "date": "x", "lines": []}
                with open(p, "w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                # ...then B resets DEMO0001: OTHER123 must survive, DEMO0001 must go
                b.cal.reset_correction()
                with open(p, encoding="utf-8") as fh:
                    wl = json.load(fh)["wavelength"]
                self.assertIn("OTHER123", wl)
                self.assertNotIn("DEMO0001", wl)
            finally:
                b.on_close()

            # ---- unreachable shared folder: falls back to the app folder with a warning
            with open(os.path.join(tmp, "laptopB", "settings.json"), "w", encoding="utf-8") as fh:
                json.dump({"data_dir": os.path.join(tmp, "not-mounted")}, fh)
            c, pump = self.make_app(tmp, "laptopB")
            try:
                self.assertEqual(c.logs, os.path.join(tmp, "laptopB", "logs"))
                self.assertIn("not reachable", c.status.cget("text"))
            finally:
                c.on_close()


if __name__ == "__main__":
    unittest.main()
