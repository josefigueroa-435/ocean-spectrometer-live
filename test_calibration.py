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

                # ---- wavelength: Hg-Ar lamp, find lines, apply, survives reconnect, reset
                dev.lamp, dev.sample_in = "lines", False
                fresh()
                c.find_lines()
                self.assertGreaterEqual(len(c.found), 12)
                for d in c.found:  # the demo's built-in error: 0.30 nm + 0.06 nm per 100 nm above 600
                    self.assertAlmostEqual(d["error"], 0.30 + 0.0006 * (d["ref"] - 600), delta=0.04)
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
                for conc in (2, 4, 6, 8):
                    dev.sample_conc = conc
                    fresh()
                    c.v_conc.set(str(conc))
                    n = len(c.standards)
                    c.record_standard()
                    self.assertTrue(pump(5, lambda: len(c.standards) > n), f"standard {conc} not recorded")
                self.assertEqual(str(c.e_cwl.cget("state")), "disabled")  # wavelength locked by standards
                self.assertGreater(c.fit["r2"], 0.999)
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


if __name__ == "__main__":
    unittest.main()
