"""Tests: python -m unittest test_spectro   (no spectrometer needed)."""
import json
import os
import tempfile
import time
import unittest

import numpy as np

import spectro
from acquire import Acquirer


class TestMath(unittest.TestCase):
    def test_boxcar(self):
        y = np.array([0, 0, 3, 0, 0], dtype=float)
        np.testing.assert_allclose(spectro.boxcar(y, 1), [0, 1, 1, 1, 0])
        np.testing.assert_array_equal(spectro.boxcar(y, 0), y)
        flat = np.full(50, 7.0)
        np.testing.assert_allclose(spectro.boxcar(flat, 5), flat)  # edges padded, no droop

    def test_modes(self):
        d = np.full(4, 100.0)
        r = np.array([1100.0, 1100.0, 1100.0, 100.0])  # last pixel: reference == dark
        s = np.array([1100.0, 200.0, 600.0, 500.0])
        np.testing.assert_array_equal(spectro.process("Scope", s), s)
        np.testing.assert_array_equal(spectro.process("Scope - dark", s, d), s - d)
        t = spectro.process("Transmittance", s, d, r)
        np.testing.assert_allclose(t[:3], [100.0, 10.0, 50.0])
        self.assertTrue(np.isnan(t[3]))
        a = spectro.process("Absorbance", s, d, r)
        np.testing.assert_allclose(a[:3], [0.0, 1.0, -np.log10(0.5)])
        self.assertTrue(np.isnan(a[3]))
        with self.assertRaises(ValueError):
            spectro.process("Absorbance", s, d, None)
        with self.assertRaises(ValueError):
            spectro.process("Scope - dark", s, None)

    def test_absorbance_below_dark_is_nan(self):
        a = spectro.process("Absorbance", np.array([50.0]), np.array([100.0]), np.array([1000.0]))
        self.assertTrue(np.isnan(a[0]))

    def test_peak_and_fwhm(self):
        wl = np.linspace(400, 700, 3001)
        sigma = 2.0
        y = 50 + 1000 * np.exp(-0.5 * ((wl - 546.07) / sigma) ** 2)
        w, v, fwhm = spectro.peak_in_window(wl, y, 500, 600)
        self.assertAlmostEqual(w, 546.07, delta=0.06)
        self.assertAlmostEqual(v, 1050, delta=1)
        self.assertAlmostEqual(fwhm, 2.3548 * sigma, delta=0.05)
        # window where the trace only rises toward the edge: no FWHM
        ramp = wl.copy()
        w2, _, fwhm2 = spectro.peak_in_window(wl, ramp, 400, 500)
        self.assertAlmostEqual(w2, 500, delta=0.2)
        self.assertIsNone(fwhm2)
        self.assertIsNone(spectro.peak_in_window(wl, y, 800, 900))
        # flat spectrum: no crash, no FWHM
        self.assertIsNone(spectro.peak_in_window(wl, np.ones_like(wl), 450, 460)[2])

    def test_reliable_pixels(self):
        dark = np.full(5, 100.0)
        ref = np.array([100.0, 110.0, 1100.0, 600.0, 120.0])  # signal 0, 10, 1000, 500, 20
        np.testing.assert_array_equal(spectro.reliable_pixels(dark, ref), [False, False, True, True, False])
        self.assertFalse(np.any(spectro.reliable_pixels(dark, dark)))  # reference never rose

    def test_band_value(self):
        wl = np.arange(500.0, 541.0)
        y = wl - 500
        self.assertAlmostEqual(spectro.band_value(wl, y, 520, 2), 20.0)
        self.assertAlmostEqual(spectro.band_value(wl, y, 520.2, 0), 20.0)  # nearest pixel

    def test_csv_round_trip(self):
        wl = np.linspace(200, 1100, 3840)
        raw = np.arange(3840, dtype=float)
        dark = np.full(3840, 90.0)
        proc = raw - dark
        proc[5] = np.nan
        with tempfile.TemporaryDirectory() as tmp:
            path = spectro.write_spectrum_csv(os.path.join(tmp, "a", "s.csv"), wl, raw, dark, None, proc,
                                              "Scope - dark", {"Mode": "Scope - dark", "Sample": "x"})
            meta, wl2, cols = spectro.read_spectrum_csv(path)
        self.assertEqual(meta["Mode"], "Scope - dark")
        np.testing.assert_allclose(wl2, wl, atol=1e-4)
        np.testing.assert_allclose(cols["raw_counts"], raw)
        self.assertTrue(np.all(np.isnan(cols["reference_counts"])))
        self.assertTrue(np.isnan(cols["Scope - dark [Counts (dark subtracted)]"][5]))

    def test_safe_name(self):
        self.assertEqual(spectro.safe_name(" Cu(II) 0.1 M / run 2 "), "Cu_II__0.1_M___run_2")

    def test_no_device_hints(self):
        self.assertIn("cable", spectro.no_device_hint([]))
        hint = spectro.no_device_hint([("Ocean Optics HR4000", "Error", "28")])
        self.assertIn("Install HR4000 driver", hint)
        self.assertIn("seabreeze.os_setup", spectro.no_device_hint([("Ocean Optics USB4000", "Error", "28")]))
        self.assertIn("other program", spectro.no_device_hint([("Ocean Optics HR4000 (WinUSB)", "OK", "0")]))


class TestLightSource(unittest.TestCase):
    def test_working_mask(self):
        wl = np.linspace(195, 1126, 1000)
        m = spectro.working_mask(wl, "Deuterium (UV)")
        self.assertTrue(np.all((wl[m] >= 200) & (wl[m] <= 400)))
        self.assertTrue(spectro.working_mask(wl, "Full range (any source)").all())
        self.assertTrue(spectro.working_mask(wl, "no such source").all())
        self.assertTrue(spectro.working_mask(np.linspace(1500, 1600, 10), "Halogen (visible-NIR)").all())  # no overlap

    def test_suggest_integration(self):
        raw = np.zeros(100)
        mask = np.zeros(100, dtype=bool)
        mask[20:60] = True
        raw[30] = 4096.0                                     # 25% of 16383 inside the range
        raw[80] = 16383.0                                    # saturated, but outside the range: ignored
        us, note = spectro.suggest_integration_us(100_000, raw, mask, 16383.0, (10, 655_350_000))
        self.assertAlmostEqual(us, 100_000 * 0.75 * 16383 / 4096, delta=2)
        self.assertIn("25%", note)
        raw[40] = 16383.0                                    # saturated inside: cut to a third
        us, note = spectro.suggest_integration_us(90_000, raw, mask, 16383.0, (10, 655_350_000))
        self.assertEqual(us, 30_000)
        self.assertIn("saturated", note)
        us, note = spectro.suggest_integration_us(100_000, np.full(100, 5.0), mask, 16383.0, (10, 655_350_000))
        self.assertEqual(us, 100_000)                        # no light: leave it
        self.assertIn("no light", note)
        # electric dark correction off: a 1050-count offset that does not scale with time
        raw = np.full(100, 1050.0)
        raw[30] = 1050.0 + 750.0                             # 750 counts of real signal
        us, _ = spectro.suggest_integration_us(100_000, raw, mask, 16383.0, (10, 655_350_000), offset=1050.0)
        self.assertAlmostEqual(us, 100_000 * (0.75 * 16383 - 1050) / 750, delta=2)  # lands at 75% incl. offset
        raw = np.zeros(100)
        raw[30] = 0.02 * 16383                               # 2%: would need 3.75 s; limit 1 s
        us, note = spectro.suggest_integration_us(100_000, raw, mask, 16383.0, (10, 1_000_000))
        self.assertEqual(us, 1_000_000)
        self.assertIn("limited", note)


class TestDemo(unittest.TestCase):
    def test_shape_and_limits(self):
        d = spectro.DemoSpectrometer(seed=1)
        d.set_integration_us(1000)
        y = d.intensities()
        self.assertEqual(y.shape, (3840,))
        self.assertEqual(d.wavelengths().shape, (3840,))
        self.assertTrue(np.all(np.diff(d.wavelengths()) > 0))
        self.assertLessEqual(y.max(), d.max_intensity)
        d.set_integration_us(1)  # clamped to the HR4000's 10 us minimum
        self.assertEqual(d._int_us, 10)

    def test_saturation_clips(self):
        d = spectro.DemoSpectrometer(seed=2)
        d.sample_in = False
        d._int_us = 1_000_000  # set directly so the test does not sleep for the integration
        sleep = spectro.time.sleep
        spectro.time.sleep = lambda s: None
        try:
            y = d.intensities()
        finally:
            spectro.time.sleep = sleep
        self.assertEqual(y.max(), d.max_intensity)

    def test_absorbance_recovers_sample(self):
        """Dark + reference + sample through the demo gives back the demo's true absorbance."""
        sleep = spectro.time.sleep
        spectro.time.sleep = lambda s: None
        try:
            d = spectro.DemoSpectrometer(seed=3, start=time.monotonic())
            d.set_integration_us(80_000)

            def avg(n=30):
                return np.mean([d.intensities() for _ in range(n)], axis=0)

            d.shutter_closed = True
            dark = avg()
            d.shutter_closed = False
            d.sample_in = False
            ref = avg()
            d.sample_in = True
            sample = avg()
            truth = d.absorbance_now()
        finally:
            spectro.time.sleep = sleep
        a = spectro.process("Absorbance", sample, dark, ref, half_width=3)
        wl = d.wavelengths()
        sel = (wl > 480) & (wl < 560)
        self.assertLess(np.nanmax(np.abs(a[sel] - truth[sel])), 0.03)
        i = int(np.argmin(np.abs(wl - 520)))
        self.assertGreater(a[i], 0.5)

    def test_electric_dark_removes_offset(self):
        d = spectro.DemoSpectrometer(seed=4)
        d._int_us = 10
        sleep = spectro.time.sleep
        spectro.time.sleep = lambda s: None
        try:
            d.shutter_closed = True
            plain = np.mean(d.intensities())
            corrected = np.mean(d.intensities(dark=True))
        finally:
            spectro.time.sleep = sleep
        self.assertGreater(plain, 80)
        self.assertLess(abs(corrected), 5)


class CountingDevice(spectro.SpectrometerBase):
    """Stub that records calls; each 'spectrum' is filled with its read number."""

    def __init__(self):
        self.reads = 0
        self.int_calls = []
        self.pixels = 4
        self.closed = False

    def wavelengths(self):
        return np.arange(4.0)

    def set_integration_us(self, us):
        self.int_calls.append(us)

    def intensities(self, dark=False, nonlin=False):
        self.reads += 1
        time.sleep(0.002)
        return np.full(4, float(self.reads))

    def close(self):
        self.closed = True


def wait_for(acq, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        item = acq.take()
        if item is not None:
            return item
        time.sleep(0.005)
    raise AssertionError("no spectrum arrived")


class TestAcquirer(unittest.TestCase):
    def test_average_and_discard_after_integration_change(self):
        acq = Acquirer()
        dev = CountingDevice()
        try:
            acq.attach(dev)
            acq.params = {"int_us": 5000, "avg": 3, "edark": False, "nonlin": False}
            acq.single = True
            y, info = wait_for(acq)
            # read 1 is discarded (first after integration change); reads 2,3,4 averaged
            np.testing.assert_allclose(y, 3.0)
            self.assertEqual(info["avg"], 3)
            self.assertEqual(info["int_us"], 5000)
            self.assertEqual(dev.int_calls, [5000])
            time.sleep(0.05)
            self.assertIsNone(acq.take())  # single means one
            self.assertFalse(acq.single)
            # same integration time again: no new discard
            acq.params = {"int_us": 5000, "avg": 1, "edark": False, "nonlin": False}
            acq.single = True
            y, _ = wait_for(acq)
            np.testing.assert_allclose(y, 5.0)
            self.assertEqual(dev.int_calls, [5000])
        finally:
            acq.stop()
        self.assertTrue(dev.closed)

    def test_error_stops_running(self):
        class Broken(CountingDevice):
            def intensities(self, dark=False, nonlin=False):
                raise OSError("USB gone")

        acq = Acquirer()
        try:
            acq.attach(Broken())
            acq.running = True
            end = time.time() + 2
            while acq.error is None and time.time() < end:
                time.sleep(0.01)
            self.assertIn("USB gone", acq.error)
            self.assertFalse(acq.running)
        finally:
            acq.stop()


class TestGui(unittest.TestCase):
    """Drive the real window against the demo spectrometer."""

    def test_demo_session(self):
        import tkinter as tk

        import app as app_mod

        with tempfile.TemporaryDirectory() as tmp:
            root = tk.Tk()
            root.withdraw()
            settings = os.path.join(tmp, "settings.json")
            a = app_mod.App(root, demo=True, settings_path=settings, logs_dir=os.path.join(tmp, "logs"),
                            auto_connect=False)

            def pump(seconds, until=None):
                end = time.time() + seconds
                while time.time() < end:
                    root.update()
                    if until and until():
                        return True
                    time.sleep(0.01)
                return False

            try:
                a.v_int.set("20")
                a.refresh_devices(auto=True)
                self.assertIsNotNone(a.dev)
                self.assertTrue(pump(5, lambda: a.raw is not None), "no spectrum arrived")
                self.assertEqual(a.acq.params["int_us"], 20000)

                # absorbance before dark/reference: shows raw with a message, does not crash
                a.v_mode.set("Absorbance")
                a._on_mode()
                self.assertFalse(a.y_ok)

                a.v_shutter.set(True)
                a._demo_controls()
                a.request_store("dark")
                self.assertTrue(pump(5, lambda: a.dark is not None), "dark not stored")
                self.assertLess(np.mean(a.dark), 200)
                a.v_shutter.set(False)
                a.v_sample_in.set(False)
                a._demo_controls()
                a.request_store("reference")
                self.assertTrue(pump(5, lambda: a.ref is not None), "reference not stored")
                a.v_sample_in.set(True)
                a._demo_controls()
                t_after = time.time()
                pump(5, lambda: a.info and a.info["t_start"] > t_after and a.y_ok)
                self.assertTrue(a.y_ok)
                i = int(np.argmin(np.abs(a.wl - 520)))
                self.assertGreater(np.nanmean(a.y[i - 5:i + 5]), 0.4)

                # integration change makes stored dark stale
                a.v_int.set("40")
                a.apply_params()
                self.assertIn("store again", a.l_dark.cget("text"))
                a.v_int.set("20")
                a.apply_params()
                self.assertNotIn("store again", a.l_dark.cget("text"))

                # trend logs absorbance at 520 nm
                a.v_track.set("520, 700")
                a.toggle_trend()
                self.assertIsNotNone(a.trend)
                pump(3, lambda: len(a.trend_data["t"]) >= 5)
                a.toggle_trend()
                trend_files = os.listdir(os.path.join(tmp, "logs", "trends"))
                self.assertEqual(len(trend_files), 1)
                with open(os.path.join(tmp, "logs", "trends", trend_files[0]), encoding="utf-8") as fh:
                    rows = [ln for ln in fh if not ln.startswith("#")]
                self.assertEqual(rows[0].strip(), "time,elapsed_s,mode,520nm,700nm")
                self.assertGreaterEqual(len(rows) - 1, 5)
                self.assertGreaterEqual(float(rows[1].split(",")[1]), 0.0)  # no point from before Start
                self.assertGreater(float(rows[1].split(",")[3]), 0.3)  # absorbing at 520
                self.assertLess(abs(float(rows[1].split(",")[4])), 0.1)  # not at 700

                # save spectrum + graph, read the spectrum back
                a.v_sample.set("demo dye")
                path = a.save_spectrum()
                meta, wl, cols = spectro.read_spectrum_csv(path)
                self.assertEqual(meta["Mode"], "Absorbance")
                self.assertEqual(meta["Sample"], "demo dye")
                self.assertEqual(len(wl), 3840)
                self.assertIn("Absorbance [Absorbance (AU)]", cols)
                png = a.save_graph()
                self.assertGreater(os.path.getsize(png), 10_000)

                # hold + overlay + clear
                a.hold()
                self.assertEqual(len(a.held), 1)
                a.clear_held()
                self.assertEqual(len(a.held), 0)

                # light-source preset: deuterium range ignores the lines above 400 nm
                a.v_mode.set("Scope")
                a._on_mode()
                a.dev.lamp, a.dev.sample_in = "lines", False
                a.v_source.set("Deuterium (UV)")
                a.set_light_source()
                self.assertEqual(tuple(round(v) for v in a.ax.get_xlim()), (200, 400))
                t = time.time()
                pump(5, lambda: a.info and a.info["t_start"] > t)
                self.assertIn("in 200-400 nm", a.l_sat.cget("text"))
                a.full_range()  # zoomed out, the peak search still stays inside the working range
                self.assertLess(float(a.l_peak.cget("text").split(":")[1].split("nm")[0]), 400)
                # halogen preset + Set integration fills the range to ~75%
                a.dev.lamp = "continuum"
                a.v_source.set("Halogen (visible-NIR)")
                a.set_light_source()
                t = time.time()
                pump(5, lambda: a.info and a.info["t_start"] > t)
                a.auto_integration()
                t = time.time()
                pump(8, lambda: a.info and a.info["t_start"] > t and a.info["int_us"] == a.acq.params["int_us"])
                mask = a._work_mask()
                pct = 100 * float(np.max(a.raw[mask])) / a.dev.max_intensity
                self.assertAlmostEqual(pct, 75, delta=8)
                # same with electric dark correction off (raw counts include the electronic offset)
                a.v_edark.set(False)
                a.apply_params()
                a.v_int.set("20")
                a.apply_params()
                t = time.time()
                pump(5, lambda: a.info and a.info["t_start"] > t and not a.info["edark"])
                a.auto_integration()
                t = time.time()
                pump(8, lambda: a.info and a.info["t_start"] > t and a.info["int_us"] == a.acq.params["int_us"])
                pct = 100 * float(np.max(a.raw[a._work_mask()])) / a.dev.max_intensity
                self.assertAlmostEqual(pct, 75, delta=8)
                a.v_edark.set(True)
                a.apply_params()
                self.assertEqual(a._meta()["Light source"], "Halogen (visible-NIR)")
                a.v_source.set("Full range (any source)")
                a.set_light_source()

                # pause / single
                a.toggle_run()
                self.assertFalse(a.acq.running)
                n = a.acq.delivered
                a.single_shot()
                self.assertTrue(pump(3, lambda: a.acq.delivered > n))
            finally:
                a.on_close()
            with open(settings, encoding="utf-8") as fh:
                saved = json.load(fh)
            self.assertEqual(saved["mode"], "Scope")  # last mode used in the session
            self.assertGreater(float(saved["int_ms"]), 20)  # raised by Set integration


if __name__ == "__main__":
    unittest.main()
