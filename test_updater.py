"""Tests for the self-updater: python -m unittest test_updater (no network: a fake repo on disk)."""
import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import updater


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FakeRepo:
    """Mimics raw.githubusercontent.com/<owner>/<repo>/<ref>/<path> on disk (file:// URLs)."""

    def __init__(self, root: Path):
        self.root = root

    def publish(self, version: str, commit: str, files: dict, notes: str = "") -> dict:
        for rel, data in files.items():
            p = self.root / commit / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
        manifest = {"version": version, "commit": commit, "notes": notes,
                    "files": [{"path": rel, "sha256": sha(d), "size": len(d)} for rel, d in files.items()]}
        m = self.root / "main" / "release.json"
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest


class TestUpdater(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.repo = FakeRepo(self.tmp / "server")
        self.app = self.tmp / "Ocean Spectrometer Live"
        self.app.mkdir()
        self._base = updater.BASE
        updater.BASE = (self.tmp / "server").as_uri()
        # an installed 1.0.0 copy, with user data next to it
        self.old = {"app.py": b"print('old')\n", "version.py": b'__version__ = "1.0.0"\n',
                    "READ ME FIRST.txt": b"same\r\n", "driver/OOI_HR4000.inf": b"inf"}
        for rel, data in self.old.items():
            (self.app / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.app / rel).write_bytes(data)
        (self.app / "settings.json").write_text('{"int_ms": "42"}')
        (self.app / "logs" / "spectra").mkdir(parents=True)
        (self.app / "logs" / "spectra" / "mine.csv").write_text("data")

    def tearDown(self):
        updater.BASE = self._base
        shutil.rmtree(self.tmp, ignore_errors=True)

    def new_files(self):
        files = dict(self.old)
        files["app.py"] = b"print('new')\n"
        files["version.py"] = b'__version__ = "1.1.0"\n'
        files["Make desktop shortcut.bat"] = b"@echo off\r\n"  # a file that did not exist before
        return files

    def test_versions(self):
        pv = updater.parse_version
        self.assertGreater(pv("1.10.0"), pv("1.9.3"))
        self.assertGreater(pv("v2"), pv("1.9"))
        self.assertEqual(pv("1.2"), (1, 2, 0))

    def test_check(self):
        self.repo.publish("1.1.0", "c1", self.new_files(), notes="faster")
        m = updater.check("1.0.0", self.app)
        self.assertEqual(m["version"], "1.1.0")
        self.assertIsNone(updater.check("1.1.0", self.app))  # up to date
        self.assertIsNone(updater.check("1.2.0", self.app))  # newer than published
        (self.app / ".git").mkdir()
        self.assertIsNone(updater.check("1.0.0", self.app))  # development copy never updates

    def test_apply_replaces_only_changed_files_and_keeps_user_data(self):
        m = self.repo.publish("1.1.0", "c1", self.new_files())
        readme_mtime = (self.app / "READ ME FIRST.txt").stat().st_mtime_ns
        changed = updater.apply(m, self.app)
        self.assertEqual(sorted(changed), ["Make desktop shortcut.bat", "app.py", "version.py"])
        self.assertEqual((self.app / "app.py").read_bytes(), b"print('new')\n")
        self.assertEqual((self.app / "version.py").read_bytes(), b'__version__ = "1.1.0"\n')
        self.assertTrue((self.app / "Make desktop shortcut.bat").exists())
        self.assertEqual((self.app / "READ ME FIRST.txt").stat().st_mtime_ns, readme_mtime)  # untouched
        self.assertEqual((self.app / "settings.json").read_text(), '{"int_ms": "42"}')
        self.assertEqual((self.app / "logs" / "spectra" / "mine.csv").read_text(), "data")
        self.assertEqual((self.app / ".update_backup" / "app.py").read_bytes(), b"print('old')\n")
        self.assertFalse((self.app / ".update_tmp").exists())
        self.assertEqual(updater.changed_files(m, self.app), [])  # second run: nothing to do

    def test_bad_download_changes_nothing(self):
        m = self.repo.publish("1.1.0", "c1", self.new_files())
        (self.tmp / "server" / "c1" / "version.py").write_bytes(b"tampered")
        with self.assertRaises(updater.UpdateError):
            updater.apply(m, self.app)
        for rel, data in self.old.items():
            self.assertEqual((self.app / rel).read_bytes(), data)
        self.assertFalse((self.app / ".update_tmp").exists())

    def test_missing_file_on_server_changes_nothing(self):
        m = self.repo.publish("1.1.0", "c1", self.new_files())
        (self.tmp / "server" / "c1" / "Make desktop shortcut.bat").unlink()
        with self.assertRaises(Exception):
            updater.apply(m, self.app)
        self.assertEqual((self.app / "app.py").read_bytes(), b"print('old')\n")

    def test_unsafe_paths_rejected(self):
        self.repo.publish("1.1.0", "c1", {"app.py": b"x"})
        mpath = self.tmp / "server" / "main" / "release.json"
        m = json.loads(mpath.read_text())
        m["files"].append({"path": "../evil.py", "sha256": sha(b"x"), "size": 1})
        mpath.write_text(json.dumps(m))
        with self.assertRaises(updater.UpdateError):
            updater.fetch_manifest()

    def test_dev_copy_refuses_apply(self):
        m = self.repo.publish("1.1.0", "c1", self.new_files())
        (self.app / ".git").mkdir()
        with self.assertRaises(updater.UpdateError):
            updater.apply(m, self.app)

    def test_shipped_list_matches_repo(self):
        """Every shipped file exists here, so release.json can always be built."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent / "tools"))
        import shipping

        missing = [f for f in shipping.SHIPPED if not (Path(__file__).parent / f).exists()
                   and f != shipping.GUIDE_PDF]
        self.assertEqual(missing, [])
        self.assertIn("updater.py", shipping.SHIPPED)
        self.assertIn("version.py", shipping.SHIPPED)


if __name__ == "__main__":
    unittest.main()
