"""Build the hand-over zip: refresh the PDF guide, run the tests on a clean copy, check for personal
data, then zip.

    python tools/build_package.py            # writes dist/Ocean_Spectrometer_Live_<version>.zip
    python tools/build_package.py OUT_DIR    # somewhere else

The zip holds one folder, "Ocean Spectrometer Live", containing exactly the files in
shipping.SHIPPED. Copies made from it update themselves from GitHub (see updater.py).
"""
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shipping  # noqa: E402

SRC = shipping.SRC
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else SRC / "dist"
TOP = "Ocean Spectrometer Live"


def run_tests(stage: Path) -> None:
    """Run the tests against the staged copy (tests are not shipped)."""
    for t in shipping.TESTS:
        shutil.copy2(SRC / f"{t}.py", stage / f"{t}.py")
    r = subprocess.run([sys.executable, "-m", "unittest", *shipping.TESTS], cwd=stage, capture_output=True,
                       text=True, timeout=300, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print("tests:", " ".join(r.stderr.strip().splitlines()[-3:]))
    assert r.returncode == 0, r.stderr
    for t in shipping.TESTS:
        (stage / f"{t}.py").unlink()
    for junk in ("__pycache__", "logs", ".update_tmp", ".update_backup"):
        shutil.rmtree(stage / junk, ignore_errors=True)
    (stage / "settings.json").unlink(missing_ok=True)


def build(out_dir: Path = OUT_DIR, render: bool = True) -> Path:
    scratch = Path(tempfile.mkdtemp(prefix="ocean_pkg_"))
    if render:
        shipping.render_guide(scratch)
    stage = scratch / TOP
    for rel in shipping.SHIPPED:
        dst = stage / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / rel, dst)  # byte-for-byte: the updater compares checksums with these
    run_tests(stage)

    files = sorted(p for p in stage.rglob("*") if p.is_file())
    names = [p.relative_to(stage).as_posix() for p in files]
    assert sorted(names) == sorted(shipping.SHIPPED), set(names) ^ set(shipping.SHIPPED)
    shipping.check_no_personal_data(stage, names)

    out_dir.mkdir(parents=True, exist_ok=True)
    zp = out_dir / f"Ocean_Spectrometer_Live_{shipping.read_version()}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(scratch).as_posix())
    shutil.rmtree(scratch, ignore_errors=True)
    print(zp, f"{zp.stat().st_size:,} bytes, {len(files)} files")
    return zp


if __name__ == "__main__":
    build()
