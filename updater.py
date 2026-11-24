"""Self-update from the app's public GitHub repository.

How it works
------------
`tools/publish.py` (run on the maintainer's computer) commits a release and then writes
`release.json` on the main branch:

    {"version": "1.1.0", "commit": "<sha>", "date": "...", "notes": "...",
     "files": [{"path": "app.py", "sha256": "...", "size": 1234}, ...]}

Each installed copy reads that file at start-up. If the version is newer, the user can update:
only files whose SHA-256 differs are downloaded (from that exact commit), every download is
verified, the old files are kept in `.update_backup/`, and then the new files replace the old
ones. Logs, settings and anything not listed in `files` are never touched.

A development copy (a folder with `.git`) never updates itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from version import __version__

REPO = "josefigueroa-435/ocean-spectrometer-live"
# OCEAN_UPDATE_BASE points tests at a local copy of the repository layout.
BASE = os.environ.get("OCEAN_UPDATE_BASE", f"https://raw.githubusercontent.com/{REPO}").rstrip("/")
MANIFEST_REF = "main"
TIMEOUT = 8
APP_DIR = Path(__file__).resolve().parent
UA = {"User-Agent": "OceanSpectrometerLive-updater"}


class UpdateError(Exception):
    pass


def parse_version(text: str) -> tuple[int, ...]:
    parts = []
    for p in str(text).strip().lstrip("vV").split("."):
        digits = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def is_dev_copy(app_dir: Path = APP_DIR) -> bool:
    """The maintainer's git working copy: never overwrite it from GitHub."""
    return (app_dir / ".git").exists()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _url(ref: str, path: str) -> str:
    return f"{BASE}/{ref}/{urllib.parse.quote(path)}"


def _fetch(url: str, timeout: float = TIMEOUT) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def fetch_manifest(timeout: float = TIMEOUT) -> dict:
    data = json.loads(_fetch(_url(MANIFEST_REF, "release.json"), timeout).decode("utf-8"))
    for key in ("version", "commit", "files"):
        if key not in data:
            raise UpdateError(f"release.json has no '{key}'")
    for f in data["files"]:
        p = Path(f["path"])
        if p.is_absolute() or ".." in p.parts:
            raise UpdateError(f"unsafe path in release.json: {f['path']}")
    return data


def check(current: str = __version__, app_dir: Path = APP_DIR, timeout: float = TIMEOUT) -> dict | None:
    """The newer release's manifest, or None if up to date. Raises on network/format errors."""
    if is_dev_copy(app_dir):
        return None
    manifest = fetch_manifest(timeout)
    if parse_version(manifest["version"]) > parse_version(current):
        return manifest
    return None


def changed_files(manifest: dict, app_dir: Path = APP_DIR) -> list[dict]:
    return [f for f in manifest["files"] if sha256_file(app_dir / f["path"]) != f["sha256"]]


def apply(manifest: dict, app_dir: Path = APP_DIR, progress=None) -> list[str]:
    """Download, verify and install the release. Returns the paths that changed.

    Everything is downloaded and checked before any file is replaced, so a failed download
    leaves the installed copy exactly as it was.
    """
    if is_dev_copy(app_dir):
        raise UpdateError("This is a development copy (it has .git); it does not update itself.")
    todo = changed_files(manifest, app_dir)
    staging = app_dir / ".update_tmp"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    try:
        for i, f in enumerate(todo, 1):
            if progress:
                progress(f"Downloading {f['path']} ({i}/{len(todo)})")
            data = _fetch(_url(manifest["commit"], f["path"]), timeout=60)
            if hashlib.sha256(data).hexdigest() != f["sha256"]:
                raise UpdateError(f"{f['path']} did not download correctly (checksum mismatch)")
            dst = staging / f["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)

        backup = app_dir / ".update_backup"
        shutil.rmtree(backup, ignore_errors=True)
        backup.mkdir()
        (backup / "VERSION.txt").write_text(f"Files replaced by the update to {manifest['version']}, "
                                            f"from version {__version__}\n", encoding="utf-8")
        # version.py last, so an interrupted install still reports the old version
        todo.sort(key=lambda f: f["path"] == "version.py")
        for f in todo:
            target = app_dir / f["path"]
            if target.exists():
                (backup / f["path"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup / f["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging / f["path"], target)
    except PermissionError as exc:
        raise UpdateError(f"Windows would not let the app replace its files ({exc.filename}). "
                          "Move the app folder somewhere you can write to, such as Documents.") from exc
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return [f["path"] for f in todo]


def install_requirements(app_dir: Path = APP_DIR) -> tuple[bool, str]:
    """pip install -r requirements.txt with this Python (console-less)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
        exe = exe.parent / "python.exe"
    r = subprocess.run([str(exe), "-m", "pip", "install", "-r", str(app_dir / "requirements.txt")],
                       capture_output=True, text=True, timeout=900,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.returncode == 0, (r.stdout + r.stderr)[-2000:]


def restart(args: list[str] | None = None, app_dir: Path = APP_DIR) -> None:
    """Start a fresh copy of the app (the caller then closes this one)."""
    exe = Path(sys.executable)
    if exe.name.lower() == "python.exe" and (exe.parent / "pythonw.exe").exists():
        exe = exe.parent / "pythonw.exe"
    subprocess.Popen([str(exe), str(app_dir / "app.py"), *(args or [])], cwd=str(app_dir),
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
