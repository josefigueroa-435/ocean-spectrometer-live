"""What gets shipped to users, shared by build_package.py (hand-over zip) and publish.py (updates)."""
import getpass
import hashlib
import subprocess
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
GUIDE_HTML = SRC / "docs" / "Ocean_Spectrometer_Live_Guide.html"
GUIDE_PDF = "Ocean Spectrometer Live - Guide.pdf"
EDGE = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"]

# Installed layout == repository layout for these files. Updates replace exactly these.
SHIPPED = [
    "app.py", "spectro.py", "acquire.py", "updater.py", "version.py", "requirements.txt", "ocean_live.ico",
    "READ ME FIRST.txt", GUIDE_PDF,
    "Ocean Spectrometer Live.bat", "Ocean Spectrometer Live (demo).bat", "Install requirements.bat",
    "Install HR4000 driver (run once).bat", "Make desktop shortcut.bat",
    "driver/OOI_HR4000.inf", "driver/OOI_HR4000.cat", "driver/amd64/WdfCoInstaller01007.dll",
    "driver/amd64/WinUSBCoInstaller.dll", "driver/amd64/WUDFUpdate_01007.dll",
]
TESTS = ["test_spectro", "test_updater"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_version() -> str:
    ns: dict = {}
    exec((SRC / "version.py").read_text(encoding="utf-8"), ns)
    return ns["__version__"]


def write_version(version: str) -> None:
    (SRC / "version.py").write_text(
        '"""Version of this copy of Ocean Spectrometer Live. tools/publish.py bumps it."""\n'
        f'__version__ = "{version}"\n', encoding="utf-8", newline="\n")


def render_guide(scratch: Path) -> Path:
    """Print docs/Ocean_Spectrometer_Live_Guide.html to the shipped PDF with Edge (headless)."""
    edge = next((e for e in EDGE if Path(e).exists()), None)
    assert edge, "Microsoft Edge is needed to print the guide to PDF"
    pdf = SRC / GUIDE_PDF
    tmp = scratch / "guide.pdf"
    subprocess.run([edge, "--headless=new", "--disable-gpu", f"--user-data-dir={scratch / 'edge-profile'}",
                    "--no-pdf-header-footer", f"--print-to-pdf={tmp}", GUIDE_HTML.as_uri()],
                   capture_output=True, timeout=180)
    size, deadline = -1, time.monotonic() + 120  # msedge may return before the PDF is written
    while time.monotonic() < deadline:
        now = tmp.stat().st_size if tmp.exists() else -1
        if now > 200_000 and now == size:
            break
        size = now
        time.sleep(1)
    assert tmp.exists() and tmp.stat().st_size > 200_000, "guide PDF not written"
    # Edge stamps creation dates into the PDF; only replace the shipped copy if the content changed,
    # so an unchanged guide is not re-downloaded by every update.
    new = tmp.read_bytes()
    if not pdf.exists() or _pdf_body(pdf.read_bytes()) != _pdf_body(new):
        pdf.write_bytes(new)
    return pdf


def _pdf_body(data: bytes) -> bytes:
    import re
    return re.sub(rb"/(CreationDate|ModDate) \(D:[^)]*\)", b"", data)


def personal_strings() -> list[str]:
    """This computer's user name and home folder: must never appear in shipped text files."""
    out = {getpass.getuser(), Path.home().name, str(Path.home())}
    return [s for s in out if s and len(s) >= 3]


def check_no_personal_data(root: Path, names) -> None:
    bad = personal_strings()
    for rel in names:
        p = root / rel
        if p.suffix.lower() in (".py", ".bat", ".txt", ".inf", ".json", ".html", ".md"):
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
            hits = [b for b in bad if b.lower() in text]
            assert not hits, f"personal data {hits} in {rel}"
