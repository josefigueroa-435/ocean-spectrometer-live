"""Publish an update: every installed copy of the app will offer it the next time it starts.

    python tools/publish.py 1.1.0 "What changed, in one sentence"
    python tools/publish.py 1.1.0 "..." --no-push     # do everything except git push

Steps: tests -> version.py -> guide PDF -> personal-data check -> commit (the release) -> release.json
(points at that commit, lists every shipped file with its SHA-256) -> commit -> tag -> git push ->
hand-over zip in dist/.

Day-to-day commits and pushes to GitHub do NOT reach users; only release.json does, and only this
script writes it.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shipping  # noqa: E402

SRC = shipping.SRC
sys.path.insert(0, str(SRC))
from updater import parse_version  # noqa: E402


def git(*args, check=True) -> str:
    r = subprocess.run(["git", *args], cwd=SRC, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def last_published() -> str:
    p = SRC / "release.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))["version"]
    return "0.0.0"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("version", help="new version, e.g. 1.1.0")
    ap.add_argument("notes", help="what changed (shown to users in the update bar)")
    ap.add_argument("--no-push", action="store_true", help="commit and tag, but do not push")
    ap.add_argument("--trailer", action="append", default=[], help="extra commit-message trailer line")
    a = ap.parse_args()

    if not (SRC / ".git").exists():
        raise SystemExit("Not a git repository.")
    if git("rev-parse", "--abbrev-ref", "HEAD") != "main":
        raise SystemExit("Publish from the main branch.")
    prev = last_published()
    if parse_version(a.version) <= parse_version(prev):
        raise SystemExit(f"Version {a.version} must be newer than the last published {prev}.")

    print("1/7 tests")
    r = subprocess.run([sys.executable, "-m", "unittest", *shipping.TESTS], cwd=SRC, capture_output=True,
                       text=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    print("   ", " ".join(r.stderr.strip().splitlines()[-3:]))
    if r.returncode != 0:
        raise SystemExit("Tests failed; nothing was published.\n" + r.stderr[-3000:])

    print("2/7 version.py ->", a.version)
    shipping.write_version(a.version)
    print("3/7 guide PDF")
    shipping.render_guide(Path(tempfile.mkdtemp(prefix="ocean_pub_")))
    print("4/7 personal-data check")
    shipping.check_no_personal_data(SRC, shipping.SHIPPED)
    shipping.check_no_personal_data(SRC, ["README.md", "docs/Ocean_Spectrometer_Live_Guide.html"])

    trailers = "".join(f"\n{t}" for t in a.trailer)
    print("5/7 commit release")
    git("add", "-A")
    # the whole repository is public, not just the shipped files
    shipping.check_no_personal_data(SRC, git("ls-files").splitlines())
    git("commit", "-m", f"Release v{a.version}: {a.notes}\n{trailers}".rstrip() + "\n", "--allow-empty")
    commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain")
    if dirty:
        raise SystemExit(f"Working tree not clean after commit:\n{dirty}")

    print("6/7 release.json ->", commit[:10])
    manifest = {
        "version": a.version,
        "commit": commit,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": a.notes,
        "files": [{"path": rel, "sha256": shipping.sha256(SRC / rel), "size": (SRC / rel).stat().st_size}
                  for rel in shipping.SHIPPED],
    }
    (SRC / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    git("add", "release.json")
    git("commit", "-m", f"Publish v{a.version}\n{trailers}".rstrip() + "\n")
    git("tag", "-f", f"v{a.version}", commit)

    if a.no_push:
        print("7/7 push skipped (--no-push). Run:  git push origin main --tags")
    else:
        print("7/7 git push")
        r = subprocess.run(["git", "push", "origin", "main", "--tags"], cwd=SRC)
        if r.returncode != 0:
            raise SystemExit("Push failed; the release is committed locally. Fix the problem and run:\n"
                             "    git push origin main --tags")

    import build_package
    zp = build_package.build(SRC / "dist", render=False)
    print(f"\nPublished {a.version}. Hand-over zip for new users: {zp}")


if __name__ == "__main__":
    main()
