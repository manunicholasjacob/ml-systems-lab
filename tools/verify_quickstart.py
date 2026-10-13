#!/usr/bin/env python3
"""Run the README's first command block against an installed wheel, outside the checkout.

The bug this exists to catch: the README once opened with `mlsys run
configs/example-smoke.yaml`, which works from a clone and fails after `pip install`,
because `configs/` is not part of the package. A test suite run from the source tree
cannot see that, so this does what a new user does: install the built wheel into a
fresh environment, change into an empty directory, and run every `mlsys` line in the
first fenced block of README.md.

    python -m build                          # or: python -m build --wheel
    python tools/verify_quickstart.py        # uses the newest wheel in dist/
    python tools/verify_quickstart.py --membw-mb 64

`mlsys membw` is run with a small working set (--membw-mb) so the check stays fast; the
point is that the command exists and runs without a config, not the number it prints.

Exits non-zero on the first failing command and says which one.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def first_block_commands(readme: str) -> list:
    """The `mlsys` command lines in the first fenced code block of the README."""
    m = re.search(r"^```[^\n]*\n(.*?)^```", readme, flags=re.S | re.M)
    if not m:
        raise SystemExit("README.md has no fenced code block")
    cmds = []
    for line in m.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("mlsys "):
            cmds.append(line)
    if not cmds:
        raise SystemExit("the first README code block has no mlsys command")
    return cmds


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wheel", help="wheel to install (default: newest in dist/)")
    ap.add_argument("--membw-mb", type=int, default=64,
                    help="working set for `mlsys membw` (default 64)")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args(argv)

    wheel = args.wheel
    if not wheel:
        found = sorted(glob.glob(os.path.join(ROOT, "dist", "*.whl")), key=os.path.getmtime)
        if not found:
            print("nothing in dist/. Run: python -m build", file=sys.stderr)
            return 2
        wheel = found[-1]

    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        cmds = first_block_commands(fh.read())

    with tempfile.TemporaryDirectory(prefix="mlsys-quickstart-") as tmp:
        venv = os.path.join(tmp, "venv")
        subprocess.run([sys.executable, "-m", "venv", venv], check=True)
        bindir = os.path.join(venv, "Scripts" if os.name == "nt" else "bin")
        py = os.path.join(bindir, "python")
        subprocess.run([py, "-m", "pip", "install", "-q", "--disable-pip-version-check", wheel], check=True)
        work = os.path.join(tmp, "empty")
        os.makedirs(work)
        env = dict(os.environ, PATH=bindir + os.pathsep + os.environ.get("PATH", ""))
        for cmd in cmds:
            argv_ = shlex.split(cmd)
            if argv_[1] == "membw" and "--mb" not in argv_:
                argv_ += ["--mb", str(args.membw_mb), "--reps", "1"]
            exe = os.path.join(bindir, "mlsys")
            try:
                r = subprocess.run([exe] + argv_[1:], cwd=work, env=env,
                                   capture_output=True, text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"FAIL  {cmd}  (timed out after {args.timeout:.0f} s)", file=sys.stderr)
                return 1
            if r.returncode != 0:
                print(f"FAIL  {cmd}  (exit {r.returncode})", file=sys.stderr)
                print(r.stdout[-2000:], r.stderr[-2000:], sep="\n", file=sys.stderr)
                return 1
            print(f"  ok    {' '.join(argv_)}")
    print(f"\nthe README quickstart ({len(cmds)} command(s)) runs from {os.path.basename(wheel)}"
          " in an empty directory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
