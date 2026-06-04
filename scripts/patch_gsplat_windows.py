from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


ORIGINAL = '        extra_cflags = [opt_level, "-Wno-attributes"]'
PATCHED = (
    "        extra_cflags = [opt_level]\n"
    '        if os.name != "nt":\n'
    '            extra_cflags += ["-Wno-attributes"]'
)


def backend_path() -> Path:
    spec = importlib.util.find_spec("gsplat.cuda._backend")
    if spec is None or spec.origin is None:
        raise RuntimeError("Could not locate gsplat.cuda._backend in the active Python environment.")
    return Path(spec.origin)


def patch_backend(check_only: bool) -> int:
    path = backend_path()
    text = path.read_text(encoding="utf-8")

    if PATCHED in text:
        print(f"gsplat Windows patch present: {path}")
        return 0

    if ORIGINAL not in text:
        print(f"gsplat Windows patch expected line not found: {path}", file=sys.stderr)
        return 2

    if check_only:
        print(f"gsplat Windows patch missing: {path}", file=sys.stderr)
        return 1

    path.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print(f"Applied gsplat Windows patch: {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Patch gsplat 1.5.x JIT flags for Windows MSVC.")
    parser.add_argument("--check", action="store_true", help="Only verify that the patch is present.")
    args = parser.parse_args()
    return patch_backend(check_only=args.check)


if __name__ == "__main__":
    raise SystemExit(main())

