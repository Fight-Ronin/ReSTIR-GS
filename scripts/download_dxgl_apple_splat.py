from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.render.ply_loader import load_gaussian_ply_with_stats
from scripts.download_dxgl_apple import REQUEST_HEADERS


DXGL_APPLE_SPLAT_URL = "https://dx.gl/splat/apple.ply"
DEFAULT_SPLAT_PATH = Path("outputs/aligned_assets/dxgl/apple_splat/apple.ply")
DEFAULT_SUMMARY_PATH = Path("outputs/aligned_assets/dxgl/apple_splat_intake_summary.json")
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DxglAppleSplatPlan:
    url: str
    splat_path: Path
    summary_path: Path
    exists: bool
    size_bytes: int


def plan_dxgl_apple_splat(
    splat_path: str | Path = DEFAULT_SPLAT_PATH,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    url: str = DXGL_APPLE_SPLAT_URL,
) -> DxglAppleSplatPlan:
    splat_path = Path(splat_path)
    size_bytes = splat_path.stat().st_size if splat_path.exists() else 0
    return DxglAppleSplatPlan(
        url=url,
        splat_path=splat_path,
        summary_path=Path(summary_path),
        exists=size_bytes > 0,
        size_bytes=size_bytes,
    )


def download_dxgl_splat(plan: DxglAppleSplatPlan) -> None:
    if plan.exists:
        return
    plan.splat_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = plan.splat_path.with_name(plan.splat_path.name + ".part")
    if partial_path.exists():
        partial_path.unlink()

    request = urllib.request.Request(plan.url, headers=REQUEST_HEADERS)
    try:
        with urllib.request.urlopen(request) as response:
            expected_size = response.headers.get("Content-Length")
            expected_size_int = int(expected_size) if expected_size and expected_size.isdigit() else None
            with partial_path.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Failed to download DXGL Apple splat from {plan.url}: {exc}") from exc

    size = partial_path.stat().st_size if partial_path.exists() else 0
    if size <= 0:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Downloaded empty DXGL Apple splat: {plan.url}")
    if expected_size_int is not None and size != expected_size_int:
        partial_path.unlink()
        raise RuntimeError(f"Partial DXGL Apple splat download: expected {expected_size_int} bytes, got {size} bytes.")
    partial_path.replace(plan.splat_path)


def validate_dxgl_splat_file(path: str | Path, max_probe_gaussians: int = 16) -> dict[str, object]:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty DXGL Apple splat: {path}")
    loaded = load_gaussian_ply_with_stats(path, device="cpu", max_gaussians=max_probe_gaussians)
    return {
        "path": str(path),
        "valid": True,
        "original_count": loaded.stats.original_count,
        "probe_loaded_count": loaded.stats.loaded_count,
        "color_source": loaded.stats.color_source,
        "size_bytes": path.stat().st_size,
    }


def write_splat_summary(plan: DxglAppleSplatPlan, validation: dict[str, object]) -> None:
    payload = {
        "version": 1,
        "dataset": "dxgl_polyhaven_10_apple",
        "asset": "pretrained_splat",
        "url": plan.url,
        "splat_path": str(plan.splat_path),
        "validation": validation,
    }
    plan.summary_path.parent.mkdir(parents=True, exist_ok=True)
    plan.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate the DXGL Apple pretrained 3DGS splat.")
    parser.add_argument("--url", default=DXGL_APPLE_SPLAT_URL)
    parser.add_argument("--splat-path", type=Path, default=DEFAULT_SPLAT_PATH)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = plan_dxgl_apple_splat(args.splat_path, args.summary_path, args.url)
    print(f"url:        {plan.url}")
    print(f"splat:      {plan.splat_path}")
    print(f"summary:    {plan.summary_path}")
    print(f"exists:     {plan.exists}")
    print(f"size_bytes: {plan.size_bytes}")
    if args.dry_run:
        print("dry-run: no files written")
        return 0

    download_dxgl_splat(plan)
    validation = validate_dxgl_splat_file(plan.splat_path)
    write_splat_summary(plan_dxgl_apple_splat(args.splat_path, args.summary_path, args.url), validation)
    print(f"validated:  {plan.splat_path}")
    print(f"gaussians:  {validation['original_count']}")
    print(f"color:      {validation['color_source']}")
    print(f"wrote:      {plan.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
