from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import urllib.error
import urllib.request
import zipfile


DXGL_APPLE_URL = "https://dx.gl/api/v/EJbs8npt2RVM/vCHDLxjWG65d/dataset"
DEFAULT_EXTRACT_DIR = Path("outputs/aligned_assets/dxgl/apple")
DEFAULT_SUMMARY_PATH = Path("outputs/aligned_assets/dxgl/apple_intake_summary.json")
CHUNK_SIZE = 1024 * 1024
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 ReSTIR-GS-DXGL-Intake/1.0",
    "Accept": "application/zip,application/octet-stream,*/*",
}
REQUIRED_ENTRIES = (
    ("transforms.json", "file"),
    ("images", "dir"),
    ("depth", "dir"),
    ("depth_16bit", "dir"),
    ("normals", "dir"),
    ("masks", "dir"),
    ("points3D.ply", "file"),
)


@dataclass(frozen=True)
class DxglApplePlan:
    url: str
    zip_path: Path
    extract_dir: Path
    summary_path: Path
    zip_exists: bool
    dataset_root: Path | None
    dataset_valid: bool


def plan_dxgl_apple(
    extract_dir: str | Path = DEFAULT_EXTRACT_DIR,
    zip_path: str | Path | None = None,
    summary_path: str | Path = DEFAULT_SUMMARY_PATH,
    url: str = DXGL_APPLE_URL,
) -> DxglApplePlan:
    extract_dir = Path(extract_dir)
    zip_path = Path(zip_path) if zip_path is not None else extract_dir.with_suffix(".zip")
    summary_path = Path(summary_path)
    dataset_root = find_dxgl_dataset_root(extract_dir, required=False)
    dataset_valid = False
    if dataset_root is not None:
        dataset_valid = validate_dxgl_dataset_root(dataset_root, raise_on_missing=False)["valid"]
    return DxglApplePlan(
        url=url,
        zip_path=zip_path,
        extract_dir=extract_dir,
        summary_path=summary_path,
        zip_exists=zip_path.exists() and zip_path.stat().st_size > 0,
        dataset_root=dataset_root,
        dataset_valid=dataset_valid,
    )


def download_dxgl_zip(url: str, zip_path: str | Path) -> None:
    zip_path = Path(zip_path)
    if zip_path.exists() and zip_path.stat().st_size > 0:
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = zip_path.with_name(zip_path.name + ".part")
    if partial_path.exists():
        partial_path.unlink()

    request = urllib.request.Request(url, headers=REQUEST_HEADERS)
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
        raise RuntimeError(f"Failed to download DXGL Apple dataset from {url}: {exc}") from exc

    size = partial_path.stat().st_size if partial_path.exists() else 0
    if size <= 0:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Downloaded empty DXGL Apple dataset: {url}")
    if expected_size_int is not None and size != expected_size_int:
        partial_path.unlink()
        raise RuntimeError(f"Partial DXGL Apple download: expected {expected_size_int} bytes, got {size} bytes.")
    partial_path.replace(zip_path)


def extract_dxgl_zip(zip_path: str | Path, extract_dir: str | Path) -> Path:
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    if not zip_path.exists() or zip_path.stat().st_size <= 0:
        raise RuntimeError(f"Missing DXGL zip file: {zip_path}")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    dataset_root = find_dxgl_dataset_root(extract_dir, required=True)
    validate_dxgl_dataset_root(dataset_root, raise_on_missing=True)
    return dataset_root


def find_dxgl_dataset_root(extract_dir: str | Path, required: bool = True) -> Path | None:
    extract_dir = Path(extract_dir)
    candidates = [extract_dir]
    if extract_dir.exists():
        candidates.extend(path for path in extract_dir.iterdir() if path.is_dir())
    for candidate in candidates:
        if (candidate / "transforms.json").exists():
            return candidate
    if required:
        raise RuntimeError(f"Could not find transforms.json under {extract_dir}")
    return None


def validate_dxgl_dataset_root(root: str | Path, raise_on_missing: bool = True) -> dict[str, object]:
    root = Path(root)
    entries: dict[str, bool] = {}
    missing: list[str] = []
    for name, kind in REQUIRED_ENTRIES:
        path = root / name
        exists = path.is_dir() if kind == "dir" else path.is_file()
        entries[name] = bool(exists)
        if not exists:
            missing.append(name)
    result: dict[str, object] = {"root": str(root), "valid": not missing, "entries": entries, "missing": missing}
    if missing and raise_on_missing:
        raise RuntimeError(f"DXGL Apple dataset is missing required entries {missing} under {root}")
    return result


def write_intake_summary(plan: DxglApplePlan, dataset_root: Path, validation: dict[str, object]) -> None:
    summary = {
        "version": 1,
        "dataset": "dxgl_polyhaven_10_apple",
        "url": plan.url,
        "zip_path": str(plan.zip_path),
        "extract_dir": str(plan.extract_dir),
        "dataset_root": str(dataset_root),
        "validation": validation,
    }
    plan.summary_path.parent.mkdir(parents=True, exist_ok=True)
    plan.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate the DXGL Polyhaven Apple aligned dataset.")
    parser.add_argument("--url", default=DXGL_APPLE_URL)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--summary-path", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    plan = plan_dxgl_apple(args.extract_dir, args.zip_path, args.summary_path, args.url)
    print(f"url:          {plan.url}")
    print(f"zip:          {plan.zip_path}")
    print(f"extract_dir:  {plan.extract_dir}")
    print(f"summary:      {plan.summary_path}")
    print(f"zip_exists:   {plan.zip_exists}")
    print(f"dataset_root: {plan.dataset_root if plan.dataset_root is not None else 'missing'}")
    print(f"dataset_valid:{plan.dataset_valid}")
    if args.dry_run:
        print("dry-run: no files written")
        return 0

    download_dxgl_zip(plan.url, plan.zip_path)
    dataset_root = extract_dxgl_zip(plan.zip_path, plan.extract_dir)
    validation = validate_dxgl_dataset_root(dataset_root, raise_on_missing=True)
    write_intake_summary(plan, dataset_root, validation)
    print(f"validated:    {dataset_root}")
    print(f"wrote:        {plan.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
