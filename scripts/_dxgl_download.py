from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import urllib.error
import urllib.request
import zipfile

from restir_gs.render.ply_loader import load_gaussian_ply_with_stats


CHUNK_SIZE = 1024 * 1024
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 ReSTIR-GS-DXGL-Intake/1.0",
    "Accept": "application/zip,application/octet-stream,*/*",
}
REQUIRED_DXGL_ENTRIES = (
    ("transforms.json", "file"),
    ("images", "dir"),
    ("depth", "dir"),
    ("depth_16bit", "dir"),
    ("normals", "dir"),
    ("masks", "dir"),
    ("points3D.ply", "file"),
)


@dataclass(frozen=True)
class DxglSplatPlan:
    url: str
    splat_path: Path
    summary_path: Path
    exists: bool
    size_bytes: int


def download_dxgl_file(url: str, output_path: str | Path, label: str) -> None:
    output_path = Path(output_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".part")
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
        raise RuntimeError(f"Failed to download {label} from {url}: {exc}") from exc

    size = partial_path.stat().st_size if partial_path.exists() else 0
    if size <= 0:
        if partial_path.exists():
            partial_path.unlink()
        raise RuntimeError(f"Downloaded empty {label}: {url}")
    if expected_size_int is not None and size != expected_size_int:
        partial_path.unlink()
        raise RuntimeError(f"Partial {label} download: expected {expected_size_int} bytes, got {size} bytes.")
    partial_path.replace(output_path)


def download_dxgl_zip(url: str, zip_path: str | Path) -> None:
    download_dxgl_file(url, zip_path, label="DXGL dataset")


def download_dxgl_splat(plan: DxglSplatPlan) -> None:
    if plan.exists:
        return
    download_dxgl_file(plan.url, plan.splat_path, label="DXGL splat")


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
    for name, kind in REQUIRED_DXGL_ENTRIES:
        path = root / name
        exists = path.is_dir() if kind == "dir" else path.is_file()
        entries[name] = bool(exists)
        if not exists:
            missing.append(name)
    result: dict[str, object] = {"root": str(root), "valid": not missing, "entries": entries, "missing": missing}
    if missing and raise_on_missing:
        raise RuntimeError(f"DXGL dataset is missing required entries {missing} under {root}")
    return result


def validate_dxgl_splat_file(path: str | Path, max_probe_gaussians: int = 16) -> dict[str, object]:
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Missing or empty DXGL splat: {path}")
    loaded = load_gaussian_ply_with_stats(path, device="cpu", max_gaussians=max_probe_gaussians)
    return {
        "path": str(path),
        "valid": True,
        "original_count": loaded.stats.original_count,
        "probe_loaded_count": loaded.stats.loaded_count,
        "color_source": loaded.stats.color_source,
        "size_bytes": path.stat().st_size,
    }
