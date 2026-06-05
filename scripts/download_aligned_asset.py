from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.render.aligned_asset_registry import (
    DEFAULT_MANIFEST_PATH,
    get_aligned_asset_spec,
    load_aligned_asset_manifest,
    resolve_aligned_asset_paths,
)
from scripts.download_dxgl_apple import download_dxgl_zip, extract_dxgl_zip, validate_dxgl_dataset_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate a manifest-registered aligned dataset.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--zip-path", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_aligned_asset_manifest(args.manifest)
    spec = get_aligned_asset_spec(manifest, args.asset_id)
    resolved = resolve_aligned_asset_paths(spec, repo_root=manifest.repo_root)
    if spec.dataset_type != "dxgl":
        raise ValueError(f"Unsupported aligned dataset_type '{spec.dataset_type}'. Phase 28 supports 'dxgl' only.")
    zip_path = args.zip_path if args.zip_path is not None else resolved.dataset_root.with_suffix(".zip")
    dataset_root_exists = resolved.dataset_root.exists()
    dataset_valid = False
    if dataset_root_exists:
        dataset_valid = bool(validate_dxgl_dataset_root(resolved.dataset_root, raise_on_missing=False)["valid"])

    print(f"asset_id:      {spec.asset_id}")
    print(f"dataset_type:  {spec.dataset_type}")
    print(f"url:           {spec.dataset_url}")
    print(f"zip:           {zip_path}")
    print(f"dataset_root:  {resolved.dataset_root}")
    print(f"dataset_valid: {dataset_valid}")
    if args.dry_run:
        print("dry-run: no files written")
        return 0

    download_dxgl_zip(spec.dataset_url, zip_path)
    dataset_root = extract_dxgl_zip(zip_path, resolved.dataset_root)
    validation = validate_dxgl_dataset_root(dataset_root, raise_on_missing=True)
    summary_path = resolved.dataset_root.parent / f"{spec.asset_id}_intake_summary.json"
    summary = {
        "version": 1,
        "asset_id": spec.asset_id,
        "dataset_type": spec.dataset_type,
        "url": spec.dataset_url,
        "zip_path": str(zip_path),
        "dataset_root": str(dataset_root),
        "validation": validation,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"validated:     {dataset_root}")
    print(f"wrote:         {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
