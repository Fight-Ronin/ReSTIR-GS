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
    resolve_requested_asset_ids,
)
from scripts._dxgl_download import DxglSplatPlan, download_dxgl_splat, validate_dxgl_splat_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate a manifest-registered aligned Gaussian splat.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--asset-id")
    selection.add_argument("--asset-set")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_aligned_asset_manifest(args.manifest)
    asset_ids = resolve_requested_asset_ids(
        manifest,
        asset_ids=[args.asset_id] if args.asset_id is not None else None,
        asset_set=args.asset_set,
    )
    for asset_id in asset_ids:
        spec = get_aligned_asset_spec(manifest, asset_id)
        _run_one(spec, manifest.repo_root, args.dry_run)
    return 0


def _run_one(spec, repo_root: Path, dry_run: bool) -> None:
    resolved = resolve_aligned_asset_paths(spec, repo_root=repo_root)
    size_bytes = resolved.splat_path.stat().st_size if resolved.splat_path.exists() else 0
    summary_path = resolved.splat_path.parent / f"{spec.asset_id}_splat_intake_summary.json"
    plan = DxglSplatPlan(
        url=spec.splat_url,
        splat_path=resolved.splat_path,
        summary_path=summary_path,
        exists=size_bytes > 0,
        size_bytes=size_bytes,
    )

    print()
    print(f"asset_id:    {spec.asset_id}")
    print(f"dataset_type:{spec.dataset_type}")
    print(f"url:         {plan.url}")
    print(f"splat:       {plan.splat_path}")
    print(f"summary:     {plan.summary_path}")
    print(f"exists:      {plan.exists}")
    print(f"size_bytes:  {plan.size_bytes}")
    if dry_run:
        print("dry-run: no files written")
        return

    download_dxgl_splat(plan)
    validation = validate_dxgl_splat_file(plan.splat_path)
    payload = {
        "version": 1,
        "asset_id": spec.asset_id,
        "dataset_type": spec.dataset_type,
        "asset": "pretrained_splat",
        "url": plan.url,
        "splat_path": str(plan.splat_path),
        "gaussian_schema": spec.gaussian_schema,
        "validation": validation,
    }
    plan.summary_path.parent.mkdir(parents=True, exist_ok=True)
    plan.summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"validated:   {plan.splat_path}")
    print(f"gaussians:   {validation['original_count']}")
    print(f"color:       {validation['color_source']}")
    print(f"wrote:       {plan.summary_path}")


if __name__ == "__main__":
    raise SystemExit(main())
