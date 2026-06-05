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
from scripts.download_dxgl_apple_splat import DxglAppleSplatPlan, download_dxgl_splat, validate_dxgl_splat_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and validate a manifest-registered aligned Gaussian splat.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_aligned_asset_manifest(args.manifest)
    spec = get_aligned_asset_spec(manifest, args.asset_id)
    resolved = resolve_aligned_asset_paths(spec, repo_root=manifest.repo_root)
    size_bytes = resolved.splat_path.stat().st_size if resolved.splat_path.exists() else 0
    summary_path = resolved.splat_path.parent / f"{spec.asset_id}_splat_intake_summary.json"
    plan = DxglAppleSplatPlan(
        url=spec.splat_url,
        splat_path=resolved.splat_path,
        summary_path=summary_path,
        exists=size_bytes > 0,
        size_bytes=size_bytes,
    )

    print(f"asset_id:    {spec.asset_id}")
    print(f"dataset_type:{spec.dataset_type}")
    print(f"url:         {plan.url}")
    print(f"splat:       {plan.splat_path}")
    print(f"summary:     {plan.summary_path}")
    print(f"exists:      {plan.exists}")
    print(f"size_bytes:  {plan.size_bytes}")
    if args.dry_run:
        print("dry-run: no files written")
        return 0

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
