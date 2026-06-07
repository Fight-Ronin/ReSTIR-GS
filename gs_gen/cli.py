from __future__ import annotations

import argparse
import json
from pathlib import Path

from gs_gen.config import load_config
from gs_gen.nerfstudio import build_nerfstudio_plan
from gs_gen.paths import DEFAULT_WORKSPACE, validate_asset_id
from gs_gen.source_probe import make_source, probe_source
from gs_gen.stage import stage_asset
from gs_gen.validate import format_validation_summary, validate_exported_asset
from gs_gen.video_extract import extract_video_frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independent room Gaussian Splatting generation pipeline helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="Print a Nerfstudio/Splatfacto pipeline plan.")
    plan.add_argument("--config", type=Path, default=None)
    plan.add_argument("--asset-id", default=None)
    plan.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    plan_source = plan.add_mutually_exclusive_group()
    plan_source.add_argument("--images", type=Path, default=None)
    plan_source.add_argument("--video", type=Path, default=None)
    plan.set_defaults(func=run_plan)

    source = subparsers.add_parser("probe-source", help="Validate an image directory or video file before processing.")
    source_group = source.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--images", type=Path, default=None)
    source_group.add_argument("--video", type=Path, default=None)
    source.add_argument("--json", action="store_true")
    source.set_defaults(func=run_probe_source)

    extract = subparsers.add_parser("extract-frames", help="Extract an image sequence from a video source.")
    extract.add_argument("--video", type=Path, required=True)
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--target-fps", type=float, default=5.0)
    extract.add_argument("--max-frames", type=int, default=None)
    extract.add_argument("--dry-run", action="store_true")
    extract.add_argument("--json", action="store_true")
    extract.set_defaults(func=run_extract_frames)

    validate = subparsers.add_parser("validate", help="Validate processed transforms and exported splat.")
    validate.add_argument("--dataset-root", type=Path, required=True)
    validate.add_argument("--splat", type=Path, required=True)
    validate.add_argument("--json", action="store_true")
    validate.set_defaults(func=run_validate)

    stage = subparsers.add_parser("stage", help="Stage a validated local GS asset.")
    stage.add_argument("--asset-id", required=True)
    stage.add_argument("--dataset-root", type=Path, required=True)
    stage.add_argument("--splat", type=Path, required=True)
    stage.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    stage.add_argument("--copy-images", action="store_true")
    stage.add_argument("--dry-run", action="store_true")
    stage.add_argument("--json", action="store_true")
    stage.set_defaults(func=run_stage)

    args = parser.parse_args(argv)
    return int(args.func(args))


def run_plan(args: argparse.Namespace) -> int:
    if args.config is not None:
        config = load_config(args.config)
        asset_id = config.asset_id
        source = config.source
        workspace = config.workspace
    else:
        if args.asset_id is None:
            raise ValueError("--asset-id is required when --config is not provided")
        asset_id = validate_asset_id(args.asset_id)
        source = make_source(images=args.images, video=args.video)
        workspace = args.workspace

    plan = build_nerfstudio_plan(asset_id, source=source, workspace=workspace)
    print(f"asset_id: {plan.asset_id}")
    print(f"source_kind: {plan.source.kind}")
    print(f"source_path: {plan.source.path}")
    print(f"workspace: {plan.paths.workspace}")
    print(f"processed_dir: {plan.paths.processed_dir}")
    print(f"train_dir: {plan.paths.train_dir}")
    print(f"export_dir: {plan.paths.export_dir}")
    print(f"staged_dir: {plan.paths.staged_dir}")
    print()
    print("commands:")
    for command in plan.commands:
        print(f"  {command}")
    return 0


def run_probe_source(args: argparse.Namespace) -> int:
    result = probe_source(make_source(images=args.images, video=args.video))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"valid: {result['valid']}")
        print(f"kind: {result['kind']}")
        print(f"path: {result['path']}")
        print(f"errors: {result['errors']}")
        if result["kind"] == "images":
            print(f"image_count: {result['image_count']}")
        if result["kind"] == "video":
            print(f"suffix: {result['suffix']}")
    return 0 if bool(result["valid"]) else 1


def run_validate(args: argparse.Namespace) -> int:
    result = validate_exported_asset(args.dataset_root, args.splat)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_validation_summary(result))
    return 0 if bool(result["valid"]) else 1


def run_extract_frames(args: argparse.Namespace) -> int:
    result = extract_video_frames(
        video_path=args.video,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        max_frames=args.max_frames,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"video_path: {result['video_path']}")
        print(f"output_dir: {result['output_dir']}")
        print(f"source_fps: {result['source_fps']}")
        print(f"frame_count: {result['frame_count']}")
        print(f"size: {result['width']}x{result['height']}")
        print(f"target_fps: {result['target_fps']}")
        print(f"planned_frame_count: {result['planned_frame_count']}")
        print(f"written: {result['written']}")
        if args.dry_run:
            print("dry-run: no files written")
    return 0


def run_stage(args: argparse.Namespace) -> int:
    result = stage_asset(
        asset_id=args.asset_id,
        dataset_root=args.dataset_root,
        splat_path=args.splat,
        workspace=args.workspace,
        copy_images=args.copy_images,
        dry_run=args.dry_run,
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"valid: {result['validation']['valid']}")
        print(f"staged_dir: {result['staged_dir']}")
        print(f"copy_images: {result['images_copied']}")
        if args.dry_run:
            print("dry-run: no files written")
    return 0 if bool(result["validation"]["valid"]) else 1
