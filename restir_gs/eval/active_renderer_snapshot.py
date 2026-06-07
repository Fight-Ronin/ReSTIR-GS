from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from restir_gs.restir.renderer import RESTIR_TIMING_FIELDS


GPU_STAGE_TIMING_FIELDS = tuple(
    field
    for field in RESTIR_TIMING_FIELDS
    if field not in {"frame_gpu_ms", "frame_wall_ms", "shadow_bundle_asset_gpu_ms"}
)

SNAPSHOT_IMAGE_FILES = (
    ("Reference", "final_reference.png"),
    ("Initial RIS", "final_initial_ris.png"),
    ("Temporal Filtered", "final_temporal_filtered_ris.png"),
    ("Filtered Error", "final_temporal_filtered_abs_error.png"),
    ("Filter Alpha", "final_temporal_filter_alpha.png"),
    ("Reuse Mask", "final_reuse_mask.png"),
)


def summarize_estimator_contribution_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("reference_quantity")) == "contribution_rgb":
            groups[str(row["estimator"])].append(row)

    summary: dict[str, dict[str, float | int]] = {}
    for estimator, group in sorted(groups.items()):
        mae = [float(row["mae"]) for row in group]
        rmse = [float(row["rmse"]) for row in group]
        summary[estimator] = {
            "count": len(group),
            "mae_mean": _mean(mae),
            "mae_max": max(mae) if mae else 0.0,
            "rmse_mean": _mean(rmse),
        }
    return summary


def rank_gpu_timing_stages(timing_summary: dict[str, Any]) -> list[dict[str, float | str]]:
    rows = []
    frame_mean = float(timing_summary.get("frame_gpu_ms", {}).get("mean", 0.0))
    for field in GPU_STAGE_TIMING_FIELDS:
        stats = timing_summary.get(field, {})
        mean_ms = float(stats.get("mean", 0.0))
        rows.append(
            {
                "stage": field,
                "mean_ms": mean_ms,
                "max_ms": float(stats.get("max", 0.0)),
                "frame_fraction": mean_ms / frame_mean if frame_mean > 0.0 else 0.0,
            }
        )
    return sorted(rows, key=lambda row: float(row["mean_ms"]), reverse=True)


def build_active_renderer_snapshot_summary(
    renderer_summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    settings = renderer_summary.get("settings", {})
    row_count = int(renderer_summary.get("row_count", len(rows)))
    timing_summary = renderer_summary.get("timing_summary", {})
    return {
        "version": 1,
        "row_count": row_count,
        "asset_ids": list(renderer_summary.get("asset_ids", [])),
        "active_policy": {
            "target_mode": settings.get("target_mode"),
            "proposal": settings.get("proposal"),
            "preferred_output": "temporal_filtered_ris",
            "num_lights": settings.get("num_lights"),
            "temporal_reprojection_search_radius": settings.get("temporal_reprojection_search_radius"),
            "temporal_history_m_cap": settings.get("temporal_history_m_cap"),
        },
        "validation": {
            "all_numeric_finite": bool(renderer_summary.get("all_numeric_finite", False)),
            "has_timing_summary": bool(timing_summary),
            "has_asset_timing": bool(renderer_summary.get("asset_timing")),
            "expected_active_row_count": 72,
            "row_count_matches_active_default": row_count == 72,
        },
        "estimator_contribution_summary": summarize_estimator_contribution_rows(rows),
        "gpu_stage_rank": rank_gpu_timing_stages(timing_summary),
        "timing_summary": timing_summary,
        "asset_timing": renderer_summary.get("asset_timing", {}),
    }


def make_active_renderer_contact_sheet(
    renderer_output_dir: Path,
    output_path: Path,
    asset_ids: list[str],
) -> None:
    if not asset_ids:
        raise ValueError("Expected at least one asset id for active renderer contact sheet.")

    thumb_w, thumb_h = 150, 96
    label_w = 190
    header_h = 34
    row_h = 132
    col_count = len(SNAPSHOT_IMAGE_FILES)
    sheet = Image.new("RGB", (label_w + col_count * thumb_w, header_h + len(asset_ids) * row_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 9), "Active ReSTIR renderer snapshot", fill=(0, 0, 0))
    for col, (label, _) in enumerate(SNAPSHOT_IMAGE_FILES):
        draw.text((label_w + col * thumb_w + 6, 9), label, fill=(0, 0, 0))

    for row, asset_id in enumerate(asset_ids):
        y = header_h + row * row_h
        draw.text((8, y + 8), asset_id, fill=(0, 0, 0))
        asset_dir = renderer_output_dir / asset_id
        for col, (_, filename) in enumerate(SNAPSHOT_IMAGE_FILES):
            path = asset_dir / filename
            if not path.is_file():
                raise FileNotFoundError(f"Missing active renderer preview image: {path}")
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb_w - 12, thumb_h))
            x = label_w + col * thumb_w
            paste_xy = (x + (thumb_w - image.width) // 2, y + 26 + (thumb_h - image.height) // 2)
            sheet.paste(image, paste_xy)
            draw.rectangle((x + 6, y + 26, x + thumb_w - 6, y + 26 + thumb_h), outline=(180, 180, 180))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _mean(values: list[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0
