from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

from restir_gs.eval.reference_inventory import (
    classify_tree_paths,
    reference_image_local_path,
    tree_entries_to_paths,
)
from scripts import demo_16_render_fidelity_triage as fidelity
from scripts import download_voxel51_references as refs


def test_reference_url_and_path_builders_match_voxel51_scene_root() -> None:
    remote_path = "FO_dataset/playroom/DSC05572.jpg"

    assert refs.voxel51_tree_api_url("playroom") == (
        "https://huggingface.co/api/datasets/Voxel51/gaussian_splatting/tree/main/"
        "FO_dataset/playroom?recursive=true"
    )
    assert refs.voxel51_resolve_url(remote_path) == (
        "https://huggingface.co/datasets/Voxel51/gaussian_splatting/resolve/main/"
        "FO_dataset/playroom/DSC05572.jpg"
    )
    assert reference_image_local_path(remote_path, "playroom", "outputs/references") == Path(
        "outputs/references/voxel51_playroom/DSC05572.jpg"
    )


def test_reference_dry_run_lists_downloads_without_writing(tmp_path, monkeypatch, capsys) -> None:
    def fake_tree(scene: str) -> list[dict[str, str]]:
        return [{"path": f"FO_dataset/{scene}/reference.png"}]

    monkeypatch.setattr(refs, "fetch_scene_tree_entries", fake_tree)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["download_voxel51_references.py", "--dry-run"])

    assert refs.main() == 0
    captured = capsys.readouterr()

    assert "dry-run: no files written" in captured.out
    assert "reference.png" in captured.out
    assert not (tmp_path / "outputs").exists()


def test_existing_non_empty_reference_files_are_skipped(tmp_path) -> None:
    remote_path = "FO_dataset/train/reference.jpg"
    local_path = reference_image_local_path(remote_path, "train", tmp_path)
    local_path.parent.mkdir(parents=True)
    local_path.write_bytes(b"image")

    downloads = refs.make_reference_downloads("train", [remote_path], tmp_path)

    assert downloads[0].exists is True
    assert downloads[0].size_bytes == 5


def test_inventory_classifies_images_and_camera_metadata() -> None:
    paths = tree_entries_to_paths(
        [
            {"path": "FO_dataset/playroom/DSC05572.jpg"},
            {"path": "FO_dataset/playroom/point_cloud/iteration_7000/point_cloud.ply"},
            {"path": "FO_dataset/playroom/sparse/0/cameras.txt"},
            {"path": "FO_dataset/playroom/nested/not_scene_root.png"},
        ]
    )

    inventory = classify_tree_paths("playroom", paths)

    assert inventory.image_paths == ["FO_dataset/playroom/DSC05572.jpg"]
    assert inventory.camera_metadata_paths == ["FO_dataset/playroom/sparse/0/cameras.txt"]
    assert inventory.camera_alignment == "available"


def test_missing_camera_metadata_marks_alignment_unavailable() -> None:
    inventory = classify_tree_paths("truck", ["FO_dataset/truck/reference.jpeg"])

    assert inventory.camera_alignment == "unavailable"
    assert inventory.camera_metadata_paths == []


def test_contact_sheet_handles_one_or_multiple_reference_images(tmp_path) -> None:
    ref_a = tmp_path / "ref_a.png"
    ref_b = tmp_path / "ref_b.png"
    preview = tmp_path / "preview.png"
    Image.new("RGB", (64, 64), "red").save(ref_a)
    Image.new("RGB", (64, 64), "green").save(ref_b)
    Image.new("RGB", (64, 64), "blue").save(preview)
    summary = {
        "phase17_benchmark_role": "algorithm smoke benchmark",
        "scenes": [
            {
                "scene_id": "voxel51_playroom",
                "reference_images_found": 2,
                "reference_image_paths": [str(ref_a), str(ref_b)],
                "camera_alignment": "unavailable",
                "loaded_count": 10,
                "original_count": 20,
                "selected_views": [
                    {
                        "view_id": "view_00",
                        "preview_path": str(preview),
                        "camera_score": 1.0,
                        "valid_pixels": 4,
                        "comparison_status": "not camera-aligned",
                    }
                ],
            }
        ],
    }

    output = tmp_path / "contact.png"
    fidelity.make_contact_sheet(summary, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_fidelity_summary_records_four_scenes_and_twelve_views(tmp_path) -> None:
    benchmark_summary = {"scenes": []}
    inventory = {"scenes": []}
    for scene in ["drjohnson", "playroom", "train", "truck"]:
        scene_id = f"voxel51_{scene}"
        benchmark_summary["scenes"].append(
            {
                "scene_id": scene_id,
                "loaded_count": 5,
                "original_count": 10,
                "selected_views": [
                    {
                        "view_id": f"view_{index:02d}",
                        "preview_path": str(tmp_path / scene_id / f"view_{index:02d}" / "preview_rgb.png"),
                        "camera_score": 1.0,
                        "valid_pixels": 100,
                    }
                    for index in range(3)
                ],
            }
        )
        inventory["scenes"].append(
            {
                "scene_id": scene,
                "downloads": [
                    {
                        "local_path": str(tmp_path / f"{scene}.jpg"),
                        "exists": True,
                    }
                ],
                "camera_metadata_paths": [],
                "camera_alignment": "unavailable",
            }
        )

    summary = fidelity.build_fidelity_summary(benchmark_summary, inventory, tmp_path)

    assert summary["scene_count"] == 4
    assert summary["selected_view_count"] == 12
    assert summary["reference_images"] == "found"
    assert summary["camera_metadata"] == "missing"
    assert summary["phase17_benchmark_role"] == "algorithm smoke benchmark"


def test_fidelity_summary_json_round_trip(tmp_path) -> None:
    payload = {"version": 1, "scenes": []}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert fidelity.load_json(path) == payload
