from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

from restir_gs.render.orbit_camera import OrbitCameraState, save_orbit_camera_config
from restir_gs.render.synthetic_scene import PinholeCamera, SyntheticGaussians
from scripts.demo_22_interactive_viewer import configure_viewer_runtime_environment, load_viewer_asset


def test_generic_ply_viewer_asset_uses_auto_camera(tmp_path: Path) -> None:
    ply = tmp_path / "tiny.ply"
    _write_tiny_3dgs_ply(ply)
    args = _viewer_args(ply=ply)

    asset = load_viewer_asset(args, device=torch.device("cpu"))

    assert asset.label == "3DGS tiny"
    assert asset.metadata["source_mode"] == "generic_3dgs_ply"
    assert asset.metadata["camera_source"]["mode"] == "auto_asset_camera"
    assert len(asset.frame_cameras) == 1
    assert asset.frame_cameras[0].width == 64
    assert asset.frame_cameras[0].height == 48


def test_generic_ply_viewer_asset_accepts_camera_config(tmp_path: Path) -> None:
    ply = tmp_path / "tiny.ply"
    camera_config = tmp_path / "camera.json"
    _write_tiny_3dgs_ply(ply)
    save_orbit_camera_config(
        OrbitCameraState((0.0, 0.0, 2.0), 0.0, 0.0, 2.0, 1.25, 80, 60),
        camera_config,
    )
    args = _viewer_args(ply=ply, camera_config=camera_config)

    asset = load_viewer_asset(args, device=torch.device("cpu"))

    assert asset.metadata["camera_source"]["mode"] == "camera_config"
    assert asset.frame_cameras[0].width == 80
    assert asset.frame_cameras[0].height == 60


def test_cuda_runtime_preflight_sets_compile_defaults(monkeypatch) -> None:
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("MAX_JOBS", raising=False)
    monkeypatch.delenv("TORCH_EXTENSIONS_DIR", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr("scripts.demo_22_interactive_viewer.platform.system", lambda: "Linux")

    configure_viewer_runtime_environment(torch.device("cuda"))

    assert torch.cuda is not None
    assert __import__("os").environ["TORCH_CUDA_ARCH_LIST"] == "8.9"
    assert __import__("os").environ["MAX_JOBS"] == "4"
    assert "torch_extensions_restirgs" in __import__("os").environ["TORCH_EXTENSIONS_DIR"]
    assert "matplotlib_cache" in __import__("os").environ["MPLCONFIGDIR"]


def test_cuda_runtime_preflight_warns_without_windows_cl(monkeypatch, capsys) -> None:
    monkeypatch.setattr("scripts.demo_22_interactive_viewer.platform.system", lambda: "Windows")
    monkeypatch.setattr("scripts.demo_22_interactive_viewer.shutil.which", lambda name: None)

    configure_viewer_runtime_environment(torch.device("cuda"))

    captured = capsys.readouterr()
    assert "MSVC cl.exe" in captured.err
    assert "run_interactive_viewer_windows.bat" in captured.err


def test_registered_viewer_asset_uses_manifest_loader(monkeypatch, tmp_path: Path) -> None:
    calls = {}
    spec = SimpleNamespace(asset_id="dxgl_test", dataset_type="dxgl")
    resolved = SimpleNamespace(dataset_root=tmp_path / "dataset", splat_path=tmp_path / "test.ply")
    fake_asset = SimpleNamespace(
        loaded=SimpleNamespace(
            scene=_tiny_scene(),
            stats=SimpleNamespace(original_count=2, loaded_count=2),
        ),
        transforms=SimpleNamespace(frames=[SimpleNamespace(index=7, camera=_camera())]),
    )
    monkeypatch.setattr("scripts.demo_22_interactive_viewer.load_aligned_asset_manifest", lambda path: SimpleNamespace(repo_root=tmp_path))
    monkeypatch.setattr("scripts.demo_22_interactive_viewer.get_aligned_asset_spec", lambda manifest, asset_id: spec)
    monkeypatch.setattr("scripts.demo_22_interactive_viewer.resolve_aligned_asset_paths", lambda item, repo_root: resolved)

    def fake_load_registered(resolved_spec, device="cuda", max_gaussians_override=None):
        calls["resolved"] = resolved_spec
        calls["max_gaussians_override"] = max_gaussians_override
        return fake_asset

    monkeypatch.setattr("scripts.demo_22_interactive_viewer.load_registered_aligned_asset", fake_load_registered)
    args = _viewer_args(ply=None, asset_id="dxgl_test", manifest=tmp_path / "manifest.json")

    asset = load_viewer_asset(args, device=torch.device("cpu"))

    assert calls["resolved"] is resolved
    assert calls["max_gaussians_override"] == 0
    assert asset.label == "Aligned dxgl_test"
    assert asset.metadata["source_mode"] == "aligned_registry"
    assert asset.metadata["asset_id"] == "dxgl_test"
    assert asset.frame_labels == ["7"]


def _viewer_args(
    ply: Path | None,
    camera_config: Path | None = None,
    asset_id: str | None = None,
    manifest: Path | None = None,
) -> Namespace:
    return Namespace(
        ply=ply,
        camera_config=camera_config,
        manifest=manifest or Path("unused_manifest.json"),
        asset_id=asset_id,
        max_gaussians=0,
        width=64,
        height=48,
        auto_camera_bbox_percentile=1.0,
        auto_camera_radius_scale=1.8,
        dataset_root=Path("unused"),
        splat=Path("unused"),
        normalization_bbox_percentile=0.98,
    )


def _camera() -> PinholeCamera:
    return PinholeCamera(
        viewmats=torch.eye(4, dtype=torch.float32)[None],
        intrinsics=torch.eye(3, dtype=torch.float32)[None],
        width=64,
        height=48,
    )


def _tiny_scene() -> SyntheticGaussians:
    return SyntheticGaussians(
        means=torch.tensor([[0.0, 0.0, 2.0], [0.2, 0.0, 2.1]], dtype=torch.float32),
        quats=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        scales=torch.full((2, 3), 0.05, dtype=torch.float32),
        opacities=torch.ones((2,), dtype=torch.float32),
        colors=torch.ones((2, 3), dtype=torch.float32),
    )


def _write_tiny_3dgs_ply(path: Path) -> None:
    properties = [
        ("float", "x"),
        ("float", "y"),
        ("float", "z"),
        ("float", "opacity"),
        ("float", "scale_0"),
        ("float", "scale_1"),
        ("float", "scale_2"),
        ("float", "rot_0"),
        ("float", "rot_1"),
        ("float", "rot_2"),
        ("float", "rot_3"),
        ("float", "f_dc_0"),
        ("float", "f_dc_1"),
        ("float", "f_dc_2"),
    ]
    rows = [
        [0.0, 0.0, 2.0, 4.0, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
        [0.2, 0.0, 2.1, 4.0, -3.0, -3.0, -3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
    ]
    lines = ["ply", "format ascii 1.0", f"element vertex {len(rows)}"]
    lines.extend(f"property {kind} {name}" for kind, name in properties)
    lines.append("end_header")
    lines.extend(" ".join(str(value) for value in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
