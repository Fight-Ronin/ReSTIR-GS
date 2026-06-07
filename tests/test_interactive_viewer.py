from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import torch

from restir_gs.render.orbit_camera import OrbitCameraState, orbit_state_eye, save_orbit_camera_config
from interactive.camera import camera_state_look, camera_state_translate_local
from interactive.layers import view_from_key, viewer_computed_views, viewer_render_requirements
from interactive.rendering import (
    ViewerAsset,
    ViewerRenderResult,
    ViewerSettings,
    ViewerTimings,
    ViewerVisibilityResult,
    save_outputs,
    view_image,
    viewer_save_metadata,
)
from interactive.session import InteractiveSession
from interactive.viewer import (
    InteractiveViewer,
    camera_command_from_key,
    configure_matplotlib_viewer_keymaps,
    save_command_from_key,
    view_accent,
    view_title,
    viewer_footer,
    viewer_header,
    viewer_view_accent,
    viewer_view_chips,
)
from restir_gs.render.synthetic_scene import PinholeCamera, SyntheticGaussians
from interactive.launcher import (
    configure_viewer_runtime_environment,
    load_viewer_asset,
)


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
    monkeypatch.setattr("interactive.launcher.platform.system", lambda: "Linux")

    configure_viewer_runtime_environment(torch.device("cuda"))

    assert torch.cuda is not None
    assert __import__("os").environ["TORCH_CUDA_ARCH_LIST"] == "8.9"
    assert __import__("os").environ["MAX_JOBS"] == "4"
    assert "torch_extensions_restirgs" in __import__("os").environ["TORCH_EXTENSIONS_DIR"]
    assert "matplotlib_cache" in __import__("os").environ["MPLCONFIGDIR"]


def test_cuda_runtime_preflight_warns_without_windows_cl(monkeypatch, capsys) -> None:
    monkeypatch.setattr("interactive.launcher.platform.system", lambda: "Windows")
    monkeypatch.setattr("interactive.launcher.shutil.which", lambda name: None)

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
    monkeypatch.setattr("interactive.launcher.load_aligned_asset_manifest", lambda path: SimpleNamespace(repo_root=tmp_path))
    monkeypatch.setattr("interactive.launcher.get_aligned_asset_spec", lambda manifest, asset_id: spec)
    monkeypatch.setattr("interactive.launcher.resolve_aligned_asset_paths", lambda item, repo_root: resolved)

    def fake_load_registered(resolved_spec, device="cuda", max_gaussians_override=None):
        calls["resolved"] = resolved_spec
        calls["max_gaussians_override"] = max_gaussians_override
        return fake_asset

    monkeypatch.setattr("interactive.launcher.load_registered_aligned_asset", fake_load_registered)
    args = _viewer_args(ply=None, asset_id="dxgl_test", manifest=tmp_path / "manifest.json")

    asset = load_viewer_asset(args, device=torch.device("cpu"))

    assert calls["resolved"] is resolved
    assert calls["max_gaussians_override"] == 0
    assert asset.label == "Aligned dxgl_test"
    assert asset.metadata["source_mode"] == "aligned_registry"
    assert asset.metadata["asset_id"] == "dxgl_test"
    assert asset.frame_labels == ["7"]


def test_viewer_header_and_footer_surface_runtime_context() -> None:
    assert ViewerSettings().width == 768
    assert ViewerSettings().height == 768

    asset = ViewerAsset(
        label="Aligned dxgl_test",
        scene=_tiny_scene(),
        source_path=Path("test.ply"),
        frame_cameras=[_camera(), _camera(), _camera()],
        frame_labels=["0", "7", "9"],
        metadata={},
    )
    result = ViewerRenderResult(
        frame_index=1,
        state=OrbitCameraState((0.0, 0.0, 2.0), 12.0, -8.0, 1.5, 1.25, 64, 48),
        gbuffer=None,
        lambertian=None,
        blinn_phong=None,
        restir=None,
        visibility=None,
        valid_pixels=123,
        render_ms=45.6,
        light_info={},
    )

    header = viewer_header(result, asset, "blinn_phong")
    footer = viewer_footer("saved 5 outputs")

    assert "Aligned dxgl_test" in header
    assert "frame 2/3 (7)" in header
    assert "view Blinn-Phong" in header
    assert "render 45.6 ms" in header
    assert "1 RGB" in footer
    assert "6 Blinn-Phong" in footer
    assert "WASD move" in footer
    assert "Ctrl+S save" in footer
    assert "Left drag orbit" in footer
    assert "Shift/middle drag pan" in footer
    assert "Status: saved 5 outputs" in footer
    assert viewer_footer("x" * 120).endswith("...")

    chips = viewer_view_chips("normal")
    assert chips == [
        ("1 RGB", False),
        ("2 Alpha", False),
        ("3 Depth", False),
        ("4 Normal", True),
        ("5 Lambertian", False),
        ("6 Blinn-Phong", False),
    ]
    assert view_from_key("6") == "blinn_phong"
    assert view_from_key("7") is None
    assert view_title("RGB") == "RGB"
    assert view_accent("depth") != view_accent("lambertian")
    assert view_accent("normal") != view_accent("blinn_phong")
    assert viewer_view_accent("blinn_phong") != viewer_view_accent("depth")
    assert camera_command_from_key("w") == "forward"
    assert camera_command_from_key("s") == "backward"
    assert camera_command_from_key("shift") == "up"
    assert camera_command_from_key("ctrl") == "down"
    assert camera_command_from_key("ctrl+s") is None
    assert save_command_from_key("ctrl+s") is True
    assert save_command_from_key("s") is False
    assert camera_command_from_key("7") is None

    metadata = viewer_save_metadata(ViewerSettings(width=64, height=48), result, asset)
    assert metadata["display_backend_renderer"] == "restir_gs.restir.renderer.evaluate_restir_display_frame_from_gbuffer"
    assert metadata["evaluation_backend_renderer"] == "restir_gs.restir.renderer.evaluate_restir_frame_from_gbuffer"
    assert metadata["timings"]["total_wall_ms"] == 0.0


def test_save_visibility_display_outputs_do_not_require_reference(tmp_path: Path) -> None:
    rgb = torch.ones((2, 3, 3), dtype=torch.float32)
    scalar = torch.ones((2, 3), dtype=torch.float32)
    mask = torch.ones((2, 3), dtype=torch.bool)
    result = ViewerRenderResult(
        frame_index=0,
        state=OrbitCameraState((0.0, 0.0, 2.0), 0.0, 0.0, 1.5, 1.25, 3, 2),
        gbuffer=SimpleNamespace(rgb=rgb, alpha=scalar, depth=scalar, valid_mask=mask, normal_cam=rgb, normal_mask=mask),
        lambertian=None,
        blinn_phong=SimpleNamespace(composite_rgb=rgb),
        restir=None,
        visibility=ViewerVisibilityResult(
            reference=None,
            geometric_mc=None,
            initial_ris=SimpleNamespace(composite_rgb=rgb),
            error=None,
        ),
        valid_pixels=6,
        render_ms=1.0,
        light_info={},
    )

    paths = save_outputs(result, tmp_path)

    assert "visibility_ris" in paths
    assert "visibility_reference" not in paths
    assert "visibility_error" not in paths
    assert Path(paths["visibility_ris"]).is_file()


def test_viewer_timings_export_stable_dict() -> None:
    timings = ViewerTimings(render_rgbd_gpu_ms=1.0, total_gpu_ms=3.0, total_wall_ms=4.0)

    data = timings.as_dict()

    assert data["render_rgbd_gpu_ms"] == 1.0
    assert data["total_gpu_ms"] == 3.0
    assert data["total_wall_ms"] == 4.0
    assert set(data) == {
        "render_rgbd_gpu_ms",
        "gbuffer_gpu_ms",
        "world_lights_gpu_ms",
        "diffuse_restir_gpu_ms",
        "blinn_phong_gpu_ms",
        "proposal_confidence_gpu_ms",
        "visibility_gpu_ms",
        "total_gpu_ms",
        "total_wall_ms",
    }


def test_viewer_render_requirements_are_view_scoped() -> None:
    base = viewer_render_requirements("rgb")
    assert base == {
        "world_lights": False,
        "diffuse_restir": False,
        "blinn_phong": False,
        "visibility": False,
    }
    assert viewer_computed_views(base) == ("rgb", "alpha", "depth", "normal")

    lambertian = viewer_render_requirements("lambertian")
    assert lambertian["world_lights"] is True
    assert lambertian["diffuse_restir"] is True
    assert lambertian["blinn_phong"] is False
    assert "lambertian" in viewer_computed_views(lambertian)

    blinn_visibility = viewer_render_requirements("blinn_phong", include_visibility=True)
    assert blinn_visibility["world_lights"] is True
    assert blinn_visibility["diffuse_restir"] is False
    assert blinn_visibility["blinn_phong"] is True
    assert blinn_visibility["visibility"] is True
    assert viewer_computed_views(blinn_visibility) == ("rgb", "alpha", "depth", "normal", "blinn_phong", "visibility")


def test_free_camera_local_translation_and_look_are_eye_stable() -> None:
    state = OrbitCameraState((0.0, 0.0, 2.0), 0.0, 0.0, 1.0, 1.25, 64, 48)

    moved = camera_state_translate_local(state, delta_right=0.5, delta_up=0.25, delta_forward=0.75)
    looked = camera_state_look(state, delta_yaw_degrees=20.0, delta_pitch_degrees=10.0)

    assert moved.target == (0.5, 0.25, 2.75)
    assert looked.yaw_degrees == 20.0
    assert looked.pitch_degrees == 10.0
    assert looked.radius == state.radius
    assert looked.target != state.target
    assert torch.allclose(orbit_state_eye(looked, device="cpu"), orbit_state_eye(state, device="cpu"))


def test_interactive_viewer_key_dispatch_keeps_save_and_movement_distinct() -> None:
    calls = []

    class FakeSession:
        status_message = "ready"

        def set_view(self, view):
            calls.append(("view", view))

        def move_camera(self, command):
            calls.append(("move", command))

    viewer = object.__new__(InteractiveViewer)
    viewer.session = FakeSession()
    viewer.draw = lambda: calls.append(("draw", None))
    viewer.save_current = lambda: calls.append(("save", None))

    InteractiveViewer.on_key(viewer, SimpleNamespace(key="s"))
    InteractiveViewer.on_key(viewer, SimpleNamespace(key="control+s"))
    InteractiveViewer.on_key(viewer, SimpleNamespace(key="3"))
    InteractiveViewer.on_key(viewer, SimpleNamespace(key="shift"))
    InteractiveViewer.on_key(viewer, SimpleNamespace(key="control"))

    assert calls == [
        ("move", "backward"),
        ("draw", None),
        ("save", None),
        ("view", "depth"),
        ("draw", None),
        ("move", "up"),
        ("draw", None),
        ("move", "down"),
        ("draw", None),
    ]


def test_interactive_viewer_disables_matplotlib_builtin_s_save() -> None:
    plt = SimpleNamespace(rcParams={"keymap.save": ["s", "ctrl+s"]})

    configure_matplotlib_viewer_keymaps(plt)

    assert plt.rcParams["keymap.save"] == []


def test_interactive_session_switches_cached_layers_without_rerender() -> None:
    calls = []
    asset = ViewerAsset(
        label="fake",
        scene=_tiny_scene(),
        source_path=Path("fake.ply"),
        frame_cameras=[_camera(), _camera()],
        frame_labels=["0", "1"],
        metadata={},
    )

    def fake_reset(_asset, frame_index, width, height, _device):
        return OrbitCameraState((0.0, 0.0, 2.0 + frame_index), 0.0, 0.0, 1.0, 1.25, width, height)

    def fake_render(_asset, frame_index, state, _settings, _visibility_cache, _device, required_view):
        calls.append(required_view)
        return SimpleNamespace(
            frame_index=frame_index,
            state=state,
            computed_views=viewer_computed_views(viewer_render_requirements(required_view)),
            valid_pixels=1,
            render_ms=1.0,
        )

    session = InteractiveSession(
        asset,
        ViewerSettings(width=64, height=48),
        torch.device("cpu"),
        0,
        render_fn=fake_render,
        reset_fn=fake_reset,
    )

    assert calls == ["rgb"]
    assert session.set_view("depth") is False
    assert calls == ["rgb"]
    assert session.set_view("blinn_phong") is True
    assert calls == ["rgb", "blinn_phong"]

    session.move_camera("forward")
    assert calls[-1] == "blinn_phong"
    assert session.step_frame(1) is True
    assert session.frame_index == 1


def test_interactive_session_orbit_keeps_target_for_object_inspection() -> None:
    asset = ViewerAsset(
        label="fake",
        scene=_tiny_scene(),
        source_path=Path("fake.ply"),
        frame_cameras=[_camera()],
        frame_labels=["0"],
        metadata={},
    )

    def fake_reset(_asset, _frame_index, width, height, _device):
        return OrbitCameraState((0.0, 0.0, 2.0), 0.0, 0.0, 1.0, 1.25, width, height)

    def fake_render(_asset, frame_index, state, _settings, _visibility_cache, _device, required_view):
        return SimpleNamespace(
            frame_index=frame_index,
            state=state,
            computed_views=viewer_computed_views(viewer_render_requirements(required_view)),
            valid_pixels=1,
            render_ms=1.0,
        )

    session = InteractiveSession(
        asset,
        ViewerSettings(width=64, height=48),
        torch.device("cpu"),
        0,
        render_fn=fake_render,
        reset_fn=fake_reset,
    )
    target_before = session.state.target
    eye_before = orbit_state_eye(session.state, device="cpu")

    session.orbit(delta_yaw_degrees=20.0, delta_pitch_degrees=10.0)

    assert session.state.target == target_before
    assert not torch.allclose(orbit_state_eye(session.state, device="cpu"), eye_before)


def test_interactive_session_does_not_commit_view_after_failed_render() -> None:
    asset = ViewerAsset(
        label="fake",
        scene=_tiny_scene(),
        source_path=Path("fake.ply"),
        frame_cameras=[_camera()],
        frame_labels=["0"],
        metadata={},
    )

    def fake_reset(_asset, _frame_index, width, height, _device):
        return OrbitCameraState((0.0, 0.0, 2.0), 0.0, 0.0, 1.0, 1.25, width, height)

    def fake_render(_asset, frame_index, state, _settings, _visibility_cache, _device, required_view):
        if required_view == "blinn_phong":
            raise RuntimeError("boom")
        return SimpleNamespace(
            frame_index=frame_index,
            state=state,
            computed_views=viewer_computed_views(viewer_render_requirements(required_view)),
            valid_pixels=1,
            render_ms=1.0,
        )

    session = InteractiveSession(
        asset,
        ViewerSettings(width=64, height=48),
        torch.device("cpu"),
        0,
        render_fn=fake_render,
        reset_fn=fake_reset,
    )

    try:
        session.set_view("blinn_phong")
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected failed layer render to raise.")

    assert session.view == "rgb"
    assert session.result.computed_views == ("rgb", "alpha", "depth", "normal")


def test_view_image_returns_single_requested_layer() -> None:
    rgb = torch.ones((2, 3, 3), dtype=torch.float32)
    scalar = torch.ones((2, 3), dtype=torch.float32)
    mask = torch.ones((2, 3), dtype=torch.bool)
    result = ViewerRenderResult(
        frame_index=0,
        state=OrbitCameraState((0.0, 0.0, 2.0), 0.0, 0.0, 1.5, 1.25, 3, 2),
        gbuffer=SimpleNamespace(rgb=rgb, alpha=scalar, depth=scalar, valid_mask=mask, normal_cam=rgb, normal_mask=mask),
        lambertian=SimpleNamespace(composite_rgb=rgb),
        blinn_phong=SimpleNamespace(composite_rgb=rgb),
        restir=None,
        visibility=None,
        valid_pixels=6,
        render_ms=1.0,
        light_info={},
    )

    assert view_image(result, "rgb")[0] == "RGB"
    assert view_image(result, "rgb")[1].shape == (2, 3, 3)
    assert view_image(result, "alpha")[0] == "Alpha"
    assert view_image(result, "alpha")[1].shape == (2, 3)
    assert view_image(result, "depth")[0] == "Depth"
    assert view_image(result, "normal")[0] == "Normal"
    assert view_image(result, "lambertian")[0] == "Lambertian"
    assert view_image(result, "blinn_phong")[0] == "Blinn-Phong"


def test_view_image_fails_for_uncomputed_lighting_layer() -> None:
    rgb = torch.ones((2, 3, 3), dtype=torch.float32)
    scalar = torch.ones((2, 3), dtype=torch.float32)
    mask = torch.ones((2, 3), dtype=torch.bool)
    result = ViewerRenderResult(
        frame_index=0,
        state=OrbitCameraState((0.0, 0.0, 2.0), 0.0, 0.0, 1.5, 1.25, 3, 2),
        gbuffer=SimpleNamespace(rgb=rgb, alpha=scalar, depth=scalar, valid_mask=mask, normal_cam=rgb, normal_mask=mask),
        lambertian=None,
        blinn_phong=None,
        restir=None,
        visibility=None,
        valid_pixels=6,
        render_ms=1.0,
        light_info={},
        computed_views=("rgb", "alpha", "depth", "normal"),
    )

    assert view_image(result, "rgb")[0] == "RGB"
    try:
        view_image(result, "blinn_phong")
    except RuntimeError as exc:
        assert "Blinn-Phong view was not computed" in str(exc)
    else:
        raise AssertionError("Expected missing Blinn-Phong view to fail loudly.")


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
