from __future__ import annotations

import builtins
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from interactive.web_server import WebViewerState, build_parser, create_app, image_array_to_png_bytes, require_web_dependencies
from restir_gs.render.orbit_camera import OrbitCameraState


def test_image_array_to_png_bytes_round_trips_rgb() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[0, 1] = [10, 20, 30]

    png = image_array_to_png_bytes(image)
    decoded = np.asarray(Image.open(BytesIO(png)))

    assert png.startswith(b"\x89PNG")
    assert decoded.shape == image.shape
    assert decoded[0, 1].tolist() == [10, 20, 30]


def test_image_array_to_png_bytes_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="Expected uint8"):
        image_array_to_png_bytes(np.zeros((2, 3, 3), dtype=np.float32))


def test_web_viewer_state_routes_actions_and_save(tmp_path: Path) -> None:
    session = _FakeSession()

    def fake_save(save_session, output_dir):
        assert save_session is session
        assert output_dir == tmp_path
        return {"camera": str(output_dir / "current_camera.json")}

    state = WebViewerState(session=session, asset_label="fake asset", output_dir=tmp_path, save_fn=fake_save)

    assert state.snapshot()["asset"] == {"label": "fake asset"}
    assert state.set_view("depth")["view"] == "depth"
    state.apply_action({"action": "move", "command": "forward"})
    state.apply_action({"action": "orbit", "dx": 4.0, "dy": -8.0})
    state.apply_action({"action": "pan", "dx": 16.0, "dy": 8.0})
    state.apply_action({"action": "dolly", "scale": 0.9})
    state.apply_action({"action": "frame", "delta": 1})
    state.apply_action({"action": "reset"})
    save_result = state.save()

    assert save_result["paths"]["camera"].endswith("current_camera.json")
    assert save_result["snapshot"]["status"] == "saved 1 outputs"
    assert session.calls == [
        ("view", "depth"),
        ("move", "forward"),
        ("orbit", 1.0, 2.0),
        ("pan", -0.25, 0.125),
        ("dolly", 0.9),
        ("frame", 1),
        ("reset",),
    ]


def test_web_viewer_state_rejects_unknown_action(tmp_path: Path) -> None:
    state = WebViewerState(
        session=_FakeSession(),
        asset_label="fake asset",
        output_dir=tmp_path,
        save_fn=lambda _session, _output_dir: {},
    )

    with pytest.raises(ValueError, match="Unsupported web viewer action"):
        state.apply_action({"action": "teleport"})


def test_web_parser_exposes_host_port_and_viewer_resolution() -> None:
    args = build_parser().parse_args(["--host", "0.0.0.0", "--port", "9000", "--width", "320", "--height", "240"])

    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.width == 320
    assert args.height == 240


def test_create_app_registers_web_viewer_routes(tmp_path: Path) -> None:
    state = WebViewerState(
        session=_FakeSession(),
        asset_label="fake asset",
        output_dir=tmp_path,
        save_fn=lambda _session, _output_dir: {"camera": "current_camera.json"},
    )

    app = create_app(state)
    routes = {
        (route.path, method)
        for route in app.routes
        if getattr(route, "methods", None)
        for method in route.methods
    }

    assert ("/", "GET") in routes
    assert ("/api/snapshot", "GET") in routes
    assert ("/api/image.png", "GET") in routes
    assert ("/api/view", "POST") in routes
    assert ("/api/action", "POST") in routes
    assert ("/api/save", "POST") in routes


def test_require_web_dependencies_error_mentions_pinned_install(monkeypatch) -> None:
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "fastapi" or name.startswith("fastapi."):
            raise ImportError("missing fastapi")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError) as excinfo:
        require_web_dependencies()

    message = str(excinfo.value)
    assert "fastapi==0.136.3" in message
    assert "uvicorn==0.49.0" in message


class _FakeSession:
    def __init__(self) -> None:
        self.calls = []
        self.view = "rgb"
        self.status_message = "ready"
        self.state = OrbitCameraState((0.0, 0.0, 2.0), 0.0, 0.0, 1.0, 1.25, 64, 32)
        self.result = SimpleNamespace(valid_pixels=12, render_ms=3.5, computed_views=("rgb", "depth"))

    def snapshot(self) -> dict[str, object]:
        return {
            "frame_index": 0,
            "view": self.view,
            "status": self.status_message,
            "camera": {
                "target": self.state.target,
                "yaw_degrees": self.state.yaw_degrees,
                "pitch_degrees": self.state.pitch_degrees,
                "radius": self.state.radius,
                "focal_scale": self.state.focal_scale,
                "width": self.state.width,
                "height": self.state.height,
            },
            "render": {
                "valid_pixels": self.result.valid_pixels,
                "render_ms": self.result.render_ms,
                "computed_views": list(self.result.computed_views),
            },
        }

    def set_view(self, view: str) -> bool:
        self.calls.append(("view", view))
        self.view = view
        self.status_message = f"view: {view}"
        return True

    def move_camera(self, command: str) -> bool:
        self.calls.append(("move", command))
        self.status_message = f"camera {command}"
        return True

    def orbit(self, delta_yaw_degrees: float, delta_pitch_degrees: float) -> object:
        self.calls.append(("orbit", delta_yaw_degrees, delta_pitch_degrees))
        return self.result

    def pan(self, delta_right: float, delta_up: float) -> object:
        self.calls.append(("pan", delta_right, delta_up))
        return self.result

    def dolly(self, scale: float) -> object:
        self.calls.append(("dolly", scale))
        return self.result

    def step_frame(self, delta: int) -> bool:
        self.calls.append(("frame", delta))
        return True

    def reset_camera(self) -> object:
        self.calls.append(("reset",))
        return self.result
