from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Callable

import numpy as np
from PIL import Image
import torch

from interactive.launcher import (
    configure_viewer_runtime_environment,
    load_viewer_asset,
    make_viewer_settings,
)
from interactive.rendering import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_VIEWER_HEIGHT,
    DEFAULT_VIEWER_WIDTH,
    render_view,
    reset_state_from_frame,
    save_outputs,
    view_image,
    viewer_save_metadata,
)
from interactive.session import InteractiveSession
from restir_gs.render.aligned_asset_registry import DEFAULT_MANIFEST_PATH


WEB_DIR = Path(__file__).resolve().parent / "web"


@dataclass
class WebViewerState:
    session: InteractiveSession
    asset_label: str
    output_dir: Path
    save_fn: Callable[[Any, Path], dict[str, str]]
    lock: Lock = field(default_factory=Lock)

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return self._snapshot_unlocked()

    def image_png(self) -> bytes:
        with self.lock:
            _title, image = view_image(self.session.result, self.session.view)
            return image_array_to_png_bytes(image)

    def set_view(self, view: str) -> dict[str, object]:
        with self.lock:
            self.session.set_view(view)
            return self._snapshot_unlocked()

    def apply_action(self, payload: dict[str, object]) -> dict[str, object]:
        action = str(payload.get("action", ""))
        with self.lock:
            if action == "move":
                self.session.move_camera(str(payload.get("command", "")))
            elif action == "orbit":
                dx = float(payload.get("dx", 0.0))
                dy = float(payload.get("dy", 0.0))
                self.session.orbit(delta_yaw_degrees=dx * 0.25, delta_pitch_degrees=-dy * 0.25)
            elif action == "pan":
                dx = float(payload.get("dx", 0.0))
                dy = float(payload.get("dy", 0.0))
                state = self.session.state
                pan_scale = state.radius / float(max(state.width, state.height))
                self.session.pan(delta_right=-dx * pan_scale, delta_up=dy * pan_scale)
            elif action == "dolly":
                self.session.dolly(scale=float(payload.get("scale", 1.0)))
            elif action == "frame":
                self.session.step_frame(int(payload.get("delta", 0)))
            elif action == "reset":
                self.session.reset_camera()
            else:
                raise ValueError(f"Unsupported web viewer action '{action}'.")
            return self._snapshot_unlocked()

    def save(self) -> dict[str, object]:
        with self.lock:
            paths = self.save_fn(self.session, self.output_dir)
            self.session.status_message = f"saved {len(paths)} outputs"
            return {"paths": paths, "snapshot": self._snapshot_unlocked()}

    def _snapshot_unlocked(self) -> dict[str, object]:
        data = self.session.snapshot()
        data["asset"] = {"label": self.asset_label}
        return data


def image_array_to_png_bytes(image: np.ndarray) -> bytes:
    if image.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image array, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected image rank 2 or 3, got {image.ndim}")
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def save_session_outputs(session: InteractiveSession, output_dir: Path) -> dict[str, str]:
    result = session.render_for_view("blinn_phong")
    metadata = viewer_save_metadata(session.settings, result, session.asset)
    return save_outputs(result, output_dir, metadata=metadata)


def create_app(state: WebViewerState):
    FastAPI, Request, FileResponse, JSONResponse, Response, StaticFiles, _ = require_web_dependencies()
    globals()["_FastAPIRequest"] = Request

    app = FastAPI(title="ReSTIR-GS Interactive Viewer")
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(WEB_DIR / "index.html")

    @app.get("/api/snapshot")
    def api_snapshot():
        return JSONResponse(state.snapshot())

    @app.get("/api/image.png")
    def api_image():
        return Response(content=state.image_png(), media_type="image/png")

    @app.post("/api/view")
    async def api_view(request: _FastAPIRequest):
        payload = await request.json()
        return JSONResponse(state.set_view(str(payload.get("view", ""))))

    @app.post("/api/action")
    async def api_action(request: _FastAPIRequest):
        payload = await request.json()
        try:
            return JSONResponse(state.apply_action(payload))
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @app.post("/api/save")
    def api_save(_request: _FastAPIRequest):
        return JSONResponse(state.save())

    return app


def make_web_viewer_state(args: argparse.Namespace) -> WebViewerState:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    settings = make_viewer_settings(args)
    device = torch.device(args.device)
    configure_viewer_runtime_environment(device)
    asset = load_viewer_asset(args, device=device)
    frame_index = args.frame_index
    if frame_index is None:
        frame_index = 49 if len(asset.frame_cameras) > 49 else 0
    if frame_index < 0 or frame_index >= len(asset.frame_cameras):
        raise ValueError(f"Frame index {frame_index} exceeds frame count {len(asset.frame_cameras)}.")
    session = InteractiveSession(
        asset,
        settings,
        device,
        frame_index,
        render_fn=render_view,
        reset_fn=reset_state_from_frame,
    )
    return WebViewerState(
        session=session,
        asset_label=asset.label,
        output_dir=settings.output_dir,
        save_fn=save_session_outputs,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the browser-based ReSTIR-GS interactive viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--ply", type=Path, default=None, help="Generic compatible 3DGS PLY to view. Omit to use a registered aligned asset.")
    parser.add_argument("--camera-config", type=Path, default=None, help="Optional camera config for --ply mode.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--asset-id", default="dxgl_apple", help="Registered aligned asset id to view. Ignored when --ply is provided.")
    parser.add_argument("--frame-index", type=int, default=None)
    parser.add_argument("--width", type=int, default=DEFAULT_VIEWER_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_VIEWER_HEIGHT)
    parser.add_argument("--max-gaussians", type=int, default=0)
    parser.add_argument("--auto-camera-bbox-percentile", type=float, default=0.98)
    parser.add_argument("--auto-camera-radius-scale", type=float, default=1.8)
    parser.add_argument("--num-lights", type=int, default=128)
    parser.add_argument("--light-seed", type=int, default=2027)
    parser.add_argument("--restir-candidate-count", type=int, default=8)
    parser.add_argument("--restir-candidate-seed", type=int, default=34100)
    parser.add_argument("--restir-selection-seed", type=int, default=35100)
    parser.add_argument("--visibility-num-lights", type=int, default=16)
    parser.add_argument("--visibility-light-seed", type=int, default=2027)
    parser.add_argument("--visibility-candidate-count", type=int, default=8)
    parser.add_argument("--visibility-candidate-seed", type=int, default=36100)
    parser.add_argument("--visibility-selection-seed", type=int, default=37100)
    parser.add_argument("--visibility-shadow-resolution", type=int, default=128)
    parser.add_argument("--visibility-shadow-bias-scale", type=float, default=0.02)
    parser.add_argument("--visibility-shadow-alpha-threshold", type=float, default=1e-4)
    parser.add_argument("--visibility-shadow-pcf-radius", type=int, default=1)
    parser.add_argument("--ambient", type=float, default=0.2)
    parser.add_argument("--specular-strength", type=float, default=0.15)
    parser.add_argument("--shininess", type=float, default=24.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Expected positive viewer size, got {args.width}x{args.height}")
    if args.port <= 0:
        raise ValueError(f"Expected positive port, got {args.port}")
    *_, uvicorn = require_web_dependencies()
    state = make_web_viewer_state(args)
    app = create_app(state)
    print(f"serving ReSTIR-GS web viewer at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def require_web_dependencies():
    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import FileResponse, JSONResponse, Response
        from fastapi.staticfiles import StaticFiles
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI WebUI dependencies are not installed. Install them with "
            "`pip install fastapi==0.136.3 uvicorn==0.49.0` or reinstall from requirements.txt."
        ) from exc
    return FastAPI, Request, FileResponse, JSONResponse, Response, StaticFiles, uvicorn


if __name__ == "__main__":
    raise SystemExit(main())
