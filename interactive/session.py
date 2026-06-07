from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import torch

from restir_gs.render.orbit_camera import (
    OrbitCameraState,
    orbit_state_dolly,
    orbit_state_orbit,
    orbit_state_pan,
)

from interactive.camera import camera_move_step, camera_state_translate_local
from interactive.layers import VIEWER_VIEW_LABELS


RenderFn = Callable[[Any, int, OrbitCameraState, Any, Any, torch.device, str], Any]
ResetFn = Callable[[Any, int, int, int, torch.device], OrbitCameraState]


@dataclass
class InteractiveSession:
    asset: Any
    settings: Any
    device: torch.device
    frame_index: int
    render_fn: RenderFn
    reset_fn: ResetFn
    view: str = "rgb"
    status_message: str = "ready"
    state: OrbitCameraState = field(init=False)
    result: Any = field(init=False)

    def __post_init__(self) -> None:
        self.frame_index = int(self.frame_index)
        self.state = self.reset_fn(self.asset, self.frame_index, self.settings.width, self.settings.height, self.device)
        self.result = self.render(required_view=self.view)

    def render(self, required_view: str | None = None, status_message: str | None = None) -> Any:
        view = self.view if required_view is None else required_view
        self.result = self.render_fn(
            self.asset,
            self.frame_index,
            self.state,
            self.settings,
            None,
            self.device,
            view,
        )
        if status_message is not None:
            self.status_message = status_message
        return self.result

    def render_for_view(self, view: str) -> Any:
        if view not in VIEWER_VIEW_LABELS:
            raise ValueError(f"Unsupported viewer view '{view}'.")
        return self.render_fn(
            self.asset,
            self.frame_index,
            self.state,
            self.settings,
            None,
            self.device,
            view,
        )

    def set_view(self, view: str) -> bool:
        if view not in VIEWER_VIEW_LABELS:
            raise ValueError(f"Unsupported viewer view '{view}'.")
        computed_views = tuple(getattr(self.result, "computed_views", ()))
        if view in computed_views:
            self.view = view
            self.status_message = f"view: {VIEWER_VIEW_LABELS[view]}"
            return False
        self.render(required_view=view, status_message=f"view: {VIEWER_VIEW_LABELS[view]}")
        self.view = view
        return True

    def orbit(self, delta_yaw_degrees: float, delta_pitch_degrees: float) -> Any:
        self.state = orbit_state_orbit(
            self.state,
            delta_yaw_degrees=delta_yaw_degrees,
            delta_pitch_degrees=delta_pitch_degrees,
        )
        return self.render()

    def pan(self, delta_right: float, delta_up: float) -> Any:
        self.state = orbit_state_pan(
            self.state,
            delta_right=delta_right,
            delta_up=delta_up,
            device="cpu",
        )
        return self.render()

    def dolly(self, scale: float) -> Any:
        self.state = orbit_state_dolly(self.state, scale=scale)
        return self.render()

    def resize(self, width: int, height: int) -> bool:
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            raise ValueError(f"Expected positive session size, got {width}x{height}")
        if self.state.width == width and self.state.height == height:
            return False
        self.settings = replace(self.settings, width=width, height=height)
        self.state = replace(self.state, width=width, height=height)
        self.render(status_message=f"viewport {width}x{height}")
        return True

    def move_camera(self, command: str) -> bool:
        step = camera_move_step(self.state)
        if command == "forward":
            delta = (0.0, 0.0, step)
        elif command == "backward":
            delta = (0.0, 0.0, -step)
        elif command == "left":
            delta = (-step, 0.0, 0.0)
        elif command == "right":
            delta = (step, 0.0, 0.0)
        elif command == "up":
            delta = (0.0, step, 0.0)
        elif command == "down":
            delta = (0.0, -step, 0.0)
        else:
            return False
        self.state = camera_state_translate_local(
            self.state,
            delta_right=delta[0],
            delta_up=delta[1],
            delta_forward=delta[2],
            device="cpu",
        )
        self.render(status_message=f"camera {command}")
        return True

    def reset_camera(self) -> Any:
        self.state = self.reset_fn(self.asset, self.frame_index, self.settings.width, self.settings.height, self.device)
        return self.render(status_message=f"reset to frame {self.frame_index + 1}/{len(self.asset.frame_cameras)}")

    def step_frame(self, delta: int) -> bool:
        frame_index = max(0, min(len(self.asset.frame_cameras) - 1, self.frame_index + int(delta)))
        if frame_index == self.frame_index:
            edge = "first" if delta < 0 else "last"
            self.status_message = f"already at {edge} frame"
            return False
        self.frame_index = frame_index
        self.state = self.reset_fn(self.asset, self.frame_index, self.settings.width, self.settings.height, self.device)
        self.render(status_message=f"frame {self.frame_index + 1}/{len(self.asset.frame_cameras)}")
        return True

    def snapshot(self) -> dict[str, object]:
        return {
            "frame_index": self.frame_index,
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
                "valid_pixels": int(getattr(self.result, "valid_pixels", 0)),
                "render_ms": float(getattr(self.result, "render_ms", 0.0)),
                "computed_views": list(getattr(self.result, "computed_views", ())),
            },
        }
