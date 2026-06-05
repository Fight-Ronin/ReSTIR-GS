from __future__ import annotations

import json

import pytest
import torch

from restir_gs.render.camera_probe import load_camera_config
from restir_gs.render.orbit_camera import (
    OrbitCameraState,
    camera_eye,
    orbit_state_dolly,
    orbit_state_from_camera,
    orbit_state_orbit,
    orbit_state_pan,
    orbit_state_to_camera,
    save_orbit_camera_config,
)


def _world_to_camera_point(camera, point: torch.Tensor) -> torch.Tensor:
    point_h = torch.cat((point, torch.ones(1)))
    return camera.viewmats[0] @ point_h


def test_orbit_camera_maps_target_to_positive_camera_z() -> None:
    state = OrbitCameraState(
        target=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        pitch_degrees=0.0,
        radius=2.0,
        focal_scale=1.25,
        width=64,
        height=48,
    )

    camera = orbit_state_to_camera(state, device="cpu")
    target_cam = _world_to_camera_point(camera, torch.tensor(state.target))

    assert torch.allclose(target_cam[:2], torch.zeros(2), atol=1e-6)
    assert target_cam[2].item() > 0.0
    assert camera.viewmats.shape == (1, 4, 4)
    assert camera.intrinsics.shape == (1, 3, 3)


def test_orbit_changes_eye_and_preserves_target_and_radius() -> None:
    state = OrbitCameraState((0.1, -0.2, 0.3), 0.0, 0.0, 2.5, 1.0, 80, 60)
    before = camera_eye(orbit_state_to_camera(state, device="cpu"))

    moved = orbit_state_orbit(state, delta_yaw_degrees=20.0, delta_pitch_degrees=10.0)
    after = camera_eye(orbit_state_to_camera(moved, device="cpu"))

    assert moved.target == state.target
    assert moved.radius == pytest.approx(state.radius)
    assert not torch.allclose(before, after)
    assert torch.linalg.norm(after - torch.tensor(moved.target)).item() == pytest.approx(state.radius, abs=1e-6)


def test_dolly_changes_radius_and_clamps_positive() -> None:
    state = OrbitCameraState((0.0, 0.0, 0.0), 0.0, 0.0, 2.0, 1.0, 64, 64)

    closer = orbit_state_dolly(state, scale=0.5)
    clamped = orbit_state_dolly(state, scale=0.001, min_radius=0.25)

    assert closer.radius == pytest.approx(1.0)
    assert clamped.radius == pytest.approx(0.25)


def test_pan_shifts_target_in_camera_right_and_up_directions() -> None:
    state = OrbitCameraState((0.0, 0.0, 0.0), 0.0, 0.0, 2.0, 1.0, 64, 64)

    panned = orbit_state_pan(state, delta_right=0.5, delta_up=0.25, device="cpu")

    assert panned.target[0] == pytest.approx(0.5, abs=1e-6)
    assert panned.target[1] == pytest.approx(0.25, abs=1e-6)
    assert panned.target[2] == pytest.approx(0.0, abs=1e-6)
    assert panned.radius == pytest.approx(state.radius)


def test_reset_from_camera_reconstructs_valid_orbit_state() -> None:
    state = OrbitCameraState((0.25, 0.0, 0.75), 17.0, -8.0, 3.0, 1.2, 96, 64)
    camera = orbit_state_to_camera(state, device="cpu")

    restored = orbit_state_from_camera(camera, target=state.target)
    restored_camera = orbit_state_to_camera(restored, device="cpu")

    assert restored.target == pytest.approx(state.target)
    assert restored.radius == pytest.approx(state.radius, abs=1e-5)
    assert torch.allclose(camera_eye(restored_camera), camera_eye(camera), atol=1e-5)


def test_saved_camera_payload_round_trips_through_existing_loader(tmp_path) -> None:
    state = OrbitCameraState((0.0, 0.1, 0.2), 5.0, 3.0, 1.5, 1.1, 72, 40)
    path = tmp_path / "camera.json"

    save_orbit_camera_config(state, path, metadata={"source": "unit_test"})
    loaded = load_camera_config(path, device="cpu")
    data = json.loads(path.read_text(encoding="utf-8"))

    assert loaded.width == state.width
    assert loaded.height == state.height
    assert loaded.viewmats.shape == (1, 4, 4)
    assert data["metadata"]["source"] == "unit_test"
    assert data["orbit_camera_state"]["radius"] == pytest.approx(state.radius)


@pytest.mark.parametrize(
    "state",
    [
        OrbitCameraState((0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 1.0, 64, 64),
        OrbitCameraState((0.0, 0.0, 0.0), 0.0, 0.0, 1.0, 0.0, 64, 64),
        OrbitCameraState((0.0, 0.0, 0.0), 0.0, 0.0, 1.0, 1.0, 0, 64),
    ],
)
def test_invalid_orbit_state_fails_loudly(state: OrbitCameraState) -> None:
    with pytest.raises(ValueError):
        orbit_state_to_camera(state, device="cpu")


def test_invalid_dolly_scale_fails_loudly() -> None:
    state = OrbitCameraState((0.0, 0.0, 0.0), 0.0, 0.0, 1.0, 1.0, 64, 64)

    with pytest.raises(ValueError):
        orbit_state_dolly(state, scale=0.0)
