from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewLayer:
    key: str
    label: str
    value: str
    accent: str


VIEW_LAYERS = (
    ViewLayer("1", "RGB", "rgb", "#7cc7ff"),
    ViewLayer("2", "Alpha", "alpha", "#8fb8ff"),
    ViewLayer("3", "Depth", "depth", "#b9a2ff"),
    ViewLayer("4", "Normal", "normal", "#6fd1b5"),
    ViewLayer("5", "Lambertian", "lambertian", "#f0c36a"),
    ViewLayer("6", "Blinn-Phong", "blinn_phong", "#ff9b73"),
)

VIEWER_VIEW_LABELS = {layer.value: layer.label for layer in VIEW_LAYERS}
VIEWER_VIEW_KEYS = tuple((layer.key, layer.label, layer.value) for layer in VIEW_LAYERS)
VIEWER_VIEW_ACCENTS = {layer.value: layer.accent for layer in VIEW_LAYERS}
VIEWER_VIEW_HELP = "1 RGB  |  2 Alpha  |  3 Depth  |  4 Normal  |  5 Lambertian  |  6 Blinn-Phong"
VIEWER_CONTROL_HELP = "WASD move  |  Shift/Ctrl up/down  |  Left drag orbit  |  Right drag look  |  Shift/middle drag pan  |  Wheel dolly  |  [ ] frame  |  r reset  |  Ctrl+S save  |  q quit"


def view_from_key(key: str) -> str | None:
    for layer in VIEW_LAYERS:
        if key == layer.key:
            return layer.value
    return None


def view_label(view: str) -> str:
    return VIEWER_VIEW_LABELS.get(view, view)


def view_chips(view: str) -> list[tuple[str, bool]]:
    return [(f"{layer.key} {layer.label}", layer.value == view) for layer in VIEW_LAYERS]


def viewer_render_requirements(view: str, include_visibility: bool = False) -> dict[str, bool]:
    if view not in VIEWER_VIEW_LABELS:
        raise ValueError(f"Unsupported viewer view '{view}'.")
    needs_diffuse = view == "lambertian"
    needs_blinn = view == "blinn_phong"
    return {
        "world_lights": needs_diffuse or needs_blinn,
        "diffuse_restir": needs_diffuse,
        "blinn_phong": needs_blinn,
        "visibility": include_visibility,
    }


def viewer_computed_views(requirements: dict[str, bool]) -> tuple[str, ...]:
    views = ["rgb", "alpha", "depth", "normal"]
    if requirements["diffuse_restir"]:
        views.append("lambertian")
    if requirements["blinn_phong"]:
        views.append("blinn_phong")
    if requirements["visibility"]:
        views.append("visibility")
    return tuple(views)
