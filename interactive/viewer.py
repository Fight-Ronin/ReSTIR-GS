from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

import interactive.rendering as rendering
from interactive.layers import (
    VIEWER_CONTROL_HELP,
    VIEWER_VIEW_ACCENTS,
    VIEWER_VIEW_HELP,
    VIEWER_VIEW_KEYS,
    view_chips,
    view_from_key,
    view_label,
)
from interactive.session import InteractiveSession

if TYPE_CHECKING:
    from interactive.rendering import ViewerAsset, ViewerRenderResult, ViewerSettings


VIEWER_COLORS = {
    "background": "#0f1114",
    "panel": "#15181d",
    "panel_edge": "#2b323b",
    "text": "#edf1f5",
    "muted": "#9da7b3",
    "accent": "#6fd1b5",
    "chip": "#1c2229",
    "chip_active": "#6fd1b5",
    "chip_edge": "#38424e",
    "chip_active_text": "#07110e",
    "status_bg": "#241f13",
    "status": "#f0c36a",
}


def view_title(title: str) -> str:
    return title


def view_accent(view: str) -> str:
    return VIEWER_VIEW_ACCENTS.get(view, VIEWER_COLORS["accent"])


def viewer_title(asset: ViewerAsset) -> str:
    return asset.label


def viewer_metrics(result: ViewerRenderResult, asset: ViewerAsset, view: str) -> str:
    state = result.state
    frame_label = asset.frame_labels[result.frame_index]
    layer_label = view_label(view)
    return (
        f"frame {result.frame_index + 1}/{len(asset.frame_cameras)} ({frame_label}) | view {layer_label} | "
        f"render {result.render_ms:.1f} ms | valid {result.valid_pixels} px | "
        f"yaw {state.yaw_degrees:.1f} | pitch {state.pitch_degrees:.1f} | radius {state.radius:.3f}"
    )


def viewer_view_chips(view: str) -> list[tuple[str, bool]]:
    return view_chips(view)


def viewer_view_accent(view: str) -> str:
    return view_accent(view)


def _clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        raise ValueError(f"Expected positive max_chars, got {max_chars}")
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return "." * max_chars
    return text[: max_chars - 3] + "..."


def viewer_status_line(status_message: str) -> str:
    return f"Status: {_clip_text(status_message, max_chars=96)}"


def viewer_header(result: ViewerRenderResult, asset: ViewerAsset, view: str) -> str:
    return f"{viewer_title(asset)}\n{viewer_metrics(result, asset, view)}"


def viewer_footer(status_message: str) -> str:
    return f"{VIEWER_VIEW_HELP}\n{VIEWER_CONTROL_HELP}\n{viewer_status_line(status_message)}"


def camera_command_from_key(key: str) -> str | None:
    normalized = key.strip().lower().replace("ctrl", "control")
    if not normalized:
        return None
    if save_command_from_key(normalized):
        return None
    primary = normalized.split("+")[-1]
    if primary == "w":
        return "forward"
    if primary == "s":
        return "backward"
    if primary == "a":
        return "left"
    if primary == "d":
        return "right"
    if primary in ("shift", "shift_l", "shift_r"):
        return "up"
    if primary in ("control", "control_l", "control_r"):
        return "down"
    return None


def save_command_from_key(key: str) -> bool:
    normalized = key.strip().lower().replace("ctrl", "control")
    return normalized in ("control+s", "control_l+s", "control_r+s")


def configure_matplotlib_viewer_keymaps(plt_module: Any) -> None:
    plt_module.rcParams["keymap.save"] = []


class InteractiveViewer:
    def __init__(self, asset: ViewerAsset, settings: ViewerSettings, device: torch.device, frame_index: int) -> None:
        self.asset = asset
        self.settings = settings
        self.device = device
        self.drag_start: tuple[float, float] | None = None
        self.drag_mode: str | None = None
        self.closed = False
        self.session = InteractiveSession(
            asset,
            settings,
            device,
            frame_index,
            render_fn=rendering.render_view,
            reset_fn=rendering.reset_state_from_frame,
        )

    def run(self) -> None:
        import matplotlib.pyplot as plt

        configure_matplotlib_viewer_keymaps(plt)
        self.fig, ax = plt.subplots(1, 1, figsize=(12.8, 8.2))
        self.fig.patch.set_facecolor(VIEWER_COLORS["background"])
        self.fig.subplots_adjust(left=0.045, right=0.955, top=0.80, bottom=0.15)
        self.ax = ax
        self.title_artist = self.fig.text(
            0.035,
            0.955,
            "",
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color=VIEWER_COLORS["text"],
        )
        self.metrics_artist = self.fig.text(
            0.035,
            0.912,
            "",
            ha="left",
            va="top",
            fontsize=9.5,
            color=VIEWER_COLORS["muted"],
        )
        self.view_artists = [
            self.fig.text(0.515 + index * 0.076, 0.946, "", ha="center", va="center", fontsize=8.5)
            for index in range(len(VIEWER_VIEW_KEYS))
        ]
        self.status_artist = self.fig.text(
            0.035,
            0.073,
            "",
            ha="left",
            va="center",
            fontsize=9,
            color=VIEWER_COLORS["status"],
        )
        self.help_artist = self.fig.text(
            0.965,
            0.073,
            "",
            ha="right",
            va="center",
            fontsize=8.5,
            color=VIEWER_COLORS["muted"],
        )
        self.header_rule = plt.Line2D(
            [0.035, 0.965],
            [0.842, 0.842],
            transform=self.fig.transFigure,
            color=VIEWER_COLORS["panel_edge"],
            linewidth=0.8,
        )
        self.fig.add_artist(self.header_rule)
        if hasattr(self.fig.canvas.manager, "set_window_title"):
            self.fig.canvas.manager.set_window_title(f"{self.asset.label} viewer")
        self.fig.canvas.mpl_connect("button_press_event", self.on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_button_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.draw()
        plt.show()

    def draw(self) -> None:
        view_accent_color = view_accent(self.session.view)
        title, image = rendering.view_image(self.session.result, self.session.view)
        self.ax.clear()
        self.ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        self.ax.set_title(view_title(title), loc="left", fontsize=11, color=view_accent_color, pad=9)
        self.ax.set_facecolor(VIEWER_COLORS["panel"])
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        for spine in self.ax.spines.values():
            spine.set_visible(True)
            spine.set_color(view_accent_color)
            spine.set_linewidth(1.2)
        self.title_artist.set_text(viewer_title(self.asset))
        self.title_artist.set_color(VIEWER_COLORS["text"])
        self.metrics_artist.set_text(viewer_metrics(self.session.result, self.asset, self.session.view))
        for artist, (label, active) in zip(self.view_artists, viewer_view_chips(self.session.view), strict=True):
            artist.set_text(label)
            artist.set_color(VIEWER_COLORS["chip_active_text"] if active else VIEWER_COLORS["muted"])
            artist.set_fontweight("bold" if active else "normal")
            artist.set_bbox(
                {
                    "boxstyle": "round,pad=0.35,rounding_size=0.18",
                    "facecolor": view_accent_color if active else VIEWER_COLORS["chip"],
                    "edgecolor": view_accent_color if active else VIEWER_COLORS["chip_edge"],
                    "linewidth": 0.8,
                }
            )
        self.header_rule.set_color(view_accent_color)
        self.status_artist.set_text(viewer_status_line(self.session.status_message))
        self.status_artist.set_bbox(
            {
                "boxstyle": "round,pad=0.35,rounding_size=0.15",
                "facecolor": VIEWER_COLORS["status_bg"],
                "edgecolor": VIEWER_COLORS["status_bg"],
                "linewidth": 0.0,
            }
        )
        self.help_artist.set_text(VIEWER_CONTROL_HELP)
        self.fig.canvas.draw_idle()

    def rerender_and_draw(self, status_message: str | None = None) -> None:
        try:
            self.session.render(status_message=status_message)
        except RuntimeError as exc:
            self.session.status_message = f"render failed: {exc}"
            print(self.session.status_message)
            self.draw()
            return
        self.draw()

    def on_button_press(self, event: Any) -> None:
        if event.x is None or event.y is None:
            return
        shift = isinstance(event.key, str) and "shift" in event.key.lower()
        if event.button == 2 or (event.button == 1 and shift):
            self.drag_mode = "pan"
        elif event.button == 3:
            self.drag_mode = "look"
        elif event.button == 1:
            self.drag_mode = "orbit"
        else:
            self.drag_mode = None
        self.drag_start = (float(event.x), float(event.y))

    def on_button_release(self, event: Any) -> None:
        self.drag_start = None
        self.drag_mode = None

    def on_motion(self, event: Any) -> None:
        if self.drag_start is None or self.drag_mode is None or event.x is None or event.y is None:
            return
        x0, y0 = self.drag_start
        dx = float(event.x) - x0
        dy = float(event.y) - y0
        self.drag_start = (float(event.x), float(event.y))
        if self.drag_mode == "orbit":
            self.session.orbit(delta_yaw_degrees=dx * 0.25, delta_pitch_degrees=-dy * 0.25)
        elif self.drag_mode == "look":
            self.session.look(delta_yaw_degrees=dx * 0.25, delta_pitch_degrees=-dy * 0.25)
        else:
            state = self.session.state
            pan_scale = state.radius / float(max(state.width, state.height))
            self.session.pan(
                delta_right=-dx * pan_scale,
                delta_up=dy * pan_scale,
            )
        self.draw()

    def on_scroll(self, event: Any) -> None:
        step = float(getattr(event, "step", 0.0))
        if step == 0.0:
            return
        self.session.dolly(scale=0.9**step)
        self.draw()

    def on_key(self, event: Any) -> None:
        key = event.key
        if key in ("q", "escape"):
            self.session.status_message = "closing"
            self.on_close(event)
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            return
        view = view_from_key(str(key))
        if view is not None:
            try:
                self.session.set_view(view)
            except RuntimeError as exc:
                self.session.status_message = f"render failed: {exc}"
                print(self.session.status_message)
            self.draw()
            return
        if save_command_from_key(str(key)):
            self.save_current()
            return
        camera_command = camera_command_from_key(str(key))
        if camera_command is not None:
            try:
                self.session.move_camera(camera_command)
            except RuntimeError as exc:
                self.session.status_message = f"render failed: {exc}"
                print(self.session.status_message)
            self.draw()
            return
        if key == "r":
            try:
                self.session.reset_camera()
            except RuntimeError as exc:
                self.session.status_message = f"render failed: {exc}"
                print(self.session.status_message)
            self.draw()
            return
        if key in ("[", "]"):
            delta = -1 if key == "[" else 1
            try:
                self.session.step_frame(delta)
            except RuntimeError as exc:
                self.session.status_message = f"render failed: {exc}"
                print(self.session.status_message)
            self.draw()

    def save_current(self) -> None:
        save_result = self.session.render_for_view("blinn_phong")
        metadata = rendering.viewer_save_metadata(self.settings, save_result, self.asset)
        paths = rendering.save_outputs(save_result, self.settings.output_dir, metadata=metadata)
        print("saved interactive viewer outputs:")
        for key, path in paths.items():
            print(f"  {key}: {Path(path).resolve()}")
        self.session.status_message = f"saved {len(paths)} outputs"
        self.draw()

    def on_close(self, event: Any) -> None:
        self.closed = True
