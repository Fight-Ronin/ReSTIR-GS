from __future__ import annotations

from pathlib import Path


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    target_fps: float = 5.0,
    max_frames: int | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError(f"max_frames must be positive or None, got {max_frames}")
    if not video_path.is_file():
        raise FileNotFoundError(f"video file does not exist: {video_path}")

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    indices = compute_extract_indices(frame_count, source_fps, target_fps, max_frames=max_frames)
    if dry_run:
        cap.release()
        return _result(video_path, output_dir, source_fps, frame_count, width, height, target_fps, indices, written=0, dry_run=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    wanted = set(indices)
    current = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if current in wanted:
            out_path = output_dir / f"frame_{written:06d}.png"
            if not cv2.imwrite(str(out_path), frame):
                cap.release()
                raise RuntimeError(f"failed to write frame: {out_path}")
            written += 1
            if max_frames is not None and written >= max_frames:
                break
        current += 1
    cap.release()
    return _result(video_path, output_dir, source_fps, frame_count, width, height, target_fps, indices, written=written, dry_run=False)


def compute_extract_indices(
    frame_count: int,
    source_fps: float,
    target_fps: float,
    max_frames: int | None = None,
) -> list[int]:
    if frame_count <= 0:
        return []
    if target_fps <= 0.0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    step = 1 if source_fps <= 0.0 else max(1, round(float(source_fps) / float(target_fps)))
    indices = list(range(0, frame_count, step))
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError(f"max_frames must be positive or None, got {max_frames}")
        indices = indices[:max_frames]
    return indices


def _result(
    video_path: Path,
    output_dir: Path,
    source_fps: float,
    frame_count: int,
    width: int,
    height: int,
    target_fps: float,
    indices: list[int],
    written: int,
    dry_run: bool,
) -> dict[str, object]:
    return {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "source_fps": source_fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "target_fps": target_fps,
        "planned_frame_count": len(indices),
        "first_indices": indices[:10],
        "written": written,
        "dry_run": dry_run,
    }
