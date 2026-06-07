from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

import requests


TNT_GOOGLE_DRIVE_IDS = {
    "Barn": "0B-ePgl6HF260ZlBZcHFrTHFLdGM",
    "Caterpillar": "0B-ePgl6HF260Z00xVWgyN2c3WEU",
    "Courthouse": "0B-ePgl6HF260TEpnajBqRFJ1enM",
    "Family": "0B-ePgl6HF260UmNxYmlQeDhmeFE",
    "Francis": "0B-ePgl6HF260emtkUElRT0lXQ3M",
    "Ignatius": "0B-ePgl6HF260T19oUTIyUTRwTE0",
    "Meetingroom": "0B-ePgl6HF260V3BFSFFTZFJwSWc",
    "Truck": "0B-ePgl6HF260aVVZMzhSdVc5Njg",
}


def _confirm_token(response: requests.Response) -> str | None:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    set_cookie = response.headers.get("Set-Cookie", "")
    if "download_warning" in set_cookie:
        return set_cookie.split("=", 1)[1].split(";", 1)[0]
    return None


def _hidden_form_params(html: str) -> dict[str, str]:
    return {name: value for name, value in re.findall(r'name="([^"]+)" value="([^"]*)"', html)}


def _download_warning_size(html: str) -> str:
    match = re.search(r"\(([^()]+)\)</span>", html)
    return match.group(1) if match else "unknown"


def probe_google_drive_file(file_id: str) -> str:
    response = requests.get(
        "https://docs.google.com/uc",
        params={"export": "download", "id": file_id},
        timeout=60,
    )
    response.raise_for_status()
    if "Google Drive - Virus scan warning" in response.text:
        return _download_warning_size(response.text)
    size = response.headers.get("Content-Length")
    return f"{int(size) / (1024 * 1024):.1f} MiB" if size else "unknown"


def download_google_drive_file(file_id: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    url = "https://docs.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
    token = _confirm_token(response)
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True, timeout=60)
    elif "Google Drive - Virus scan warning" in response.text:
        action = re.search(r'<form[^>]+action="([^"]+)"', response.text)
        if action is None:
            raise RuntimeError("Could not find Google Drive download form action.")
        response = session.get(
            urljoin(response.url, action.group(1)),
            params=_hidden_form_params(response.text),
            stream=True,
            timeout=60,
        )
    response.raise_for_status()
    if "text/html" in response.headers.get("Content-Type", ""):
        raise RuntimeError("Google Drive returned HTML instead of the video file.")

    bytes_written = 0
    with output_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            bytes_written += len(chunk)
            print(f"\r{bytes_written / (1024 * 1024):.1f} MiB", end="", flush=True)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download one Tanks and Temples video.")
    parser.add_argument("--scene", choices=sorted(TNT_GOOGLE_DRIVE_IDS), default="Meetingroom")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gsgen/tnt/videos"))
    parser.add_argument("--probe-sizes", action="store_true", help="Print visible Google Drive sizes without downloading.")
    args = parser.parse_args()

    if args.probe_sizes:
        for scene, file_id in sorted(TNT_GOOGLE_DRIVE_IDS.items()):
            print(f"{scene}: {probe_google_drive_file(file_id)}")
        return 0

    output_path = args.output_dir / f"{args.scene}.mp4"
    download_google_drive_file(TNT_GOOGLE_DRIVE_IDS[args.scene], output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
