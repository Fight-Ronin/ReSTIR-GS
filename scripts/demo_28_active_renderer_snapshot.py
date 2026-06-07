from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restir_gs.eval.active_renderer_snapshot import (
    build_active_renderer_snapshot_summary,
    make_active_renderer_contact_sheet,
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing active renderer rows CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Active renderer rows CSV is empty: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact active renderer demo/performance snapshot from existing outputs.")
    parser.add_argument("--renderer-output-dir", type=Path, default=Path("outputs/aligned_restir"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/active_demo"))
    args = parser.parse_args()

    rows_path = args.renderer_output_dir / "restir_renderer_rows.csv"
    summary_path = args.renderer_output_dir / "restir_renderer_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing active renderer summary JSON: {summary_path}")

    rows = load_rows(rows_path)
    renderer_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshot = build_active_renderer_snapshot_summary(renderer_summary, rows)

    contact_path = args.output_dir / "active_renderer_snapshot_contact.png"
    summary_out = args.output_dir / "active_renderer_snapshot_summary.json"
    make_active_renderer_contact_sheet(args.renderer_output_dir, contact_path, list(snapshot["asset_ids"]))
    snapshot["sources"] = {
        "renderer_rows": str(rows_path),
        "renderer_summary": str(summary_path),
    }
    snapshot["outputs"] = {
        "contact_sheet": str(contact_path),
        "summary": str(summary_out),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    print(f"assets:    {snapshot['asset_ids']}")
    print(f"rows:      {snapshot['row_count']}")
    print(f"finite:    {snapshot['validation']['all_numeric_finite']}")
    print(f"wrote:     {contact_path.resolve()}")
    print(f"wrote:     {summary_out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
