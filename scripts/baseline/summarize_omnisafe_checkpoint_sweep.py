#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from _common import artifact_dir, ensure_directory, load_config, read_json, relative_to_repo, write_json
from evaluate_checkpoint_sweep import composite_score, selected_row, selection_status


DEFAULT_CONFIG = "configs/methods/omnisafe_ppolag_eval_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize OmniSafe checkpoint metrics with shared selection rule.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoints", required=True, help="Comma-separated checkpoint ids.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    run_name = args.run_name or config["run_name"]
    output_dir = ensure_directory(artifact_dir(config, run_name))
    selection_cfg = config.get("selection", {})

    rows: list[dict[str, Any]] = []
    for raw_checkpoint in args.checkpoints.split(","):
        raw_checkpoint = raw_checkpoint.strip()
        if not raw_checkpoint:
            continue
        checkpoint = int(raw_checkpoint)
        metrics_path = output_dir / f"metrics_checkpoint_{checkpoint}.json"
        metrics = read_json(metrics_path)
        row = {
            "checkpoint": checkpoint,
            "metrics_path": relative_to_repo(metrics_path),
            "composite_score": composite_score(metrics),
            **metrics,
        }
        rows.append(row)

    selected, selection_details = selected_row(
        rows,
        tracking_tolerance=float(selection_cfg.get("tracking_tolerance", 0.10)),
        fall_tolerance=float(selection_cfg.get("fall_tolerance", 0.05)),
    )
    summary = {
        "status": "complete",
        "run_name": run_name,
        "method_id": config.get("method", {}).get("id"),
        "selection_status": selection_status(rows),
        "selection_rule": "task_floor_then_smoothest_eligible",
        "selection_details": selection_details,
        "selected_checkpoint": int(selected["checkpoint"]),
        "selected_metrics_path": selected["metrics_path"],
        "selected_metrics": selected,
        "checkpoints": [int(row["checkpoint"]) for row in rows],
        "rows": rows,
    }
    output_path = output_dir / "checkpoint_sweep_summary.json"
    write_json(output_path, summary)
    print(f"Wrote {relative_to_repo(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
