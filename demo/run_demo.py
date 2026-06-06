#!/usr/bin/env python3
"""Build a lightweight, standalone demo bundle for SCO-humanoid.

The demo is intentionally offline-first and self-contained: it reads only the
data and figures shipped inside the `demo/` directory, assembles a compact
markdown report, copies a few key figures, and writes a small JSON summary.
This makes the demo usable on a laptop or review machine without requiring the
rest of the repository, Isaac Gym, or MuJoCo runtime access.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_ROOT / "data"
FIGURE_DIR = PACKAGE_ROOT / "assets" / "figures"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "demo_output"

ISAAC_TABLE = DATA_DIR / "table_full_paper_isaac_mechanism_comparison.csv"
MUJOCO_TABLE = DATA_DIR / "table_matched_mujoco_mechanism_comparison.csv"
STAT_SUMMARY = DATA_DIR / "full_paper_statistics_summary.json"
PAPER_FIGURE_MANIFEST = DATA_DIR / "paper_figures_manifest.json"
UPSTREAM_SOURCE_MANIFEST = DATA_DIR / "upstream_sources.json"
FIGURE_PATHS = [
    FIGURE_DIR / "figure_mechanism_chain.png",
    FIGURE_DIR / "figure_sensitivity_vs_degradation.png",
    FIGURE_DIR / "figure_task_vs_smoothness.png",
]

QUESTION = (
    "Is policy-local-sensitivity regularization a useful smooth-control mechanism for "
    "humanoid locomotion, and how do hard constraints, soft penalties, and reward "
    "shaping trade off under the same Isaac/MuJoCo protocol?"
)
THESIS = (
    "Policy-local-sensitivity regularization is a useful smooth-control lens, but "
    "enforcement details matter and no single method dominates every metric."
)
METHODS = [
    "LCP-style soft Jacobian/Lipschitz penalty",
    "SC-PPO 3.8 with a hard policy-local-sensitivity constraint and PID-Lagrangian enforcement",
    "PPO with a revised heuristic action-rate reward-shaping anchor",
]
BOUNDARIES = [
    "LCP-style is a same-task local adaptation, not official LCP code/checkpoint parity.",
    "MuJoCo evidence is sim-to-sim replay evidence, not hardware validation.",
    "The main comparison is a five-seed selected-checkpoint study, not a broad hyperparameter sweep.",
    "The current paper is a mechanism-comparison result, not an all-metrics-winner claim for one method.",
]

DISPLAY_NAME_BY_METHOD_ID = {
    "lcp": "LCP-style soft penalty",
    "scppo": "SC-PPO 3.8 PID",
    "heuristic": "Revised heuristic",
}
DISPLAY_NAME_BY_METHOD_TEXT = {
    "LCP-style soft Jacobian/Lipschitz penalty": "LCP-style soft penalty",
    "SC-PPO 3.8 PID-Lagrangian": "SC-PPO 3.8 PID",
    "Revised heuristic action-rate penalty": "Revised heuristic",
}
ISAAC_METRICS = [
    ("fall_rate", "Fall", True),
    ("velocity_error", "Vel. err", True),
    ("joint_acceleration", "Jnt acc", True),
    ("action_jitter", "Jitter", True),
    ("episode_return", "Return", False),
    ("policy_sensitivity", "Sensitivity", True),
]
MUJOCO_METRICS = [
    ("fall_rate", "Fall", True),
    ("velocity_error", "Vel. err", True),
    ("joint_acceleration", "Jnt acc", True),
    ("action_jitter", "Jitter", True),
    ("episode_return", "Return", False),
]


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing demo source files:\n" + "\n".join(missing))


def normalize_method_name(name: str) -> str:
    return DISPLAY_NAME_BY_METHOD_TEXT.get(name, name)


def parse_metric_cell(cell: str) -> float | None:
    text = cell.strip()
    if not text:
        return None
    if "+/-" in text:
        text = text.split("+/-", 1)[0].strip()
    return float(text)


def normalize_table_rows(rows: list[dict[str, str]], metrics: list[tuple[str, str, bool]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        normalized_row = {
            "method": normalize_method_name(row["method"]),
            "checkpoints": row.get("checkpoints", ""),
            "seeds": row.get("seeds", ""),
        }
        for key, _, _ in metrics:
            normalized_row[key] = row.get(key, "")
        normalized.append(normalized_row)
    return normalized


def metric_winners(rows: list[dict[str, Any]], metrics: list[tuple[str, str, bool]]) -> dict[str, str]:
    winners: dict[str, str] = {}
    for key, label, lower_is_better in metrics:
        candidates = []
        for row in rows:
            value = parse_metric_cell(row[key])
            if value is None:
                continue
            candidates.append((value, row["method"]))
        if not candidates:
            continue
        sort_key = min if lower_is_better else max
        winning_value = sort_key(value for value, _ in candidates)
        winner_methods = [method for value, method in candidates if value == winning_value]
        winners[label] = " / ".join(winner_methods)
    return winners


def reliability_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.get("reliability_rows", []):
        rows.append(
            {
                "method": DISPLAY_NAME_BY_METHOD_ID.get(row["method_id"], row["method"]),
                "selected_task_valid": f"{row['selected_task_valid_seed_count']}/5",
                "selected_zero_fall": f"{row['selected_zero_fall_seed_count']}/5",
                "changed_seeds": f"{row['changed_seed_count']}/5",
                "selected_equals_final": f"{row['selected_equals_final_seed_count']}/5",
                "checkpoint_classification": row["checkpoint_classification"],
            }
        )
    return rows


def key_takeaways(
    isaac_winners: dict[str, str],
    mujoco_winners: dict[str, str],
    reliability: list[dict[str, Any]],
) -> list[str]:
    reliability_by_method = {row["method"]: row for row in reliability}
    lcp_reliability = reliability_by_method.get("LCP-style soft penalty", {})
    scppo_reliability = reliability_by_method.get("SC-PPO 3.8 PID", {})
    heuristic_reliability = reliability_by_method.get("Revised heuristic", {})
    mujoco_joint_winner = mujoco_winners.get("Jnt acc", "Revised heuristic")
    mujoco_return_winner = mujoco_winners.get("Return", "Revised heuristic")
    if mujoco_joint_winner == mujoco_return_winner:
        mujoco_tradeoff_text = (
            f"{mujoco_joint_winner} is best on joint acceleration and return."
        )
    else:
        mujoco_tradeoff_text = (
            f"{mujoco_joint_winner} is best on joint acceleration, while "
            f"{mujoco_return_winner} is best on return."
        )
    return [
        (
            "Isaac selected-checkpoint comparison favors "
            f"{isaac_winners.get('Fall', 'LCP-style soft penalty')} on fall rate, "
            f"{isaac_winners.get('Vel. err', 'LCP-style soft penalty')} on velocity error, "
            f"{isaac_winners.get('Sensitivity', 'LCP-style soft penalty')} on policy sensitivity, "
            f"while {isaac_winners.get('Jnt acc', 'Revised heuristic')} remains best on joint acceleration."
        ),
        (
            "Matched MuJoCo replay is intentionally mixed: "
            f"{mujoco_winners.get('Jitter', 'LCP-style soft penalty')} is best on action jitter, while "
            f"{mujoco_tradeoff_text}"
        ),
        (
            "Checkpoint dependence is part of the story: "
            f"LCP changes {lcp_reliability.get('changed_seeds', '?')} seeds, "
            f"SC-PPO changes {scppo_reliability.get('changed_seeds', '?')}, and "
            f"the heuristic changes {heuristic_reliability.get('changed_seeds', '?')} under the selected-vs-final audit."
        ),
    ]


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    header_line = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header_line, divider, *body])


def render_report(bundle: dict[str, Any]) -> str:
    isaac_table = render_table(
        ["Method", "Checkpoints", "Fall", "Vel. err", "Jnt acc", "Jitter", "Return", "Sensitivity"],
        [
            [
                row["method"],
                row["checkpoints"],
                row["fall_rate"],
                row["velocity_error"],
                row["joint_acceleration"],
                row["action_jitter"],
                row["episode_return"],
                row["policy_sensitivity"],
            ]
            for row in bundle["isaac_rows"]
        ],
    )
    mujoco_table = render_table(
        ["Method", "Checkpoints", "Fall", "Vel. err", "Jnt acc", "Jitter", "Return"],
        [
            [
                row["method"],
                row["checkpoints"],
                row["fall_rate"],
                row["velocity_error"],
                row["joint_acceleration"],
                row["action_jitter"],
                row["episode_return"],
            ]
            for row in bundle["mujoco_rows"]
        ],
    )
    reliability_table = render_table(
        ["Method", "Selected task-valid", "Zero-fall", "Changed seeds", "Selected=final", "Checkpoint class"],
        [
            [
                row["method"],
                row["selected_task_valid"],
                row["selected_zero_fall"],
                row["changed_seeds"],
                row["selected_equals_final"],
                row["checkpoint_classification"],
            ]
            for row in bundle["reliability_rows"]
        ],
    )
    figure_lines = [
        f"### {figure['label']}\n\n![{figure['label']}]({figure['relative_path']})"
        for figure in bundle["figures"]
    ]
    source_list = "\n".join(f"- `{source}`" for source in bundle["source_files"])
    boundaries = "\n".join(f"- {line}" for line in BOUNDARIES)
    methods = "\n".join(f"- {line}" for line in METHODS)
    takeaways = "\n".join(f"1. {line}" for line in bundle["takeaways"])
    return "\n".join(
        [
            "# SCO-humanoid Demo Report",
            "",
            f"Generated at: `{bundle['generated_at']}`",
            "",
            "## Research Question",
            "",
            QUESTION,
            "",
            "## Current Thesis",
            "",
            THESIS,
            "",
            "## Methods Compared",
            "",
            methods,
            "",
            "## Isaac Selected-Checkpoint Comparison",
            "",
            isaac_table,
            "",
            "## Matched MuJoCo Replay",
            "",
            mujoco_table,
            "",
            "## Reliability Snapshot",
            "",
            reliability_table,
            "",
            "## Key Takeaways",
            "",
            takeaways,
            "",
            "## Key Figures",
            "",
            *figure_lines,
            "",
            "## Boundaries",
            "",
            boundaries,
            "",
            "## Demo Source Files",
            "",
            source_list,
            "",
        ]
    )


def copy_figures(output_dir: Path) -> list[dict[str, str]]:
    figures_dir = ensure_dir(output_dir / "figures")
    copied: list[dict[str, str]] = []
    label_by_name = {
        "figure_mechanism_chain.png": "Mechanism chain",
        "figure_sensitivity_vs_degradation.png": "Sensitivity vs degradation",
        "figure_task_vs_smoothness.png": "Task vs smoothness",
    }
    for source in FIGURE_PATHS:
        destination = figures_dir / source.name
        shutil.copy2(source, destination)
        copied.append(
            {
                "label": label_by_name.get(source.name, source.stem),
                "source": str(source.relative_to(PACKAGE_ROOT)),
                "relative_path": str(destination.relative_to(output_dir)),
            }
        )
    return copied


def build_demo_bundle(output_dir: Path, *, copy_demo_figures: bool = True) -> dict[str, Any]:
    require_paths(
        [ISAAC_TABLE, MUJOCO_TABLE, STAT_SUMMARY, PAPER_FIGURE_MANIFEST, UPSTREAM_SOURCE_MANIFEST, *FIGURE_PATHS]
    )
    ensure_dir(output_dir)

    isaac_rows = normalize_table_rows(read_csv_rows(ISAAC_TABLE), ISAAC_METRICS)
    mujoco_rows = normalize_table_rows(read_csv_rows(MUJOCO_TABLE), MUJOCO_METRICS)
    stats_summary = read_json(STAT_SUMMARY)
    paper_manifest = read_json(PAPER_FIGURE_MANIFEST)
    upstream_manifest = read_json(UPSTREAM_SOURCE_MANIFEST)
    figures = copy_figures(output_dir) if copy_demo_figures else []
    isaac_winners = metric_winners(isaac_rows, ISAAC_METRICS)
    mujoco_winners = metric_winners(mujoco_rows, MUJOCO_METRICS)
    reliability = reliability_rows(stats_summary)

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": QUESTION,
        "thesis": THESIS,
        "isaac_rows": isaac_rows,
        "mujoco_rows": mujoco_rows,
        "isaac_metric_winners": isaac_winners,
        "mujoco_metric_winners": mujoco_winners,
        "reliability_rows": reliability,
        "takeaways": key_takeaways(isaac_winners, mujoco_winners, reliability),
        "figures": figures,
        "source_files": [
            str(ISAAC_TABLE.relative_to(PACKAGE_ROOT)),
            str(MUJOCO_TABLE.relative_to(PACKAGE_ROOT)),
            str(STAT_SUMMARY.relative_to(PACKAGE_ROOT)),
            str(PAPER_FIGURE_MANIFEST.relative_to(PACKAGE_ROOT)),
            str(UPSTREAM_SOURCE_MANIFEST.relative_to(PACKAGE_ROOT)),
            *[entry["source"] for entry in figures],
        ],
        "source_artifacts": stats_summary.get("source_artifacts", []),
        "paper_manifest_outputs": [output.get("file", "") for output in paper_manifest.get("outputs", [])],
        "upstream_sources": upstream_manifest,
    }

    report_text = render_report(bundle)
    report_path = output_dir / "DEMO_REPORT.md"
    report_path.write_text(report_text, encoding="utf-8")

    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=True), encoding="utf-8")

    manifest = {
        "generated_at": bundle["generated_at"],
        "report": str(report_path.relative_to(output_dir)),
        "summary_json": str(summary_path.relative_to(output_dir)),
        "figures": [figure["relative_path"] for figure in figures],
        "source_files": bundle["source_files"],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")

    bundle["report_path"] = str(report_path)
    bundle["summary_path"] = str(summary_path)
    bundle["manifest_path"] = str(manifest_path)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the lightweight SCO-humanoid demo bundle.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for the generated markdown report, JSON summary, and copied figures.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Do not copy key figures into the demo output directory.",
    )
    parser.add_argument(
        "--print-report",
        action="store_true",
        help="Print the generated markdown report to stdout after writing it.",
    )
    args = parser.parse_args()

    bundle = build_demo_bundle(Path(args.output_dir).expanduser(), copy_demo_figures=not args.skip_figures)
    print(f"Demo report written to {bundle['report_path']}")
    print(f"Demo summary written to {bundle['summary_path']}")
    if args.print_report:
        report_text = Path(bundle["report_path"]).read_text(encoding="utf-8")
        print()
        print(report_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
