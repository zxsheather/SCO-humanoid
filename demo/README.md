# SCO-humanoid Offline Demo

This package provides a compact offline presentation of the current
`SCO-humanoid` mechanism-comparison result for smooth humanoid locomotion.

It generates a short report from bundled tables, statistics, and figures. The
package does not retrain policies and does not launch Isaac Gym or MuJoCo.

The generated report includes:

- the research question and current thesis
- the five-seed Isaac selected-checkpoint comparison
- the matched five-seed MuJoCo replay comparison
- a checkpoint-reliability summary
- key figures for mechanism interpretation

## Requirements

- Python 3.9 or newer

## Quick Start

Run from the package directory:

```bash
python run_demo.py --output-dir demo_output --print-report
```

This writes:

```text
demo_output/
  DEMO_REPORT.md
  demo_summary.json
  manifest.json
  figures/
    figure_mechanism_chain.png
    figure_sensitivity_vs_degradation.png
    figure_task_vs_smoothness.png
```

## Included Data

The package reads only the files shipped inside this directory:

- `data/table_full_paper_isaac_mechanism_comparison.csv`
- `data/table_matched_mujoco_mechanism_comparison.csv`
- `data/full_paper_statistics_summary.json`
- `data/paper_figures_manifest.json`
- `data/upstream_sources.json`
- `assets/figures/figure_mechanism_chain.png`
- `assets/figures/figure_sensitivity_vs_degradation.png`
- `assets/figures/figure_task_vs_smoothness.png`

## Recommended Reading Order

1. Open `demo_output/DEMO_REPORT.md`.
2. Read the `Current Thesis` section.
3. Inspect the Isaac and MuJoCo comparison tables.
4. Use `Reliability Snapshot` to interpret checkpoint dependence.
5. Finish with `figure_mechanism_chain.png` and `figure_sensitivity_vs_degradation.png`.

## Scope

This is an offline presentation package:

- it summarizes precomputed results
- it does not run training
- it does not run Isaac Gym evaluation
- it does not run MuJoCo replay
- it does not provide hardware validation

## Provenance

The bundled data and figures are traced in `data/upstream_sources.json`.
