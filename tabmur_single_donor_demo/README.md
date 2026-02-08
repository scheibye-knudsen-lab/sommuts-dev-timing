# TabMur single-donor demo (1-M-62)

This package provides source code and a small dataset to compute cohort-standardized developmental stage mutation burden for TabMur donor 1-M-62 and save a cumulative plot. The demo runs using only files inside the `Lore2026_Nature/` download.

## Package contents

- Source code: `run_stage_burden_demo_single_donor.py`, `build_merged_outputs_single_donor.py`, `build_cell_metadata_table_single_donor.py`, `build_reference_cohort_files.py`
- Demo dataset: `demo_data/merged_outputs_demo/1-M-62/`, `demo_data/cell_metadata_table_1-M-62.csv`
- Cohort reference data (precomputed): `demo_data/reference_weights_cells/*.csv`, `demo_data/reference_stage_labels_1-M-62.csv`

## System requirements

- OS tested: Ubuntu 24.04.1 LTS
- Python: 3.12.2
- Python packages: pandas 2.2.3, numpy 2.1.3, matplotlib 3.10.1
- Non-standard hardware: none

## Installation guide

1) Download and unzip `Lore2026_Nature/`, then change into this directory.
2) Create and activate a virtual environment.
3) Install dependencies.

```bash
cd Lore2026_Nature/tabmur_single_donor_demo
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install pandas==2.2.3 numpy==2.1.3 matplotlib==3.10.1
```

Typical install time on a normal desktop: ~5 minutes.

## Demo

Run the demo (no external data needed):

```bash
python3 run_stage_burden_demo_single_donor.py
```

Expected outputs:

- `demo_results/stage_burden_per_donor_perkb_STANDARDIZED_3LEVEL_cells.csv`
- `demo_results/stage_burden_per_donor_perkb_cumsum_STANDARDIZED_3LEVEL_cells.csv`
- `demo_results/plots/cumulative_mutations_per_kb_STANDARDIZED_3LEVEL_cells_blues_by_age_months.png`

Expected runtime on a normal desktop: ~10 seconds.

## Instructions for use (demo only)

This zip is a standalone demo for TabMur donor 1-M-62. No additional data or configuration is required beyond the demo steps above.
