# TabMur single-donor demo (1-M-62)

This package provides source code and a small dataset to compute cohort-standardized developmental stage mutation burden for TabMur donor 1-M-62 and save a cumulative plot. The demo runs using only files inside the `Lore2026_Nature/` download.

## Package contents

- Source code only (no compiled binaries): `run_stage_burden_demo_single_donor.py`, `build_merged_outputs_single_donor.py`, `build_cell_metadata_table_single_donor.py`, `build_reference_cohort_files.py`
- Demo dataset (small, real): `demo_data/merged_outputs_demo/1-M-62/`, `demo_data/cell_metadata_table_1-M-62.csv`
- Cohort reference data (precomputed for 1-M-62): `demo_data/reference_weights_cells/*.csv`, `demo_data/reference_stage_labels_1-M-62.csv`

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

## Instructions for use (your data; external inputs required)

To run on your own data, you need external inputs that are not included in this zip:
- SingleCellAlleles outputs for your donor.
- UniqueCellCallableSites outputs for your donor.
- Cell barcode metadata and germ-layer mapping tables.
- Optional (for cohort-standardized results): full cohort outputs and cohort CB table.

1) Build merged outputs for your donor:

```bash
python3 build_merged_outputs_single_donor.py \
  --donor <DONOR_ID> \
  --geno-root <SingleCellAlleles_root> \
  --sites-root <UniqueCellCallableSites_root> \
  --dest-root demo_data/merged_outputs_demo
```

2) Build a cell metadata table (donor, CB, tissue, germ_layer):

```bash
python3 build_cell_metadata_table_single_donor.py \
  --outputs-root demo_data/merged_outputs_demo \
  --donor <DONOR_ID> \
  --cellbarcodes <cellbarcodes_with_tissue.csv> \
  --germ-map <cell_type_to_germ_layer_map.csv> \
  --out demo_data/cell_metadata_table_<DONOR_ID>.csv
```

3) Update paths and donor ID in `run_stage_burden_demo_single_donor.py`:
- Set `ROOT` to your merged outputs directory.
- Set `CB_TABLE` to your donor’s cell metadata table.
- Set `DONOR_FILTER` to your donor ID.
- If you have cohort reference files, set `REF_WEIGHTS_DIR` and `REF_STAGE_LABELS` to those paths. If not, remove/rename the bundled reference files so the script computes donor-only weights/labels.

4) Run:

```bash
python3 run_stage_burden_demo_single_donor.py
```

Expected outputs are the same as in the demo section (written to `demo_results/`).

## Reproduction instructions (optional)

To reproduce manuscript-scale results, run the full cohort pipeline on all donors to generate the full `outputs` tree and `cb_table__any_type.csv`, then generate cohort reference weights/stage labels with `build_reference_cohort_files.py` and run the stage burden script per donor.

## License

MIT License (see `LICENSE`).

## Code repository

[https://github.com/sierra-lore/Lore2026_Nature](https://github.com/scheibye-knudsen-lab/sommuts-dev-timing)

## Additional pseudocode location in manuscript

Methods section.
