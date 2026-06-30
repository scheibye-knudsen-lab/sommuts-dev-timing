# Lore 2026 Runnable Analysis Package

This repository contains the runnable analysis package for the revised Lore 2026 manuscript. It includes the analysis notebooks, helper scripts, plotting style, and a lightweight set of derived input tables needed to regenerate the manuscript-facing summary tables and most figure panels.

The package intentionally does not include raw sequencing files, per-cell SComatic output directories, full single-cell expression matrices, full FASTA/GTF references, journal-submitted DOCX/PDF files, or other large intermediate products.

## Quick Start

Run from the package root:

```bash
cd Lore2026_GitHub
python3 -m pip install -r requirements.txt
python3 -m py_compile scripts/*.py scripts/revision/*.py scripts/validation/*.py
```

Regenerate the supplementary table CSVs:

```bash
python3 scripts/run_reproduction.py --only tables
```

Run the main and extended notebooks in lightweight mode:

```bash
python3 scripts/run_reproduction.py --only main
python3 scripts/run_reproduction.py --only extended
```

Lightweight mode is the default. It runs cells that can be reproduced from bundled derived data and prints a skip message for cells that require omitted full assets: full expression matrices, external GTF annotations, FASTA sequence, or precomputed heavy sensitivity tables.

For a full local rerun with external resources available, use:

```bash
python3 scripts/run_reproduction.py --only extended --full   --human-gtf /path/to/gencode_v44.gtf   --mouse-gtf /path/to/gencode.vM36.annotation.gtf   --mouse-fasta /path/to/mm10.fa
```

The same paths can be supplied through `LORE2026_HUMAN_GTF`, `LORE2026_MOUSE_GTF`, and `LORE2026_MM10_FASTA`.

## SComatic Variant Calling

Raw BAMs and intermediate SComatic outputs are not bundled, but the package includes shell entry points for rerunning SComatic when those external files are available:

```bash
bash run_scomatic_human.sh \
  --manifest /path/to/human_manifest.tsv \
  --scomatic /path/to/SComatic \
  --ref /path/to/gencode_v41_ercc.fa \
  --outdir /path/to/scomatic_human_outputs

bash run_scomatic_mouse.sh \
  --manifest /path/to/mouse_manifest.tsv \
  --scomatic /path/to/SComatic \
  --ref /path/to/mm10.fa \
  --outdir /path/to/scomatic_mouse_outputs
```

The manifest is tab-delimited with columns `sample`, `library_id`, `bam`, and `metadata`. These commands require SComatic, `samtools`, `bedtools`, indexed BAM files, per-cell metadata files, matching reference FASTA files, species-specific SComatic RNA-editing and panel-of-normal resources, and the SComatic mappability BED.

## Contents

- `Figures.ipynb`: main manuscript figure-generation notebook.
- `Supp.ipynb`: Extended Data and supplemental robustness figure-generation notebook.
- `run_scomatic_human.sh`, `run_scomatic_mouse.sh`: manifest-driven SComatic variant-calling entry points for external BAM/metadata resources.
- `scripts/run_reproduction.py`: ordered runner for supplementary tables and notebooks.
- `scripts/regenerate_current_supplementary_tables.py`: rebuilds the current supplementary table CSV exports from bundled derived tables where possible and records source status in a manifest.
- `scripts/variant_calling/`: shared SComatic runner used by the human and mouse shell entry points.
- `scripts/revision/`: scripts for developmental-stage, breadth, and genome-position revision analyses.
- `scripts/validation/`: validation timing summary builder.
- `data/`: lightweight derived inputs, validation summaries, small reference indexes, and revision summary tables.
- `results/`: lightweight support-metric tables used by supplementary table regeneration.
- `styles/`: plotting style used by the notebooks.
- `DATA_AVAILABILITY.md`: data manifest and notes on omitted full assets.

## Figure Order

`Figures.ipynb` is ordered to match the main manuscript:

1. Fig. 1a-c
2. Fig. 2a-d
3. Fig. 3a-e

`Supp.ipynb` is ordered to match the Extended Data document:

1. Extended Data Fig. 1, schematic/artwork note
2. Extended Data Fig. 2
3. Extended Data Fig. 3, orthogonal and external validation
4. Extended Data Fig. 4
5. Extended Data Fig. 5
6. Extended Data Fig. 6
7. Extended Data Fig. 7
8. Extended Data Fig. 8
9. Extended Data Fig. 9
10. Extended Data Fig. 10
11. Extended Data Fig. 11
12. Extended Data Fig. 12, external EnrichmentMap/Cytoscape artwork note
13. Extended Data Fig. 13, external EnrichmentMap/Cytoscape artwork note
14. Extended Data Fig. 14
15. Extended Data Fig. 15

The notebook runner executes cells in document order and closes figures after each cell. It does not write a full figure export directory.

## Regenerating Bundled Derived Tables

The validation timing tables can be regenerated with:

```bash
python3 scripts/validation/build_validation_timing.py
```

Revision-stage summaries that use bundled aggregate mutation tables can be regenerated with:

```bash
python3 scripts/revision/analyze_genome_stage_distribution.py
python3 scripts/revision/developmental_stage_breadth_distributions.py
```

Supplementary table outputs are written to `results/supplementary_tables/csv/`; the accompanying `TABLE_SOURCE_MANIFEST.tsv` records whether each table was rebuilt from current bundled inputs or copied from a bundled EnrichmentMap export because the corresponding EnrichmentMap/Cytoscape export is not rebuildable from the lightweight files.
