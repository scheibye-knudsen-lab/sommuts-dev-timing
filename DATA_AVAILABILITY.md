# Data Availability

This is a runnable lightweight reviewer package. It includes derived CSV/TSV inputs sufficient for table regeneration and lightweight notebook execution, but it does not redistribute large raw or intermediate data products.

## Bundled Data

| Path | Contents |
| --- | --- |
| `data/TabMur/aggregates/` | Mouse aggregate mutation, callable-base, burden, and stage-summary tables |
| `data/TabSap/aggregates/` | Human aggregate mutation, callable-base, burden, and stage-summary tables |
| `data/TabMur/expression_coupling/` | Mouse expression-coupling slopes and GSEA report exports |
| `data/TabSap/expression_coupling/` | Human expression-coupling slopes and GSEA report exports |
| `data/TabMur/expression_coupling_by_group/`, `data/TabSap/expression_coupling_by_group/` | Cell-type/tissue grouped expression-coupling slopes and GO enrichment summaries |
| `data/TabMur/single_cell_metadata/`, `data/TabSap/single_cell_metadata/` | Lightweight cell metadata and gene/barcode lookup tables used by bundled notebook panels |
| `data/validation/` | Orthogonal, external, and colon validation inputs, stage counts, and timing-curve tables |
| `data/revision/` | Derived revision summary tables used by the extended notebook |
| `data/ref_genome/*.fai` | Small chromosome-length indexes used for chromosome-position summaries |
| `results/intermediate/`, `results/supplementary_tables/` | Lightweight support-metric tables used by supplementary table regeneration |

The bundled `*.fai` files provide chromosome lengths for position-normalized summaries. They are not FASTA sequence files.

## Omitted Full Assets

The following categories are intentionally omitted because they are raw data, large intermediate products, or external references:

| Asset type | Example expected path | Needed for |
| --- | --- | --- |
| Full single-cell expression matrices | `data/TabSap/single_cell_metadata/expression_matrix_all_genes.npz`, `data/TabMur/single_cell_metadata/expression_matrix_all_genes.npz` | Recomputing expression-matrix dependent panels from raw matrix inputs |
| Full FASTA references | `data/ref_genome/gencode_v41_ercc.fa`, `data/ref_genome/mm10.fa` | Trinucleotide sequence-context panels |
| GTF annotations | `data/ref_genome/gencode_v44.gtf`, `data/TabMur/gencode.vM36.annotation.gtf` | Active-gene span normalization panels |
| Raw sequencing and SComatic outputs | BAM/VCF/intermediate SComatic directories | Full preprocessing and mutation calling reruns |
| EnrichmentMap/Cytoscape project exports | Cytoscape session or cluster export files | Rebuilding graphical pathway cluster exports from scratch |
| Manuscript files | `Files/` | Submitted separately through the journal system |

When these external resources are available locally, the runner can be used in full mode with `--full` and the optional reference-path arguments documented in `README.md`.

## Lightweight Execution Notes

Default notebook execution is designed for the bundled reviewer package. The runner skips only cells whose required full assets are not included and prints the reason for each skipped cell. Table regeneration still writes the current supplementary table CSV set and a source manifest under `results/supplementary_tables/csv/`.
