from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
FALLBACK_DIR = DATA_DIR / "supplementary_tables" / "enrichmentmap_cluster_exports"
OUT_DIR = RESULTS_DIR / "supplementary_tables" / "csv"
MANIFEST_PATH = OUT_DIR / "TABLE_SOURCE_MANIFEST.tsv"

STAGE_OLD_LABELS = {
    "Embryoblast": "Pre-gastrulation",
    "Germ layer-specific": "Post-gastrulation",
    "Tissue-specific": "Tissue-specific",
    "Adult-specific": "Cell-type specific",
}

def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUT_DIR.glob("*.csv"):
        path.unlink()
    if MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()

def write_csv(df: pd.DataFrame, table_num: int) -> Path:
    path = OUT_DIR / f"{table_num:02d}_table_S{table_num}.csv"
    df.to_csv(path, index=False)
    return path

def copy_fallback_csv(table_num: int, reason: str, manifest: list[dict[str, str]]) -> None:
    src = FALLBACK_DIR / f"{table_num:02d}_table_S{table_num}.csv"
    dst = OUT_DIR / src.name
    shutil.copyfile(src, dst)
    manifest.append(
        {
            "table": f"S{table_num}",
            "status": "bundled_cluster_export",
            "source": str(src.relative_to(REPO_ROOT)),
            "notes": reason,
        }
    )

def load_gsea(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")

def build_lookup_table(path: Path, cell_col: str, tissue_col_out: str) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=[cell_col, "tissue", "germ_layer"])
    out = (
        df.rename(columns={cell_col: "cell_type", "tissue": tissue_col_out})
        .dropna()
        .drop_duplicates()
        .sort_values(["cell_type", tissue_col_out, "germ_layer"])
        .reset_index(drop=True)
    )
    return out

def build_concordant_table(mouse: pd.DataFrame, human: pd.DataFrame, direction: str) -> pd.DataFrame:
    merged = mouse.merge(
        human,
        on="NAME",
        how="inner",
        suffixes=("_mouse", "_human"),
    )
    out = pd.DataFrame(
        {
            "NAME": merged["NAME"],
            "NES_avg": (pd.to_numeric(merged["NES_mouse"]) + pd.to_numeric(merged["NES_human"])) / 2.0,
            "pvalue": pd.concat(
                [
                    pd.to_numeric(merged["NOM p-val_mouse"], errors="coerce"),
                    pd.to_numeric(merged["NOM p-val_human"], errors="coerce"),
                ],
                axis=1,
            ).max(axis=1),
            "FDR": pd.concat(
                [
                    pd.to_numeric(merged["FDR q-val_mouse"], errors="coerce"),
                    pd.to_numeric(merged["FDR q-val_human"], errors="coerce"),
                ],
                axis=1,
            ).max(axis=1),
        }
    )
    ascending = direction == "down"
    return out.sort_values("NES_avg", ascending=ascending).reset_index(drop=True)

def build_nonconcordant_table(primary: pd.DataFrame, opposite: pd.DataFrame, ascending: bool) -> pd.DataFrame:
    names = sorted(set(primary["NAME"]) & set(opposite["NAME"]))
    out = (
        primary[primary["NAME"].isin(names)][["NAME", "NES", "NOM p-val", "FDR q-val"]]
        .rename(columns={"NOM p-val": "pvalue", "FDR q-val": "FDR"})
        .sort_values("NES", ascending=ascending)
        .reset_index(drop=True)
    )
    return out

def build_stage_weighting_table() -> pd.DataFrame:
    stage_weight = pd.read_csv(DATA_DIR / "revision" / "stage_weighting_summary.csv")
    stage_weight["stage_old"] = stage_weight["stage"].map(STAGE_OLD_LABELS)
    stage_weight["dataset"] = stage_weight["species"].map({"Mouse": "Tabula Muris", "Human": "Tabula Sapiens"})
    stage_weight["weighting_label"] = stage_weight["weighting"].map(
        {"supporting-read-weighted": "Weighted", "cells-weighted": "Unweighted"}
    )
    sub = stage_weight[stage_weight["weighting_label"].notna()].copy()
    pivot = (
        sub.pivot_table(
            index=["dataset", "weighting_label"],
            columns="stage_old",
            values="mean_percent",
            aggfunc="first",
        )
        .reindex(columns=["Pre-gastrulation", "Post-gastrulation", "Tissue-specific", "Cell-type specific"])
        / 100.0
    )
    pivot = pivot.reset_index().rename(columns={"weighting_label": "weighting"})
    return pivot.sort_values(["dataset", "weighting"]).reset_index(drop=True)

def main() -> int:
    ensure_out_dir()
    manifest: list[dict[str, str]] = []

    # Tables S1-S2: germ-layer lookup tables.
    s1 = build_lookup_table(
        DATA_DIR / "TabMur" / "aggregates" / "cb_table__any_type.csv",
        "Cell_type_observed",
        "tissue_type",
    )
    write_csv(s1, 1)
    manifest.append(
        {
            "table": "S1",
            "status": "current_derived",
            "source": "data/TabMur/aggregates/cb_table__any_type.csv",
            "notes": "Unique mouse cell_type/tissue/germ_layer combinations from current packaged aggregate data.",
        }
    )

    s2 = build_lookup_table(
        DATA_DIR / "TabSap" / "aggregates" / "cb_table__any_type.csv",
        "Cell_type_observed",
        "donor_tissue",
    )
    write_csv(s2, 2)
    manifest.append(
        {
            "table": "S2",
            "status": "current_derived",
            "source": "data/TabSap/aggregates/cb_table__any_type.csv",
            "notes": "Unique human cell_type/tissue/germ_layer combinations from current packaged aggregate data; column name preserved from the source sheet.",
        }
    )

    # Tables S3-S6: direct GSEA exports.
    gsea_sources = {
        3: DATA_DIR / "TabMur" / "expression_coupling" / "gsea_report_for_na_pos_1758807683509.tsv",
        4: DATA_DIR / "TabMur" / "expression_coupling" / "gsea_report_for_na_neg_1758807683509.tsv",
        5: DATA_DIR / "TabSap" / "expression_coupling" / "gsea_report_for_na_pos_1758807192566.tsv",
        6: DATA_DIR / "TabSap" / "expression_coupling" / "gsea_report_for_na_neg_1758807192566.tsv",
    }
    for table_num, src in gsea_sources.items():
        write_csv(load_gsea(src), table_num)
        manifest.append(
            {
                "table": f"S{table_num}",
                "status": "current_direct",
                "source": str(src.relative_to(REPO_ROOT)),
                "notes": "Direct current packaged GSEA report.",
            }
        )

    # Tables S7-S10: bundled EnrichmentMap cluster exports.
    for table_num in [7, 8, 9, 10]:
        copy_fallback_csv(
            table_num,
            "Current EnrichmentMap cluster export is not rebuildable from packaged artifacts; copied from bundled EnrichmentMap cluster export.",
            manifest,
        )

    # Direct current tables that are already materialized by the analysis workflow.
    direct_tsv_tables = {
        11: DATA_DIR / "TabMur" / "expression_coupling_by_group" / "go_enrichment_by_group" / "go_combined.tsv",
        12: DATA_DIR / "TabSap" / "expression_coupling_by_group" / "go_enrichment_by_group" / "go_combined.tsv",
        25: DATA_DIR / "revision" / "developmental_stage_unified_support_threshold_sensitivity.csv",
        26: DATA_DIR / "revision" / "developmental_stage_unified_retained_counts.csv",
        27: DATA_DIR / "revision" / "developmental_stage_unified_substitution_class_sensitivity.csv",
        28: DATA_DIR / "revision" / "developmental_stage_unified_rna_editing_sensitivity.csv",
        30: RESULTS_DIR / "intermediate" / "per_donor_variant_support_metrics.tsv",
        31: RESULTS_DIR / "supplementary_tables" / "Supplementary_Table_stage_specific_SComatic_support_metrics.tsv",
    }
    for table_num, src in direct_tsv_tables.items():
        df = pd.read_csv(src, sep="\t" if src.suffix == ".tsv" else ",")
        write_csv(df, table_num)
        manifest.append(
            {
                "table": f"S{table_num}",
                "status": "current_direct",
                "source": str(src.relative_to(REPO_ROOT)),
                "notes": "Direct current packaged table.",
            }
        )

    # Tables S13-S24: concordant and discordant pathway intersections.
    mur_pos = load_gsea(gsea_sources[3])
    mur_neg = load_gsea(gsea_sources[4])
    sap_pos = load_gsea(gsea_sources[5])
    sap_neg = load_gsea(gsea_sources[6])

    write_csv(build_concordant_table(mur_pos, sap_pos, "up"), 13)
    manifest.append(
        {
            "table": "S13",
            "status": "current_derived",
            "source": "; ".join(
                [
                    str(gsea_sources[3].relative_to(REPO_ROOT)),
                    str(gsea_sources[5].relative_to(REPO_ROOT)),
                ]
            ),
            "notes": "Intersection of current mouse-positive and human-positive GSEA terms with average NES and conservative max(pvalue/FDR).",
        }
    )
    copy_fallback_csv(
        14,
        "Current EnrichmentMap cluster export for concordant up pathways is not rebuildable from packaged artifacts; copied from bundled EnrichmentMap cluster export.",
        manifest,
    )

    write_csv(build_concordant_table(mur_neg, sap_neg, "down"), 15)
    manifest.append(
        {
            "table": "S15",
            "status": "current_derived",
            "source": "; ".join(
                [
                    str(gsea_sources[4].relative_to(REPO_ROOT)),
                    str(gsea_sources[6].relative_to(REPO_ROOT)),
                ]
            ),
            "notes": "Intersection of current mouse-negative and human-negative GSEA terms with average NES and conservative max(pvalue/FDR).",
        }
    )
    copy_fallback_csv(
        16,
        "Current EnrichmentMap cluster export for concordant down pathways is not rebuildable from packaged artifacts; copied from bundled EnrichmentMap cluster export.",
        manifest,
    )

    write_csv(build_nonconcordant_table(mur_pos, sap_neg, ascending=False), 17)
    manifest.append(
        {
            "table": "S17",
            "status": "current_derived",
            "source": "; ".join(
                [
                    str(gsea_sources[3].relative_to(REPO_ROOT)),
                    str(gsea_sources[6].relative_to(REPO_ROOT)),
                ]
            ),
            "notes": "Current mouse-up / human-down pathway intersection using mouse-side NES.",
        }
    )
    copy_fallback_csv(
        18,
        "Current EnrichmentMap cluster export for mouse-up/human-down pathways is not rebuildable from packaged artifacts; copied from bundled EnrichmentMap cluster export.",
        manifest,
    )

    write_csv(build_nonconcordant_table(mur_neg, sap_pos, ascending=True), 19)
    manifest.append(
        {
            "table": "S19",
            "status": "current_derived",
            "source": "; ".join(
                [
                    str(gsea_sources[4].relative_to(REPO_ROOT)),
                    str(gsea_sources[5].relative_to(REPO_ROOT)),
                ]
            ),
            "notes": "Current mouse-down / human-up pathway intersection using mouse-side NES.",
        }
    )
    copy_fallback_csv(
        20,
        "Current EnrichmentMap cluster export for mouse-down/human-up pathways is not rebuildable from packaged artifacts; copied from bundled EnrichmentMap cluster export.",
        manifest,
    )

    write_csv(build_nonconcordant_table(sap_pos, mur_neg, ascending=False), 21)
    manifest.append(
        {
            "table": "S21",
            "status": "current_derived",
            "source": "; ".join(
                [
                    str(gsea_sources[5].relative_to(REPO_ROOT)),
                    str(gsea_sources[4].relative_to(REPO_ROOT)),
                ]
            ),
            "notes": "Current human-up / mouse-down pathway intersection using human-side NES.",
        }
    )
    copy_fallback_csv(
        22,
        "Current EnrichmentMap cluster export for human-up/mouse-down pathways is not rebuildable from packaged artifacts; copied from bundled EnrichmentMap cluster export.",
        manifest,
    )

    write_csv(build_nonconcordant_table(sap_neg, mur_pos, ascending=True), 23)
    manifest.append(
        {
            "table": "S23",
            "status": "current_derived",
            "source": "; ".join(
                [
                    str(gsea_sources[6].relative_to(REPO_ROOT)),
                    str(gsea_sources[3].relative_to(REPO_ROOT)),
                ]
            ),
            "notes": "Current human-down / mouse-up pathway intersection using human-side NES.",
        }
    )
    copy_fallback_csv(
        24,
        "Current EnrichmentMap cluster export for human-down/mouse-up pathways is not rebuildable from packaged artifacts; copied from bundled EnrichmentMap cluster export.",
        manifest,
    )

    # Table S29: stage weighting summary.
    write_csv(build_stage_weighting_table(), 29)
    manifest.append(
        {
            "table": "S29",
            "status": "current_derived",
            "source": "data/revision/stage_weighting_summary.csv",
            "notes": "Derived from current stage weighting summary: weighted=supporting-read-weighted, unweighted=cells-weighted, values stored as fractions.",
        }
    )

    pd.DataFrame(manifest).to_csv(MANIFEST_PATH, sep="\t", index=False)
    print(f"Wrote regenerated supplementary tables to {OUT_DIR}")
    print(f"Wrote source manifest to {MANIFEST_PATH}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
