#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROOT = REPO_ROOT / "data" / "validation"

STAGE_ORDER = [
    "Embryoblast",
    "Germ layer-specific",
    "Tissue-specific",
    "Adult-specific",
]
PLOT_ORDER = ["Zygote", *STAGE_ORDER]

ORTHO_TISSUE_TOKENS = {
    "Ctx": "cortex",
    "Cbl": "cerebellum",
    "Heart": "heart",
    "Liver": "liver",
    "Kidney": "kidney",
}
ORTHO_TISSUE_TO_LAYER = {
    "cortex": "ectoderm",
    "cerebellum": "ectoderm",
    "heart": "mesoderm",
    "kidney": "mesoderm",
    "liver": "endoderm",
}
ORTHO_REGION = {"F", "O", "P", "T", "PF"}
ORTHO_LATERAL = {"L", "R"}
ORTHO_IGNORE = set(ORTHO_TISSUE_TOKENS) | ORTHO_REGION | ORTHO_LATERAL | {"Tissue", "DAPI"}
ORTHO_TOKEN_SPLIT = re.compile(r"[_-]")
ORTHO_WELL_RE = re.compile(r"^[A-H]\d{1,2}$")
ORTHO_SIDE_NUM_RE = re.compile(r"^[LR]\d+$")

DATASET_CONFIG = {
    "orth_val_1": {
        "title": "Orthogonal Validation 1",
        "legend_max": 12,
    },
    "orth_val_2": {
        "title": "Orthogonal Validation 2",
        "legend_max": 12,
    },
    "moore_val": {
        "title": "Moore Validation",
        "legend_max": 12,
    },
    "kim_val": {
        "title": "Kim Validation",
        "legend_max": 12,
    },
}


def classify_stage(n_layers: int, n_tissues: int, n_cells: int) -> str:
    if n_layers > 1:
        return "Embryoblast"
    if n_tissues > 1:
        return "Germ layer-specific"
    if n_cells > 1:
        return "Tissue-specific"
    return "Adult-specific"


def bool_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).gt(0)
    values = series.astype(str).str.strip().str.lower()
    return values.isin({"1", "true", "t", "yes", "y"})


def normalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.div(frame.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def add_zygote_zero(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "Zygote", 0.0)
    return out[PLOT_ORDER]


def timing_curves_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    frac = normalize_rows(counts.reindex(columns=STAGE_ORDER, fill_value=0.0))
    return add_zygote_zero(frac).cumsum(axis=1)


def tissue_from_ortho_id(value: str) -> str | None:
    for token in ORTHO_TOKEN_SPLIT.split(str(value)):
        if token in ORTHO_TISSUE_TOKENS:
            return ORTHO_TISSUE_TOKENS[token]
    return None


def cell_type_from_ortho_id(value: str) -> str | None:
    tokens = [token for token in ORTHO_TOKEN_SPLIT.split(str(value)) if token]
    if not tokens:
        return None
    out = []
    for token in tokens[1:]:
        if token in ORTHO_IGNORE or ORTHO_WELL_RE.match(token) or ORTHO_SIDE_NUM_RE.match(token) or token.isdigit():
            continue
        out.append(token)
    return "-".join(out) if out else None


def donor_from_ortho_id(value: str) -> str:
    tokens = [token for token in ORTHO_TOKEN_SPLIT.split(str(value)) if token]
    return tokens[0] if tokens else "unknown"


def load_orth_val_1_counts(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "orthogonal_val.csv", low_memory=False)
    presence_col = "IN_SAMPLE" if "IN_SAMPLE" in df.columns else "IN_TISSUE"
    donor = donor_from_ortho_id(df["ID"].iloc[0]) if len(df) else "unknown"
    df["tissue"] = df["ID"].map(tissue_from_ortho_id)
    df["cell_type"] = df["ID"].map(cell_type_from_ortho_id)
    df["germ_layer"] = df["tissue"].map(ORTHO_TISSUE_TO_LAYER)
    present = df[bool_mask(df[presence_col])].copy()
    present["cell_type_key"] = present["tissue"].fillna("unknown") + ":" + present["cell_type"].fillna("bulk")

    grouped = present.groupby("CHR_POS_REF_ALT", observed=True)
    breadth = pd.DataFrame(
        {
            "n_tissues": grouped["tissue"].nunique(dropna=True),
            "n_layers": grouped["germ_layer"].nunique(dropna=True),
            "n_cells": grouped["cell_type_key"].nunique(dropna=True),
        }
    )
    breadth["stage_label"] = breadth.apply(
        lambda row: classify_stage(row["n_layers"], row["n_tissues"], row["n_cells"]),
        axis=1,
    )
    counts = pd.DataFrame(
        [breadth.groupby("stage_label").size().reindex(STAGE_ORDER, fill_value=0)],
        index=[donor],
    ).rename_axis("donor")
    return counts


def load_orth_val_2_counts(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "orth_val_2_wgs_stage_metrics_by_donor_strict_3level.csv", low_memory=False)
    df = df[(df["filter_name"] == "all_variants") & (df["metric_type"] == "site_count")].copy()
    counts = (
        df.pivot_table(index="donor", columns="stage", values="value", aggfunc="first", fill_value=0.0)
        .reindex(columns=STAGE_ORDER, fill_value=0.0)
        .sort_index()
    )
    return counts


def load_moore_counts(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "moore_3donor_stage_summary.csv", low_memory=False)
    counts = (
        df.set_index("donor")[STAGE_ORDER]
        .fillna(0)
        .sort_index()
    )
    return counts


def load_kim_counts(dataset_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_dir / "kim_val_donor_stage_summary_control_donors.csv", low_memory=False)
    df = df[df["total_variants"] > 0].copy()
    counts = (
        df.set_index("donor")[STAGE_ORDER]
        .fillna(0)
        .sort_index()
    )
    return counts


def load_dataset_counts(dataset: str) -> pd.DataFrame:
    dataset_dir = VALIDATION_ROOT / dataset
    loaders = {
        "orth_val_1": load_orth_val_1_counts,
        "orth_val_2": load_orth_val_2_counts,
        "moore_val": load_moore_counts,
        "kim_val": load_kim_counts,
    }
    if dataset not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return loaders[dataset](dataset_dir)


def write_dataset_tables(dataset: str, counts: pd.DataFrame) -> tuple[Path, Path]:
    dataset_dir = VALIDATION_ROOT / dataset
    counts_path = dataset_dir / "stage_counts_by_donor.csv"
    curves_path = dataset_dir / "developmental_timing_curves_by_donor.csv"

    counts_out = counts.reset_index().rename(columns={"index": "donor"})
    counts_out.to_csv(counts_path, index=False)

    curves = timing_curves_from_counts(counts)
    curves.reset_index().rename(columns={"index": "donor"}).to_csv(curves_path, index=False)
    return counts_path, curves_path


def write_combined_tables(dataset_counts: dict[str, pd.DataFrame]) -> None:
    rows = []
    curve_rows = []
    for dataset, counts in dataset_counts.items():
        counts = counts.copy()
        counts.index.name = "donor"
        counts_reset = counts.reset_index()
        counts_reset["dataset"] = dataset
        rows.append(
            counts_reset.melt(
                id_vars=["dataset", "donor"],
                value_vars=STAGE_ORDER,
                var_name="stage_label",
                value_name="variant_count",
            )
        )

        curves = timing_curves_from_counts(counts)
        curves.index.name = "donor"
        curves_reset = curves.reset_index()
        curves_reset["dataset"] = dataset
        curve_rows.append(
            curves_reset.melt(
                id_vars=["dataset", "donor"],
                value_vars=PLOT_ORDER,
                var_name="stage_label",
                value_name="cumulative_fraction",
            )
        )

    pd.concat(rows, ignore_index=True).to_csv(
        VALIDATION_ROOT / "validation_stage_counts_all_datasets.csv",
        index=False,
    )
    pd.concat(curve_rows, ignore_index=True).to_csv(
        VALIDATION_ROOT / "validation_timing_curves_all_datasets.csv",
        index=False,
    )


def plot_dataset(dataset: str, curves: pd.DataFrame) -> None:
    config = DATASET_CONFIG[dataset]

    n_donors = len(curves)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    if n_donors <= config["legend_max"]:
        cmap = plt.get_cmap("tab20", max(n_donors, 1))
        handles = []
        for idx, (donor, row) in enumerate(curves.iterrows()):
            color = cmap(idx % max(n_donors, 1))
            ax.plot(
                PLOT_ORDER,
                row[PLOT_ORDER].to_numpy(),
                marker="o",
                linestyle="-",
                color=color,
                alpha=0.9,
                markeredgecolor="black",
                markeredgewidth=0.3,
                linewidth=2,
            )
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    markeredgecolor="black",
                    markeredgewidth=0.3,
                    linewidth=2,
                    label=donor,
                )
            )
        ax.legend(handles=handles, title="Donor", bbox_to_anchor=(1.02, 0.5), loc="center left", frameon=False)
        plt.subplots_adjust(right=0.8)
    else:
        for _, row in curves.iterrows():
            ax.plot(
                PLOT_ORDER,
                row[PLOT_ORDER].to_numpy(),
                marker="o",
                linestyle="-",
                color="#9AA3B2",
                alpha=0.22,
                markeredgecolor="none",
                linewidth=1.1,
            )
        mean_curve = curves[PLOT_ORDER].mean(axis=0)
        ax.plot(
            PLOT_ORDER,
            mean_curve.to_numpy(),
            marker="o",
            linestyle="-",
            color="#123A73",
            markeredgecolor="black",
            markeredgewidth=0.3,
            linewidth=3,
            label="Mean",
        )
        ax.legend(frameon=False, loc="lower right")

    ax.set_title(f"{config['title']} Developmental Timing")
    ax.set_xlabel("Developmental stage")
    ax.set_ylabel("Cumulative fraction of staged variants")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.tight_layout()
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def main() -> None:
    dataset_counts: dict[str, pd.DataFrame] = {}
    for dataset in DATASET_CONFIG:
        counts = load_dataset_counts(dataset)
        dataset_counts[dataset] = counts
        counts_path, curves_path = write_dataset_tables(dataset, counts)
        plot_dataset(dataset, timing_curves_from_counts(counts))
        print(f"[OK] {dataset}: wrote {counts_path}")
        print(f"[OK] {dataset}: wrote {curves_path}")

    write_combined_tables(dataset_counts)
    print(f"[OK] wrote {VALIDATION_ROOT / 'validation_stage_counts_all_datasets.csv'}")
    print(f"[OK] wrote {VALIDATION_ROOT / 'validation_timing_curves_all_datasets.csv'}")


if __name__ == "__main__":
    main()
