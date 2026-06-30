from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "revision"

STAGE_ORDER = [
    "Embryoblast",
    "Germ layer-specific",
    "Tissue-specific",
    "Adult-specific",
]
STAGE_LABEL_MAP = {
    "Embryoblast": "Embryoblast",
    "Germ layer-specific": "Germ layer-specific",
    "Tissue-specific": "Tissue-specific",
    "Adult-specific": "Adult/cell type-specific",
}
DISPLAY_STAGE_ORDER = [STAGE_LABEL_MAP[stage] for stage in STAGE_ORDER]
STAGE_COLORS = {
    "Embryoblast": "#FFE5CC",
    "Germ layer-specific": "#FFB366",
    "Tissue-specific": "#FF7F00",
    "Adult/cell type-specific": "#CC5500",
}
THRESHOLDS = [1, 2, 3, 5, 10]
SUBSTITUTION_FIGURE_THRESHOLDS = [1, 2]
MUT_CLASS_ORDER = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
DATASET_SPECS = [
    ("TabMur", "Tabula Muris"),
    ("TabSap", "Tabula Sapiens"),
]


def classify_stage(n_layers: int, n_tissues: int, n_cells: int) -> str:
    if n_layers > 1:
        return "Embryoblast"
    if n_tissues > 1:
        return "Germ layer-specific"
    if n_cells > 1:
        return "Tissue-specific"
    return "Adult-specific"


def complement(base: str) -> str:
    return {"A": "T", "T": "A", "C": "G", "G": "C"}.get(str(base).upper(), "N")


def to_pyrimidine_class(ref: str, alt: str) -> str | None:
    ref = str(ref).upper()
    alt = str(alt).upper()
    if ref in {"A", "G"}:
        ref = complement(ref)
        alt = complement(alt)
    if ref in {"C", "T"} and alt in {"A", "C", "G", "T"} and ref != alt:
        return f"{ref}>{alt}"
    return None


def _stage_count_columns() -> list[str]:
    return [
        "Embryoblast_count",
        "Embryoblast_percent",
        "Germ_layer_specific_count",
        "Germ_layer_specific_percent",
        "Tissue_specific_count",
        "Tissue_specific_percent",
        "Adult_cell_type_specific_count",
        "Adult_cell_type_specific_percent",
    ]


def _empty_stage_summary() -> dict[str, float]:
    return {column: 0.0 for column in _stage_count_columns()}


def _stage_summary_from_calls(call_stage: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    if call_stage.empty:
        long = pd.DataFrame(
            {
                "stage_label": DISPLAY_STAGE_ORDER,
                "count": [0, 0, 0, 0],
                "percent": [0.0, 0.0, 0.0, 0.0],
            }
        )
        return _empty_stage_summary(), long

    counts = (
        call_stage["stage_label"]
        .value_counts()
        .reindex(STAGE_ORDER, fill_value=0)
        .astype(int)
    )
    percents = (counts / counts.sum() * 100.0).reindex(STAGE_ORDER, fill_value=0.0)
    summary = {
        "Embryoblast_count": int(counts["Embryoblast"]),
        "Embryoblast_percent": round(float(percents["Embryoblast"]), 2),
        "Germ_layer_specific_count": int(counts["Germ layer-specific"]),
        "Germ_layer_specific_percent": round(float(percents["Germ layer-specific"]), 2),
        "Tissue_specific_count": int(counts["Tissue-specific"]),
        "Tissue_specific_percent": round(float(percents["Tissue-specific"]), 2),
        "Adult_cell_type_specific_count": int(counts["Adult-specific"]),
        "Adult_cell_type_specific_percent": round(float(percents["Adult-specific"]), 2),
    }
    long = pd.DataFrame(
        {
            "stage_internal": STAGE_ORDER,
            "stage_label": [STAGE_LABEL_MAP[stage] for stage in STAGE_ORDER],
            "count": counts.reindex(STAGE_ORDER).tolist(),
            "percent": [round(float(percents[stage]), 2) for stage in STAGE_ORDER],
        }
    )
    return summary, long


def _load_stage_timing_donors(dataset_name: str) -> set[str]:
    path = (
        DATA_DIR
        / dataset_name
        / "aggregates"
        / "stage_burden_per_donor_perkb_STANDARDIZED_3LEVEL_cells.csv"
    )
    return {str(donor) for donor in pd.read_csv(path, index_col=0).index}


def _load_baseline_candidate_set(dataset_name: str) -> pd.DataFrame:
    path = DATA_DIR / dataset_name / "aggregates" / "mut_table_with_stage.csv"
    donors = _load_stage_timing_donors(dataset_name)
    usecols = [
        "donor",
        "CB",
        "var_id",
        "germ_layer",
        "tissue",
        "cell_type_stage",
        "REF",
        "ALT_expected",
    ]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    df["donor"] = df["donor"].astype(str)
    df = df[df["donor"].isin(donors)].copy()
    df = df.dropna(
        subset=[
            "donor",
            "CB",
            "var_id",
            "germ_layer",
            "tissue",
            "cell_type_stage",
            "REF",
            "ALT_expected",
        ]
    ).copy()
    df["CB"] = df["CB"].astype(str).str.strip()
    df["var_id"] = df["var_id"].astype(str)
    df["germ_layer"] = df["germ_layer"].astype(str).str.strip()
    df["tissue"] = df["tissue"].astype(str).str.strip()
    df["cell_type_stage"] = df["cell_type_stage"].astype(str).str.strip()
    df["REF"] = df["REF"].astype(str).str.upper().str.strip()
    df["ALT_expected"] = df["ALT_expected"].astype(str).str.upper().str.strip()
    df = df.drop_duplicates(subset=["donor", "CB", "var_id"]).copy()
    return df


def _collapse_to_presence(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["donor", "var_id", "germ_layer", "tissue", "cell_type_stage"],
            observed=True,
        )
        .agg(
            supporting_cells=("CB", "nunique"),
            REF=("REF", "first"),
            ALT_expected=("ALT_expected", "first"),
        )
        .reset_index()
    )


def _recompute_stage_assignments(presence: pd.DataFrame) -> pd.DataFrame:
    if presence.empty:
        return pd.DataFrame(
            columns=[
                "donor",
                "var_id",
                "n_layers",
                "n_tissues",
                "n_cell_types",
                "REF",
                "ALT_expected",
                "stage_label",
                "stage_label_display",
                "mut_class",
            ]
        )

    call_stage = (
        presence.groupby(["donor", "var_id"], observed=True)
        .agg(
            n_layers=("germ_layer", "nunique"),
            n_tissues=("tissue", "nunique"),
            n_cell_types=("cell_type_stage", "nunique"),
            REF=("REF", "first"),
            ALT_expected=("ALT_expected", "first"),
        )
        .reset_index()
    )
    call_stage["stage_label"] = call_stage.apply(
        lambda row: classify_stage(
            int(row["n_layers"]),
            int(row["n_tissues"]),
            int(row["n_cell_types"]),
        ),
        axis=1,
    )
    call_stage["stage_label_display"] = call_stage["stage_label"].map(STAGE_LABEL_MAP)
    call_stage["mut_class"] = call_stage.apply(
        lambda row: to_pyrimidine_class(row["REF"], row["ALT_expected"]),
        axis=1,
    )
    return call_stage


def _threshold_tables_for_dataset(dataset_name: str, species_label: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = _load_baseline_candidate_set(dataset_name)
    presence = _collapse_to_presence(baseline)

    stage_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []

    for threshold in THRESHOLDS:
        retained_presence = presence[presence["supporting_cells"] >= threshold].copy()
        call_stage = _recompute_stage_assignments(retained_presence)
        summary, long_summary = _stage_summary_from_calls(call_stage)

        stage_rows.append(
            {
                "species": species_label,
                "threshold": threshold,
                "retained_donor_variant_calls": int(len(call_stage)),
                **summary,
            }
        )
        count_rows.append(
            {
                "species": species_label,
                "threshold": threshold,
                "retained_donor_variant_calls": int(len(call_stage)),
                "retained_donor_variant_compartments": int(len(retained_presence)),
                "retained_unique_loci": int(call_stage["var_id"].nunique()) if not call_stage.empty else 0,
                "retained_donors": int(call_stage["donor"].nunique()) if not call_stage.empty else 0,
            }
        )
        long_summary["species"] = species_label
        long_summary["threshold"] = threshold
        long_summary["retained_donor_variant_calls"] = int(len(call_stage))
        plot_rows.append(long_summary)

    return (
        pd.DataFrame(stage_rows),
        pd.DataFrame(count_rows),
        pd.concat(plot_rows, ignore_index=True),
    )


def _substitution_table_for_dataset(dataset_name: str, species_label: str) -> pd.DataFrame:
    baseline = _load_baseline_candidate_set(dataset_name)
    presence = _collapse_to_presence(baseline)

    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        retained_presence = presence[presence["supporting_cells"] >= threshold].copy()
        call_stage = _recompute_stage_assignments(retained_presence)
        call_stage = call_stage[call_stage["mut_class"].isin(MUT_CLASS_ORDER)].copy()

        for mut_class in MUT_CLASS_ORDER:
            sub = call_stage[call_stage["mut_class"] == mut_class].copy()
            summary, _ = _stage_summary_from_calls(sub)
            rows.append(
                {
                    "species": species_label,
                    "threshold": threshold,
                    "substitution_class": mut_class,
                    "retained_donor_variant_calls": int(len(sub)),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _editing_sensitivity_for_dataset(dataset_name: str, species_label: str) -> pd.DataFrame:
    baseline = _load_baseline_candidate_set(dataset_name)
    presence = _collapse_to_presence(baseline)

    rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        retained_presence = presence[presence["supporting_cells"] >= threshold].copy()
        call_stage = _recompute_stage_assignments(retained_presence)
        call_stage = call_stage[call_stage["mut_class"].isin(MUT_CLASS_ORDER)].copy()

        subset_specs = [
            (
                "all_substitutions",
                "All pyrimidine-normalized substitution classes",
                call_stage,
            ),
            (
                "exclude_T_to_C",
                "Exclude T>C, the primary A-to-I editing-sensitive class after pyrimidine normalization",
                call_stage[call_stage["mut_class"] != "T>C"].copy(),
            ),
            (
                "exclude_C_to_T_and_T_to_C",
                "Exclude both transition classes C>T and T>C as a broader conservative sensitivity analysis",
                call_stage[~call_stage["mut_class"].isin({"C>T", "T>C"})].copy(),
            ),
        ]

        for subset_name, subset_description, sub in subset_specs:
            summary, _ = _stage_summary_from_calls(sub)
            rows.append(
                {
                    "species": species_label,
                    "threshold": threshold,
                    "subset": subset_name,
                    "subset_description": subset_description,
                    "retained_donor_variant_calls": int(len(sub)),
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def _build_unified_figure(
    threshold_long: pd.DataFrame,
    retained_counts: pd.DataFrame,
    substitution_table: pd.DataFrame,
):
    n_substitution_rows = len(SUBSTITUTION_FIGURE_THRESHOLDS)
    nrows = 2 + n_substitution_rows
    fig, axes = plt.subplots(nrows, 2, figsize=(15, 5 * nrows), dpi=600)

    for col_idx, species_label in enumerate(["Tabula Muris", "Tabula Sapiens"]):
        ax = axes[0, col_idx]
        sub = threshold_long[threshold_long["species"] == species_label].copy()
        pivot = (
            sub.pivot(index="threshold", columns="stage_label", values="percent")
            .reindex(index=THRESHOLDS, columns=DISPLAY_STAGE_ORDER, fill_value=0.0)
        )
        bottom = np.zeros(len(pivot), dtype=float)
        x = np.arange(len(pivot.index))
        for stage_label in DISPLAY_STAGE_ORDER:
            vals = pivot[stage_label].to_numpy(float)
            ax.bar(
                x,
                vals,
                bottom=bottom,
                color=STAGE_COLORS[stage_label],
                edgecolor="black",
                linewidth=0.5,
                label=stage_label,
            )
            bottom += vals
        ax.set_xticks(x)
        ax.set_xticklabels([f"\u2265{threshold}" for threshold in pivot.index])
        ax.set_ylim(0, 100)
        ax.set_ylabel("Stage composition (%)")
        ax.set_xlabel("Minimum supporting cells per donor-variant-compartment")
        ax.set_title(f"Panel A: {species_label}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col_idx == 1:
            ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    axes[0, 0].text(-0.22, 1.08, "A", transform=axes[0, 0].transAxes, fontsize=18, fontweight="bold")

    for col_idx, species_label in enumerate(["Tabula Muris", "Tabula Sapiens"]):
        ax = axes[1, col_idx]
        sub = retained_counts[retained_counts["species"] == species_label].sort_values("threshold")
        x = np.arange(len(sub))
        bars = ax.bar(
            x,
            sub["retained_donor_variant_calls"],
            color="#BDBDBD",
            edgecolor="black",
            linewidth=0.5,
        )
        for bar, count in zip(bars, sub["retained_donor_variant_calls"].tolist()):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{int(count):,}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"\u2265{threshold}" for threshold in sub["threshold"]])
        ax.set_ylabel("Retained donor-variant calls")
        ax.set_xlabel("Minimum supporting cells per donor-variant-compartment")
        ax.set_title(f"Panel B: {species_label}")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[1, 0].text(-0.22, 1.08, "B", transform=axes[1, 0].transAxes, fontsize=18, fontweight="bold")

    stage_percent_cols = {
        "Embryoblast": "Embryoblast_percent",
        "Germ layer-specific": "Germ_layer_specific_percent",
        "Tissue-specific": "Tissue_specific_percent",
        "Adult/cell type-specific": "Adult_cell_type_specific_percent",
    }
    panel_letters = ["C", "D", "E", "F", "G", "H"]
    for row_offset, threshold in enumerate(SUBSTITUTION_FIGURE_THRESHOLDS, start=2):
        for col_idx, species_label in enumerate(["Tabula Muris", "Tabula Sapiens"]):
            ax = axes[row_offset, col_idx]
            sub = substitution_table[
                (substitution_table["species"] == species_label)
                & (substitution_table["threshold"] == threshold)
            ].copy()
            pivot = pd.DataFrame(index=MUT_CLASS_ORDER)
            for stage_label, pct_col in stage_percent_cols.items():
                pivot[stage_label] = (
                    sub.set_index("substitution_class")
                    .reindex(MUT_CLASS_ORDER)[pct_col]
                    .fillna(0.0)
                )
            bottom = np.zeros(len(pivot), dtype=float)
            x = np.arange(len(pivot.index))
            for stage_label in DISPLAY_STAGE_ORDER:
                vals = pivot[stage_label].to_numpy(float)
                ax.bar(
                    x,
                    vals,
                    bottom=bottom,
                    color=STAGE_COLORS[stage_label],
                    edgecolor="black",
                    linewidth=0.5,
                )
                bottom += vals
            ax.set_xticks(x)
            ax.set_xticklabels(MUT_CLASS_ORDER)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Stage composition (%)")
            ax.set_xlabel("Pyrimidine-normalized substitution class")
            ax.set_title(f"Panel {panel_letters[row_offset - 2]}: {species_label} (>= {threshold} cells)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[row_offset, 0].text(
            -0.22,
            1.08,
            panel_letters[row_offset - 2],
            transform=axes[row_offset, 0].transAxes,
            fontsize=18,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def _threshold_plot_table_from_summary(threshold_table: pd.DataFrame) -> pd.DataFrame:
    stage_columns = [
        ("Embryoblast", "Embryoblast_count", "Embryoblast_percent"),
        ("Germ layer-specific", "Germ_layer_specific_count", "Germ_layer_specific_percent"),
        ("Tissue-specific", "Tissue_specific_count", "Tissue_specific_percent"),
        ("Adult/cell type-specific", "Adult_cell_type_specific_count", "Adult_cell_type_specific_percent"),
    ]
    rows = []
    for _, row in threshold_table.iterrows():
        for stage_label, count_col, percent_col in stage_columns:
            rows.append(
                {
                    "species": row["species"],
                    "threshold": int(row["threshold"]),
                    "stage_label": stage_label,
                    "count": int(row[count_col]),
                    "percent": float(row[percent_col]),
                    "retained_donor_variant_calls": int(row["retained_donor_variant_calls"]),
                }
            )
    return pd.DataFrame(rows)


def _print_threshold_summary(threshold_table: pd.DataFrame, retained_counts: pd.DataFrame) -> None:
    stage_cols = [
        "Embryoblast_percent",
        "Germ_layer_specific_percent",
        "Tissue_specific_percent",
        "Adult_cell_type_specific_percent",
    ]
    label_map = {
        "Embryoblast_percent": "Embryoblast",
        "Germ_layer_specific_percent": "Germ layer-specific",
        "Tissue_specific_percent": "Tissue-specific",
        "Adult_cell_type_specific_percent": "Adult/cell type-specific",
    }

    print("\nSupport-threshold sensitivity")
    for species_label in ["Tabula Muris", "Tabula Sapiens"]:
        merged = threshold_table[threshold_table["species"] == species_label].merge(
            retained_counts,
            on=["species", "threshold", "retained_donor_variant_calls"],
            how="left",
        )
        print(f"\n## {species_label}")
        for _, row in merged.sort_values("threshold").iterrows():
            summary = ", ".join(
                f"{label_map[col]} {row[col]:.2f}%"
                for col in stage_cols
            )
            print(
                f">= {int(row['threshold'])} cells: "
                f"{int(row['retained_donor_variant_calls']):,} donor-variant calls; "
                f"{int(row['retained_donor_variant_compartments']):,} donor-variant-compartment observations; "
                f"{summary}"
            )

        ordered = merged.sort_values("threshold")
        start = ordered.iloc[0]
        end = ordered.iloc[-1]
        start_threshold = int(start["threshold"])
        end_threshold = int(end["threshold"])
        start_n = int(start["retained_donor_variant_calls"])
        end_n = int(end["retained_donor_variant_calls"])
        lost = start_n - end_n
        lost_pct = (lost / start_n * 100.0) if start_n else 0.0
        deltas = {
            label_map[col]: float(end[col]) - float(start[col])
            for col in stage_cols
        }
        max_stage = max(deltas, key=lambda key: abs(deltas[key]))
        print(
            f"Threshold tradeoff: {start_n:,} -> {end_n:,} retained calls "
            f"({lost:,} lost; {lost_pct:.2f}% reduction). "
            f"Largest stage shift from >={start_threshold} to >={end_threshold} is {max_stage} "
            f"({deltas[max_stage]:+.2f} percentage points)."
        )


def _print_substitution_summary(substitution_table: pd.DataFrame, editing_table: pd.DataFrame) -> None:
    print("\nSubstitution-class and RNA-editing sensitivity")
    for species_label in ["Tabula Muris", "Tabula Sapiens"]:
        print(f"\n## {species_label}")
        sub = substitution_table[substitution_table["species"] == species_label].copy()
        for threshold in THRESHOLDS:
            print(f"Threshold >= {threshold} cells")
            threshold_sub = sub[sub["threshold"] == threshold].copy()
            for _, row in threshold_sub.set_index("substitution_class").reindex(MUT_CLASS_ORDER).reset_index().iterrows():
                print(
                    f"{row['substitution_class']}: "
                    f"{int(row['retained_donor_variant_calls']):,} calls; "
                    f"Embryoblast {row['Embryoblast_percent']:.2f}%, "
                    f"Germ layer-specific {row['Germ_layer_specific_percent']:.2f}%, "
                    f"Tissue-specific {row['Tissue_specific_percent']:.2f}%, "
                    f"Adult/cell type-specific {row['Adult_cell_type_specific_percent']:.2f}%"
                )

            edit_sub = editing_table[
                (editing_table["species"] == species_label)
                & (editing_table["threshold"] == threshold)
            ].copy()
            for _, row in edit_sub.iterrows():
                print(
                    f"{row['subset']}: "
                    f"{int(row['retained_donor_variant_calls']):,} calls; "
                    f"Embryoblast {row['Embryoblast_percent']:.2f}%, "
                    f"Germ layer-specific {row['Germ_layer_specific_percent']:.2f}%, "
                    f"Tissue-specific {row['Tissue_specific_percent']:.2f}%, "
                    f"Adult/cell type-specific {row['Adult_cell_type_specific_percent']:.2f}%"
                )


def _return_unified_outputs(
    threshold_table: pd.DataFrame,
    retained_counts: pd.DataFrame,
    substitution_table: pd.DataFrame,
    editing_table: pd.DataFrame,
) -> dict[str, object]:
    return {
        "threshold_table": threshold_table,
        "retained_counts_table": retained_counts,
        "substitution_table": substitution_table,
        "editing_sensitivity_table": editing_table,
        "output_dir": OUT_DIR,
    }


def run_unified_sensitivity_analysis(*, show_plot: bool = True, verbose: bool = True) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold_path = OUT_DIR / "developmental_stage_unified_support_threshold_sensitivity.csv"
    retained_counts_path = OUT_DIR / "developmental_stage_unified_retained_counts.csv"
    substitution_path = OUT_DIR / "developmental_stage_unified_substitution_class_sensitivity.csv"
    editing_path = OUT_DIR / "developmental_stage_unified_rna_editing_sensitivity.csv"
    bundled_outputs = [threshold_path, retained_counts_path, substitution_path, editing_path]
    if os.environ.get("LORE2026_FULL_RUN") != "1" and all(path.exists() for path in bundled_outputs):
        threshold_table = pd.read_csv(threshold_path)
        retained_counts = pd.read_csv(retained_counts_path)
        substitution_table = pd.read_csv(substitution_path)
        editing_table = pd.read_csv(editing_path)
        if show_plot:
            figure = _build_unified_figure(
                _threshold_plot_table_from_summary(threshold_table),
                retained_counts,
                substitution_table,
            )
            plt.show()
            plt.close(figure)
        if verbose:
            print("Using bundled developmental-stage unified sensitivity outputs.")
            for path in bundled_outputs:
                print(f"- {path.relative_to(REPO_ROOT)}")
            _print_threshold_summary(threshold_table, retained_counts)
            _print_substitution_summary(substitution_table, editing_table)
            print(
                "\nInterpretation note: stricter supporting-cell thresholds reduce the retained donor-variant call set "
                "and can shift stage percentages. These values should be interpreted as threshold-dependent estimates, "
                "not fixed quantities."
            )
        return _return_unified_outputs(
            threshold_table,
            retained_counts,
            substitution_table,
            editing_table,
        )

    threshold_tables = []
    retained_count_tables = []
    threshold_plot_tables = []
    substitution_tables = []
    editing_tables = []

    for dataset_name, species_label in DATASET_SPECS:
        threshold_table, retained_counts, threshold_long = _threshold_tables_for_dataset(
            dataset_name,
            species_label,
        )
        threshold_tables.append(threshold_table)
        retained_count_tables.append(retained_counts)
        threshold_plot_tables.append(threshold_long)
        substitution_tables.append(_substitution_table_for_dataset(dataset_name, species_label))
        editing_tables.append(_editing_sensitivity_for_dataset(dataset_name, species_label))

    threshold_table = pd.concat(threshold_tables, ignore_index=True)
    retained_counts = pd.concat(retained_count_tables, ignore_index=True)
    threshold_long = pd.concat(threshold_plot_tables, ignore_index=True)
    substitution_table = pd.concat(substitution_tables, ignore_index=True)
    editing_table = pd.concat(editing_tables, ignore_index=True)

    threshold_table.to_csv(threshold_path, index=False)
    retained_counts.to_csv(retained_counts_path, index=False)
    substitution_table.to_csv(substitution_path, index=False)
    editing_table.to_csv(editing_path, index=False)
    if show_plot:
        figure = _build_unified_figure(threshold_long, retained_counts, substitution_table)
        plt.show()
        plt.close(figure)

    if verbose:
        print("Saved reviewer sensitivity outputs:")
        for path in [
            threshold_path,
            retained_counts_path,
            substitution_path,
            editing_path,
        ]:
            print(f"- {path.relative_to(REPO_ROOT)}")
        _print_threshold_summary(threshold_table, retained_counts)
        _print_substitution_summary(substitution_table, editing_table)
        print(
            "\nInterpretation note: stricter supporting-cell thresholds reduce the retained donor-variant call set "
            "and can shift stage percentages. These values should be interpreted as threshold-dependent estimates, "
            "not fixed quantities."
        )

    return _return_unified_outputs(
        threshold_table,
        retained_counts,
        substitution_table,
        editing_table,
    )


def main() -> None:
    run_unified_sensitivity_analysis(show_plot=False, verbose=True)


if __name__ == "__main__":
    main()
