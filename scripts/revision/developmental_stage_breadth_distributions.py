from __future__ import annotations

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
STAGE_COLORS = {
    "Embryoblast": "#f0c4a6",
    "Germ layer-specific": "#e39a58",
    "Tissue-specific": "#c96f1c",
    "Adult-specific": "#7e4314",
}
DATASET_SPECS = [
    ("TabMur", "Mouse"),
    ("TabSap", "Human"),
]
METRIC_SPECS = [
    ("n_layers", "Germ layers spanned"),
    ("n_tissues", "Tissues spanned"),
    ("n_cell_types", "Cell types spanned"),
]


def _load_variant_breadth(dataset_name: str, species_label: str) -> pd.DataFrame:
    path = DATA_DIR / dataset_name / "aggregates" / "mut_table_with_stage.csv"
    df = pd.read_csv(
        path,
        usecols=["donor", "var_id", "germ_layer", "tissue", "cell_type_stage", "stage_label"],
        low_memory=False,
    )
    df = df.dropna(
        subset=["donor", "var_id", "germ_layer", "tissue", "cell_type_stage", "stage_label"]
    ).copy()
    df["donor"] = df["donor"].astype(str)
    df["var_id"] = df["var_id"].astype(str)
    df["germ_layer"] = df["germ_layer"].astype(str).str.strip()
    df["tissue"] = df["tissue"].astype(str).str.strip()
    df["cell_type_stage"] = df["cell_type_stage"].astype(str).str.strip()
    df["stage_label"] = df["stage_label"].astype(str).str.strip()
    df = df[df["stage_label"].isin(STAGE_ORDER)].copy()

    collapsed = df.drop_duplicates(
        subset=["donor", "var_id", "germ_layer", "tissue", "cell_type_stage"]
    ).copy()
    breadth = (
        collapsed.groupby(["donor", "var_id", "stage_label"], observed=True)
        .agg(
            n_layers=("germ_layer", "nunique"),
            n_tissues=("tissue", "nunique"),
            n_cell_types=("cell_type_stage", "nunique"),
        )
        .reset_index()
    )
    breadth.insert(0, "species", species_label)
    return breadth


def _summarize_breadth(breadth: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for species_label in ["Mouse", "Human"]:
        species_df = breadth[breadth["species"] == species_label].copy()
        for stage_label in STAGE_ORDER:
            sub = species_df[species_df["stage_label"] == stage_label].copy()
            row: dict[str, object] = {
                "species": species_label,
                "stage_label": stage_label,
                "n_donor_variant_calls": int(len(sub)),
            }
            for metric, _ in METRIC_SPECS:
                vals = sub[metric].astype(int)
                row[f"{metric}_median"] = float(vals.median()) if not vals.empty else np.nan
                row[f"{metric}_mean"] = round(float(vals.mean()), 2) if not vals.empty else np.nan
                row[f"{metric}_max"] = int(vals.max()) if not vals.empty else 0
                row[f"{metric}_multi_percent"] = (
                    round(float((vals > 1).mean() * 100.0), 2) if not vals.empty else np.nan
                )
                row[f"{metric}_ge_3_percent"] = (
                    round(float((vals >= 3).mean() * 100.0), 2) if not vals.empty else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _metric_support(breadth: pd.DataFrame, metric: str) -> np.ndarray:
    max_val = int(breadth[metric].max())
    return np.arange(1, max_val + 1, dtype=int)


def _plot_distribution_figure(breadth: pd.DataFrame):
    fig, axes = plt.subplots(len(METRIC_SPECS), 2, figsize=(14, 12), dpi=400, sharey="row")
    for col_idx, species_label in enumerate(["Mouse", "Human"]):
        species_df = breadth[breadth["species"] == species_label].copy()
        for row_idx, (metric, axis_label) in enumerate(METRIC_SPECS):
            ax = axes[row_idx, col_idx]
            x = _metric_support(species_df, metric)
            for stage_label in STAGE_ORDER:
                vals = species_df.loc[species_df["stage_label"] == stage_label, metric].astype(int)
                counts = vals.value_counts(normalize=True).reindex(x, fill_value=0.0).sort_index()
                ax.plot(
                    x,
                    counts.to_numpy(float) * 100.0,
                    marker="o",
                    linewidth=2.0,
                    markersize=4.5,
                    color=STAGE_COLORS[stage_label],
                    label=stage_label,
                )
            ax.set_xticks(x)
            ax.set_xlabel(axis_label)
            if col_idx == 0:
                ax.set_ylabel("Donor-variant calls (%)")
            ax.set_title(f"{species_label}: {axis_label}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    return fig


def _plot_cdf_figure(breadth: pd.DataFrame):
    fig, axes = plt.subplots(len(METRIC_SPECS), 2, figsize=(14, 12), dpi=400, sharey="row")
    for col_idx, species_label in enumerate(["Mouse", "Human"]):
        species_df = breadth[breadth["species"] == species_label].copy()
        for row_idx, (metric, axis_label) in enumerate(METRIC_SPECS):
            ax = axes[row_idx, col_idx]
            x = _metric_support(species_df, metric)
            for stage_label in STAGE_ORDER:
                vals = species_df.loc[species_df["stage_label"] == stage_label, metric].astype(int)
                counts = vals.value_counts(normalize=True).reindex(x, fill_value=0.0).sort_index()
                cdf = counts.cumsum()
                ax.step(
                    x,
                    cdf.to_numpy(float) * 100.0,
                    where="post",
                    linewidth=2.0,
                    color=STAGE_COLORS[stage_label],
                    label=stage_label,
                )
                ax.plot(
                    x,
                    cdf.to_numpy(float) * 100.0,
                    marker="o",
                    linewidth=0.0,
                    markersize=4.5,
                    color=STAGE_COLORS[stage_label],
                )
            ax.set_xticks(x)
            ax.set_ylim(0, 102)
            ax.set_xlabel(axis_label)
            if col_idx == 0:
                ax.set_ylabel("Cumulative donor-variant calls (%)")
            ax.set_title(f"{species_label}: {axis_label}")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.18, linewidth=0.6)
            if row_idx == 0 and col_idx == 1:
                ax.legend(frameon=False, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    plt.tight_layout()
    return fig


def _print_summary(summary: pd.DataFrame) -> None:
    print("\nDevelopmental-stage breadth summary")
    for species_label in ["Mouse", "Human"]:
        sub = (
            summary[summary["species"] == species_label]
            .set_index("stage_label")
            .reindex(STAGE_ORDER)
            .reset_index()
        )
        print(f"\n## {species_label}")
        print(
            sub[
                [
                    "stage_label",
                    "n_donor_variant_calls",
                    "n_layers_median",
                    "n_layers_mean",
                    "n_tissues_median",
                    "n_tissues_mean",
                    "n_cell_types_median",
                    "n_cell_types_mean",
                    "n_tissues_multi_percent",
                    "n_tissues_ge_3_percent",
                ]
            ]
            .rename(
                columns={
                    "stage_label": "stage",
                    "n_donor_variant_calls": "n_calls",
                    "n_layers_median": "median_layers",
                    "n_layers_mean": "mean_layers",
                    "n_tissues_median": "median_tissues",
                    "n_tissues_mean": "mean_tissues",
                    "n_cell_types_median": "median_cell_types",
                    "n_cell_types_mean": "mean_cell_types",
                    "n_tissues_multi_percent": "multi_tissue_percent",
                    "n_tissues_ge_3_percent": "tissues_ge_3_percent",
                }
            )
            .round(
                {
                    "median_layers": 1,
                    "mean_layers": 2,
                    "median_tissues": 1,
                    "mean_tissues": 2,
                    "median_cell_types": 1,
                    "mean_cell_types": 2,
                    "multi_tissue_percent": 2,
                    "tissues_ge_3_percent": 2,
                }
            )
            .to_string(index=False)
        )


def run_stage_breadth_distribution_analysis(
    *, show_plot: bool = True, verbose: bool = True
) -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    breadth_tables = []
    for dataset_name, species_label in DATASET_SPECS:
        breadth_tables.append(_load_variant_breadth(dataset_name, species_label))
    breadth = pd.concat(breadth_tables, ignore_index=True)
    summary = _summarize_breadth(breadth)

    breadth_path = OUT_DIR / "developmental_stage_breadth_long.csv"
    summary_path = OUT_DIR / "developmental_stage_breadth_distribution_summary.csv"
    breadth.to_csv(breadth_path, index=False)
    summary.to_csv(summary_path, index=False)

    if show_plot:
        distribution_figure = _plot_distribution_figure(breadth)
        cdf_figure = _plot_cdf_figure(breadth)
        plt.show()
        plt.close(distribution_figure)
        plt.close(cdf_figure)

    if verbose:
        print("Saved developmental breadth tables:")
        for path in [breadth_path, summary_path]:
            print(f"- {path.relative_to(REPO_ROOT)}")
        _print_summary(summary)
        print(
            "\nInterpretation note: the tissue-spanning columns directly quantify how often variants remain "
            "shared across multiple tissues within each developmental stage label."
        )

    return {
        "breadth_table": breadth,
        "summary_table": summary,
        "output_dir": OUT_DIR,
    }


def main() -> None:
    run_stage_breadth_distribution_analysis(show_plot=False, verbose=True)


if __name__ == "__main__":
    main()
