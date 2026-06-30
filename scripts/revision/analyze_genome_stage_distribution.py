from __future__ import annotations

import json
from math import comb
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
OVERALL_SCOPE = "Overall"
SCOPE_ORDER = [OVERALL_SCOPE, *STAGE_ORDER]
STAGE_COLORS = {
    "Embryoblast": "#f0c4a6",
    "Germ layer-specific": "#e39a58",
    "Tissue-specific": "#c96f1c",
    "Adult-specific": "#7e4314",
}
SCOPE_COLORS = {
    OVERALL_SCOPE: "#7f7f7f",
    **STAGE_COLORS,
}
N_MONTE_CARLO = 20000
RNG = np.random.default_rng(20260520)
SMALL_WINDOW_SIZES_BP = [100, 1_000]
TOP_SMALL_WINDOWS_PER_STAGE = 10
CLUSTER_WINDOW_SIZES_BP = [10, 100, 1_000]

DATASET_SPECS = [
    {
        "dataset": "TabMur",
        "species": "Mouse",
        "plot_label": "Mouse (TabMur)",
        "mut_csv": DATA_DIR / "TabMur" / "aggregates" / "mut_table_with_stage.csv",
        "stage_donor_csv": DATA_DIR / "TabMur" / "aggregates" / "stage_burden_per_donor_perkb_STANDARDIZED_3LEVEL_cells.csv",
        "fai": DATA_DIR / "ref_genome" / "mm10.fa.fai",
        "canonical_autosomes": [f"chr{i}" for i in range(1, 20)],
        "bin_size_bp": 10_000_000,
    },
    {
        "dataset": "TabSap",
        "species": "Human",
        "plot_label": "Human (TabSap)",
        "mut_csv": DATA_DIR / "TabSap" / "aggregates" / "mut_table_with_stage.csv",
        "stage_donor_csv": DATA_DIR / "TabSap" / "aggregates" / "stage_burden_per_donor_perkb_STANDARDIZED_3LEVEL_cells.csv",
        "fai": DATA_DIR / "ref_genome" / "gencode_v41_ercc.fa.fai",
        "canonical_autosomes": [f"chr{i}" for i in range(1, 23)],
        "bin_size_bp": 10_000_000,
    },
]


def canonical_autosomes_by_species() -> dict[str, list[str]]:
    return {str(spec["species"]): list(spec["canonical_autosomes"]) for spec in DATASET_SPECS}


def ensure_output_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_lengths(fai_path: Path, chromosomes: list[str]) -> pd.Series:
    cols = ["chromosome", "length_bp", "offset", "line_bases", "line_width"]
    fai = pd.read_csv(fai_path, sep="\t", header=None, names=cols, usecols=[0, 1])
    fai["chromosome"] = fai["chromosome"].astype(str)
    fai = fai[fai["chromosome"].isin(chromosomes)].copy()
    if fai.empty:
        raise SystemExit(f"[ERROR] No matching chromosome lengths found in {fai_path}")
    lengths = fai.set_index("chromosome")["length_bp"].astype(float).reindex(chromosomes)
    missing = lengths[lengths.isna()].index.tolist()
    if missing:
        raise SystemExit(f"[ERROR] Missing chromosome lengths in {fai_path}: {missing}")
    return lengths


def load_stage_timing_donors(path: Path) -> set[str]:
    return {str(donor) for donor in pd.read_csv(path, index_col=0).index}


def load_unique_variants(
    mut_csv: Path,
    donor_whitelist: set[str],
    include_position: bool = False,
) -> pd.DataFrame:
    usecols = ["donor", "var_id", "#CHROM", "stage_label"]
    if include_position:
        usecols.append("Start")
    df = pd.read_csv(mut_csv, usecols=usecols, low_memory=False)
    df["donor"] = df["donor"].astype(str)
    df = df[df["donor"].isin(donor_whitelist)].copy()
    required = ["donor", "var_id", "#CHROM", "stage_label"]
    if include_position:
        required.append("Start")
    df = df.dropna(subset=required).copy()
    df["#CHROM"] = df["#CHROM"].astype(str).str.strip()
    df["stage_label"] = df["stage_label"].astype(str).str.strip()
    df["var_id"] = df["var_id"].astype(str)
    df = df[df["stage_label"].isin(STAGE_ORDER)].copy()
    if include_position:
        df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
        df = df.dropna(subset=["Start"]).copy()
        df["Start"] = df["Start"].astype(int)
        df = df.sort_values(["donor", "var_id", "Start"]).drop_duplicates(subset=["donor", "var_id"])
    else:
        df = df.drop_duplicates(subset=["donor", "var_id"])
    return df.reset_index(drop=True)


def load_cell_level_variants(spec: dict[str, object]) -> pd.DataFrame:
    donor_whitelist = load_stage_timing_donors(spec["stage_donor_csv"])
    usecols = [
        "donor",
        "CB",
        "var_id",
        "#CHROM",
        "Start",
        "stage_label",
        "REF",
        "ALT_expected",
        "tissue",
        "cell_type_parsed",
        "germ_layer",
    ]
    df = pd.read_csv(spec["mut_csv"], usecols=usecols, low_memory=False)
    df["donor"] = df["donor"].astype(str)
    df = df[df["donor"].isin(donor_whitelist)].copy()
    required = ["donor", "CB", "var_id", "#CHROM", "Start", "stage_label"]
    df = df.dropna(subset=required).copy()
    df["CB"] = df["CB"].astype(str)
    df["cell_id"] = df["donor"] + "::" + df["CB"]
    df["var_id"] = df["var_id"].astype(str)
    df["stage_label"] = df["stage_label"].astype(str).str.strip()
    df["#CHROM"] = df["#CHROM"].astype(str).str.strip()
    df = df[df["stage_label"].isin(STAGE_ORDER)].copy()
    df["Start"] = pd.to_numeric(df["Start"], errors="coerce")
    df = df.dropna(subset=["Start"]).copy()
    df["Start"] = df["Start"].astype(int)
    for col in ["REF", "ALT_expected", "tissue", "cell_type_parsed", "germ_layer"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()
    df = df.drop_duplicates(subset=["stage_label", "cell_id", "var_id"]).reset_index(drop=True)
    return df


def load_unique_donor_stage_sites(spec: dict[str, object]) -> pd.DataFrame:
    chromosomes = list(spec["canonical_autosomes"])
    df = load_cell_level_variants(spec)
    df = df[df["#CHROM"].isin(chromosomes)].copy()
    df = df.rename(columns={"#CHROM": "chromosome", "Start": "position_bp"})
    df["chromosome"] = pd.Categorical(
        df["chromosome"],
        categories=chromosomes,
        ordered=True,
    )
    df = df.sort_values(
        ["donor", "stage_label", "chromosome", "position_bp", "var_id", "cell_id"]
    ).reset_index(drop=True)
    out = (
        df.groupby(
            ["donor", "stage_label", "var_id", "chromosome", "position_bp"],
            observed=True,
            sort=False,
        )
        .agg(
            supporting_germ_layers=(
                "germ_layer",
                lambda values: ";".join(
                    sorted({str(value).strip() for value in values if str(value).strip()})
                ),
            ),
            n_supporting_germ_layers=(
                "germ_layer",
                lambda values: len({str(value).strip() for value in values if str(value).strip()}),
            ),
        )
        .reset_index()
    )
    return out[
        [
            "donor",
            "stage_label",
            "var_id",
            "chromosome",
            "position_bp",
            "supporting_germ_layers",
            "n_supporting_germ_layers",
        ]
    ].copy()


def _sem_or_zero(values: pd.Series) -> float:
    return float(values.sem(ddof=1)) if len(values) > 1 else 0.0


def monte_carlo_pvalue(obs: np.ndarray, probs: np.ndarray, n_simulations: int) -> tuple[float, float]:
    n_obs = int(obs.sum())
    exp = probs * n_obs
    stat_obs = float(np.sum((obs - exp) ** 2 / exp))
    simulated = RNG.multinomial(n_obs, probs, size=n_simulations)
    stat_sim = ((simulated - exp) ** 2 / exp).sum(axis=1)
    pvalue = (1.0 + float((stat_sim >= stat_obs).sum())) / (n_simulations + 1.0)
    return stat_obs, pvalue


def build_chrom_offsets(lengths: pd.Series, chromosomes: list[str]) -> pd.DataFrame:
    offsets = lengths.rename("chrom_length_bp").reset_index()
    offsets.columns = ["chromosome", "chrom_length_bp"]
    offsets["chromosome"] = pd.Categorical(offsets["chromosome"], categories=chromosomes, ordered=True)
    offsets = offsets.sort_values("chromosome").reset_index(drop=True)
    offsets["chrom_length_bp"] = offsets["chrom_length_bp"].astype(int)
    offsets["chrom_offset_bp"] = offsets["chrom_length_bp"].cumsum().shift(fill_value=0).astype(int)
    offsets["chrom_center_mb"] = (
        offsets["chrom_offset_bp"].to_numpy(float) + offsets["chrom_length_bp"].to_numpy(float) / 2.0
    ) / 1_000_000.0
    offsets["label"] = [chrom.replace("chr", "") for chrom in offsets["chromosome"].astype(str)]
    offsets["chromosome"] = offsets["chromosome"].astype(str)
    return offsets


def build_autosomal_bins(lengths: pd.Series, chromosomes: list[str], bin_size_bp: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for chromosome in chromosomes:
        chrom_length_bp = int(lengths[chromosome])
        n_bins = int(np.ceil(chrom_length_bp / bin_size_bp))
        for bin_idx in range(n_bins):
            start_bp = bin_idx * bin_size_bp + 1
            end_bp = min((bin_idx + 1) * bin_size_bp, chrom_length_bp)
            rows.append(
                {
                    "chromosome": chromosome,
                    "bin_idx": bin_idx,
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "bin_width_bp": end_bp - start_bp + 1,
                    "bin_mid_bp": (start_bp + end_bp) / 2.0,
                }
            )
    bins = pd.DataFrame(rows)
    bins["chromosome"] = pd.Categorical(bins["chromosome"], categories=chromosomes, ordered=True)
    bins = bins.sort_values(["chromosome", "bin_idx"]).reset_index(drop=True)
    bins["chromosome"] = bins["chromosome"].astype(str)
    bins["start_mb"] = bins["start_bp"] / 1_000_000.0
    bins["end_mb"] = bins["end_bp"] / 1_000_000.0
    bins["region"] = (
        bins["chromosome"]
        + ":"
        + bins["start_mb"].map(lambda value: f"{value:.1f}")
        + "-"
        + bins["end_mb"].map(lambda value: f"{value:.1f}")
        + " Mb"
    )
    return bins


def _pair_fraction(n_hits: int, universe_size: int) -> float:
    if universe_size < 2 or n_hits < 2:
        return 0.0
    return float(comb(int(n_hits), 2) / comb(int(universe_size), 2))


def _total_pair_count(universe_size: int) -> int:
    if universe_size < 2:
        return 0
    return int(comb(int(universe_size), 2))


def _same_site_pair_count(counts: pd.Series) -> int:
    if counts.empty:
        return 0
    numer = ((counts.astype(int) * (counts.astype(int) - 1)) // 2).sum()
    return int(numer)


def _collision_pair_stats(counts: pd.Series, n_variants: int) -> dict[str, object]:
    total_pairs = _total_pair_count(n_variants)
    same_site_pairs = _same_site_pair_count(counts)
    return {
        "n_unique_exact_sites": int(counts.shape[0]),
        "same_site_pairs": same_site_pairs,
        "total_pairs": total_pairs,
        "collision_rate": float(same_site_pairs / total_pairs) if total_pairs else 0.0,
    }


def _shared_exact_site_rate(counts: pd.Series, universe_size: int) -> float:
    return float(_collision_pair_stats(counts, universe_size)["collision_rate"])


def _same_window_pair_stats(window_counts: pd.Series, n_sites: int) -> dict[str, object]:
    total_pairs = _total_pair_count(n_sites)
    same_window_pairs = _same_site_pair_count(window_counts)
    return {
        "same_window_pairs": same_window_pairs,
        "total_site_pairs": total_pairs,
        "empirical_same_window_probability": float(same_window_pairs / total_pairs) if total_pairs else 0.0,
    }


def _summarize_unique_labels(values: pd.Series, limit: int = 3) -> str:
    labels = sorted({str(value).strip() for value in values if str(value).strip()})
    if not labels:
        return ""
    if len(labels) <= limit:
        return ", ".join(labels)
    return ", ".join(labels[:limit]) + f" (+{len(labels) - limit} more)"


def _split_semicolon_labels(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def _merge_semicolon_label_values(values: pd.Series) -> list[str]:
    labels: set[str] = set()
    for value in values:
        labels.update(_split_semicolon_labels(value))
    return sorted(labels)


def summarize_dataset(spec: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    donor_whitelist = load_stage_timing_donors(spec["stage_donor_csv"])
    variants = load_unique_variants(spec["mut_csv"], donor_whitelist)
    lengths = load_lengths(spec["fai"], spec["canonical_autosomes"])

    autosome_variants = variants[variants["#CHROM"].isin(lengths.index)].copy()
    excluded_non_autosomal = int(len(variants) - len(autosome_variants))
    probs = (lengths / lengths.sum()).astype(float)

    stage_rows: list[dict[str, object]] = []
    chrom_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []

    stage_counts_all = variants["stage_label"].value_counts().reindex(STAGE_ORDER, fill_value=0).astype(int)
    stage_perc_all = (stage_counts_all / max(stage_counts_all.sum(), 1) * 100.0).astype(float)
    for stage in STAGE_ORDER:
        stage_rows.append(
            {
                "species": spec["species"],
                "dataset": spec["dataset"],
                "stage_label": stage,
                "n_unique_donor_variants": int(stage_counts_all[stage]),
                "percent_unique_donor_variants": float(stage_perc_all[stage]),
                "n_stage_timing_donors": int(len(donor_whitelist)),
            }
        )

    for scope in SCOPE_ORDER:
        if scope == OVERALL_SCOPE:
            sub = autosome_variants
        else:
            sub = autosome_variants[autosome_variants["stage_label"] == scope]
        obs = sub["#CHROM"].value_counts().reindex(lengths.index, fill_value=0).astype(int)
        n_obs = int(obs.sum())
        exp = probs * n_obs
        residual = (obs - exp) / np.sqrt(exp)
        log2_oe = np.log2((obs + 0.5) / (exp + 0.5))

        for chrom in lengths.index:
            chrom_rows.append(
                {
                    "species": spec["species"],
                    "dataset": spec["dataset"],
                    "scope": scope,
                    "chromosome": chrom,
                    "chrom_length_bp": int(lengths[chrom]),
                    "observed_n_variants": int(obs[chrom]),
                    "expected_n_variants": float(exp[chrom]),
                    "observed_fraction": float(obs[chrom] / n_obs) if n_obs else 0.0,
                    "expected_fraction": float(probs[chrom]),
                    "variants_per_100mb": float(obs[chrom] / (lengths[chrom] / 100_000_000.0)),
                    "log2_observed_over_expected": float(log2_oe[chrom]),
                    "standardized_residual": float(residual[chrom]),
                }
            )

        if n_obs == 0:
            continue

        stat_obs, pvalue = monte_carlo_pvalue(obs.to_numpy(), probs.to_numpy(), N_MONTE_CARLO)
        top_enriched = residual.sort_values(ascending=False).index[0]
        top_depleted = residual.sort_values(ascending=True).index[0]
        test_rows.append(
            {
                "species": spec["species"],
                "dataset": spec["dataset"],
                "scope": scope,
                "tested_chromosomes": "autosomes",
                "n_stage_timing_donors": int(len(donor_whitelist)),
                "n_unique_donor_variants_tested": n_obs,
                "n_unique_donor_variants_excluded_non_autosomal": excluded_non_autosomal,
                "chi_square_statistic": float(stat_obs),
                "monte_carlo_pvalue": float(pvalue),
                "n_monte_carlo_simulations": int(N_MONTE_CARLO),
                "top_enriched_chromosome": top_enriched,
                "top_enriched_residual": float(residual[top_enriched]),
                "top_enriched_log2_observed_over_expected": float(log2_oe[top_enriched]),
                "top_depleted_chromosome": top_depleted,
                "top_depleted_residual": float(residual[top_depleted]),
                "top_depleted_log2_observed_over_expected": float(log2_oe[top_depleted]),
            }
        )

    return pd.DataFrame(stage_rows), pd.DataFrame(chrom_rows), pd.DataFrame(test_rows)


def summarize_stage_overlap_and_hotspots(
    spec: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chromosomes = list(spec["canonical_autosomes"])
    lengths = load_lengths(spec["fai"], chromosomes)
    offsets = build_chrom_offsets(lengths, chromosomes)
    cell_variants = load_cell_level_variants(spec)
    cell_variants = cell_variants[cell_variants["#CHROM"].isin(chromosomes)].copy()
    cell_variants = cell_variants.rename(columns={"#CHROM": "chromosome", "Start": "position_bp"})
    cell_variants["chromosome"] = pd.Categorical(
        cell_variants["chromosome"],
        categories=chromosomes,
        ordered=True,
    )
    cell_variants = cell_variants.sort_values(
        ["stage_label", "chromosome", "position_bp", "donor", "cell_id", "var_id"]
    ).reset_index(drop=True)
    cell_variants["donor_var_id"] = cell_variants["donor"] + "::" + cell_variants["var_id"]

    donor_variant_summary = (
        cell_variants.groupby(
            ["stage_label", "donor", "var_id", "chromosome", "position_bp"],
            observed=True,
            sort=False,
        )
        .agg(
            n_supporting_cells=("cell_id", "nunique"),
            n_supporting_tissues=("tissue", "nunique"),
            n_supporting_cell_types=("cell_type_parsed", "nunique"),
            tissues=("tissue", _summarize_unique_labels),
            cell_types=("cell_type_parsed", _summarize_unique_labels),
            germ_layers=("germ_layer", _summarize_unique_labels),
            ref=("REF", lambda values: next((value for value in values if str(value).strip()), "")),
            alt=("ALT_expected", lambda values: next((value for value in values if str(value).strip()), "")),
        )
        .reset_index()
    )
    donor_variant_summary["species"] = spec["species"]
    donor_variant_summary["dataset"] = spec["dataset"]
    donor_variant_summary["plot_label"] = spec["plot_label"]
    donor_variant_summary = donor_variant_summary.merge(
        offsets[["chromosome", "chrom_length_bp", "chrom_offset_bp", "chrom_center_mb", "label"]],
        on="chromosome",
        how="left",
    )
    donor_variant_summary["position_percent"] = (
        donor_variant_summary["position_bp"] / donor_variant_summary["chrom_length_bp"] * 100.0
    )
    donor_variant_summary["genome_position_mb"] = (
        donor_variant_summary["chrom_offset_bp"] + donor_variant_summary["position_bp"]
    ) / 1_000_000.0
    donor_variant_summary["chromosome_order"] = pd.Categorical(
        donor_variant_summary["chromosome"].astype(str),
        categories=chromosomes,
        ordered=True,
    ).codes
    donor_variant_summary["chromosome"] = donor_variant_summary["chromosome"].astype(str)

    overlap_rows: list[dict[str, object]] = []
    hotspot_rows: list[dict[str, object]] = []

    for stage in STAGE_ORDER:
        stage_cells = cell_variants[cell_variants["stage_label"] == stage].copy()
        stage_donor_variants = donor_variant_summary[donor_variant_summary["stage_label"] == stage].copy()
        n_stage_cells = int(stage_cells["cell_id"].nunique())
        n_stage_donors = int(stage_cells["donor"].nunique())
        n_cell_variant_observations = int(len(stage_cells))
        n_unique_sites = int(stage_cells["var_id"].nunique())
        n_unique_donor_variants = int(len(stage_donor_variants))

        cell_counts_by_var = stage_cells.groupby("var_id", observed=True)["cell_id"].nunique().sort_values(ascending=False)
        donor_counts_by_var = (
            stage_donor_variants.groupby("var_id", observed=True)["donor"].nunique().sort_values(ascending=False)
        )
        n_sites_seen_in_ge2_cells = int((cell_counts_by_var >= 2).sum())
        n_sites_seen_in_ge2_donors = int((donor_counts_by_var >= 2).sum())
        most_recurrent_site = cell_counts_by_var.index[0] if not cell_counts_by_var.empty else ""
        most_recurrent_cell_support = int(cell_counts_by_var.iloc[0]) if not cell_counts_by_var.empty else 0
        most_recurrent_donor_support = int(donor_counts_by_var.max()) if not donor_counts_by_var.empty else 0

        overlap_rows.append(
            {
                "species": spec["species"],
                "dataset": spec["dataset"],
                "stage_label": stage,
                "n_stage_cells": n_stage_cells,
                "n_stage_donors": n_stage_donors,
                "n_cell_variant_observations": n_cell_variant_observations,
                "n_unique_sites": n_unique_sites,
                "n_unique_donor_variants": n_unique_donor_variants,
                "n_sites_seen_in_ge2_cells": n_sites_seen_in_ge2_cells,
                "pct_unique_sites_seen_in_ge2_cells": (
                    float(n_sites_seen_in_ge2_cells / n_unique_sites * 100.0) if n_unique_sites else 0.0
                ),
                "pct_cell_variant_observations_in_ge2_cell_sites": (
                    float(cell_counts_by_var[cell_counts_by_var >= 2].sum() / n_cell_variant_observations * 100.0)
                    if n_cell_variant_observations
                    else 0.0
                ),
                "n_sites_seen_in_ge2_donors": n_sites_seen_in_ge2_donors,
                "pct_unique_sites_seen_in_ge2_donors": (
                    float(n_sites_seen_in_ge2_donors / n_unique_sites * 100.0) if n_unique_sites else 0.0
                ),
                "expected_shared_exact_sites_per_random_cell_pair": _shared_exact_site_rate(
                    cell_counts_by_var,
                    n_stage_cells,
                ),
                "expected_shared_exact_sites_per_random_donor_pair": _shared_exact_site_rate(
                    donor_counts_by_var,
                    n_stage_donors,
                ),
                "most_recurrent_exact_site": most_recurrent_site,
                "max_supporting_cells_for_exact_site": most_recurrent_cell_support,
                "max_supporting_donors_for_exact_site": most_recurrent_donor_support,
            }
        )

        if stage_cells.empty:
            continue

        for window_size_bp in SMALL_WINDOW_SIZES_BP:
            stage_windows = stage_cells.copy()
            stage_windows["window_start_bp"] = (
                ((stage_windows["position_bp"].astype(int) - 1) // window_size_bp) * window_size_bp + 1
            )
            window_summary = (
                stage_windows.groupby(["chromosome", "window_start_bp"], observed=True)
                .agg(
                    n_cell_variant_observations=("var_id", "size"),
                    n_unique_cells=("cell_id", "nunique"),
                    n_unique_donors=("donor", "nunique"),
                    n_unique_sites=("var_id", "nunique"),
                    n_unique_donor_variants=("donor_var_id", "nunique"),
                )
                .reset_index()
            )
            if window_summary.empty:
                continue
            window_summary["window_end_bp"] = window_summary["window_start_bp"] + window_size_bp - 1
            window_summary["region"] = (
                window_summary["chromosome"].astype(str)
                + ":"
                + window_summary["window_start_bp"].astype(int).astype(str)
                + "-"
                + window_summary["window_end_bp"].astype(int).astype(str)
            )
            top_windows = (
                window_summary.sort_values(
                    ["n_unique_cells", "n_unique_donor_variants", "n_unique_sites", "n_cell_variant_observations"],
                    ascending=[False, False, False, False],
                )
                .head(TOP_SMALL_WINDOWS_PER_STAGE)
                .reset_index(drop=True)
            )

            for rank, window_row in enumerate(top_windows.itertuples(index=False), start=1):
                sub = stage_windows[
                    (stage_windows["chromosome"].astype(str) == str(window_row.chromosome))
                    & (stage_windows["window_start_bp"] == int(window_row.window_start_bp))
                ].copy()
                cell_counts_within_window = (
                    sub.groupby("var_id", observed=True)["cell_id"].nunique().sort_values(ascending=False)
                )
                donor_counts_within_window = (
                    sub.groupby("var_id", observed=True)["donor"].nunique().sort_values(ascending=False)
                )
                top_exact_site = cell_counts_within_window.index[0] if not cell_counts_within_window.empty else ""
                top_exact_site_cells = int(cell_counts_within_window.iloc[0]) if not cell_counts_within_window.empty else 0
                top_exact_site_donors = int(donor_counts_within_window.max()) if not donor_counts_within_window.empty else 0
                hotspot_rows.append(
                    {
                        "species": spec["species"],
                        "dataset": spec["dataset"],
                        "stage_label": stage,
                        "window_size_bp": int(window_size_bp),
                        "rank_within_stage_window_size": int(rank),
                        "chromosome": str(window_row.chromosome),
                        "window_start_bp": int(window_row.window_start_bp),
                        "window_end_bp": int(window_row.window_end_bp),
                        "region": str(window_row.region),
                        "n_stage_cells": n_stage_cells,
                        "n_stage_donors": n_stage_donors,
                        "n_cell_variant_observations": int(window_row.n_cell_variant_observations),
                        "n_unique_cells": int(window_row.n_unique_cells),
                        "n_unique_donors": int(window_row.n_unique_donors),
                        "n_unique_sites": int(window_row.n_unique_sites),
                        "n_unique_donor_variants": int(window_row.n_unique_donor_variants),
                        "fraction_of_stage_cells_with_any_mutation_in_window": (
                            float(window_row.n_unique_cells / n_stage_cells) if n_stage_cells else 0.0
                        ),
                        "fraction_of_stage_donors_with_any_mutation_in_window": (
                            float(window_row.n_unique_donors / n_stage_donors) if n_stage_donors else 0.0
                        ),
                        "approx_p_two_random_cells_both_hit_window": _pair_fraction(
                            int(window_row.n_unique_cells),
                            n_stage_cells,
                        ),
                        "approx_p_two_random_donors_both_hit_window": _pair_fraction(
                            int(window_row.n_unique_donors),
                            n_stage_donors,
                        ),
                        "expected_shared_exact_sites_per_random_cell_pair_from_window": _shared_exact_site_rate(
                            cell_counts_within_window,
                            n_stage_cells,
                        ),
                        "expected_shared_exact_sites_per_random_donor_pair_from_window": _shared_exact_site_rate(
                            donor_counts_within_window,
                            n_stage_donors,
                        ),
                        "conditional_expected_shared_exact_sites_per_cell_pair_given_both_hit_window": _shared_exact_site_rate(
                            cell_counts_within_window,
                            int(window_row.n_unique_cells),
                        ),
                        "conditional_expected_shared_exact_sites_per_donor_pair_given_both_hit_window": _shared_exact_site_rate(
                            donor_counts_within_window,
                            int(window_row.n_unique_donors),
                        ),
                        "uniform_same_position_p_given_both_hits": float(1.0 / window_size_bp),
                        "top_exact_site_in_window": top_exact_site,
                        "top_exact_site_supporting_cells": top_exact_site_cells,
                        "top_exact_site_supporting_donors": top_exact_site_donors,
                    }
                )

    overlap_summary = pd.DataFrame(overlap_rows)
    hotspot_summary = pd.DataFrame(hotspot_rows)
    return overlap_summary, hotspot_summary, donor_variant_summary, offsets


def _summarize_same_window_scale(
    stage_sites: pd.DataFrame,
    window_size_bp: int,
    label: str,
    total_site_pairs: int,
) -> dict[str, object]:
    empty_summary = {
        f"same_window_pairs_{label}": 0,
        f"empirical_same_window_probability_{label}": 0.0,
        f"top_{label}_chrom": "",
        f"top_{label}_start": pd.NA,
        f"top_{label}_end": pd.NA,
        f"top_{label}_n_unique_sites": 0,
        f"top_window_pairs_{label}": 0,
        f"top_window_pair_fraction_{label}": 0.0,
    }
    if stage_sites.empty:
        return empty_summary

    stage_windows = stage_sites.copy()
    stage_windows["window_start_bp"] = (
        ((stage_windows["position_bp"].astype(int) - 1) // window_size_bp) * window_size_bp + 1
    )
    window_summary = (
        stage_windows.groupby(["chromosome", "window_start_bp"], observed=True)
        .size()
        .reset_index(name="n_unique_sites")
    )
    if window_summary.empty:
        return empty_summary

    pair_stats = _same_window_pair_stats(window_summary["n_unique_sites"], int(len(stage_sites)))
    top_window = (
        window_summary.sort_values(
            ["n_unique_sites", "chromosome", "window_start_bp"],
            ascending=[False, True, True],
        )
        .head(1)
        .iloc[0]
    )
    top_window_n_unique_sites = int(top_window["n_unique_sites"])
    top_window_pairs = _total_pair_count(top_window_n_unique_sites)
    top_window_start = int(top_window["window_start_bp"])
    return {
        f"same_window_pairs_{label}": int(pair_stats["same_window_pairs"]),
        f"empirical_same_window_probability_{label}": float(pair_stats["empirical_same_window_probability"]),
        f"top_{label}_chrom": str(top_window["chromosome"]),
        f"top_{label}_start": top_window_start,
        f"top_{label}_end": top_window_start + window_size_bp - 1,
        f"top_{label}_n_unique_sites": top_window_n_unique_sites,
        f"top_window_pairs_{label}": top_window_pairs,
        f"top_window_pair_fraction_{label}": float(top_window_pairs / total_site_pairs) if total_site_pairs else 0.0,
    }


def _summarize_top_two_germ_layer_window(
    stage_sites: pd.DataFrame,
    window_size_bp: int,
    label: str,
    total_site_pairs: int,
) -> dict[str, object]:
    prefix = f"top_two_germ_layer_{label}"
    empty_summary = {
        f"{prefix}_chrom": "",
        f"{prefix}_start": pd.NA,
        f"{prefix}_end": pd.NA,
        f"{prefix}_n_unique_sites": 0,
        f"{prefix}_n_germ_layers": 0,
        f"{prefix}_germ_layers": "",
        f"{prefix}_window_pairs": 0,
        f"{prefix}_window_pair_fraction": 0.0,
        f"{prefix}_mutation_frequency_per_bp": 0.0,
        f"{prefix}_mutation_frequency_per_kb": 0.0,
        f"{prefix}_random_exact_site_probability": 0.0,
        f"{prefix}_uniform_same_position_probability_given_window_hits": float(1.0 / window_size_bp),
    }
    if stage_sites.empty:
        return empty_summary

    stage_windows = stage_sites.copy()
    stage_windows["window_start_bp"] = (
        ((stage_windows["position_bp"].astype(int) - 1) // window_size_bp) * window_size_bp + 1
    )
    window_summary = (
        stage_windows.groupby(["chromosome", "window_start_bp"], observed=True)
        .agg(
            n_unique_sites=("var_id", "nunique"),
            germ_layers=("supporting_germ_layers", lambda values: ";".join(_merge_semicolon_label_values(values))),
            n_germ_layers=("supporting_germ_layers", lambda values: len(_merge_semicolon_label_values(values))),
        )
        .reset_index()
    )
    if window_summary.empty:
        return empty_summary

    qualifying = window_summary[window_summary["n_germ_layers"] >= 2].copy()
    if qualifying.empty:
        return empty_summary

    top_window = (
        qualifying.sort_values(
            ["n_unique_sites", "n_germ_layers", "chromosome", "window_start_bp"],
            ascending=[False, False, True, True],
        )
        .head(1)
        .iloc[0]
    )
    n_unique_sites = int(top_window["n_unique_sites"])
    top_window_pairs = _total_pair_count(n_unique_sites)
    top_window_pair_fraction = float(top_window_pairs / total_site_pairs) if total_site_pairs else 0.0
    mutation_frequency_per_bp = float(n_unique_sites / window_size_bp)
    return {
        f"{prefix}_chrom": str(top_window["chromosome"]),
        f"{prefix}_start": int(top_window["window_start_bp"]),
        f"{prefix}_end": int(top_window["window_start_bp"]) + window_size_bp - 1,
        f"{prefix}_n_unique_sites": n_unique_sites,
        f"{prefix}_n_germ_layers": int(top_window["n_germ_layers"]),
        f"{prefix}_germ_layers": str(top_window["germ_layers"]),
        f"{prefix}_window_pairs": top_window_pairs,
        f"{prefix}_window_pair_fraction": top_window_pair_fraction,
        f"{prefix}_mutation_frequency_per_bp": mutation_frequency_per_bp,
        f"{prefix}_mutation_frequency_per_kb": float(mutation_frequency_per_bp * 1000.0),
        f"{prefix}_random_exact_site_probability": float(top_window_pair_fraction / window_size_bp),
        f"{prefix}_uniform_same_position_probability_given_window_hits": float(1.0 / window_size_bp),
    }


def summarize_donor_stage_same_window_clustering(
    spec: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    donor_order = sorted(load_stage_timing_donors(spec["stage_donor_csv"]))
    lengths = load_lengths(spec["fai"], list(spec["canonical_autosomes"]))
    genome_size_bp = int(lengths.sum())
    variants = load_unique_donor_stage_sites(spec)
    variants = variants.sort_values(
        ["stage_label", "chromosome", "position_bp", "donor", "var_id"]
    ).reset_index(drop=True)

    donor_rows: list[dict[str, object]] = []
    for donor in donor_order:
        donor_variants = variants[variants["donor"] == donor].copy()
        for stage in STAGE_ORDER:
            donor_stage_sites = donor_variants[donor_variants["stage_label"] == stage].copy()
            n_variants = int(len(donor_stage_sites))
            total_site_pairs = _total_pair_count(n_variants)
            exact_site_counts = (
                donor_stage_sites.groupby(["chromosome", "position_bp"], observed=True).size().sort_values(ascending=False)
            )
            exact_site_stats = _collision_pair_stats(exact_site_counts, n_variants)
            row = {
                "species": spec["species"],
                "dataset": spec["dataset"],
                "donor": donor,
                "stage_label": stage,
                "n_variants": n_variants,
                "n_unique_exact_sites": int(exact_site_stats["n_unique_exact_sites"]),
                "exact_site_same_site_pairs": int(exact_site_stats["same_site_pairs"]),
                "total_site_pairs": total_site_pairs,
                "exact_site_collision_rate": float(exact_site_stats["collision_rate"]),
            }
            for window_size_bp, label in [(10, "10bp"), (100, "100bp"), (1_000, "1kb")]:
                row.update(_summarize_same_window_scale(donor_stage_sites, window_size_bp, label, total_site_pairs))
                row.update(
                    _summarize_top_two_germ_layer_window(
                        donor_stage_sites,
                        window_size_bp,
                        label,
                        total_site_pairs,
                    )
                )
            donor_rows.append(row)

    donor_summary = pd.DataFrame(donor_rows)[
        [
            "species",
            "dataset",
            "donor",
            "stage_label",
            "n_variants",
            "n_unique_exact_sites",
            "exact_site_same_site_pairs",
            "total_site_pairs",
            "exact_site_collision_rate",
            "same_window_pairs_10bp",
            "empirical_same_window_probability_10bp",
            "top_10bp_chrom",
            "top_10bp_start",
            "top_10bp_end",
            "top_10bp_n_unique_sites",
            "top_window_pairs_10bp",
            "top_window_pair_fraction_10bp",
            "top_two_germ_layer_10bp_chrom",
            "top_two_germ_layer_10bp_start",
            "top_two_germ_layer_10bp_end",
            "top_two_germ_layer_10bp_n_unique_sites",
            "top_two_germ_layer_10bp_n_germ_layers",
            "top_two_germ_layer_10bp_germ_layers",
            "top_two_germ_layer_10bp_window_pairs",
            "top_two_germ_layer_10bp_window_pair_fraction",
            "top_two_germ_layer_10bp_mutation_frequency_per_bp",
            "top_two_germ_layer_10bp_mutation_frequency_per_kb",
            "top_two_germ_layer_10bp_random_exact_site_probability",
            "top_two_germ_layer_10bp_uniform_same_position_probability_given_window_hits",
            "same_window_pairs_100bp",
            "empirical_same_window_probability_100bp",
            "top_100bp_chrom",
            "top_100bp_start",
            "top_100bp_end",
            "top_100bp_n_unique_sites",
            "top_window_pairs_100bp",
            "top_window_pair_fraction_100bp",
            "top_two_germ_layer_100bp_chrom",
            "top_two_germ_layer_100bp_start",
            "top_two_germ_layer_100bp_end",
            "top_two_germ_layer_100bp_n_unique_sites",
            "top_two_germ_layer_100bp_n_germ_layers",
            "top_two_germ_layer_100bp_germ_layers",
            "top_two_germ_layer_100bp_window_pairs",
            "top_two_germ_layer_100bp_window_pair_fraction",
            "top_two_germ_layer_100bp_mutation_frequency_per_bp",
            "top_two_germ_layer_100bp_mutation_frequency_per_kb",
            "top_two_germ_layer_100bp_random_exact_site_probability",
            "top_two_germ_layer_100bp_uniform_same_position_probability_given_window_hits",
            "same_window_pairs_1kb",
            "empirical_same_window_probability_1kb",
            "top_1kb_chrom",
            "top_1kb_start",
            "top_1kb_end",
            "top_1kb_n_unique_sites",
            "top_window_pairs_1kb",
            "top_window_pair_fraction_1kb",
            "top_two_germ_layer_1kb_chrom",
            "top_two_germ_layer_1kb_start",
            "top_two_germ_layer_1kb_end",
            "top_two_germ_layer_1kb_n_unique_sites",
            "top_two_germ_layer_1kb_n_germ_layers",
            "top_two_germ_layer_1kb_germ_layers",
            "top_two_germ_layer_1kb_window_pairs",
            "top_two_germ_layer_1kb_window_pair_fraction",
            "top_two_germ_layer_1kb_mutation_frequency_per_bp",
            "top_two_germ_layer_1kb_mutation_frequency_per_kb",
            "top_two_germ_layer_1kb_random_exact_site_probability",
            "top_two_germ_layer_1kb_uniform_same_position_probability_given_window_hits",
        ]
    ]

    summary_rows: list[dict[str, object]] = []
    null_probabilities = {
        "10bp": float(10 / genome_size_bp),
        "100bp": float(100 / genome_size_bp),
        "1kb": float(1_000 / genome_size_bp),
    }
    for stage in STAGE_ORDER:
        stage_df = donor_summary[donor_summary["stage_label"] == stage].copy()
        row = {
            "species": spec["species"],
            "dataset": spec["dataset"],
            "stage_label": stage,
            "n_donors": int(stage_df["donor"].nunique()),
            "median_n_unique_sites": float(stage_df["n_variants"].median()),
            "sum_total_site_pairs": int(stage_df["total_site_pairs"].sum()),
            "sum_exact_site_same_site_pairs": int(stage_df["exact_site_same_site_pairs"].sum()),
            "pair_weighted_exact_site_collision_rate": (
                float(stage_df["exact_site_same_site_pairs"].sum() / stage_df["total_site_pairs"].sum())
                if int(stage_df["total_site_pairs"].sum())
                else 0.0
            ),
            "mean_exact_site_collision_rate": float(stage_df["exact_site_collision_rate"].mean()),
            "median_exact_site_collision_rate": float(stage_df["exact_site_collision_rate"].median()),
        }
        for label in ["10bp", "100bp", "1kb"]:
            sum_same_window_pairs = int(stage_df[f"same_window_pairs_{label}"].sum())
            sum_top_window_pairs = int(stage_df[f"top_window_pairs_{label}"].sum())
            sum_total_site_pairs = int(stage_df["total_site_pairs"].sum())
            sum_top_two_germ_layer_window_pairs = int(stage_df[f"top_two_germ_layer_{label}_window_pairs"].sum())
            two_germ_layer_mask = stage_df[f"top_two_germ_layer_{label}_n_germ_layers"].fillna(0).astype(int) >= 2
            row.update(
                {
                    f"sum_same_window_pairs_{label}": sum_same_window_pairs,
                    f"pair_weighted_empirical_same_window_probability_{label}": (
                        float(sum_same_window_pairs / sum_total_site_pairs) if sum_total_site_pairs else 0.0
                    ),
                    f"mean_empirical_same_window_probability_{label}": float(
                        stage_df[f"empirical_same_window_probability_{label}"].mean()
                    ),
                    f"median_empirical_same_window_probability_{label}": float(
                        stage_df[f"empirical_same_window_probability_{label}"].median()
                    ),
                    f"median_top_{label}_n_unique_sites": float(stage_df[f"top_{label}_n_unique_sites"].median()),
                    f"sum_top_window_pairs_{label}": sum_top_window_pairs,
                    f"pair_weighted_top_window_pair_fraction_{label}": (
                        float(sum_top_window_pairs / sum_total_site_pairs) if sum_total_site_pairs else 0.0
                    ),
                    f"null_probability_{label}": float(null_probabilities[label]),
                    f"n_donors_with_top_two_germ_layer_{label}": int(two_germ_layer_mask.sum()),
                    f"median_top_two_germ_layer_{label}_n_unique_sites": float(
                        stage_df.loc[two_germ_layer_mask, f"top_two_germ_layer_{label}_n_unique_sites"].median()
                    )
                    if bool(two_germ_layer_mask.any())
                    else 0.0,
                    f"mean_top_two_germ_layer_{label}_mutation_frequency_per_bp": float(
                        stage_df.loc[two_germ_layer_mask, f"top_two_germ_layer_{label}_mutation_frequency_per_bp"].mean()
                    )
                    if bool(two_germ_layer_mask.any())
                    else 0.0,
                    f"median_top_two_germ_layer_{label}_mutation_frequency_per_bp": float(
                        stage_df.loc[two_germ_layer_mask, f"top_two_germ_layer_{label}_mutation_frequency_per_bp"].median()
                    )
                    if bool(two_germ_layer_mask.any())
                    else 0.0,
                    f"sum_top_two_germ_layer_window_pairs_{label}": sum_top_two_germ_layer_window_pairs,
                    f"pair_weighted_top_two_germ_layer_window_pair_fraction_{label}": (
                        float(sum_top_two_germ_layer_window_pairs / sum_total_site_pairs) if sum_total_site_pairs else 0.0
                    ),
                    f"pair_weighted_top_two_germ_layer_random_exact_site_probability_{label}": (
                        float(
                            stage_df[f"top_two_germ_layer_{label}_random_exact_site_probability"].mul(
                                stage_df["total_site_pairs"]
                            ).sum()
                            / sum_total_site_pairs
                        )
                        if sum_total_site_pairs
                        else 0.0
                    ),
                }
            )
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)[
        [
            "species",
            "dataset",
            "stage_label",
            "n_donors",
            "median_n_unique_sites",
            "sum_total_site_pairs",
            "sum_exact_site_same_site_pairs",
            "pair_weighted_exact_site_collision_rate",
            "mean_exact_site_collision_rate",
            "median_exact_site_collision_rate",
            "sum_same_window_pairs_10bp",
            "pair_weighted_empirical_same_window_probability_10bp",
            "mean_empirical_same_window_probability_10bp",
            "median_empirical_same_window_probability_10bp",
            "median_top_10bp_n_unique_sites",
            "sum_top_window_pairs_10bp",
            "pair_weighted_top_window_pair_fraction_10bp",
            "null_probability_10bp",
            "n_donors_with_top_two_germ_layer_10bp",
            "median_top_two_germ_layer_10bp_n_unique_sites",
            "mean_top_two_germ_layer_10bp_mutation_frequency_per_bp",
            "median_top_two_germ_layer_10bp_mutation_frequency_per_bp",
            "sum_top_two_germ_layer_window_pairs_10bp",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_10bp",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_10bp",
            "sum_same_window_pairs_100bp",
            "pair_weighted_empirical_same_window_probability_100bp",
            "mean_empirical_same_window_probability_100bp",
            "median_empirical_same_window_probability_100bp",
            "median_top_100bp_n_unique_sites",
            "sum_top_window_pairs_100bp",
            "pair_weighted_top_window_pair_fraction_100bp",
            "null_probability_100bp",
            "n_donors_with_top_two_germ_layer_100bp",
            "median_top_two_germ_layer_100bp_n_unique_sites",
            "mean_top_two_germ_layer_100bp_mutation_frequency_per_bp",
            "median_top_two_germ_layer_100bp_mutation_frequency_per_bp",
            "sum_top_two_germ_layer_window_pairs_100bp",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_100bp",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_100bp",
            "sum_same_window_pairs_1kb",
            "pair_weighted_empirical_same_window_probability_1kb",
            "mean_empirical_same_window_probability_1kb",
            "median_empirical_same_window_probability_1kb",
            "median_top_1kb_n_unique_sites",
            "sum_top_window_pairs_1kb",
            "pair_weighted_top_window_pair_fraction_1kb",
            "null_probability_1kb",
            "n_donors_with_top_two_germ_layer_1kb",
            "median_top_two_germ_layer_1kb_n_unique_sites",
            "mean_top_two_germ_layer_1kb_mutation_frequency_per_bp",
            "median_top_two_germ_layer_1kb_mutation_frequency_per_bp",
            "sum_top_two_germ_layer_window_pairs_1kb",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_1kb",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_1kb",
        ]
    ]
    return donor_summary, summary_df


def build_two_germ_layer_hotspot_probability_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    out = summary_df[
        [
            "species",
            "dataset",
            "stage_label",
            "n_donors",
            "n_donors_with_top_two_germ_layer_10bp",
            "median_top_two_germ_layer_10bp_n_unique_sites",
            "mean_top_two_germ_layer_10bp_mutation_frequency_per_bp",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_10bp",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_10bp",
            "n_donors_with_top_two_germ_layer_100bp",
            "median_top_two_germ_layer_100bp_n_unique_sites",
            "mean_top_two_germ_layer_100bp_mutation_frequency_per_bp",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_100bp",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_100bp",
            "n_donors_with_top_two_germ_layer_1kb",
            "median_top_two_germ_layer_1kb_n_unique_sites",
            "mean_top_two_germ_layer_1kb_mutation_frequency_per_bp",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_1kb",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_1kb",
        ]
    ].copy()
    out["percent_of_mutation_site_pairs_in_top_two_germ_layer_window_10bp"] = (
        out["pair_weighted_top_two_germ_layer_window_pair_fraction_10bp"] * 100.0
    )
    out["percent_of_mutation_site_pairs_in_top_two_germ_layer_window_100bp"] = (
        out["pair_weighted_top_two_germ_layer_window_pair_fraction_100bp"] * 100.0
    )
    out["percent_of_mutation_site_pairs_in_top_two_germ_layer_window_1kb"] = (
        out["pair_weighted_top_two_germ_layer_window_pair_fraction_1kb"] * 100.0
    )
    return out[
        [
            "species",
            "dataset",
            "stage_label",
            "n_donors",
            "n_donors_with_top_two_germ_layer_10bp",
            "median_top_two_germ_layer_10bp_n_unique_sites",
            "mean_top_two_germ_layer_10bp_mutation_frequency_per_bp",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_10bp",
            "percent_of_mutation_site_pairs_in_top_two_germ_layer_window_10bp",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_10bp",
            "n_donors_with_top_two_germ_layer_100bp",
            "median_top_two_germ_layer_100bp_n_unique_sites",
            "mean_top_two_germ_layer_100bp_mutation_frequency_per_bp",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_100bp",
            "percent_of_mutation_site_pairs_in_top_two_germ_layer_window_100bp",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_100bp",
            "n_donors_with_top_two_germ_layer_1kb",
            "median_top_two_germ_layer_1kb_n_unique_sites",
            "mean_top_two_germ_layer_1kb_mutation_frequency_per_bp",
            "pair_weighted_top_two_germ_layer_window_pair_fraction_1kb",
            "percent_of_mutation_site_pairs_in_top_two_germ_layer_window_1kb",
            "pair_weighted_top_two_germ_layer_random_exact_site_probability_1kb",
        ]
    ]


def build_donor_two_germ_layer_hotspot_table(donor_summary: pd.DataFrame) -> pd.DataFrame:
    return donor_summary[
        [
            "species",
            "dataset",
            "donor",
            "stage_label",
            "n_variants",
            "total_site_pairs",
            "top_two_germ_layer_10bp_chrom",
            "top_two_germ_layer_10bp_start",
            "top_two_germ_layer_10bp_end",
            "top_two_germ_layer_10bp_n_unique_sites",
            "top_two_germ_layer_10bp_n_germ_layers",
            "top_two_germ_layer_10bp_germ_layers",
            "top_two_germ_layer_10bp_window_pair_fraction",
            "top_two_germ_layer_10bp_mutation_frequency_per_bp",
            "top_two_germ_layer_10bp_random_exact_site_probability",
            "top_two_germ_layer_100bp_chrom",
            "top_two_germ_layer_100bp_start",
            "top_two_germ_layer_100bp_end",
            "top_two_germ_layer_100bp_n_unique_sites",
            "top_two_germ_layer_100bp_n_germ_layers",
            "top_two_germ_layer_100bp_germ_layers",
            "top_two_germ_layer_100bp_window_pair_fraction",
            "top_two_germ_layer_100bp_mutation_frequency_per_bp",
            "top_two_germ_layer_100bp_random_exact_site_probability",
            "top_two_germ_layer_1kb_chrom",
            "top_two_germ_layer_1kb_start",
            "top_two_germ_layer_1kb_end",
            "top_two_germ_layer_1kb_n_unique_sites",
            "top_two_germ_layer_1kb_n_germ_layers",
            "top_two_germ_layer_1kb_germ_layers",
            "top_two_germ_layer_1kb_window_pair_fraction",
            "top_two_germ_layer_1kb_mutation_frequency_per_bp",
            "top_two_germ_layer_1kb_random_exact_site_probability",
        ]
    ].copy()


def summarize_position_distributions(
    spec: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    donor_whitelist = load_stage_timing_donors(spec["stage_donor_csv"])
    donor_order = sorted(donor_whitelist)
    chromosomes = list(spec["canonical_autosomes"])
    lengths = load_lengths(spec["fai"], chromosomes)
    offsets = build_chrom_offsets(lengths, chromosomes)
    variants = load_unique_variants(spec["mut_csv"], donor_whitelist, include_position=True)

    autosome_variants = variants[variants["#CHROM"].isin(chromosomes)].copy()
    autosome_variants = autosome_variants.rename(columns={"#CHROM": "chromosome", "Start": "position_bp"})
    autosome_variants["chromosome"] = pd.Categorical(
        autosome_variants["chromosome"],
        categories=chromosomes,
        ordered=True,
    )
    autosome_variants = autosome_variants.sort_values(
        ["chromosome", "position_bp", "donor", "var_id"]
    ).reset_index(drop=True)
    autosome_variants["chromosome_order"] = autosome_variants["chromosome"].cat.codes
    autosome_variants["chromosome"] = autosome_variants["chromosome"].astype(str)
    autosome_variants = autosome_variants.merge(
        offsets[["chromosome", "chrom_length_bp", "chrom_offset_bp", "chrom_center_mb", "label"]],
        on="chromosome",
        how="left",
    )
    autosome_variants["position_percent"] = (
        autosome_variants["position_bp"] / autosome_variants["chrom_length_bp"] * 100.0
    )
    autosome_variants["genome_position_mb"] = (
        autosome_variants["chrom_offset_bp"] + autosome_variants["position_bp"]
    ) / 1_000_000.0

    scope_variants = pd.concat(
        [
            autosome_variants.assign(scope=OVERALL_SCOPE),
            autosome_variants.assign(scope=autosome_variants["stage_label"]),
        ],
        ignore_index=True,
    )
    scope_variants["species"] = spec["species"]
    scope_variants["dataset"] = spec["dataset"]
    scope_variants["plot_label"] = spec["plot_label"]

    position_scope_summary = scope_variants[
        [
            "species",
            "dataset",
            "plot_label",
            "scope",
            "stage_label",
            "donor",
            "var_id",
            "chromosome",
            "chromosome_order",
            "position_bp",
            "chrom_length_bp",
            "chrom_offset_bp",
            "chrom_center_mb",
            "position_percent",
            "genome_position_mb",
        ]
    ].copy()

    bin_size_bp = int(spec["bin_size_bp"])
    bins = build_autosomal_bins(lengths, chromosomes, bin_size_bp)
    scope_variants["bin_idx"] = ((scope_variants["position_bp"] - 1) // bin_size_bp).astype(int)

    donor_bin_counts = (
        scope_variants.groupby(["scope", "donor", "chromosome", "bin_idx"], observed=True)["var_id"]
        .nunique()
        .rename("n_variants")
        .reset_index()
    )

    donor_scope_grid = (
        bins.assign(_merge_key=1)
        .merge(pd.DataFrame({"donor": donor_order, "_merge_key": 1}), on="_merge_key", how="inner")
        .merge(pd.DataFrame({"scope": SCOPE_ORDER, "_merge_key": 1}), on="_merge_key", how="inner")
        .drop(columns="_merge_key")
    )
    donor_bin_counts = donor_scope_grid.merge(
        donor_bin_counts,
        on=["scope", "donor", "chromosome", "bin_idx"],
        how="left",
    )
    donor_bin_counts["n_variants"] = donor_bin_counts["n_variants"].fillna(0.0).astype(float)
    donor_bin_counts["variants_per_100mb"] = (
        donor_bin_counts["n_variants"] / (donor_bin_counts["bin_width_bp"] / 100_000_000.0)
    )
    donor_bin_counts = donor_bin_counts.merge(
        offsets[["chromosome", "chrom_length_bp", "chrom_offset_bp", "chrom_center_mb", "label"]],
        on="chromosome",
        how="left",
    )
    donor_bin_counts["genome_mid_bp"] = donor_bin_counts["chrom_offset_bp"] + donor_bin_counts["bin_mid_bp"]

    scope_totals = pd.Series(0, index=SCOPE_ORDER, dtype=int)
    observed_scope_totals = scope_variants.groupby("scope", observed=True).size().astype(int)
    scope_totals.update(observed_scope_totals)
    scope_totals = scope_totals.rename("n_unique_donor_variants").rename_axis("scope").reset_index()

    bin_scope_summary = (
        donor_bin_counts.groupby(
            [
                "scope",
                "chromosome",
                "bin_idx",
                "start_bp",
                "end_bp",
                "start_mb",
                "end_mb",
                "region",
                "bin_width_bp",
                "bin_mid_bp",
                "chrom_length_bp",
                "chrom_offset_bp",
                "chrom_center_mb",
                "label",
                "genome_mid_bp",
            ],
            observed=True,
        )
        .agg(
            mean_n_variants=("n_variants", "mean"),
            sem_n_variants=("n_variants", _sem_or_zero),
            mean_variants_per_100mb=("variants_per_100mb", "mean"),
            sem_variants_per_100mb=("variants_per_100mb", _sem_or_zero),
        )
        .reset_index()
    )
    bin_scope_summary = bin_scope_summary.merge(scope_totals, on="scope", how="left")
    bin_scope_summary["n_unique_donor_variants"] = (
        bin_scope_summary["n_unique_donor_variants"].fillna(0).astype(int)
    )
    bin_scope_summary["n_stage_timing_donors"] = int(len(donor_order))
    bin_scope_summary["bin_size_bp"] = bin_size_bp
    bin_scope_summary["species"] = spec["species"]
    bin_scope_summary["dataset"] = spec["dataset"]
    bin_scope_summary["plot_label"] = spec["plot_label"]
    bin_scope_summary["scope_color"] = bin_scope_summary["scope"].map(SCOPE_COLORS)
    bin_scope_summary = bin_scope_summary.sort_values(["scope", "chromosome", "bin_idx"]).reset_index(drop=True)

    top_regions = (
        bin_scope_summary.sort_values(["scope", "mean_variants_per_100mb"], ascending=[True, False])
        .groupby("scope", sort=False, observed=True)
        .head(10)
        .copy()
    )
    top_regions["rank_within_scope"] = top_regions.groupby("scope", sort=False).cumcount() + 1
    top_regions = top_regions[
        [
            "species",
            "dataset",
            "plot_label",
            "scope",
            "rank_within_scope",
            "chromosome",
            "region",
            "bin_size_bp",
            "n_stage_timing_donors",
            "n_unique_donor_variants",
            "mean_n_variants",
            "sem_n_variants",
            "mean_variants_per_100mb",
            "sem_variants_per_100mb",
        ]
    ].copy()

    return position_scope_summary, bin_scope_summary, top_regions, offsets


def plot_stage_distribution(stage_totals: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300, sharey=True)
    for ax, species in zip(axes, ["Mouse", "Human"]):
        sub = stage_totals[stage_totals["species"] == species].set_index("stage_label").reindex(STAGE_ORDER)
        vals = sub["percent_unique_donor_variants"].to_numpy(float)
        ax.bar(
            np.arange(len(STAGE_ORDER)),
            vals,
            color=[STAGE_COLORS[stage] for stage in STAGE_ORDER],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_title(species)
        ax.set_xticks(np.arange(len(STAGE_ORDER)))
        ax.set_xticklabels(STAGE_ORDER, rotation=25, ha="right")
        ax.set_ylabel("Unique donor-variant share (%)")
        ax.set_ylim(0, max(stage_totals["percent_unique_donor_variants"].max() * 1.15, 5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def plot_chromosome_heatmaps(chrom_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.8), dpi=300)
    species_chroms = canonical_autosomes_by_species()
    vmax = max(
        1.0,
        float(np.nanmax(np.abs(chrom_summary["log2_observed_over_expected"].to_numpy(float)))),
    )
    for ax, species in zip(axes, ["Mouse", "Human"]):
        sub = chrom_summary[chrom_summary["species"] == species].copy()
        pivot = (
            sub.pivot(index="scope", columns="chromosome", values="log2_observed_over_expected")
            .reindex(index=SCOPE_ORDER)
            .reindex(columns=species_chroms[species])
        )
        data = pivot.to_numpy(float)
        im = ax.imshow(data, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(f"{species}: autosomal chromosome enrichment by developmental stage")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Chromosome")
    cbar = fig.colorbar(im, ax=axes, shrink=0.92, pad=0.02)
    cbar.set_label("log2(observed / expected by chromosome length)")
    fig.subplots_adjust(left=0.07, right=0.9, top=0.93, bottom=0.08, hspace=0.38)
    plt.show()
    plt.close(fig)


def _plot_chromosome_position_strip(
    ax: plt.Axes,
    variant_df: pd.DataFrame,
    chromosomes: list[str],
    title: str,
    color: str,
    seed: int = 0,
) -> None:
    if variant_df.empty:
        ax.set_title(title)
        ax.set_xticks(np.arange(len(chromosomes)))
        ax.set_xticklabels([chrom.replace("chr", "") for chrom in chromosomes], rotation=90)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        return

    rng = np.random.default_rng(seed)
    x_base = variant_df["chromosome_order"].to_numpy(float)
    x_jitter = rng.uniform(-0.28, 0.28, size=len(variant_df))
    y = variant_df["position_percent"].to_numpy(float)

    ax.scatter(
        x_base + x_jitter,
        y,
        s=8,
        color=color,
        alpha=0.18,
        edgecolor="none",
        rasterized=True,
        zorder=2,
    )

    centerline = variant_df.groupby("chromosome", observed=True)["position_percent"].median().reindex(chromosomes)
    x_line = np.arange(len(chromosomes), dtype=float)
    ax.plot(x_line, centerline.to_numpy(float), color="black", linewidth=1.6, zorder=4)
    ax.scatter(
        x_line,
        centerline.to_numpy(float),
        s=18,
        facecolor="white",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )

    for x in np.arange(-0.5, len(chromosomes), 1.0):
        ax.axvline(x, color="#d9d9d9", linewidth=0.45, alpha=0.6, zorder=1)
    ax.axhline(50.0, color="#8c8c8c", linestyle="--", linewidth=0.8, alpha=0.8, zorder=1)

    ax.set_title(title)
    ax.set_xlabel("Autosomal chromosome")
    ax.set_xticks(np.arange(len(chromosomes)))
    ax.set_xticklabels([chrom.replace("chr", "") for chrom in chromosomes], rotation=90)
    ax.set_xlim(-0.6, len(chromosomes) - 0.4)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_genome_guides(ax: plt.Axes, offsets_df: pd.DataFrame) -> None:
    offsets_df = offsets_df.sort_values("chrom_offset_bp").reset_index(drop=True)
    for idx, row in offsets_df.iterrows():
        start = float(row["chrom_offset_bp"]) / 1_000_000.0
        end = start + float(row["chrom_length_bp"]) / 1_000_000.0
        if idx % 2 == 0:
            ax.axvspan(start, end, color="#000000", alpha=0.04, lw=0, zorder=0)
        ax.axvline(start, color="#bbbbbb", linewidth=0.5, alpha=0.8, zorder=1)
    last_row = offsets_df.iloc[-1]
    ax.axvline(
        (float(last_row["chrom_offset_bp"]) + float(last_row["chrom_length_bp"])) / 1_000_000.0,
        color="#bbbbbb",
        linewidth=0.5,
        alpha=0.8,
        zorder=1,
    )


def _set_genome_axis(ax: plt.Axes, offsets_df: pd.DataFrame) -> None:
    ax.set_xticks(offsets_df["chrom_center_mb"].to_numpy(float))
    ax.set_xticklabels(offsets_df["label"].tolist(), rotation=90)
    ax.set_xlim(0.0, float(offsets_df["chrom_center_mb"].iloc[-1] * 2.0))
    ax.set_xlabel("Autosomal genomic position (chromosomes in order)")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_genomewide_mean_density(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    offsets_df: pd.DataFrame,
    title: str,
    color: str,
) -> None:
    _add_genome_guides(ax, offsets_df)
    ax.set_title(title)
    _set_genome_axis(ax, offsets_df)
    if summary_df.empty:
        return

    x = summary_df["genome_mid_bp"].to_numpy(float) / 1_000_000.0
    y = summary_df["mean_variants_per_100mb"].to_numpy(float)
    sem = summary_df["sem_variants_per_100mb"].fillna(0).to_numpy(float)
    ax.fill_between(
        x,
        np.clip(y - sem, a_min=0, a_max=None),
        y + sem,
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.plot(x, y, color=color, linewidth=1.4, zorder=3)
    ax.scatter(x, y, s=10, color=color, alpha=0.6, edgecolor="white", linewidth=0.2, zorder=4)

    top_bins = summary_df.nlargest(min(5, len(summary_df)), "mean_variants_per_100mb")
    ax.scatter(
        top_bins["genome_mid_bp"].to_numpy(float) / 1_000_000.0,
        top_bins["mean_variants_per_100mb"].to_numpy(float),
        s=22,
        color="#c44e52",
        edgecolor="black",
        linewidth=0.35,
        zorder=5,
    )


def plot_scope_position_strips(position_scope_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        len(DATASET_SPECS),
        len(SCOPE_ORDER),
        figsize=(4.2 * len(SCOPE_ORDER), 3.8 * len(DATASET_SPECS)),
        dpi=300,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_idx, spec in enumerate(DATASET_SPECS):
        species = str(spec["species"])
        chromosomes = list(spec["canonical_autosomes"])
        species_df = position_scope_summary[position_scope_summary["species"] == species].copy()
        for col_idx, scope in enumerate(SCOPE_ORDER):
            ax = axes[row_idx, col_idx]
            sub = species_df[species_df["scope"] == scope].copy()
            title = f"{scope}\n(n={len(sub):,})"
            _plot_chromosome_position_strip(
                ax,
                sub,
                chromosomes,
                title=title,
                color=SCOPE_COLORS[scope],
                seed=1729 + row_idx * 101 + col_idx,
            )
            if col_idx == 0:
                ax.set_ylabel(f"{species}\nMutation position within chromosome (%)")
            else:
                ax.set_ylabel("")
    fig.suptitle(
        "Distribution of somatic mutation calls across autosomal chromosome positions by developmental stage",
        fontsize=14,
    )
    plt.show()
    plt.close(fig)


def plot_scope_hotspot_densities(
    bin_scope_summary: pd.DataFrame,
    offsets_by_species: dict[str, pd.DataFrame],
) -> None:
    fig, axes = plt.subplots(
        len(DATASET_SPECS),
        len(SCOPE_ORDER),
        figsize=(4.4 * len(SCOPE_ORDER), 3.8 * len(DATASET_SPECS)),
        dpi=300,
        sharey="row",
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row_idx, spec in enumerate(DATASET_SPECS):
        species = str(spec["species"])
        offsets = offsets_by_species[species]
        species_df = bin_scope_summary[bin_scope_summary["species"] == species].copy()
        for col_idx, scope in enumerate(SCOPE_ORDER):
            ax = axes[row_idx, col_idx]
            sub = species_df[species_df["scope"] == scope].sort_values(["chromosome", "bin_idx"]).copy()
            n_unique = int(sub["n_unique_donor_variants"].iloc[0]) if not sub.empty else 0
            title = f"{scope}\n(n={n_unique:,})"
            _plot_genomewide_mean_density(
                ax,
                sub,
                offsets,
                title=title,
                color=SCOPE_COLORS[scope],
            )
            if col_idx == 0:
                ax.set_ylabel(f"{species}\nMean unique donor-variants per donor per 100 Mb")
            else:
                ax.set_ylabel("")
    fig.suptitle(
        "Autosomal hotspot-region density of somatic mutation calls by developmental stage",
        fontsize=14,
    )
    plt.show()
    plt.close(fig)


def write_stage_hover_browser_html(
    hover_variant_summary: pd.DataFrame,
    hotspot_summary: pd.DataFrame,
    offsets_by_species: dict[str, pd.DataFrame],
    output_path: Path,
) -> Path:
    species_payload: dict[str, dict[str, object]] = {}
    for spec in DATASET_SPECS:
        species = str(spec["species"])
        species_hover = hover_variant_summary[hover_variant_summary["species"] == species].copy()
        offsets = offsets_by_species[species].sort_values("chrom_offset_bp").reset_index(drop=True)
        panel_payload: dict[str, list[dict[str, object]]] = {}
        for stage_idx, stage in enumerate(STAGE_ORDER):
            stage_hover = species_hover[species_hover["stage_label"] == stage].copy().reset_index(drop=True)
            if stage_hover.empty:
                panel_payload[stage] = []
                continue
            rng = np.random.default_rng(20260609 + stage_idx + (1000 if species == "Human" else 0))
            stage_hover["jitter"] = rng.uniform(-0.34, 0.34, size=len(stage_hover))
            stage_hover["alt"] = stage_hover["alt"].fillna("")
            stage_hover["ref"] = stage_hover["ref"].fillna("")
            panel_payload[stage] = [
                {
                    "x": round(float(row.genome_position_mb), 6),
                    "j": round(float(row.jitter), 6),
                    "chr": str(row.chromosome),
                    "pos": int(row.position_bp),
                    "var": str(row.var_id),
                    "donor": str(row.donor),
                    "cells": int(row.n_supporting_cells),
                    "cell_types": str(row.cell_types),
                    "tissues": str(row.tissues),
                    "germ_layers": str(row.germ_layers),
                    "ref": str(row.ref),
                    "alt": str(row.alt),
                }
                for row in stage_hover.itertuples(index=False)
            ]

        hotspot_payload: dict[str, dict[str, object]] = {}
        species_hotspots = hotspot_summary[
            (hotspot_summary["species"] == species) & (hotspot_summary["rank_within_stage_window_size"] == 1)
        ].copy()
        for stage in STAGE_ORDER:
            stage_hotspots = species_hotspots[species_hotspots["stage_label"] == stage].copy()
            window_payload: dict[str, object] = {}
            for row in stage_hotspots.itertuples(index=False):
                start_mb = (
                    float(
                        offsets.loc[offsets["chromosome"] == row.chromosome, "chrom_offset_bp"].iloc[0]
                        + row.window_start_bp
                    )
                    / 1_000_000.0
                )
                end_mb = (
                    float(
                        offsets.loc[offsets["chromosome"] == row.chromosome, "chrom_offset_bp"].iloc[0]
                        + row.window_end_bp
                    )
                    / 1_000_000.0
                )
                window_payload[str(int(row.window_size_bp))] = {
                    "start_mb": round(start_mb, 6),
                    "end_mb": round(end_mb, 6),
                    "region": str(row.region),
                    "n_cells": int(row.n_unique_cells),
                    "expected_same_site_rate": float(row.expected_shared_exact_sites_per_random_cell_pair_from_window),
                }
            hotspot_payload[stage] = window_payload

        species_payload[species] = {
            "plot_label": str(spec["plot_label"]),
            "genome_max_mb": round(
                float((offsets["chrom_offset_bp"] + offsets["chrom_length_bp"]).max()) / 1_000_000.0,
                6,
            ),
            "chromosomes": [
                {
                    "label": str(row.label),
                    "start_mb": round(float(row.chrom_offset_bp) / 1_000_000.0, 6),
                    "end_mb": round(
                        float(row.chrom_offset_bp + row.chrom_length_bp) / 1_000_000.0,
                        6,
                    ),
                    "center_mb": round(float(row.chrom_center_mb), 6),
                }
                for row in offsets.itertuples(index=False)
            ],
            "panels": panel_payload,
            "hotspots": hotspot_payload,
        }

    payload = {
        "speciesOrder": [str(spec["species"]) for spec in DATASET_SPECS],
        "stageOrder": STAGE_ORDER,
        "species": species_payload,
    }
    payload_json = json.dumps(payload, separators=(",", ":"))
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Stage-by-chromosome mutation browser</title>
<style>
body {{
  margin: 0;
  padding: 24px;
  font-family: Arial, sans-serif;
  color: #202124;
  background: #fafafa;
}}
h1 {{
  margin: 0 0 10px 0;
  font-size: 24px;
}}
h2 {{
  margin: 28px 0 8px 0;
  font-size: 20px;
}}
p {{
  max-width: 1100px;
  line-height: 1.45;
}}
.panel {{
  margin: 0 0 16px 0;
  border: 1px solid #d9d9d9;
  background: white;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}}
.panel-header {{
  display: flex;
  justify-content: space-between;
  gap: 16px;
  align-items: baseline;
  padding: 8px 12px 0 12px;
  font-size: 13px;
}}
.panel-title {{
  font-weight: 600;
}}
.panel-note {{
  color: #666;
}}
canvas {{
  display: block;
  width: 100%;
  height: 170px;
}}
#tooltip {{
  position: fixed;
  display: none;
  pointer-events: none;
  z-index: 1000;
  max-width: 340px;
  padding: 8px 10px;
  border-radius: 6px;
  background: rgba(24, 24, 24, 0.94);
  color: white;
  font-size: 12px;
  line-height: 1.4;
  box-shadow: 0 8px 20px rgba(0,0,0,0.18);
}}
.tooltip-title {{
  font-weight: 700;
  margin-bottom: 4px;
}}
.legend {{
  display: flex;
  gap: 18px;
  font-size: 13px;
  margin: 12px 0 18px 0;
  color: #555;
}}
.legend span::before {{
  content: "";
  display: inline-block;
  width: 12px;
  height: 12px;
  margin-right: 6px;
  vertical-align: -2px;
}}
.legend .w1000::before {{
  background: rgba(196, 78, 82, 0.18);
  border: 1px solid rgba(196, 78, 82, 0.8);
}}
.legend .w100::before {{
  background: #dd8452;
}}
</style>
</head>
<body>
<h1>Stage-by-chromosome mutation browser</h1>
<p>Each point is one unique donor-variant call on the canonical autosomes, positioned along concatenated chromosomes and grouped by developmental stage. Hover to inspect the exact mutation, donor, and the number of supporting cells. The red translucent span marks the top 1 kb hotspot window for that stage; the orange line marks the top 100 bp hotspot window.</p>
<div class="legend">
  <span class="w1000">Top 1 kb hotspot</span>
  <span class="w100">Top 100 bp hotspot center</span>
</div>
<div id="app"></div>
<div id="tooltip"></div>
<script>
const DATA = {payload_json};
const STAGE_COLORS = {{
  "Embryoblast": "#f0c4a6",
  "Germ layer-specific": "#e39a58",
  "Tissue-specific": "#c96f1c",
  "Adult-specific": "#7e4314"
}};
const tooltip = document.getElementById("tooltip");

function fmtProb(value) {{
  if (!Number.isFinite(value)) return "NA";
  if (value === 0) return "0";
  if (value >= 0.001) return value.toFixed(4);
  return value.toExponential(2);
}}

function showTooltip(event, point) {{
  const altLabel = point.alt ? `${{point.ref}}>${{point.alt}}` : point.var.split("_").slice(1).join("_");
  tooltip.innerHTML = `
    <div class="tooltip-title">${{point.var}}</div>
    <div><strong>Donor:</strong> ${{point.donor}}</div>
    <div><strong>Coordinate:</strong> ${{point.chr}}:${{point.pos.toLocaleString()}}</div>
    <div><strong>Base change:</strong> ${{altLabel || "NA"}}</div>
    <div><strong>Supporting cells:</strong> ${{point.cells.toLocaleString()}}</div>
    <div><strong>Cell types:</strong> ${{point.cell_types || "NA"}}</div>
    <div><strong>Tissues:</strong> ${{point.tissues || "NA"}}</div>
    <div><strong>Germ layers:</strong> ${{point.germ_layers || "NA"}}</div>
  `;
  tooltip.style.display = "block";
  tooltip.style.left = `${{event.clientX + 14}}px`;
  tooltip.style.top = `${{event.clientY + 14}}px`;
}}

function hideTooltip() {{
  tooltip.style.display = "none";
}}

function drawPanel(canvas, speciesData, stage, points, hotspots) {{
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 1200;
  const cssHeight = canvas.clientHeight || 170;
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const margin = {{left: 56, right: 18, top: 16, bottom: 30}};
  const plotW = cssWidth - margin.left - margin.right;
  const plotH = cssHeight - margin.top - margin.bottom;
  const xMax = speciesData.genome_max_mb;
  const color = STAGE_COLORS[stage] || "#4c78a8";

  function xToPx(value) {{
    return margin.left + (value / xMax) * plotW;
  }}
  function yToPx(value) {{
    return margin.top + (0.5 - value / 2) * plotH;
  }}

  const laidOut = points.map((point) => ({{
    ...point,
    sx: xToPx(point.x),
    sy: yToPx(point.j),
  }}));

  function renderScene(highlightIdx) {{
    ctx.clearRect(0, 0, cssWidth, cssHeight);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(margin.left, margin.top, plotW, plotH);
    ctx.strokeStyle = "#d0d0d0";
    ctx.lineWidth = 1;
    ctx.strokeRect(margin.left, margin.top, plotW, plotH);

    speciesData.chromosomes.forEach((chrom, idx) => {{
      const startX = xToPx(chrom.start_mb);
      const endX = xToPx(chrom.end_mb);
      if (idx % 2 === 0) {{
        ctx.fillStyle = "rgba(0,0,0,0.025)";
        ctx.fillRect(startX, margin.top, endX - startX, plotH);
      }}
      ctx.strokeStyle = "#d9d9d9";
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(startX, margin.top);
      ctx.lineTo(startX, margin.top + plotH);
      ctx.stroke();
    }});

    const hotspot1k = hotspots["1000"];
    if (hotspot1k) {{
      ctx.fillStyle = "rgba(196, 78, 82, 0.18)";
      ctx.strokeStyle = "rgba(196, 78, 82, 0.78)";
      const x0 = xToPx(hotspot1k.start_mb);
      const x1 = xToPx(hotspot1k.end_mb);
      ctx.fillRect(x0, margin.top, Math.max(2, x1 - x0), plotH);
      ctx.strokeRect(x0, margin.top, Math.max(2, x1 - x0), plotH);
    }}
    const hotspot100 = hotspots["100"];
    if (hotspot100) {{
      const centerX = xToPx((hotspot100.start_mb + hotspot100.end_mb) / 2);
      ctx.strokeStyle = "#dd8452";
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      ctx.moveTo(centerX, margin.top);
      ctx.lineTo(centerX, margin.top + plotH);
      ctx.stroke();
    }}

    ctx.strokeStyle = "#8c8c8c";
    ctx.lineWidth = 0.9;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(margin.left, margin.top + plotH / 2);
    ctx.lineTo(margin.left + plotW, margin.top + plotH / 2);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = color;
    ctx.globalAlpha = 0.22;
    laidOut.forEach((point) => {{
      ctx.beginPath();
      ctx.arc(point.sx, point.sy, 2.4, 0, Math.PI * 2);
      ctx.fill();
    }});
    ctx.globalAlpha = 1.0;

    if (highlightIdx >= 0) {{
      const point = laidOut[highlightIdx];
      ctx.fillStyle = "#000000";
      ctx.beginPath();
      ctx.arc(point.sx, point.sy, 4.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(point.sx, point.sy, 2.2, 0, Math.PI * 2);
      ctx.fill();
    }}

    ctx.fillStyle = "#444";
    ctx.font = "11px Arial, sans-serif";
    ctx.textAlign = "center";
    speciesData.chromosomes.forEach((chrom) => {{
      ctx.fillText(chrom.label, xToPx(chrom.center_mb), cssHeight - 9);
    }});
  }}

  renderScene(-1);

  canvas.onmousemove = (event) => {{
    const rect = canvas.getBoundingClientRect();
    const mx = event.clientX - rect.left;
    const my = event.clientY - rect.top;
    let bestIdx = -1;
    let bestDist = 49;
    for (let i = 0; i < laidOut.length; i += 1) {{
      const dx = laidOut[i].sx - mx;
      const dy = laidOut[i].sy - my;
      const dist = dx * dx + dy * dy;
      if (dist < bestDist) {{
        bestDist = dist;
        bestIdx = i;
      }}
    }}
    renderScene(bestIdx);
    if (bestIdx >= 0) {{
      showTooltip(event, laidOut[bestIdx]);
    }} else {{
      hideTooltip();
    }}
  }};

  canvas.onmouseleave = () => {{
    hideTooltip();
    renderScene(-1);
  }};
}}

function buildApp() {{
  const app = document.getElementById("app");
  DATA.speciesOrder.forEach((speciesName) => {{
    const speciesData = DATA.species[speciesName];
    const h2 = document.createElement("h2");
    h2.textContent = speciesData.plot_label;
    app.appendChild(h2);
    DATA.stageOrder.forEach((stage) => {{
      const panel = document.createElement("div");
      panel.className = "panel";
      const header = document.createElement("div");
      header.className = "panel-header";
      const title = document.createElement("div");
      title.className = "panel-title";
      title.textContent = stage;
      const hotspot100 = speciesData.hotspots[stage]["100"];
      const hotspot1000 = speciesData.hotspots[stage]["1000"];
      const note = document.createElement("div");
      note.className = "panel-note";
      const bits = [];
      if (hotspot1000) {{
        bits.push(`top 1 kb: ${{hotspot1000.region}} (expected same-site rate from window = ${{fmtProb(hotspot1000.expected_same_site_rate)}})`);
      }}
      if (hotspot100) {{
        bits.push(`top 100 bp: ${{hotspot100.region}}`);
      }}
      note.textContent = bits.join(" | ");
      header.appendChild(title);
      header.appendChild(note);
      panel.appendChild(header);
      const canvas = document.createElement("canvas");
      panel.appendChild(canvas);
      app.appendChild(panel);
      drawPanel(canvas, speciesData, stage, speciesData.panels[stage], speciesData.hotspots[stage]);
    }});
  }});
}}

buildApp();
</script>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_overlap_collision_analysis(write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    if write_outputs:
        ensure_output_dir()

    overlap_frames = []
    hotspot_frames = []
    hover_frames = []
    donor_clustering_frames = []
    clustering_summary_frames = []
    offsets_by_species: dict[str, pd.DataFrame] = {}

    for spec in DATASET_SPECS:
        overlap_df, hotspot_df, hover_df, offsets = summarize_stage_overlap_and_hotspots(spec)
        donor_clustering_df, clustering_summary_df = summarize_donor_stage_same_window_clustering(spec)
        overlap_frames.append(overlap_df)
        hotspot_frames.append(hotspot_df)
        hover_frames.append(hover_df)
        donor_clustering_frames.append(donor_clustering_df)
        clustering_summary_frames.append(clustering_summary_df)
        offsets_by_species[str(spec["species"])] = offsets

    overlap_summary = pd.concat(overlap_frames, ignore_index=True)
    hotspot_summary = pd.concat(hotspot_frames, ignore_index=True)
    hover_variant_summary = pd.concat(hover_frames, ignore_index=True)
    donor_clustering_summary = pd.concat(donor_clustering_frames, ignore_index=True)
    clustering_summary = pd.concat(clustering_summary_frames, ignore_index=True)
    donor_two_germ_layer_hotspots = build_donor_two_germ_layer_hotspot_table(donor_clustering_summary)
    two_germ_layer_hotspot_summary = build_two_germ_layer_hotspot_probability_table(clustering_summary)
    top_window_summary = hotspot_summary[hotspot_summary["rank_within_stage_window_size"] == 1].copy()
    html_path = OUT_DIR / "autosomal_stage_hover_browser.html"

    if write_outputs:
        overlap_summary.to_csv(OUT_DIR / "stage_overlap_metrics.csv", index=False)
        hotspot_summary.to_csv(OUT_DIR / "stage_hotspot_top_windows.csv", index=False)
        top_window_summary.to_csv(OUT_DIR / "stage_hotspot_artifact_summary.csv", index=False)
        donor_clustering_summary.to_csv(
            OUT_DIR / "donor_level_stage_specific_same_window_clustering_summary.csv",
            index=False,
        )
        clustering_summary.to_csv(OUT_DIR / "stage_specific_same_window_clustering_summary.csv", index=False)
        donor_two_germ_layer_hotspots.to_csv(
            OUT_DIR / "donor_level_stage_specific_two_germ_layer_hotspot_windows.csv",
            index=False,
        )
        two_germ_layer_hotspot_summary.to_csv(
            OUT_DIR / "stage_specific_two_germ_layer_hotspot_probability_summary.csv",
            index=False,
        )
        hover_variant_summary.to_csv(OUT_DIR / "autosomal_stage_hover_variant_summary.csv", index=False)
        write_stage_hover_browser_html(
            hover_variant_summary,
            hotspot_summary,
            offsets_by_species,
            html_path,
        )

    return overlap_summary, hotspot_summary, hover_variant_summary, html_path


def run_overlap_artifact_analysis(write_outputs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    return run_overlap_collision_analysis(write_outputs=write_outputs)


def main() -> None:
    ensure_output_dir()

    stage_frames = []
    chrom_frames = []
    test_frames = []
    position_frames = []
    bin_frames = []
    top_region_frames = []
    offsets_by_species: dict[str, pd.DataFrame] = {}
    overlap_frames = []
    hotspot_frames = []
    hover_frames = []
    donor_clustering_frames = []
    clustering_summary_frames = []

    for spec in DATASET_SPECS:
        stage_df, chrom_df, test_df = summarize_dataset(spec)
        position_df, bin_df, top_region_df, offsets = summarize_position_distributions(spec)
        overlap_df, hotspot_df, hover_df, _ = summarize_stage_overlap_and_hotspots(spec)
        donor_clustering_df, clustering_summary_df = summarize_donor_stage_same_window_clustering(spec)
        stage_frames.append(stage_df)
        chrom_frames.append(chrom_df)
        test_frames.append(test_df)
        position_frames.append(position_df)
        bin_frames.append(bin_df)
        top_region_frames.append(top_region_df)
        overlap_frames.append(overlap_df)
        hotspot_frames.append(hotspot_df)
        hover_frames.append(hover_df)
        donor_clustering_frames.append(donor_clustering_df)
        clustering_summary_frames.append(clustering_summary_df)
        offsets_by_species[str(spec["species"])] = offsets

    stage_totals = pd.concat(stage_frames, ignore_index=True)
    chrom_summary = pd.concat(chrom_frames, ignore_index=True)
    random_tests = pd.concat(test_frames, ignore_index=True)
    position_scope_summary = pd.concat(position_frames, ignore_index=True)
    bin_scope_summary = pd.concat(bin_frames, ignore_index=True)
    top_region_summary = pd.concat(top_region_frames, ignore_index=True)
    overlap_summary = pd.concat(overlap_frames, ignore_index=True)
    hotspot_summary = pd.concat(hotspot_frames, ignore_index=True)
    hover_variant_summary = pd.concat(hover_frames, ignore_index=True)
    donor_clustering_summary = pd.concat(donor_clustering_frames, ignore_index=True)
    clustering_summary = pd.concat(clustering_summary_frames, ignore_index=True)
    donor_two_germ_layer_hotspots = build_donor_two_germ_layer_hotspot_table(donor_clustering_summary)
    two_germ_layer_hotspot_summary = build_two_germ_layer_hotspot_probability_table(clustering_summary)
    top_window_summary = hotspot_summary[hotspot_summary["rank_within_stage_window_size"] == 1].copy()
    html_path = write_stage_hover_browser_html(
        hover_variant_summary,
        hotspot_summary,
        offsets_by_species,
        OUT_DIR / "autosomal_stage_hover_browser.html",
    )

    stage_totals.to_csv(OUT_DIR / "stage_variant_totals.csv", index=False)
    chrom_summary.to_csv(OUT_DIR / "chromosome_stage_distribution.csv", index=False)
    random_tests.to_csv(OUT_DIR / "chromosome_random_chance_tests.csv", index=False)
    position_scope_summary.to_csv(OUT_DIR / "autosomal_position_distribution_by_scope.csv", index=False)
    bin_scope_summary.to_csv(OUT_DIR / "autosomal_hotspot_bin_summary_by_scope.csv", index=False)
    top_region_summary.to_csv(OUT_DIR / "autosomal_hotspot_top_regions_by_scope.csv", index=False)
    overlap_summary.to_csv(OUT_DIR / "stage_overlap_metrics.csv", index=False)
    hotspot_summary.to_csv(OUT_DIR / "stage_hotspot_top_windows.csv", index=False)
    top_window_summary.to_csv(OUT_DIR / "stage_hotspot_artifact_summary.csv", index=False)
    donor_clustering_summary.to_csv(
        OUT_DIR / "donor_level_stage_specific_same_window_clustering_summary.csv",
        index=False,
    )
    clustering_summary.to_csv(OUT_DIR / "stage_specific_same_window_clustering_summary.csv", index=False)
    donor_two_germ_layer_hotspots.to_csv(
        OUT_DIR / "donor_level_stage_specific_two_germ_layer_hotspot_windows.csv",
        index=False,
    )
    two_germ_layer_hotspot_summary.to_csv(
        OUT_DIR / "stage_specific_two_germ_layer_hotspot_probability_summary.csv",
        index=False,
    )
    hover_variant_summary.to_csv(OUT_DIR / "autosomal_stage_hover_variant_summary.csv", index=False)

    plot_stage_distribution(stage_totals)
    plot_chromosome_heatmaps(chrom_summary)
    plot_scope_position_strips(position_scope_summary)
    plot_scope_hotspot_densities(bin_scope_summary, offsets_by_species)

    for _, row in random_tests.iterrows():
        print(
            f"[INFO] {row['species']} {row['scope']}: "
            f"n={int(row['n_unique_donor_variants_tested']):,}, "
            f"chi2={row['chi_square_statistic']:.2f}, "
            f"MonteCarloP={row['monte_carlo_pvalue']:.4g}"
        )
    print(f"[OK] Wrote {OUT_DIR / 'stage_variant_totals.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'chromosome_stage_distribution.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'chromosome_random_chance_tests.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'autosomal_position_distribution_by_scope.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'autosomal_hotspot_bin_summary_by_scope.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'autosomal_hotspot_top_regions_by_scope.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'stage_overlap_metrics.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'stage_hotspot_top_windows.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'stage_hotspot_artifact_summary.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'donor_level_stage_specific_same_window_clustering_summary.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'stage_specific_same_window_clustering_summary.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'donor_level_stage_specific_two_germ_layer_hotspot_windows.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'stage_specific_two_germ_layer_hotspot_probability_summary.csv'}")
    print(f"[OK] Wrote {OUT_DIR / 'autosomal_stage_hover_variant_summary.csv'}")
    print(f"[OK] Wrote {html_path}")


if __name__ == "__main__":
    main()
