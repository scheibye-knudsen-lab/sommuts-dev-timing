#!/usr/bin/env python3
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "demo_data"
ROOT = DATA_DIR / "merged_outputs_demo"
CB_TABLE = DATA_DIR / "cell_metadata_table_1-M-62.csv"
REF_WEIGHTS_DIR = DATA_DIR / "reference_weights_cells"
REF_STAGE_LABELS = DATA_DIR / "reference_stage_labels_1-M-62.csv"
OUT_DIR = BASE_DIR / "demo_results"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ORDERED_LABELS = [
    "Zygote",
    "Pre-gastrulation",
    "Post-gastrulation",
    "Tissue-specific",
    "Cell type-specific",
]
WEIGHT_BY = "cells"
DONOR_FILTER = {"1-M-62"}

GENO_RE = re.compile(r"^(?P<celltype>.+?)\.single_cell_genotype(?:\s*\d+)?\.tsv$", re.IGNORECASE)
SITES_RE = re.compile(r"^(?P<donor>[^\.]+)\.(?P<celltype>.+?)\.SitesPerCell\.tsv$", re.IGNORECASE)


def detect_sep(p: Path) -> str:
    with p.open("r", errors="ignore") as f:
        line = f.readline()
    return "\t" if ("\t" in line and line.count("\t") >= line.count(",")) else ","


def pick_sites_col(cols):
    def norm(c):
        return re.sub(r"[^a-z0-9]", "", str(c).lower())
    normed = {norm(c): c for c in cols}
    for key in [
        "sitespercell",
        "sites",
        "callablesites",
        "numcallablesites",
        "numsites",
        "sitecount",
        "sitespercellrounded",
    ]:
        if key in normed:
            return normed[key]
    for c in cols:
        if "site" in str(c).lower():
            return c
    return None


def read_sites_per_cell(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep=detect_sep(fp), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "CB" not in df.columns and "cell_barcode" in df.columns:
        df = df.rename(columns={"cell_barcode": "CB"})
    col_sites = pick_sites_col(df.columns)
    if col_sites is None:
        raise ValueError(f"No callable-sites column detected in {fp}")
    out = df[["CB", col_sites]].copy()
    out["CB"] = out["CB"].astype(str).str.strip().str.upper()
    out = out.rename(columns={col_sites: "SitesPerCell"})
    return out.groupby("CB", as_index=False)["SitesPerCell"].max()


def read_genotype(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep=detect_sep(fp), low_memory=False)
    if "CB" not in df.columns and "cell_barcode" in df.columns:
        df = df.rename(columns={"cell_barcode": "CB"})
    df["CB"] = df["CB"].astype(str).str.strip().str.upper()
    return df


def parse_celltype(name: str) -> str:
    m = GENO_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized genotype filename: {name}")
    return m.group("celltype")


def build_sites_index(root: Path):
    idx = {}
    for p in root.rglob("*.SitesPerCell.tsv"):
        m = SITES_RE.match(p.name)
        if m:
            idx[(m.group("donor"), m.group("celltype"))] = p
    return idx


def bad(s: pd.Series):
    ss = s.astype(str).str.strip()
    return ss.eq("") | ss.str.lower().isin({"na", "unknown", "nan", "none"})


def mode_one(x: pd.Series):
    x = x.dropna()
    m = x.mode(dropna=False)
    return m.iat[0] if not m.empty else np.nan


def classify_stage(n_layers, n_tissues, n_cells):
    if n_layers > 1:
        return "Pre-gastrulation"
    if n_tissues > 1:
        return "Post-gastrulation"
    if n_cells > 1:
        return "Tissue-specific"
    return "Cell type-specific"


def parse_mouse_age_months(donor_id: str):
    m = re.match(r"^(?P<age>\d+)-", str(donor_id))
    return int(m.group("age")) if m else None


def load_reference_weights(weights_dir: Path):
    w_gl = pd.read_csv(weights_dir / "weights_germ_layer.csv").set_index("germ_layer")["weight"]
    w_tissue_gl = (
        pd.read_csv(weights_dir / "weights_tissue_within_germ_layer.csv")
        .set_index(["germ_layer", "tissue"])["weight"]
    )
    w_cell_tissue = (
        pd.read_csv(weights_dir / "weights_celltype_within_tissue.csv")
        .set_index(["tissue", "cell_type_stage"])["weight"]
    )
    return w_gl, w_tissue_gl, w_cell_tissue


def build_cb_map(cb_table: Path) -> pd.DataFrame:
    m = pd.read_csv(cb_table, usecols=["donor", "CB", "tissue", "germ_layer"])
    m["CB"] = m["CB"].astype(str).str.strip().str.upper()
    m = m[~bad(m["tissue"]) & ~bad(m["germ_layer"])].copy()
    return (
        m.groupby(["donor", "CB"])
        .agg(tissue=("tissue", mode_one), germ_layer=("germ_layer", mode_one))
        .reset_index()
    )


def derive_keep_donors(cb_map: pd.DataFrame, min_layers=2, min_tissues=2):
    x = cb_map.loc[
        ~bad(cb_map["germ_layer"]) & ~bad(cb_map["tissue"]),
        ["donor", "germ_layer", "tissue"],
    ]
    summary = (
        x.groupby("donor")
        .agg(n_layers=("germ_layer", "nunique"), n_tissues=("tissue", "nunique"))
        .sort_index()
    )
    keep = set(summary.index[(summary["n_layers"] >= min_layers) & (summary["n_tissues"] >= min_tissues)])
    return keep


def build_cb_level(df: pd.DataFrame) -> pd.DataFrame:
    exp_cb = (
        df[["donor", "CB", "SitesPerCell"]]
        .dropna(subset=["SitesPerCell"])
        .groupby(["donor", "CB"], as_index=False)["SitesPerCell"]
        .max()
    )
    attrs = (
        df.groupby(["donor", "CB"], as_index=False)
        .agg(
            germ_layer=("germ_layer", mode_one),
            tissue=("tissue", mode_one),
            cell_type_stage=("cell_type_stage", mode_one),
        )
    )
    cb_full = exp_cb.merge(attrs, on=["donor", "CB"], how="left")
    cb_full = cb_full[~bad(cb_full["germ_layer"]) & ~bad(cb_full["tissue"])].copy()
    cb_full["cell_type_stage"] = cb_full["cell_type_stage"].astype(str).str.strip()
    return cb_full


def compute_reference_weights(cb_full: pd.DataFrame, keep_donors: set, weight_by="exposure"):
    x = cb_full[cb_full["donor"].isin(keep_donors)].copy()
    if x.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    if weight_by == "exposure":
        x["_w"] = x["SitesPerCell"].astype(float)
    elif weight_by == "cells":
        x["_w"] = 1.0
    else:
        raise ValueError("weight_by must be 'exposure' or 'cells'")
    w_gl = x.groupby("germ_layer")["_w"].sum()
    w_gl = w_gl / w_gl.sum() if w_gl.sum() > 0 else w_gl
    w_tissue_gl = x.groupby(["germ_layer", "tissue"])["_w"].sum()
    gl_sums = w_tissue_gl.groupby(level=0).sum()
    if not w_tissue_gl.empty:
        w_tissue_gl = w_tissue_gl / gl_sums.reindex(w_tissue_gl.index.get_level_values(0)).values
    w_tissue_gl = w_tissue_gl.fillna(0.0)
    w_cell_tissue = x.groupby(["tissue", "cell_type_stage"])["_w"].sum()
    t_sums = w_cell_tissue.groupby(level=0).sum()
    if not w_cell_tissue.empty:
        w_cell_tissue = w_cell_tissue / t_sums.reindex(w_cell_tissue.index.get_level_values(0)).values
    w_cell_tissue = w_cell_tissue.fillna(0.0)
    return w_gl, w_tissue_gl, w_cell_tissue


def standardize_all_stages_3level(
    df,
    mut,
    keep_donors,
    ordered_labels,
    weight_by="exposure",
    ref_weights=None,
):
    cb_full = build_cb_level(df)
    if ref_weights is None:
        w_gl, w_tissue_gl, w_cell_tissue = compute_reference_weights(cb_full, keep_donors, weight_by=weight_by)
    else:
        w_gl, w_tissue_gl, w_cell_tissue = ref_weights
    if cb_full.empty:
        return pd.DataFrame(), pd.DataFrame()

    den = cb_full.groupby(["donor", "germ_layer", "tissue", "cell_type_stage"])["SitesPerCell"].sum()
    if den.empty:
        return pd.DataFrame(), pd.DataFrame()

    num = mut.groupby(["donor", "germ_layer", "tissue", "cell_type_stage", "stage_label"]).size()
    num = num.unstack("stage_label", fill_value=0)
    for c in ordered_labels:
        if c not in num.columns:
            num[c] = 0
    num = num[ordered_labels]

    rates = {}
    den_safe = den.replace(0, np.nan)
    for stage in ordered_labels:
        r = (num[stage] / den_safe).fillna(0.0) * 1000.0
        rates[stage] = r

    donors = sorted(set(den.index.get_level_values("donor")))
    std_rows = {}

    for d in donors:
        den_d = den[den.index.get_level_values("donor") == d]
        if den_d.empty:
            continue
        gl_sums = den_d.groupby(level="germ_layer").sum()
        present_gl = gl_sums[gl_sums > 0].index.tolist()
        if not present_gl:
            continue
        w_gl_d = w_gl.reindex(present_gl).fillna(0.0)
        if w_gl_d.sum() > 0:
            w_gl_d = w_gl_d / w_gl_d.sum()
        acc = {stage: 0.0 for stage in ordered_labels}

        for gl in present_gl:
            den_d_gl = den_d[den_d.index.get_level_values("germ_layer") == gl]
            t_sums = den_d_gl.groupby(level="tissue").sum()
            present_t = t_sums[t_sums > 0].index.tolist()
            if not present_t:
                continue
            w_t_gl = (
                w_tissue_gl.loc[gl].reindex(present_t).fillna(0.0)
                if gl in w_tissue_gl.index.get_level_values(0)
                else pd.Series(0.0, index=present_t)
            )
            if w_t_gl.sum() > 0:
                w_t_gl = w_t_gl / w_t_gl.sum()

            for t in present_t:
                den_d_gl_t = den_d_gl[den_d_gl.index.get_level_values("tissue") == t]
                c_sums = den_d_gl_t.groupby(level="cell_type_stage").sum()
                present_ct = c_sums[c_sums > 0].index.tolist()
                if not present_ct:
                    continue
                w_c_t = (
                    w_cell_tissue.loc[t].reindex(present_ct).fillna(0.0)
                    if t in w_cell_tissue.index.get_level_values(0)
                    else pd.Series(0.0, index=present_ct)
                )
                if w_c_t.sum() > 0:
                    w_c_t = w_c_t / w_c_t.sum()
                wt_gl = float(w_gl_d.get(gl, 0.0))
                for ct in present_ct:
                    wt = wt_gl * float(w_t_gl.get(t, 0.0)) * float(w_c_t.get(ct, 0.0))
                    if wt == 0.0:
                        continue
                    idx = (d, gl, t, ct)
                    for stage in ordered_labels:
                        val = rates[stage].get(idx, 0.0)
                        acc[stage] += wt * float(val)
        if any(v != 0.0 for v in acc.values()):
            std_rows[d] = [acc[s] for s in ordered_labels]

    std_per_kb = pd.DataFrame.from_dict(std_rows, orient="index", columns=ordered_labels).sort_index()
    std_per_kb = std_per_kb.fillna(0.0)
    std_per_kb_cum = std_per_kb.cumsum(axis=1)
    return std_per_kb, std_per_kb_cum


def main():
    sites_idx = build_sites_index(ROOT)
    cb_map = build_cb_map(CB_TABLE)
    donors_present = {d.name for d in ROOT.iterdir() if d.is_dir()}
    keep_donors = derive_keep_donors(cb_map, min_layers=2, min_tissues=2) & donors_present
    if not keep_donors:
        raise SystemExit("No donors kept after filters.")

    rows = []
    for donor_dir in sorted([d for d in ROOT.iterdir() if d.is_dir()]):
        donor = donor_dir.name
        if donor not in keep_donors:
            continue
        gfiles = sorted(donor_dir.glob("*.single_cell_genotype*.tsv"))
        for gfp in gfiles:
            ct = parse_celltype(gfp.name)
            g = read_genotype(gfp)
            spath = sites_idx.get((donor, ct))
            if not spath or not spath.exists():
                continue
            s = read_sites_per_cell(spath)

            m = g.merge(s, on="CB", how="inner")
            if m.empty:
                continue

            need = {"ALT_expected", "Base_observed", "#CHROM", "Start", "REF"}
            if not need.issubset(m.columns):
                continue

            m["is_mutated"] = (
                m["Base_observed"].astype(str) == m["ALT_expected"].astype(str)
            ).astype(int)

            if "donor" not in m.columns:
                m.insert(0, "donor", donor)
            else:
                col = m.pop("donor")
                m.insert(0, "donor", col)

            if "cell_type_parsed" not in m.columns:
                m.insert(1, "cell_type_parsed", ct)

            m["var_id"] = (
                m["#CHROM"].astype(str) + ":" + m["Start"].astype(str) + "_" +
                m["REF"].astype(str) + ">" + m["ALT_expected"].astype(str)
            )

            if "Cell_type_observed" in m.columns and m["Cell_type_observed"].notna().any():
                m["cell_type_stage"] = m["Cell_type_observed"].astype(str)
            else:
                m["cell_type_stage"] = m["cell_type_parsed"].astype(str)

            if "tissue" not in m.columns or "germ_layer" not in m.columns:
                m = m.merge(cb_map, on=["donor", "CB"], how="left")

            rows.append(m)

    if not rows:
        raise SystemExit("No genotype/SitesPerCell merges produced rows.")

    df = pd.concat(rows, ignore_index=True)
    df = df[df["donor"].isin(keep_donors)]
    df = df[~bad(df["tissue"]) & ~bad(df["germ_layer"])].copy()
    if df.empty:
        raise SystemExit("No rows after tissue/germ layer filters.")

    mut = df[df["is_mutated"] == 1].copy()
    mut = mut.drop_duplicates(subset=["donor", "CB", "var_id"])
    if mut.empty:
        raise SystemExit("No mutated instances after filters.")

    if REF_STAGE_LABELS.exists():
        stage = pd.read_csv(REF_STAGE_LABELS)
        mut = mut.merge(stage, on="var_id", how="left")
    else:
        base = (
            mut.groupby("var_id")
            .agg(
                donor=("donor", "first"),
                n_tissues=("tissue", "nunique"),
                n_cells=("cell_type_stage", "nunique"),
            )
            .reset_index()
        )

        unique = mut.drop_duplicates(subset=["var_id", "cell_type_stage", "germ_layer"])
        layer_counts = (
            unique.groupby("var_id")["germ_layer"]
            .nunique()
            .rename("n_layers")
            .reset_index()
        )

        stage = base.merge(layer_counts, on="var_id", how="left")
        stage["stage_label"] = stage.apply(
            lambda r: classify_stage(r["n_layers"], r["n_tissues"], r["n_cells"]),
            axis=1,
        )

        mut = mut.merge(stage[["donor", "var_id", "stage_label"]], on=["donor", "var_id"], how="left")

    ref_weights = load_reference_weights(REF_WEIGHTS_DIR) if REF_WEIGHTS_DIR.exists() else None
    std_per_kb, std_per_kb_cum = standardize_all_stages_3level(
        df, mut, keep_donors, ORDERED_LABELS, weight_by=WEIGHT_BY, ref_weights=ref_weights
    )

    tag = "exposure" if WEIGHT_BY == "exposure" else "cells"
    if std_per_kb_cum.empty:
        raise SystemExit("Standardized table is empty.")

    if DONOR_FILTER:
        keep_out = sorted(set(DONOR_FILTER) & set(std_per_kb_cum.index))
        std_per_kb = std_per_kb.loc[keep_out]
        std_per_kb_cum = std_per_kb_cum.loc[keep_out]

    std_path = OUT_DIR / f"stage_burden_per_donor_perkb_STANDARDIZED_3LEVEL_{tag}.csv"
    std_cum_path = OUT_DIR / f"stage_burden_per_donor_perkb_cumsum_STANDARDIZED_3LEVEL_{tag}.csv"
    std_path.write_text(std_per_kb.to_csv(index=True))
    std_cum_path.write_text(std_per_kb_cum.to_csv(index=True))

    donor_to_age = {donor: parse_mouse_age_months(donor) for donor in std_per_kb_cum.index}
    age_color_map = {
        1: "#DEEBF7",
        3: "#C6DBEF",
        18: "#9ECAE1",
        21: "#6BAED6",
        24: "#4292C6",
        30: "#2171B5",
    }
    default_color = "#6BAED6"

    donor_to_color = {}
    for donor in std_per_kb_cum.index:
        age = donor_to_age.get(donor, None)
        donor_to_color[donor] = age_color_map.get(age, default_color)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=1000)
    for donor in std_per_kb_cum.index:
        row = std_per_kb_cum.loc[donor]
        y = row[ORDERED_LABELS].ffill().fillna(0).values
        ax.plot(
            ORDERED_LABELS,
            y,
            marker="o",
            linestyle="-",
            color=donor_to_color[donor],
            alpha=0.9,
            markeredgecolor="black",
            markeredgewidth=0.3,
            linewidth=2,
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Developmental stage", fontsize=16)
    ax.set_ylabel("Cumulative mutations per kilobase", fontsize=16)
    plt.setp(ax.get_xticklabels(), fontsize=14, rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), fontsize=14)

    ages_present = sorted({donor_to_age[d] for d in std_per_kb_cum.index if donor_to_age.get(d) is not None})
    age_handles = [
        Patch(facecolor=age_color_map.get(a, default_color), edgecolor="black", label=f"{a} mo")
        for a in ages_present
    ]
    ax.legend(
        handles=age_handles,
        title="Age (months)",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=12,
        title_fontsize=14,
        frameon=False,
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)
    out_png = PLOT_DIR / f"cumulative_mutations_per_kb_STANDARDIZED_3LEVEL_{tag}_blues_by_age_months.png"
    plt.savefig(out_png, dpi=1000, bbox_inches="tight")


if __name__ == "__main__":
    main()
