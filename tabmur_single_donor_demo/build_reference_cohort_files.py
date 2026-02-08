#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

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


def build_cb_map(cb_table: Path) -> pd.DataFrame:
    m = pd.read_csv(cb_table, usecols=["donor", "CB", "tissue", "germ_layer"])
    m["CB"] = m["CB"].astype(str).str.strip().str.upper()
    m = m[~bad(m["tissue"]) & ~bad(m["germ_layer"])]
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


def compute_reference_weights(cb_full: pd.DataFrame, keep_donors: set):
    x = cb_full[cb_full["donor"].isin(keep_donors)].copy()
    if x.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float)
    x["_w"] = 1.0
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


def main():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cb-table", type=Path, required=True)
    parser.add_argument("--donor", default="1-M-62")
    parser.add_argument(
        "--weights-out",
        type=Path,
        default=base_dir / "demo_data" / "reference_weights_cells",
    )
    parser.add_argument(
        "--stage-out",
        type=Path,
        default=None,
    )
    parser.add_argument("--min-layers", type=int, default=2)
    parser.add_argument("--min-tissues", type=int, default=2)
    args = parser.parse_args()

    if args.stage_out is None:
        args.stage_out = base_dir / "demo_data" / f"reference_stage_labels_{args.donor}.csv"

    sites_idx = build_sites_index(args.root)
    cb_map = build_cb_map(args.cb_table)
    donors_present = {d.name for d in args.root.iterdir() if d.is_dir()}
    keep_donors = derive_keep_donors(cb_map, min_layers=args.min_layers, min_tissues=args.min_tissues)
    keep_donors &= donors_present
    if args.donor not in keep_donors:
        raise SystemExit(f"Donor not in keep set: {args.donor}")

    rows = []
    need = {"ALT_expected", "Base_observed", "#CHROM", "Start", "REF"}

    for donor_dir in sorted([d for d in args.root.iterdir() if d.is_dir()]):
        donor = donor_dir.name
        if donor not in keep_donors:
            continue
        for gfp in sorted(donor_dir.glob("*.single_cell_genotype*.tsv")):
            ct = parse_celltype(gfp.name)
            g = read_genotype(gfp)
            spath = sites_idx.get((donor, ct))
            if not spath or not spath.exists():
                continue
            s = read_sites_per_cell(spath)
            m = g.merge(s, on="CB", how="inner")
            if m.empty:
                continue
            if not need.issubset(m.columns):
                continue
            if "donor" not in m.columns:
                m.insert(0, "donor", donor)
            if "Cell_type_observed" in m.columns and m["Cell_type_observed"].notna().any():
                m["cell_type_stage"] = m["Cell_type_observed"].astype(str)
            else:
                m["cell_type_stage"] = ct
            if "tissue" not in m.columns or "germ_layer" not in m.columns:
                m = m.merge(cb_map, on=["donor", "CB"], how="left")
            m["var_id"] = (
                m["#CHROM"].astype(str)
                + ":"
                + m["Start"].astype(str)
                + "_"
                + m["REF"].astype(str)
                + ">"
                + m["ALT_expected"].astype(str)
            )
            m["is_mutated"] = (
                m["Base_observed"].astype(str) == m["ALT_expected"].astype(str)
            ).astype(int)
            rows.append(
                m[
                    [
                        "donor",
                        "CB",
                        "SitesPerCell",
                        "germ_layer",
                        "tissue",
                        "cell_type_stage",
                        "var_id",
                        "is_mutated",
                    ]
                ]
            )

    if not rows:
        raise SystemExit("No merged rows for reference computation.")

    df = pd.concat(rows, ignore_index=True)
    df = df[df["donor"].isin(keep_donors)]
    df = df[~bad(df["tissue"]) & ~bad(df["germ_layer"])].copy()

    cb_full = build_cb_level(df)
    w_gl, w_tissue_gl, w_cell_tissue = compute_reference_weights(cb_full, keep_donors)

    args.weights_out.mkdir(parents=True, exist_ok=True)
    w_gl.reset_index().rename(columns={"germ_layer": "germ_layer", "_w": "weight"}).to_csv(
        args.weights_out / "weights_germ_layer.csv", index=False
    )
    w_tissue_gl = w_tissue_gl.reset_index().rename(columns={0: "weight"})
    w_tissue_gl.columns = ["germ_layer", "tissue", "weight"]
    w_tissue_gl.to_csv(args.weights_out / "weights_tissue_within_germ_layer.csv", index=False)
    w_cell_tissue = w_cell_tissue.reset_index().rename(columns={0: "weight"})
    w_cell_tissue.columns = ["tissue", "cell_type_stage", "weight"]
    w_cell_tissue.to_csv(args.weights_out / "weights_celltype_within_tissue.csv", index=False)

    mut = df[df["is_mutated"] == 1].copy()
    mut = mut.drop_duplicates(subset=["donor", "CB", "var_id"])

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

    mut_d = mut[mut["donor"] == args.donor].merge(
        stage[["var_id", "stage_label"]], on="var_id", how="left"
    )
    out = mut_d[["var_id", "stage_label"]].drop_duplicates()
    args.stage_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.stage_out, index=False)


if __name__ == "__main__":
    main()
