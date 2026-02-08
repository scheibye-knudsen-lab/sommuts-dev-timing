#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd

GENO_SUFFIX = ".single_cell_genotype.tsv"
BASE_DIR = Path(__file__).resolve().parent


def detect_sep(p: Path) -> str:
    with p.open("r", errors="ignore") as f:
        line = f.readline()
    return "\t" if ("\t" in line and line.count("\t") >= line.count(",")) else ","


def norm_cb(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def find_col(cols, *cands):
    norm = {c.lower().replace("_", ""): c for c in cols}
    for cand in cands:
        key = cand.lower().replace("_", "")
        if key in norm:
            return norm[key]
    return None


def mode_one(s: pd.Series):
    s = s.dropna()
    vc = s.value_counts(dropna=False)
    return vc.index[0] if not vc.empty else pd.NA


def norm_str(s):
    if pd.isna(s):
        return s
    return str(s).strip()


def norm_tissue(s):
    if pd.isna(s):
        return s
    return str(s).replace("_", "").strip()


def load_genotypes(outputs_root: Path, donor_filter=None):
    req = {
        "#CHROM",
        "Start",
        "End",
        "REF",
        "ALT_expected",
        "CB",
        "Cell_type_observed",
        "Base_observed",
        "Num_reads",
    }
    rows = []
    for donor_dir in sorted(p for p in outputs_root.iterdir() if p.is_dir()):
        donor = donor_dir.name
        if donor_filter and donor not in donor_filter:
            continue
        for fp in sorted(donor_dir.glob(f"*{GENO_SUFFIX}")):
            try:
                df = pd.read_csv(fp, sep=detect_sep(fp), low_memory=False)
            except Exception:
                continue
            missing = req - set(df.columns)
            if missing:
                continue
            if "CB" not in df.columns and "cell_barcode" in df.columns:
                df = df.rename(columns={"cell_barcode": "CB"})
            df["CB"] = norm_cb(df["CB"])
            mask = df["ALT_expected"].astype(str) == df["Base_observed"].astype(str)
            sub = df.loc[
                mask,
                [
                    "#CHROM",
                    "Start",
                    "End",
                    "REF",
                    "ALT_expected",
                    "CB",
                    "Cell_type_observed",
                    "Num_reads",
                ],
            ].copy()
            if sub.empty:
                continue
            sub["donor"] = donor
            rows.append(sub)
    if not rows:
        return pd.DataFrame(
            columns=[
                "#CHROM",
                "Start",
                "End",
                "REF",
                "ALT_expected",
                "donor",
                "CB",
                "Cell_type_observed",
                "Num_reads",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def add_tissue(cb_table: pd.DataFrame, barcode_path: Path) -> pd.DataFrame:
    bar = pd.read_csv(barcode_path)

    bar_cb = find_col(bar.columns, "CB", "cell_barcode")
    bar_donor = find_col(bar.columns, "donor", "Donor", "mouse.id", "mouse_id", "mouseid")
    bar_tissue = find_col(bar.columns, "tissue", "donor_tissue", "tissue_type")
    missing_bar = [
        n for n, v in {"CB": bar_cb, "donor": bar_donor, "tissue": bar_tissue}.items() if v is None
    ]
    if missing_bar:
        raise ValueError(f"cellbarcodes file missing columns: {missing_bar}")

    cb_table["CB"] = norm_cb(cb_table["CB"])
    bar[bar_cb] = norm_cb(bar[bar_cb])

    grp = bar.groupby([bar_donor, bar_cb])
    collapsed = grp[bar_tissue].agg(mode_one).reset_index().rename(
        columns={bar_donor: "donor", bar_cb: "CB", bar_tissue: "tissue_from_bar"}
    )

    merged = cb_table.merge(collapsed, on=["donor", "CB"], how="left")
    if "tissue" in merged.columns:
        merged["tissue"] = merged["tissue"].fillna(merged["tissue_from_bar"])
        merged = merged.drop(columns=["tissue_from_bar"])
    else:
        merged = merged.rename(columns={"tissue_from_bar": "tissue"})

    return merged


def add_germ_layer(cb_table: pd.DataFrame, germ_map_path: Path) -> pd.DataFrame:
    gmap = pd.read_csv(germ_map_path)

    if "germ_layer" in cb_table.columns:
        cb_table = cb_table.drop(columns=["germ_layer"])

    if "tissue" in cb_table.columns:
        cb_table["tissue"] = cb_table["tissue"].astype(str)
        cb_table["tissue"] = cb_table["tissue"].str.replace(
            r"^\s*FAT\s*$", "Fat", regex=True, flags=re.IGNORECASE
        )
        cb_table["tissue"] = cb_table["tissue"].str.replace(
            r"^\s*LymphNodes?\s*$", "Lymph_Node", regex=True, flags=re.IGNORECASE
        )
        cb_table["tissue"] = cb_table["tissue"].str.replace(
            r"^\s*BM\s*$", "Bone_Marrow", regex=True, flags=re.IGNORECASE
        )

    required_cb = {"Cell_type_observed", "tissue"}
    missing_cb = required_cb - set(cb_table.columns)
    if missing_cb:
        raise ValueError(f"CB table missing columns: {missing_cb}")

    g_cell = find_col(gmap.columns, "cell_type", "celltype", "Cell_type")
    g_tissue = find_col(gmap.columns, "donor_tissue", "tissue_type", "tissue")
    g_germ = find_col(gmap.columns, "germ_layer", "germlayer")
    missing_map = [
        name for name, value in {"cell_type": g_cell, "donor_tissue": g_tissue, "germ_layer": g_germ}.items() if value is None
    ]
    if missing_map:
        raise ValueError(f"Germ-layer map missing columns: {missing_map}")
    gmap = gmap.rename(columns={g_cell: "cell_type", g_tissue: "donor_tissue", g_germ: "germ_layer"})

    cb_table = cb_table[~cb_table["tissue"].isna() & (cb_table["tissue"].astype(str).str.strip() != "")]

    cb_table["Cell_type_observed"] = cb_table["Cell_type_observed"].map(norm_str)
    cb_table["tissue_norm"] = cb_table["tissue"].map(norm_tissue)

    gmap["cell_type"] = gmap["cell_type"].map(norm_str)
    gmap["donor_tissue_norm"] = gmap["donor_tissue"].map(norm_tissue)

    merged = cb_table.merge(
        gmap[["cell_type", "donor_tissue_norm", "germ_layer"]],
        left_on=["Cell_type_observed", "tissue_norm"],
        right_on=["cell_type", "donor_tissue_norm"],
        how="left",
    )

    out = merged.drop(
        columns=[c for c in ["tissue_norm", "cell_type", "donor_tissue_norm"] if c in merged.columns]
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=BASE_DIR / "demo_data" / "merged_outputs_demo",
    )
    parser.add_argument("--donor", default="1-M-62")
    parser.add_argument(
        "--cellbarcodes",
        type=Path,
        default=BASE_DIR / "demo_data" / "cellbarcodes_with_tissue.csv",
    )
    parser.add_argument(
        "--germ-map",
        type=Path,
        default=BASE_DIR / "demo_data" / "cell_type_to_germ_layer_map.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=BASE_DIR / "demo_data" / "cell_metadata_table_1-M-62.csv",
    )
    args = parser.parse_args()

    if not args.outputs_root.exists():
        raise SystemExit(f"outputs-root not found: {args.outputs_root}")

    cb_table = load_genotypes(args.outputs_root, donor_filter={args.donor})
    if cb_table.empty:
        raise SystemExit("No ALT matches found in genotype files.")

    cb_table = add_tissue(cb_table, args.cellbarcodes)
    cb_table = add_germ_layer(cb_table, args.germ_map)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cb_table.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
