#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd

SUFFIX_GENO = ".single_cell_genotype.tsv"
SUFFIX_SITES = ".SitesPerCell.tsv"
GENO_RE = re.compile(r"^(?P<ct>.+?)\.single_cell_genotype(?:\s*\d+)?\.tsv$", re.IGNORECASE)
BASE_DIR = Path(__file__).resolve().parent
GENO_KEEP_COLS = [
    "#CHROM",
    "Start",
    "End",
    "REF",
    "ALT_expected",
    "CB",
    "Cell_type_observed",
    "Base_observed",
    "Num_reads",
]


def detect_sep(p: Path) -> str:
    with p.open("r", errors="ignore") as f:
        first = f.readline()
    return "\t" if ("\t" in first and first.count("\t") >= first.count(",")) else ","


def norm_cb(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def parse_celltype_from_geno(name: str) -> str:
    m = GENO_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized genotype filename: {name}")
    return m.group("ct")


def parse_celltype_from_sites(donor: str, p: Path) -> str:
    m = re.match(rf"^{re.escape(donor)}(?:_[^\.]+)?\.(.+?)\.SitesPerCell\.tsv$", p.name)
    if not m:
        raise ValueError(f"Can't parse celltype from {p.name}")
    return m.group(1)


def read_genotype(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep=detect_sep(fp), low_memory=False)
    if "CB" not in df.columns and "cell_barcode" in df.columns:
        df = df.rename(columns={"cell_barcode": "CB"})
    if "CB" not in df.columns:
        raise ValueError(f"No CB/cell_barcode in {fp}")
    df["CB"] = norm_cb(df["CB"])
    return df


def read_sites_per_cell(fp: Path) -> pd.DataFrame:
    df = pd.read_csv(fp, sep=detect_sep(fp), low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    if "CB" not in df.columns and "cell_barcode" in df.columns:
        df = df.rename(columns={"cell_barcode": "CB"})
    if "CB" not in df.columns:
        raise ValueError(f"Missing CB column in {fp}")
    sites_col = pick_sites_col(df.columns)
    if sites_col is None:
        return pd.DataFrame(columns=["CB", "SitesPerCell"])
    df["CB"] = norm_cb(df["CB"])
    out = df[["CB", sites_col]].rename(columns={sites_col: "SitesPerCell"})
    return out


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


def collect_geno_files_for_donor(root: Path, donor: str) -> Dict[str, List[Path]]:
    base = root / donor
    ct_map: Dict[str, List[Path]] = {}
    if not base.is_dir():
        return ct_map
    for lib_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        for gfp in lib_dir.glob(f"*{SUFFIX_GENO}"):
            try:
                ct = parse_celltype_from_geno(gfp.name)
            except Exception:
                continue
            ct_map.setdefault(ct, []).append(gfp)
    return ct_map


def collect_sites_files_for_donor(root: Path, donor: str) -> Dict[str, List[Path]]:
    base = root / donor
    ct_map: Dict[str, List[Path]] = {}
    if not base.is_dir():
        return ct_map
    for lib_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
        for sfp in lib_dir.glob(f"*{SUFFIX_SITES}"):
            try:
                ct = parse_celltype_from_sites(donor, sfp)
            except Exception:
                continue
            ct_map.setdefault(ct, []).append(sfp)
    return ct_map


def merge_and_write_for_donor(donor: str, geno_root: Path, sites_root: Path, dest_root: Path):
    geno_map = collect_geno_files_for_donor(geno_root, donor)
    sites_map = collect_sites_files_for_donor(sites_root, donor)

    if not geno_map and not sites_map:
        raise SystemExit(f"No files found for donor {donor}")

    all_cts = sorted(set(geno_map.keys()) | set(sites_map.keys()))
    dest_geno_dir = dest_root / donor
    dest_geno_dir.mkdir(parents=True, exist_ok=True)

    geno_written = 0
    sites_written = 0

    for ct in all_cts:
        gfiles = geno_map.get(ct, [])
        sfiles = sites_map.get(ct, [])

        if gfiles:
            gdfs = []
            for fp in gfiles:
                df = read_genotype(fp)
                gdfs.append(df)
            if gdfs:
                g_all = pd.concat(gdfs, ignore_index=True)
                missing = [c for c in GENO_KEEP_COLS if c not in g_all.columns]
                if missing:
                    raise ValueError(f"Missing columns in genotype merge: {missing}")
                g_all = g_all[GENO_KEEP_COLS]
                out_g = dest_geno_dir / f"{ct}.single_cell_genotype.tsv"
                g_all.to_csv(out_g, sep="\t", index=False)
                geno_written += 1

        if sfiles:
            sdfs = []
            for fp in sfiles:
                df = read_sites_per_cell(fp)
                sdfs.append(df)
            if sdfs:
                s_all = pd.concat(sdfs, ignore_index=True)
                s_all = s_all.groupby("CB", as_index=False)["SitesPerCell"].max()
                out_s = dest_root / f"{donor}.{ct}.SitesPerCell.tsv"
                s_all.to_csv(out_s, sep="\t", index=False)
                sites_written += 1

    if geno_written == 0 and sites_written == 0:
        raise SystemExit(f"No merged files written for donor {donor}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", default="1-M-62")
    parser.add_argument(
        "--geno-root",
        type=Path,
        default=Path("/home/ubuntu/sommuts/TabMur/outputs/1mo_outputs/SingleCellAlleles"),
    )
    parser.add_argument(
        "--sites-root",
        type=Path,
        default=Path("/home/ubuntu/sommuts/TabMur/outputs/1mo_outputs/UniqueCellCallableSites"),
    )
    parser.add_argument(
        "--dest-root",
        type=Path,
        default=BASE_DIR / "demo_data" / "merged_outputs_demo",
    )
    args = parser.parse_args()

    if not args.geno_root.exists():
        raise SystemExit(f"geno-root not found: {args.geno_root}")
    if not args.sites_root.exists():
        raise SystemExit(f"sites-root not found: {args.sites_root}")

    args.dest_root.mkdir(parents=True, exist_ok=True)
    merge_and_write_for_donor(args.donor, args.geno_root, args.sites_root, args.dest_root)


if __name__ == "__main__":
    main()
