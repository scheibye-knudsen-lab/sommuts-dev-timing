from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate Lore 2026 supplementary tables and execute notebooks in document order.")
    parser.add_argument(
        "--only",
        choices=["all", "tables", "main", "extended"],
        default="all",
        help="Run only one output class. Default: all.",
    )
    parser.add_argument(
        "--human-gtf",
        type=Path,
        default=None,
        help="Optional GTF for the human active-gene-span panel; sets LORE2026_HUMAN_GTF for notebook execution.",
    )
    parser.add_argument(
        "--mouse-gtf",
        type=Path,
        default=None,
        help="Optional GTF for the mouse active-gene-span panel; sets LORE2026_MOUSE_GTF for notebook execution.",
    )
    parser.add_argument(
        "--mouse-fasta",
        type=Path,
        default=None,
        help="Optional mm10 FASTA for the mouse trinucleotide-context panel; sets LORE2026_MM10_FASTA for notebook execution.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run every notebook cell. Requires external full matrices, GTFs, FASTA files, and precomputed heavy revision tables.",
    )
    return parser.parse_args()

def display(*objects, **kwargs) -> None:
    del kwargs
    for obj in objects:
        if obj is not None:
            print(obj)

def run_step(label: str, command: list[str]) -> None:
    print(f"\n== {label} ==", flush=True)
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)

LIGHTWEIGHT_SKIP_CELLS = {
    "Figures.ipynb": {
        28: "full single-cell expression matrix is not bundled in the lightweight package",
    },
    "Supp.ipynb": {
        11: "96-class mutation spectrum panel requires external FASTA sequence files",
        19: "active-gene span panel requires external GTF annotations",
        21: "base-change sensitivity precompute is not bundled",
        25: "trinucleotide sequence-context panel requires external FASTA sequence files",
        32: "residualized cell-type control table is not bundled",
        47: "support/callable-threshold sensitivity precomputes are not bundled",
    },
}


def _lightweight_skip_reason(notebook_name: str, cell_index: int) -> str | None:
    if os.environ.get("LORE2026_FULL_RUN") == "1":
        return None
    return LIGHTWEIGHT_SKIP_CELLS.get(notebook_name, {}).get(cell_index)


def run_notebook(path: Path) -> None:
    print(f"\n== Execute {path.name} ==", flush=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook = nbformat.read(path, as_version=4)
    namespace = {"__name__": "__main__", "__file__": str(path), "display": display}
    original_cwd = Path.cwd()
    os.chdir(path.parent)
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    try:
        for cell_index, cell in enumerate(notebook.cells):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            if not source.strip():
                continue
            skip_reason = _lightweight_skip_reason(path.name, cell_index)
            if skip_reason is not None:
                print(f"[lightweight] Skipping {path.name} cell {cell_index}: {skip_reason}", flush=True)
                continue
            try:
                exec(compile(source, str(path), "exec"), namespace)
            except Exception as exc:
                raise RuntimeError(f"Notebook execution failed in cell {cell_index}") from exc
            plt.close("all")
    finally:
        plt.close("all")
        os.chdir(original_cwd)

def _set_optional_env(path: Path | None, env_name: str) -> None:
    if path is not None:
        os.environ[env_name] = str(path.expanduser().resolve())

def main() -> int:
    args = parse_args()
    selected = args.only
    _set_optional_env(args.human_gtf, "LORE2026_HUMAN_GTF")
    _set_optional_env(args.mouse_gtf, "LORE2026_MOUSE_GTF")
    _set_optional_env(args.mouse_fasta, "LORE2026_MM10_FASTA")
    if args.full:
        os.environ["LORE2026_FULL_RUN"] = "1"

    if selected in {"all", "tables"}:
        run_step(
            "Supplementary tables",
            [sys.executable, str(SCRIPTS_DIR / "regenerate_current_supplementary_tables.py")],
        )

    if selected in {"all", "main"}:
        run_notebook(REPO_ROOT / "Figures.ipynb")

    if selected in {"all", "extended"}:
        run_notebook(REPO_ROOT / "Supp.ipynb")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
