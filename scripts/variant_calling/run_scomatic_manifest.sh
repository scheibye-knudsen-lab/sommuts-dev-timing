#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

usage() {
  local wrapper="run_scomatic_human.sh"
  local manifest_example="/path/to/human_manifest.tsv"
  local ref_example="/path/to/gencode_v41_ercc.fa"
  local out_example="/path/to/scomatic_human_outputs"
  if [[ "${species:-}" == "mouse" ]]; then
    wrapper="run_scomatic_mouse.sh"
    manifest_example="/path/to/mouse_manifest.tsv"
    ref_example="/path/to/mm10.fa"
    out_example="/path/to/scomatic_mouse_outputs"
  fi

  printf '%s\n' \
    "Run SComatic variant calling for every BAM listed in a tab-delimited manifest." \
    "" \
    "Required manifest columns:" \
    "  sample    library_id    bam    metadata" \
    "" \
    "Example:" \
    "  bash ${wrapper} \\" \
    "    --manifest ${manifest_example} \\" \
    "    --scomatic /path/to/SComatic \\" \
    "    --ref ${ref_example} \\" \
    "    --outdir ${out_example}" \
    "" \
    "Options:" \
    "  --species human|mouse       Set by run_scomatic_human.sh or run_scomatic_mouse.sh." \
    "  --manifest PATH            TSV with sample, library_id, bam, metadata columns." \
    "  --scomatic PATH            SComatic checkout directory." \
    "  --ref PATH                 Reference FASTA used for SComatic." \
    "  --editing PATH             RNA-editing sites file. Defaults to SComatic species file." \
    "  --pon PATH                 SComatic panel-of-normals file. Defaults to SComatic species file." \
    "  --mappability-bed PATH     BED file used for PASS+mappability filtering." \
    "                              Defaults to SComatic bed_files_of_interest file." \
    "  --outdir PATH              Output directory. Default: scomatic_<species>_outputs." \
    "  --nprocs N                 Threads per SComatic job. Default: NPROCS_PER_JOB or all CPUs." \
    "  --help                     Show this message." \
    "" \
    "The script expects indexed BAM files and matching per-cell metadata files. It does" \
    "not download BAMs, references, PoNs, RNA-editing resources, or SComatic itself."
}

die() {
  echo "[ERROR] $*" >&2
  exit 1
}

need_file() {
  local path="$1"
  local label="$2"
  [[ -s "$path" ]] || die "$label missing or empty: $path"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command in PATH: $1"
}

bam_index_exists() {
  local bam="$1"
  [[ -f "${bam}.bai" || -f "${bam%.bam}.bai" ]]
}

species=""
manifest=""
scomatic="${SCOMATIC:-}"
ref="${SCOMATIC_REF:-}"
editing="${SCOMATIC_EDITING:-}"
pon="${SCOMATIC_PON:-}"
mappability_bed="${SCOMATIC_MAPPABILITY_BED:-}"
outdir=""
nprocs="${NPROCS_PER_JOB:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --species)
      species="${2:-}"
      shift 2
      ;;
    --manifest)
      manifest="${2:-}"
      shift 2
      ;;
    --scomatic)
      scomatic="${2:-}"
      shift 2
      ;;
    --ref)
      ref="${2:-}"
      shift 2
      ;;
    --editing)
      editing="${2:-}"
      shift 2
      ;;
    --pon)
      pon="${2:-}"
      shift 2
      ;;
    --mappability-bed)
      mappability_bed="${2:-}"
      shift 2
      ;;
    --outdir)
      outdir="${2:-}"
      shift 2
      ;;
    --nprocs)
      nprocs="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$species" == "human" || "$species" == "mouse" ]] || die "--species must be human or mouse"
[[ -n "$manifest" ]] || die "--manifest is required"
[[ -n "$scomatic" ]] || die "--scomatic is required"
[[ -n "$ref" ]] || die "--ref is required"

if [[ "$species" == "human" ]]; then
  editing="${editing:-${scomatic}/RNAediting/AllEditingSites.hg38.txt}"
  pon="${pon:-${scomatic}/PoNs/PoN.scRNAseq.hg38.tsv}"
else
  editing="${editing:-${scomatic}/RNAediting/AllEditingSites.mm10.txt}"
  pon="${pon:-${scomatic}/PoNs/PoN.scRNAseq.mm10.tsv}"
fi
mappability_bed="${mappability_bed:-${scomatic}/bed_files_of_interest/UCSC.k100_umap.without.repeatmasker.bed}"
outdir="${outdir:-scomatic_${species}_outputs}"
nprocs="${nprocs:-$(nproc)}"

need_cmd python3
need_cmd samtools
need_cmd bedtools
need_file "$manifest" "manifest"
need_file "$ref" "reference FASTA"
need_file "$editing" "RNA-editing sites"
need_file "$pon" "panel of normals"
need_file "$mappability_bed" "mappability BED"

for script in \
  scripts/SplitBam/SplitBamCellTypes.py \
  scripts/BaseCellCounter/BaseCellCounter.py \
  scripts/MergeCounts/MergeBaseCellCounts.py \
  scripts/BaseCellCalling/BaseCellCalling.step1.py \
  scripts/BaseCellCalling/BaseCellCalling.step2.py \
  scripts/GetCallableSites/GetAllCallableSites.py \
  scripts/SitesPerCell/SitesPerCell.py \
  scripts/SingleCellGenotype/SingleCellGenotype.py
do
  need_file "${scomatic}/${script}" "SComatic script"
done

if [[ ! -s "${ref}.fai" ]]; then
  echo "[INFO] Building FASTA index: ${ref}.fai"
  samtools faidx "$ref"
fi

mkdir -p "$outdir/logs"

run_sample() {
  local sample="$1"
  local library_id="$2"
  local bam="$3"
  local metadata="$4"
  local prefix="$library_id"

  need_file "$bam" "BAM"
  bam_index_exists "$bam" || die "BAM index missing for: $bam"
  need_file "$metadata" "metadata"

  local out1="${outdir}/Step1_BamCellTypes/${sample}/${library_id}"
  local out2="${outdir}/Step2_BaseCellCounts/${sample}/${library_id}"
  local out3="${outdir}/Step3_BaseCellCountsMerged/${sample}/${library_id}"
  local out4="${outdir}/Step4_VariantCalling/${sample}/${library_id}"
  local out5="${outdir}/CellTypeCallableSites/${sample}/${library_id}"
  local out6="${outdir}/UniqueCellCallableSites/${sample}/${library_id}"
  local out7="${outdir}/SingleCellAlleles/${sample}/${library_id}"
  local log="${outdir}/logs/${sample}_${library_id}.log"

  mkdir -p "$out1" "$out2" "$out3" "$out4" "$out5" "$out6" "$out7"

  {
    echo "[START] ${species} sample=${sample} library_id=${library_id}"
    echo "[INFO] BAM: ${bam}"
    echo "[INFO] metadata: ${metadata}"

    python3 "${scomatic}/scripts/SplitBam/SplitBamCellTypes.py" \
      --bam "$bam" \
      --meta "$metadata" \
      --id "$library_id" \
      --n_trim 5 \
      --max_nM 5 \
      --max_NH 1 \
      --outdir "$out1"

    local cell_bams=("$out1"/*.bam)
    ((${#cell_bams[@]} > 0)) || die "no split cell-type BAMs found under $out1"

    local ct tmp split_bam
    for split_bam in "${cell_bams[@]}"; do
      ct="$(basename "$split_bam" | awk -F'.' '{print $(NF-1)}')"
      tmp="${out2}/tmp_${ct}"
      mkdir -p "$tmp"
      python3 "${scomatic}/scripts/BaseCellCounter/BaseCellCounter.py" \
        --bam "$split_bam" \
        --ref "$ref" \
        --chrom all \
        --out_folder "$out2" \
        --min_bq 30 \
        --tmp_dir "$tmp" \
        --nprocs "$nprocs"
      rm -rf "$tmp"
    done

    python3 "${scomatic}/scripts/MergeCounts/MergeBaseCellCounts.py" \
      --tsv_folder "$out2" \
      --outfile "${out3}/${prefix}.BaseCellCounts.AllCellTypes.tsv"

    python3 "${scomatic}/scripts/BaseCellCalling/BaseCellCalling.step1.py" \
      --infile "${out3}/${prefix}.BaseCellCounts.AllCellTypes.tsv" \
      --outfile "${out4}/${prefix}" \
      --ref "$ref"

    python3 "${scomatic}/scripts/BaseCellCalling/BaseCellCalling.step2.py" \
      --infile "${out4}/${prefix}.calling.step1.tsv" \
      --outfile "${out4}/${prefix}" \
      --editing "$editing" \
      --pon "$pon"

    bedtools intersect -header \
      -a "${out4}/${prefix}.calling.step2.tsv" \
      -b "$mappability_bed" \
      | awk '$1 ~ /^#/ || $6 == "PASS"' > "${out4}/${prefix}.calling.step2.pass.tsv"

    python3 "${scomatic}/scripts/GetCallableSites/GetAllCallableSites.py" \
      --infile "${out4}/${prefix}.calling.step1.tsv" \
      --outfile "${out5}/${prefix}" \
      --max_cov 150 \
      --min_cell_types 2

    for split_bam in "${cell_bams[@]}"; do
      ct="$(basename "$split_bam" | awk -F'.' '{print $(NF-1)}')"
      tmp="${out6}/tmp_${ct}"
      mkdir -p "$tmp"
      python3 "${scomatic}/scripts/SitesPerCell/SitesPerCell.py" \
        --bam "$split_bam" \
        --infile "${out4}/${prefix}.calling.step1.tsv" \
        --ref "$ref" \
        --out_folder "$out6" \
        --tmp_dir "$tmp" \
        --nprocs "$nprocs"
      rm -rf "$tmp"
    done

    for split_bam in "${cell_bams[@]}"; do
      ct="$(basename "$split_bam" | awk -F'.' '{print $(NF-1)}')"
      tmp="${out7}/tmp_${ct}"
      mkdir -p "$tmp"
      python3 "${scomatic}/scripts/SingleCellGenotype/SingleCellGenotype.py" \
        --bam "$split_bam" \
        --infile "${out4}/${prefix}.calling.step2.pass.tsv" \
        --nprocs "$nprocs" \
        --meta "$metadata" \
        --outfile "${out7}/${ct}.single_cell_genotype.tsv" \
        --tmp_dir "$tmp" \
        --ref "$ref"
      rm -rf "$tmp"
    done

    echo "[DONE] ${species} sample=${sample} library_id=${library_id}"
  } 2>&1 | tee -a "$log"
}

line_no=0
processed=0
while IFS=$'\t' read -r sample library_id bam metadata extra || [[ -n "${sample:-}" ]]; do
  line_no=$((line_no + 1))
  [[ -z "${sample:-}" || "${sample:0:1}" == "#" ]] && continue
  if [[ "$line_no" -eq 1 && "$sample" == "sample" && "${library_id:-}" == "library_id" ]]; then
    continue
  fi
  [[ -n "${sample:-}" && -n "${library_id:-}" && -n "${bam:-}" && -n "${metadata:-}" ]] \
    || die "manifest line ${line_no} must contain sample, library_id, bam, metadata"
  [[ -z "${extra:-}" ]] || die "manifest line ${line_no} has too many tab-delimited fields"
  run_sample "$sample" "$library_id" "$bam" "$metadata"
  processed=$((processed + 1))
done < "$manifest"

((processed > 0)) || die "manifest contained no runnable sample rows: $manifest"
echo "[DONE] processed ${processed} ${species} sample libraries"
