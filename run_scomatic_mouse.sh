#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${repo_root}/scripts/variant_calling/run_scomatic_manifest.sh" --species mouse "$@"
