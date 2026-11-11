#!/usr/bin/env bash
# Submit a batch script as a fresh Slurm job.
#
# Needed because an interactive shell may itself live inside a Slurm
# allocation: sbatch/srun then inherit SLURM_* variables (CPU binding to a
# single core, the parent's partition) and the new job is silently crippled.
# This wrapper drops every inherited SLURM_*/SBATCH_* variable and submits
# with --export=NONE so the job starts from a clean environment.
#
# Usage:  slurm/submit.sh [sbatch options] slurm/run.sbatch <command...>
# Add --wait to block until the job ends (exit status = the job's).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
mkdir -p logs
unset_args=()
while IFS= read -r v; do unset_args+=(-u "$v"); done \
    < <(env | grep -oE '^(SLURM|SBATCH)[A-Za-z0-9_]*' || true)
exec env "${unset_args[@]}" sbatch --export=NONE "$@"
