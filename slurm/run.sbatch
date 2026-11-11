#!/usr/bin/env bash
# Generic job: run the given command from the repository root with thread
# counts pinned to the allocation. Override resources on the command line,
# e.g.  slurm/submit.sh --wait -c 16 --mem=32G -t 60 -J name slurm/run.sbatch <cmd...>
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=eumap
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
n="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$n" NUMBA_NUM_THREADS="$n" OPENBLAS_NUM_THREADS="$n" MKL_NUM_THREADS="$n"
# Per-job numba cache on node-local disk: nodes have different CPUs, so a
# cache shared through the filesystem could hold code built for another one.
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/eumap_numba_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
echo "[job $SLURM_JOB_ID] host=$(hostname) partition=$SLURM_JOB_PARTITION cpus=$n nproc=$(nproc) cpu='$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)'"
echo "[job $SLURM_JOB_ID] cmd: $*"
"$@"
