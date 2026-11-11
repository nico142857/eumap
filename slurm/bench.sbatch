#!/usr/bin/env bash
# One dataset of the eumap benchmark on one exclusive 36-core broadwell node.
# Arguments are passed to benchmarks/run_benchmark.py. Submit through the wrapper
# from the repository root, e.g.
#   slurm/submit.sh -J bench_mnist -t 04:00:00 slurm/bench.sbatch --dataset mnist
#   slurm/submit.sh -J gt_mnist    -t 01:00:00 slurm/bench.sbatch --dataset mnist --gt-only
# (slurm/submit_all.sh does this for every dataset.)
#SBATCH --partition=normal
#SBATCH --constraint=broadwell
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=360G
#SBATCH --time=12:00:00
#SBATCH --job-name=eumap_bench
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
n="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$n" NUMBA_NUM_THREADS="$n" OPENBLAS_NUM_THREADS="$n" MKL_NUM_THREADS="$n"
# Per-job numba cache on node-local disk (never a cache built on another CPU type).
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/eumap_numba_${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
echo "[job $SLURM_JOB_ID] host=$(hostname) partition=$SLURM_JOB_PARTITION cpus=$n nproc=$(nproc) cpu='$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs)'"
echo "[job $SLURM_JOB_ID] args: $*"
# Times are compared only within one node (benchmarks/PROTOCOL.md). A dataset that needs more than
# one job gets chained continuation parts named <this job>_part<N> (slurm/submit_all.sh); pin the
# pending ones to this node so they resume here. If this node becomes unavailable they wait; release
# them with  scontrol update JobId=<id> ReqNodeList=  (aggregate.py then uses one host per dataset).
node="${SLURMD_NODENAME:-$(hostname -s)}"
base="${SLURM_JOB_NAME%_part*}"
if [[ "$base" == bench_* ]]; then
    while read -r jid jname; do
        if [[ "$jname" =~ ^${base}_part[0-9]+$ ]]; then
            if scontrol update JobId="$jid" ReqNodeList="$node"; then
                echo "[job $SLURM_JOB_ID] pinned continuation $jname ($jid) to $node"
            else
                echo "[job $SLURM_JOB_ID] WARNING: could not pin $jname ($jid) to $node"
            fi
        fi
    done < <(squeue -u "$(id -un)" -h -t PENDING -o '%i %j' || true)
fi
# -I: isolated mode (no user site, script dir not trusted implicitly); -u: unbuffered (-I ignores PYTHON* vars).
exec .venv/bin/python -I -u benchmarks/run_benchmark.py "$@"
