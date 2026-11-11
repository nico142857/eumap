#!/usr/bin/env bash
# Submit the eumap benchmark: per dataset, a ground-truth job (exact kNN graph,
# cached under data/) and then the benchmark job (afterok on it), both on
# exclusive broadwell nodes via slurm/submit.sh + slurm/bench.sbatch.
#
# At most 3 jobs of this script run at once: jobs are placed in 3 lanes
# (greedy on estimated hours) and each lane is a chain
#   gt_A -> bench_A -> gt_B -> bench_B -> ...   (afterok gt->bench, afterany bench->next gt)
# A failed ground truth cancels its benchmark (--kill-on-invalid-dep) and the lane goes on.
# Datasets that may not fit in one 23.5 h job get a second, chained benchmark part; when part 1
# starts, slurm/bench.sbatch pins the pending part 2 to part 1's node (times are compared only
# within one node).
# The runner skips finished records, so resubmitting a job resumes it. To resubmit a dataset,
# run this script again with its name (e.g. slurm/submit_all.sh sift1m): it chains onto the
# benchmark jobs already queued or running, so the 3-lane limit holds across calls.
#
# Usage (from anywhere; runs from the repository root):
#   slurm/submit_all.sh [--dry-run] [core | extended | JOB ...]
#     core      (default) synthetic_small synthetic_1e6 synthetic_3e6 mnist fashion_mnist covtype sift1m glove100
#     extended  gist1m deep10m (deep10m is --knn-only with sampled recall)
#     JOB       any of the names above; "synthetic" means the three synthetic jobs
#   --dry-run   print the sbatch commands, submit nothing
#   --after ID[,ID,ID]  start lane i after job ID i (afterany). Without --after, the lanes start
#               after the last job of each chain of this benchmark's gt_*/bench_* jobs that are
#               still queued or running (found with squeue), so repeated calls keep 3 lanes.
#   --force     ignore queued/running benchmark jobs (may exceed 3 running benchmark jobs)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."

LANES=3
dry=0
force=0
sel=()
after=""
while [ $# -gt 0 ]; do
    a=$1; shift
    case "$a" in
        --dry-run) dry=1 ;;
        --after) after=${1:?--after needs job ids}; shift ;;
        --force) force=1 ;;
        -h|--help) sed -n '2,27p' "$0"; exit 0 ;;
        *) sel+=("$a") ;;
    esac
done
[ ${#sel[@]} -eq 0 ] && sel=(core)

declare -A ARGS GT_T BENCH_T EST PARTS
# Columns: name | runner arguments | ground-truth time limit | benchmark time limit per part |
#          estimated hours (gt + benchmark, from the smoke runs, see the orchestrator notes) |
#          parts: number of chained benchmark submissions (a later part resumes where the
#          previous one stopped at its time limit and exits within minutes if nothing is left).
def() { ARGS[$1]="$2"; GT_T[$1]="$3"; BENCH_T[$1]="$4"; EST[$1]="$5"; PARTS[$1]="$6"; }
def synthetic_small "--dataset synthetic --n 10000 30000 100000 300000 --out results/raw/synthetic_small.jsonl" 01:00:00 08:00:00 2 1
def synthetic_1e6   "--dataset synthetic --n 1000000 --out results/raw/synthetic_1e6.jsonl"   02:00:00 16:00:00 5 1
def synthetic_3e6   "--dataset synthetic --n 3000000 --out results/raw/synthetic_3e6.jsonl"   12:00:00 23:30:00 21 2
def mnist           "--dataset mnist"         01:00:00 04:00:00 1 1
def fashion_mnist   "--dataset fashion_mnist" 01:00:00 04:00:00 1 1
def covtype         "--dataset covtype"       01:00:00 10:00:00 2 1
def sift1m          "--dataset sift1m"        02:00:00 16:00:00 5 1
def glove100        "--dataset glove100"      02:00:00 18:00:00 6 1
def gist1m          "--dataset gist1m"        10:00:00 23:30:00 16 2
def deep10m         "--dataset deep10m --knn-only" 01:00:00 23:30:00 30 2
CORE=(synthetic_3e6 glove100 sift1m synthetic_1e6 covtype synthetic_small mnist fashion_mnist)
EXTENDED=(gist1m deep10m)

jobs=()
add() { local j; for j in "${jobs[@]}"; do [ "$j" = "$1" ] && return 0; done; jobs+=("$1"); }
for s in "${sel[@]}"; do
    case "$s" in
        core) for j in "${CORE[@]}"; do add "$j"; done ;;
        extended) for j in "${EXTENDED[@]}"; do add "$j"; done ;;
        synthetic) add synthetic_3e6; add synthetic_1e6; add synthetic_small ;;
        *) if [ -n "${ARGS[$s]+x}" ]; then add "$s"; else
               echo "unknown job '$s' (known: core extended synthetic ${!ARGS[*]})" >&2; exit 2; fi ;;
    esac
done

fake=900000
SUBMITTED=""
submit() {  # name time dependency runner-args...; sets SUBMITTED to the job id
    local name=$1 tlim=$2 dep=$3; shift 3
    local cmd=(slurm/submit.sh --parsable -J "$name" -t "$tlim")
    [ -n "$dep" ] && cmd+=(--dependency="$dep" --kill-on-invalid-dep=yes)
    cmd+=(slurm/bench.sbatch "$@")
    if [ "$dry" = 1 ]; then
        echo "  ${cmd[*]}"
        fake=$((fake + 1)); SUBMITTED=$fake
    else
        local out; out=$("${cmd[@]}"); SUBMITTED=${out%%;*}
    fi
}

load=(0 0 0); last=("" "" "")
if [ -z "$after" ] && [ "$force" = 0 ]; then
    # This benchmark's jobs already in the queue: chain tails are the jobs no other one depends on.
    names=$(printf '%s|' "${!ARGS[@]}"); names=${names%|}
    queued=$(squeue -u "$(id -un)" -h -o '%i|%j|%E' | grep -E "^[0-9]+\|(gt|bench)_($names)(_part[0-9]+)?\|" || true)
    if [ -n "$queued" ]; then
        deps=" $(cut -d'|' -f3 <<< "$queued" | grep -oE '[0-9]+' | tr '\n' ' ') "
        tails=()
        while IFS='|' read -r jid _; do
            [[ "$deps" == *" $jid "* ]] || tails+=("$jid")
        done <<< "$queued"
        echo "found $(wc -l <<< "$queued") queued/running benchmark jobs; chain tails: ${tails[*]}"
        if [ "${#tails[@]}" -gt "$LANES" ]; then
            echo "more than $LANES chains are queued; pass --after ID,ID,ID (or --force)" >&2; exit 3
        fi
        after=$(IFS=,; echo "${tails[*]}")
    fi
fi
IFS=, read -r -a prev <<< "$after"
for i in "${!prev[@]}"; do [ "$i" -lt "$LANES" ] && last[$i]=${prev[$i]}; done
[ "$dry" = 1 ] && echo "DRY RUN: nothing is submitted (job ids below are placeholders)"
for j in "${jobs[@]}"; do
    l=0
    for i in $(seq 1 $((LANES - 1))); do [ "${load[$i]}" -lt "${load[$l]}" ] && l=$i; done
    dep=""; [ -n "${last[$l]}" ] && dep="afterany:${last[$l]}"
    # shellcheck disable=SC2086  # ARGS holds space-separated runner arguments
    submit "gt_$j" "${GT_T[$j]}" "$dep" ${ARGS[$j]} --gt-only; gt=$SUBMITTED
    # shellcheck disable=SC2086
    submit "bench_$j" "${BENCH_T[$j]}" "afterok:$gt" ${ARGS[$j]}; bench=$SUBMITTED
    ids="$bench"
    for part in $(seq 2 "${PARTS[$j]}"); do
        # shellcheck disable=SC2086
        submit "bench_${j}_part$part" "${BENCH_T[$j]}" "afterany:$bench" ${ARGS[$j]}; bench=$SUBMITTED
        ids="$ids,$bench"
    done
    echo "lane $l: $j  gt=$gt  bench=$ids  (est. ${EST[$j]} h; logs logs/gt_${j}_${gt}.out, logs/bench_${j}*_<id>.out)"
    last[$l]=$bench; load[$l]=$((load[$l] + EST[$j]))
done
echo "estimated hours per lane: ${load[*]}"
echo "when done (or any time, it works on partial results): slurm/submit.sh --wait -c 2 --mem=8G -t 30 -J aggregate slurm/run.sbatch .venv/bin/python -I benchmarks/aggregate.py"
