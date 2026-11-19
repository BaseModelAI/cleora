#!/usr/bin/env bash
# Replicates the benchmark of the README from nothing: environments, datasets, every
# method, the tables and the figures.
#
#   benchmarks/replicate.sh <directory> [options]
#
# Everything is written under <directory>; finished jobs are skipped, so the script can
# be interrupted and started again, and several copies can run on disjoint --datasets.
#
#   --tiers S|SM|SML     datasets by size: S up to ~1M edges (default), M up to ~60M, L larger
#   --datasets a,b,c     these datasets instead of whole tiers
#   --dim N              embedding dimensionality (default 1024)
#   --threads N          threads per embedding method (default: all cores)
#
# Parts that can be switched off:
#
#   --no-setup           environments and baseline checkouts already exist
#   --no-baselines       do not run the nine baseline methods
#   --no-previous        do not run the pycleora 3.2 default (whitening, 40 iterations)
#   --no-tuned           no iteration sweep and no "Cleora, tuned" rows
#   --no-timing          no separate time / memory measurement
#   --no-figures         no UMAP figures
#   --no-report          do not write benchmarks/RESULTS.md and the tables of the READMEs
#
# The time and memory measurement runs one method at a time and is only meaningful on a
# machine that does nothing else.
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,27p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
mkdir -p "$1"
data=$(cd "$1" && pwd)
shift
here=$(cd "$(dirname "$0")" && pwd)

tiers=S datasets="" dim=1024 threads=$(getconf _NPROCESSORS_ONLN)
setup=1 baselines=1 previous=1 tuned=1 timing=1 figures=1 report=1
while [ $# -gt 0 ]; do
    case $1 in
        --tiers) tiers=$2; shift ;;
        --datasets) datasets=$2; shift ;;
        --dim) dim=$2; shift ;;
        --threads) threads=$2; shift ;;
        --no-setup) setup=0 ;;
        --no-baselines) baselines=0 ;;
        --no-previous) previous=0 ;;
        --no-tuned) tuned=0 ;;
        --no-timing) timing=0 ;;
        --no-figures) figures=0 ;;
        --no-report) report=0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

[ $setup = 1 ] && "$here/setup.sh" "$data"
export CLEORA_BENCH_ENVS=$data/envs CLEORA_BENCH_BASELINES=$data/baselines
python=$data/envs/main/bin/python
configurations() {  # names of a group of Cleora configurations
    "$python" -c "import sys; sys.path.insert(0, '$here'); from cleora_bench import configurations as c; print(','.join($1))"
}

# Three seeds on tier S, one on the larger graphs.
for tier in $(echo "$tiers" | grep -o .); do
    if [ -n "$datasets" ]; then selection=(--datasets "$datasets"); else selection=(--tiers "$tier"); fi
    seeds=0,1,2; [ "$tier" = S ] || seeds=0
    run() { "$python" "$here/run_benchmark.py" --data "$data" "${selection[@]}" --threads "$threads" "$@"; }

    "$python" "$here/prepare_datasets.py" --data "$data" "${selection[@]}"

    # Cleora with the iterations the library suggests for the task, and the plain default for the figures
    run --dim "$dim" --seeds "$seeds" --tasks nc --methods "$(configurations "c.automatic('classification')")"
    run --dim "$dim" --seeds "$seeds" --tasks lp,viz --methods "$(configurations "c.automatic('similarity')")"

    if [ $baselines = 1 ]; then
        run --dim "$dim" --seeds "$seeds"
        run --dim 128 --seeds "$seeds" --tasks nc,lp       # the dimensionality of the papers
    fi
    if [ $previous = 1 ]; then
        run --dim "$dim" --seeds "$seeds" --methods cleora_whiten40
        run --dim 256 --seeds "$seeds" --tasks nc,lp --methods cleora_whiten40
    fi
    if [ $tuned = 1 ]; then
        # one cheap pass over the iteration counts, then the counts that validated best in full
        run --results sweep --dim "$dim" --seeds 0 --tasks nc,lp --node-probes logistic --pair-probes cosine,dot \
            --methods "$(configurations "c.sweep('hyperedge') + c.sweep('pair')")"
        "$python" "$here/select_iterations.py" --data "$data"
        run --dim "$dim" --seeds "$seeds" --tasks nc,lp --methods cleora_tuned
    fi
    if [ $timing = 1 ]; then
        run --results timing --dim "$dim" --seeds 0,1,2 --tasks time \
            --methods "$(configurations "[c.AUTO[('hyperedge', 'left', 'general')], 'cleora_whiten40'] + list(c.DEFAULT)")"
    fi
    [ -n "$datasets" ] && break
done

if [ $figures = 1 ]; then
    "$python" "$here/make_figures.py" --data "$data" --out "$here/figures" --dim "$dim"
fi
if [ $report = 1 ]; then
    "$python" "$here/make_report.py" --data "$data" --out "$here/RESULTS.md" \
        --readme "$here/README.md" --readme "$here/../README.md"
fi
