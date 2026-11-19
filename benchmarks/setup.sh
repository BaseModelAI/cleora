#!/usr/bin/env bash
# Creates what the benchmark needs, all under one directory:
#
#   <home>/envs/main      Python 3.12: pycleora (built from this repository), probes, report,
#                         figures, and the baselines that run on current NumPy / SciPy
#   <home>/envs/walks     Python 3.11: DeepWalk / node2vec (PecanPy + gensim)
#   <home>/envs/karate    Python 3.10: GraRep, HOPE, Walklets (karateclub)
#   <home>/baselines/*    the authors' implementations of ProNE, NetMF, FastRP and RandNE,
#                         each at the commit the benchmark was run with
#
# Usage:   benchmarks/setup.sh <home>
# Needs:   uv (https://docs.astral.sh/uv/), git, and a Rust toolchain to build pycleora.
# Running it again only adds what is missing.
set -euo pipefail

[ $# -eq 1 ] || { echo "usage: $0 <home directory>" >&2; exit 2; }
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v cargo >/dev/null || { echo "a Rust toolchain is required: https://rustup.rs" >&2; exit 1; }
mkdir -p "$1"
home=$(cd "$1" && pwd)
here=$(cd "$(dirname "$0")" && pwd)

environment() {  # name, Python version
    local python="$home/envs/$1/bin/python"
    [ -x "$python" ] || uv venv --quiet --python "$2" "$home/envs/$1"
    uv pip install --quiet --python "$python" -r "$here/envs/$1.txt"
    echo "environment $1: $("$python" --version)"
}

environment main 3.12
uv pip install --quiet --python "$home/envs/main/bin/python" "$here/.."
environment walks 3.11
environment karate 3.10
uv pip install --quiet --python "$home/envs/karate/bin/python" --no-deps karateclub==1.3.3

baseline() {  # directory, repository, commit
    local target="$home/baselines/$1"
    [ -d "$target/.git" ] || git clone --quiet "$2" "$target"
    git -C "$target" -c advice.detachedHead=false checkout --quiet "$3"
    echo "baseline $1 at $(git -C "$target" rev-parse --short HEAD)"
}

baseline ProNE  https://github.com/THUDM/ProNE.git    873630a5943f3eba8f4bbc9751fdf16af98d2171
baseline NetMF  https://github.com/xptree/NetMF.git   77286b826c4af149055237cef65e2a500e15631a
baseline FastRP https://github.com/GTmac/FastRP.git   3a6a71c02a47b131445ed091db6d4ad9aed47d7d
baseline RandNE https://github.com/ZW-ZHANG/RandNE.git bf06bd77b3c53fcfa9d80eb076a86023443bab26

echo
echo "Ready. The benchmark scripts need:"
echo "  export CLEORA_BENCH_ENVS=$home/envs CLEORA_BENCH_BASELINES=$home/baselines"
echo "  $home/envs/main/bin/python benchmarks/run_benchmark.py --data $home ..."
