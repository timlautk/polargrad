#!/usr/bin/env bash
set -euo pipefail

POLARGRAD_BENCH_ROOT="results_additional/timing_memory"

mkdir -p "${POLARGRAD_BENCH_ROOT}/training"
mkdir -p "${POLARGRAD_BENCH_ROOT}/oracles"

# Optional numerical validation before benchmarking
python validate_polar_oracles.py --device=cpu

# Section 6.1: matrix quadratic regression
for seed in 42 142 242; do
    python mat_quad_reg.py \
        --device=cuda \
        --seed="${seed}" \
        --benchmark_only=True \
        --benchmark_steps=4000 \
        --benchmark_warmup=10 \
        --benchmark_repeats=3 \
        --benchmark_trace_every=0 \
        --matmul_precision=high \
        --results_dir="${POLARGRAD_BENCH_ROOT}/training"
done

# Section 6.2: matrix logistic regression
for seed in 42 142 242; do
    python mat_log_reg.py \
        --device=cuda \
        --seed="${seed}" \
        --benchmark_only=True \
        --benchmark_steps=1500 \
        --benchmark_warmup=10 \
        --benchmark_repeats=3 \
        --benchmark_trace_every=0 \
        --matmul_precision=high \
        --results_dir="${POLARGRAD_BENCH_ROOT}/training"
done

# Section 6.3: low-rank matrix completion
for seed in 42 142 242; do
    python low_rank_mat_comp.py \
        --device=cuda \
        --seed="${seed}" \
        --benchmark_only=True \
        --benchmark_steps=1000 \
        --benchmark_warmup=10 \
        --benchmark_repeats=3 \
        --benchmark_trace_every=0 \
        --matmul_precision=high \
        --results_dir="${POLARGRAD_BENCH_ROOT}/training"
done

# Aggregate the three training experiments
python summarize_benchmark_runs.py \
    --results-dir="${POLARGRAD_BENCH_ROOT}/training"

# Polar-oracle timing, memory, and numerical-accuracy benchmark
python benchmark_polar_oracles.py \
    --device=cuda \
    --shapes=500x100,1000x100,500x5,250x5,1024x1024,4096x1024 \
    --methods=qdwh,zolo-pd,ns,polar_express \
    --spectra=gaussian,ill_conditioned,rank_deficient \
    --inner-steps=2,5 \
    --condition-number=1e6 \
    --calls=20 \
    --warmup-calls=5 \
    --repeats=3 \
    --compute-reference \
    --matmul-precision=highest \
    --output-dir="${POLARGRAD_BENCH_ROOT}/oracles"