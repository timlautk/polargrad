#!/usr/bin/env bash
set -euo pipefail

mode=${1:-all}
profile_root=${POLARGRAD_PROFILE_ROOT:-results_additional}
metrics=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed

mkdir -p "${profile_root}/ncu" "${profile_root}/profile_runs"

profile_training() {
    local script=$1
    local experiment=$2
    local method_id=$3
    local range="training/${experiment}/${method_id}/repeat=0/"
    local report="${profile_root}/ncu/${experiment}_${method_id}"
    local results="${profile_root}/profile_runs/${experiment}_${method_id}"
    ncu --target-processes all --nvtx --nvtx-include "${range}" \
        --metrics "${metrics}" --export "${report}" --force-overwrite \
        python "${script}" --device=cuda --seed=42 --benchmark_only=True \
        --benchmark_steps=50 --benchmark_warmup=10 --benchmark_repeats=1 \
        --benchmark_trace_every=0 --benchmark_filter="${method_id}" \
        --benchmark_nvtx=True --results_dir="${results}"
}

profile_oracle() {
    local method=$1
    local steps=$2
    local method_tag=${method//-/_}
    local range="polar/${method}/steps=${steps}/4096x1024/gaussian/"
    local report="${profile_root}/ncu/oracle_${method_tag}_steps${steps}"
    local results="${profile_root}/profile_runs/oracle_${method_tag}_steps${steps}"
    ncu --target-processes all --nvtx --nvtx-include "${range}" \
        --metrics "${metrics}" --export "${report}" --force-overwrite \
        python benchmark_polar_oracles.py --device=cuda \
        --shapes=4096x1024 --methods="${method}" --spectra=gaussian \
        --inner-steps="${steps}" --calls=20 --warmup-calls=5 --repeats=1 \
        --compute-reference --matmul-precision=highest --nvtx \
        --output-dir="${results}"
}

run_training_profiles() {
    profile_training mat_quad_reg.py mat_quad_reg polargrad_qdwh_lr_decay
    profile_training mat_quad_reg.py mat_quad_reg muon_qdwh_lr_decay
    profile_training mat_quad_reg.py mat_quad_reg adam_lr_decay
    profile_training mat_log_reg.py mat_log_reg polarsgd_qdwh_lr_decay
    profile_training mat_log_reg.py mat_log_reg muon_qdwh_lr_decay
    profile_training mat_log_reg.py mat_log_reg adam_lr_decay
    profile_training low_rank_mat_comp.py low_rank_mat_comp polargrad_qdwh_lr_decay
    profile_training low_rank_mat_comp.py low_rank_mat_comp muon_qdwh_lr_decay
    profile_training low_rank_mat_comp.py low_rank_mat_comp adam_lr_decay
}

run_oracle_profiles() {
    profile_oracle qdwh 2
    profile_oracle qdwh 5
    profile_oracle ns 5
    profile_oracle polar_express 5
    profile_oracle zolo-pd 5
}

case "${mode}" in
    training)
        run_training_profiles
        ;;
    oracles)
        run_oracle_profiles
        ;;
    all)
        run_training_profiles
        run_oracle_profiles
        ;;
    *)
        echo "Usage: $0 [training|oracles|all]" >&2
        exit 2
        ;;
esac

echo "Nsight Compute reports were written under ${profile_root}/ncu."
