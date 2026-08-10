# Copyright 2026 Tim Tsz-Kit Lau.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Microbenchmark the polar oracles used by PolarGrad and Muon.

This complements the end-to-end experiments by isolating the numerical polar
operation.  It reports fixed-work latency together with accuracy residuals, so
faster but less accurate oracle configurations are not treated as equivalent.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from benchmarking import (
    MIB,
    ensure_clean_git,
    prepare_benchmark_output,
    resolve_device,
    seed_everything,
    synchronize,
)
from polar import polar


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_shapes(value):
    shapes = []
    for item in parse_csv(value):
        try:
            rows, columns = item.lower().split("x")
            shape = (int(rows), int(columns))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"Invalid shape {item!r}; expected comma-separated MxN values."
            ) from exc
        if min(shape) <= 0:
            raise argparse.ArgumentTypeError("Matrix dimensions must be positive.")
        shapes.append(shape)
    if not shapes:
        raise argparse.ArgumentTypeError("At least one matrix shape is required.")
    return shapes


def parse_int_csv(value):
    values = [int(item) for item in parse_csv(value)]
    if not values:
        raise argparse.ArgumentTypeError("At least one inner-iteration count is required.")
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("Iteration counts must be positive.")
    return values


def make_matrix(rows, columns, spectrum, condition_number, seed, device, dtype):
    seed_everything(seed, device)
    if spectrum == "gaussian":
        return torch.randn(rows, columns, device=device, dtype=dtype)

    rank_limit = min(rows, columns)
    rank = rank_limit if spectrum == "ill_conditioned" else max(1, rank_limit // 4)
    left = torch.linalg.qr(
        torch.randn(rows, rank, device=device, dtype=dtype), mode="reduced"
    ).Q
    right = torch.linalg.qr(
        torch.randn(columns, rank, device=device, dtype=dtype), mode="reduced"
    ).Q
    if spectrum == "ill_conditioned":
        singular_values = torch.logspace(
            0.0,
            -math.log10(condition_number),
            rank,
            device=device,
            dtype=dtype,
        )
    elif spectrum == "rank_deficient":
        singular_values = torch.logspace(
            0.0, -2.0, rank, device=device, dtype=dtype
        )
    else:
        raise ValueError(f"Unknown spectrum {spectrum!r}.")
    return (left * singular_values.unsqueeze(0)) @ right.mT


def polar_residuals(matrix, unitary, reference=None, reference_nuclear_norm=None):
    """Compute polar-specific residuals in double precision.

    Reconstruction uses a symmetrized Hermitian factor, so it cannot collapse
    to a mere column- or row-space projection test. Positive-semidefiniteness is
    checked separately because symmetry alone does not characterize a polar
    factor.
    """
    work_dtype = torch.complex128 if matrix.is_complex() else torch.float64
    matrix = matrix.to(work_dtype)
    unitary = unitary.to(work_dtype)
    rows, columns = matrix.shape
    order = min(rows, columns)
    identity = torch.eye(order, device=matrix.device, dtype=work_dtype)
    if rows >= columns:
        gram = unitary.mH @ unitary
        hermitian_raw = unitary.mH @ matrix
    else:
        gram = unitary @ unitary.mH
        hermitian_raw = matrix @ unitary.mH
    hermitian = 0.5 * (hermitian_raw + hermitian_raw.mH)
    reconstruction = (
        unitary @ hermitian if rows >= columns else hermitian @ unitary
    )
    matrix_norm = torch.linalg.matrix_norm(matrix)
    denominator = max(matrix_norm, torch.finfo(matrix.real.dtype).eps)
    orthogonality = torch.linalg.matrix_norm(gram - identity) / math.sqrt(order)
    symmetry_error = torch.linalg.matrix_norm(
        hermitian_raw - hermitian_raw.mH
    ) / denominator
    reconstruction_error = torch.linalg.matrix_norm(
        reconstruction - matrix
    ) / denominator
    eigenvalues = torch.linalg.eigvalsh(hermitian)
    negative_eigenvalues = torch.clamp(-eigenvalues, min=0)
    hermitian_norm = max(
        torch.linalg.matrix_norm(hermitian),
        torch.finfo(matrix.real.dtype).eps,
    )
    psd_violation = torch.linalg.vector_norm(negative_eigenvalues) / hermitian_norm
    direction_error = None
    if reference is not None:
        reference = reference.to(work_dtype)
        direction_error = torch.linalg.matrix_norm(unitary - reference) / max(
            torch.linalg.matrix_norm(reference),
            torch.finfo(matrix.real.dtype).eps,
        )
    objective_gap = None
    if reference_nuclear_norm is not None:
        polar_objective = torch.real(torch.trace(hermitian_raw))
        objective_gap = (
            reference_nuclear_norm - polar_objective
        ) / max(reference_nuclear_norm, torch.finfo(matrix.real.dtype).eps)
    residuals = {
        "orthogonality_residual": float(orthogonality.item()),
        "hermitian_symmetry_residual": float(symmetry_error.item()),
        "relative_psd_violation": float(psd_violation.item()),
        "reconstruction_residual": float(reconstruction_error.item()),
        "polar_objective_relative_gap": (
            float(objective_gap.item()) if objective_gap is not None else None
        ),
        "relative_direction_error": (
            float(direction_error.item()) if direction_error is not None else None
        ),
    }
    nonfinite = {
        key: value
        for key, value in residuals.items()
        if value is not None and not math.isfinite(value)
    }
    if nonfinite:
        raise RuntimeError(f"Non-finite polar residuals: {nonfinite}")
    return residuals


def exact_polar_reference(matrix):
    """Return an SVD polar factor and nuclear norm computed in float64."""
    work_dtype = torch.complex128 if matrix.is_complex() else torch.float64
    matrix = matrix.to(work_dtype)
    left, singular_values, right_h = torch.linalg.svd(
        matrix, full_matrices=False
    )
    return left @ right_h, singular_values.sum()


def benchmark_configuration(
    *,
    matrix,
    method,
    inner_steps,
    calls,
    warmup_calls,
    repeats,
    device,
    metadata,
    reference,
    reference_nuclear_norm,
    nvtx,
):
    records = []
    for repeat in range(repeats):
        for _ in range(warmup_calls):
            polar(matrix, method=method, max_iterations=inner_steps)[0]
        synchronize(device)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            baseline_allocated = torch.cuda.memory_allocated(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            stream = torch.cuda.current_stream(device)
        else:
            baseline_allocated = 0
            start_event = None
            end_event = None

        wall_start = time.perf_counter()
        if start_event is not None:
            start_event.record(stream)
        unitary = None
        if nvtx and device.type == "cuda":
            torch.cuda.nvtx.range_push(
                f"polar/{method}/steps={inner_steps}/{metadata['shape']}/"
                f"{metadata['spectrum']}"
            )
        try:
            for _ in range(calls):
                unitary = polar(
                    matrix, method=method, max_iterations=inner_steps
                )[0]
        finally:
            if nvtx and device.type == "cuda":
                torch.cuda.nvtx.range_pop()
        if end_event is not None:
            end_event.record(stream)
        synchronize(device)
        wall_time = time.perf_counter() - wall_start

        if unitary is None:
            raise RuntimeError("No oracle call was executed.")
        if unitary.shape != matrix.shape:
            raise RuntimeError(
                f"The {method} oracle returned shape {tuple(unitary.shape)} "
                f"for input shape {tuple(matrix.shape)}."
            )
        if start_event is not None:
            cuda_stream_span_ms = start_event.elapsed_time(end_event)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            cuda_stream_span_ms = None
            peak_allocated = 0
            peak_reserved = 0

        record = {
            **metadata,
            "repeat": repeat,
            "method": method,
            "inner_steps": inner_steps if method != "zolo-pd" else None,
            "output_dtype": str(unitary.dtype),
            "calls": calls,
            "warmup_calls": warmup_calls,
            "wall_time_s": wall_time,
            "time_per_call_ms": 1000.0 * wall_time / calls,
            "cuda_stream_span_per_call_ms": (
                cuda_stream_span_ms / calls
                if cuda_stream_span_ms is not None
                else None
            ),
            # A stream-span fraction is not GPU utilization, SM occupancy, or
            # achieved FLOP efficiency. Hardware efficiency must be profiled.
            "cuda_stream_span_fraction_pct": (
                100.0
                * cuda_stream_span_ms
                / max(1000.0 * wall_time, 1e-12)
                if cuda_stream_span_ms is not None
                else None
            ),
            "baseline_allocated_mib": baseline_allocated / MIB,
            "peak_allocated_mib": peak_allocated / MIB,
            "incremental_peak_allocated_mib": max(
                peak_allocated - baseline_allocated, 0
            )
            / MIB,
            "peak_reserved_mib": peak_reserved / MIB,
            **polar_residuals(
                matrix,
                unitary,
                reference,
                reference_nuclear_norm,
            ),
            "status": "ok",
        }
        records.append(record)
    return records


def summarize(records):
    groups = defaultdict(list)
    keys = ("shape", "spectrum", "condition_number", "method", "inner_steps")
    for record in records:
        if record.get("status") == "ok":
            groups[tuple(record.get(key) for key in keys)].append(record)
    metrics = (
        "time_per_call_ms",
        "cuda_stream_span_per_call_ms",
        "cuda_stream_span_fraction_pct",
        "peak_allocated_mib",
        "incremental_peak_allocated_mib",
        "peak_reserved_mib",
        "orthogonality_residual",
        "hermitian_symmetry_residual",
        "relative_psd_violation",
        "reconstruction_residual",
        "polar_objective_relative_gap",
        "relative_direction_error",
    )
    summaries = []
    for group_key, group in groups.items():
        row = dict(zip(keys, group_key))
        row["repeats"] = len(group)
        for metric in metrics:
            values = [record[metric] for record in group if record[metric] is not None]
            if values:
                row[f"{metric}_median"] = statistics.median(values)
                row[f"{metric}_mean"] = statistics.fmean(values)
                row[f"{metric}_std"] = statistics.pstdev(values)
        summaries.append(row)
    return summaries


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def profile_configuration(matrix, method, inner_steps, calls, output_path, device):
    if device.type != "cuda":
        raise RuntimeError("GPU profiling requires a CUDA device.")
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as profiler:
        with record_function(f"polar::{method}"):
            for _ in range(calls):
                polar(matrix, method=method, max_iterations=inner_steps)[0]
        synchronize(device)
    profiler.export_chrome_trace(str(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument(
        "--shapes", type=parse_shapes, default=parse_shapes("500x100,1000x100,500x5,250x5")
    )
    parser.add_argument(
        "--methods", default="qdwh,zolo-pd,ns,polar_express",
        help="Comma-separated polar methods.",
    )
    parser.add_argument(
        "--spectra", default="gaussian",
        help="Comma-separated from gaussian,ill_conditioned,rank_deficient.",
    )
    parser.add_argument("--condition-number", type=float, default=1e6)
    parser.add_argument("--inner-steps", type=parse_int_csv, default=parse_int_csv("2,5"))
    parser.add_argument("--calls", type=int, default=20)
    parser.add_argument("--warmup-calls", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute-reference", action="store_true")
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
        help="PyTorch float32 matmul precision; recorded in the output metadata.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Annotate measured oracle blocks for Nsight profiling.",
    )
    parser.add_argument("--allow-dirty-git", action="store_true")
    parser.add_argument("--allow-mixed-runs", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--profile-calls", type=int, default=5)
    parser.add_argument("--output-dir", default="results/oracles")
    args = parser.parse_args()

    if min(args.calls, args.repeats) <= 0 or args.warmup_calls < 0:
        parser.error("calls/repeats must be positive and warmup-calls nonnegative")
    if args.condition_number < 1.0:
        parser.error("condition-number must be at least 1")
    methods = parse_csv(args.methods)
    spectra = parse_csv(args.spectra)
    valid_methods = {"qdwh", "zolo-pd", "ns", "polar_express"}
    valid_spectra = {"gaussian", "ill_conditioned", "rank_deficient"}
    if not set(methods) <= valid_methods:
        parser.error(f"Unknown methods: {sorted(set(methods) - valid_methods)}")
    if not set(spectra) <= valid_spectra:
        parser.error(f"Unknown spectra: {sorted(set(spectra) - valid_spectra)}")

    ensure_clean_git(allow_dirty=args.allow_dirty_git)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    output = Path(args.output_dir)
    system = prepare_benchmark_output(
        str(output),
        device=device,
        allow_mixed_runs=args.allow_mixed_runs,
    )
    records = []

    for rows, columns in args.shapes:
        for spectrum in spectra:
            matrix = make_matrix(
                rows,
                columns,
                spectrum,
                args.condition_number,
                args.seed,
                device,
                dtype,
            )
            # The polar factor is nonunique on the null space of a rank-deficient
            # matrix, so a direction error is not well defined in that regime.
            reference = None
            reference_nuclear_norm = None
            if args.compute_reference:
                exact_reference, reference_nuclear_norm = exact_polar_reference(
                    matrix
                )
                if spectrum != "rank_deficient":
                    reference = exact_reference
            for method in methods:
                inner_steps_values = [args.inner_steps[0]] if method == "zolo-pd" else args.inner_steps
                for inner_steps in inner_steps_values:
                    metadata = {
                        "shape": f"{rows}x{columns}",
                        "rows": rows,
                        "columns": columns,
                        "spectrum": spectrum,
                        "condition_number": (
                            args.condition_number if spectrum == "ill_conditioned" else None
                        ),
                        "dtype": args.dtype,
                        "device": str(device),
                    }
                    print(
                        f"Benchmarking {method} (inner_steps={inner_steps}) on "
                        f"{rows}x{columns}, spectrum={spectrum}"
                    )
                    try:
                        records.extend(
                            benchmark_configuration(
                                matrix=matrix,
                                method=method,
                                inner_steps=inner_steps,
                                calls=args.calls,
                                warmup_calls=args.warmup_calls,
                                repeats=args.repeats,
                                device=device,
                                metadata=metadata,
                                reference=reference,
                                reference_nuclear_norm=reference_nuclear_norm,
                                nvtx=args.nvtx,
                            )
                        )
                        if args.profile:
                            trace_name = (
                                f"profile_{rows}x{columns}_{spectrum}_{method}_"
                                f"steps{inner_steps}.json"
                            )
                            profile_configuration(
                                matrix,
                                method,
                                inner_steps,
                                args.profile_calls,
                                output / trace_name,
                                device,
                            )
                    except (RuntimeError, ValueError) as exc:
                        if not args.continue_on_error:
                            raise
                        records.append(
                            {
                                **metadata,
                                "method": method,
                                "inner_steps": inner_steps if method != "zolo-pd" else None,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        print(f"  skipped after error: {exc}")

    summaries = summarize(records)
    raw_csv = output / "polar_oracle_benchmark_raw.csv"
    summary_csv = output / "polar_oracle_benchmark_summary.csv"
    json_path = output / "polar_oracle_benchmark.json"
    write_csv(raw_csv, records)
    write_csv(summary_csv, summaries)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "system": system,
                "arguments": vars(args),
                "records": records,
                "summary": summaries,
            },
            handle,
            indent=2,
            default=str,
        )
    print(f"Saved {json_path}, {raw_csv}, and {summary_csv}")


if __name__ == "__main__":
    main()
