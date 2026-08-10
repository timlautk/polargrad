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

from benchmarking import MIB, resolve_device, seed_everything, synchronize, system_metadata
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


def polar_residuals(matrix, unitary, reference=None):
    work_dtype = torch.float64 if matrix.dtype == torch.float64 else torch.float32
    matrix = matrix.to(work_dtype)
    unitary = unitary.to(work_dtype)
    rows, columns = matrix.shape
    order = min(rows, columns)
    identity = torch.eye(order, device=matrix.device, dtype=work_dtype)
    if rows >= columns:
        gram = unitary.mT @ unitary
        hermitian = unitary.mT @ matrix
        reconstruction = unitary @ hermitian
    else:
        gram = unitary @ unitary.mT
        hermitian = matrix @ unitary.mT
        reconstruction = hermitian @ unitary
    orthogonality = torch.linalg.matrix_norm(gram - identity) / math.sqrt(order)
    reconstruction_error = torch.linalg.matrix_norm(reconstruction - matrix) / max(
        torch.linalg.matrix_norm(matrix), torch.finfo(work_dtype).eps
    )
    direction_error = None
    if reference is not None:
        reference = reference.to(work_dtype)
        direction_error = torch.linalg.matrix_norm(unitary - reference) / max(
            torch.linalg.matrix_norm(reference), torch.finfo(work_dtype).eps
        )
    return {
        "orthogonality_residual": float(orthogonality.item()),
        "reconstruction_residual": float(reconstruction_error.item()),
        "relative_direction_error": (
            float(direction_error.item()) if direction_error is not None else None
        ),
    }


def exact_polar_factor(matrix):
    left, _, right_h = torch.linalg.svd(matrix, full_matrices=False)
    return left @ right_h


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
            start_event.record(stream)
        else:
            baseline_allocated = 0
            start_event = None
            end_event = None

        wall_start = time.perf_counter()
        unitary = None
        for _ in range(calls):
            unitary = polar(matrix, method=method, max_iterations=inner_steps)[0]
        if end_event is not None:
            end_event.record(stream)
        synchronize(device)
        wall_time = time.perf_counter() - wall_start

        if unitary is None:
            raise RuntimeError("No oracle call was executed.")
        if start_event is not None:
            device_time_ms = start_event.elapsed_time(end_event)
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
        else:
            device_time_ms = wall_time * 1000.0
            peak_allocated = 0
            peak_reserved = 0

        record = {
            **metadata,
            "repeat": repeat,
            "method": method,
            "inner_steps": inner_steps if method != "zolo-pd" else None,
            "calls": calls,
            "warmup_calls": warmup_calls,
            "wall_time_s": wall_time,
            "time_per_call_ms": 1000.0 * wall_time / calls,
            "device_time_per_call_ms": device_time_ms / calls,
            # Timing-based activity proxy; it is not SM occupancy or FLOP efficiency.
            "device_activity_pct": 100.0
            * device_time_ms
            / max(1000.0 * wall_time, 1e-12),
            "baseline_allocated_mib": baseline_allocated / MIB,
            "peak_allocated_mib": peak_allocated / MIB,
            "incremental_peak_allocated_mib": max(
                peak_allocated - baseline_allocated, 0
            )
            / MIB,
            "peak_reserved_mib": peak_reserved / MIB,
            **polar_residuals(matrix, unitary, reference),
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
        "device_time_per_call_ms",
        "device_activity_pct",
        "peak_allocated_mib",
        "incremental_peak_allocated_mib",
        "peak_reserved_mib",
        "orthogonality_residual",
        "reconstruction_residual",
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
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-calls", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
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

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
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
            reference = (
                exact_polar_factor(matrix)
                if args.compute_reference and spectrum != "rank_deficient"
                else None
            )
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
                "system": system_metadata(device),
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
