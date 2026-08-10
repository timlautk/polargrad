# Copyright 2026 Tim Tsz-Kit Lau.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Validate and aggregate multi-seed optimizer benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


IDENTITY_KEYS = (
    "git_commit",
    "source_tree_sha256",
    "torch_version",
    "device",
    "cuda_version",
    "gpu_name",
    "gpu_compute_capability",
    "float32_matmul_precision",
    "benchmark_schema_version",
)


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_threshold(value):
    try:
        experiment, metric, threshold = value.split(":", maxsplit=2)
        return experiment, metric, float(threshold)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Thresholds must have the form experiment:metric:value."
        ) from exc


def write_csv(path, rows):
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_identity(data, manifest, path):
    system = data.get("system", {})
    mismatches = {
        key: (manifest.get(key), system.get(key))
        for key in IDENTITY_KEYS
        if manifest.get(key) != system.get(key)
    }
    if mismatches:
        details = "; ".join(
            f"{key}: {expected!r} != {actual!r}"
            for key, (expected, actual) in mismatches.items()
        )
        raise RuntimeError(f"Incompatible run metadata in {path.name}: {details}")
    if system.get("git_dirty"):
        raise RuntimeError(f"Dirty-source publication run found in {path.name}.")


def aggregate_records(runs, expected_seeds, expected_repeats):
    groups = defaultdict(list)
    for run in runs:
        experiment = run["experiment"]
        seed = int(run["seed"])
        for record in run["records"]:
            groups[(experiment, record["name"])].append((seed, record))

    metrics = (
        "wall_time_s",
        "step_time_ms",
        "steps_per_s",
        "forward_backward_ms",
        "optimizer_step_ms",
        "optimizer_fraction_pct",
        "cuda_stream_span_fraction_pct",
        "optimizer_state_mib",
        "peak_allocated_mib",
        "incremental_peak_allocated_mib",
        "estimated_temporary_workspace_mib",
        "final_loss",
        "final_loss_per_entry",
        "objective_gap",
    )
    summaries = []
    for (experiment, name), observations in sorted(groups.items()):
        by_seed = defaultdict(list)
        for seed, record in observations:
            by_seed[seed].append(record)
        observed_seeds = sorted(by_seed)
        if expected_seeds and observed_seeds != expected_seeds:
            raise RuntimeError(
                f"{experiment}/{name} has seeds {observed_seeds}; expected {expected_seeds}."
            )
        if expected_repeats:
            bad = {
                seed: len(records)
                for seed, records in by_seed.items()
                if len(records) != expected_repeats
            }
            if bad:
                raise RuntimeError(
                    f"{experiment}/{name} has unexpected repeat counts: {bad}."
                )
        row = {
            "experiment": experiment,
            "name": name,
            "seeds": len(by_seed),
            "observations": len(observations),
        }
        for metric in metrics:
            raw_values = [
                float(record[metric])
                for _, record in observations
                if record.get(metric) is not None
            ]
            if not raw_values:
                continue
            seed_medians = [
                statistics.median(
                    float(record[metric])
                    for record in records
                    if record.get(metric) is not None
                )
                for records in by_seed.values()
                if any(record.get(metric) is not None for record in records)
            ]
            row[f"{metric}_median"] = statistics.median(raw_values)
            row[f"{metric}_repeat_std"] = statistics.pstdev(raw_values)
            row[f"{metric}_seed_mean"] = statistics.fmean(seed_medians)
            row[f"{metric}_seed_std"] = statistics.pstdev(seed_medians)
        summaries.append(row)
    return summaries


def aggregate_thresholds(runs, thresholds):
    output = []
    for experiment, metric, threshold in thresholds:
        times = defaultdict(dict)
        for run in runs:
            if run["experiment"] != experiment:
                continue
            seed = int(run["seed"])
            by_name = defaultdict(list)
            for point in run.get("trace", []):
                by_name[point["name"]].append(point)
            for name, trace in by_name.items():
                eligible = [
                    point
                    for point in sorted(trace, key=lambda row: row["step"])
                    if point.get(metric) is not None
                    and float(point[metric]) <= threshold
                ]
                times[name][seed] = (
                    float(eligible[0]["estimated_wall_time_s"])
                    if eligible
                    else None
                )
        for name, seed_times in sorted(times.items()):
            reached = [value for value in seed_times.values() if value is not None]
            output.append(
                {
                    "experiment": experiment,
                    "metric": metric,
                    "threshold": threshold,
                    "name": name,
                    "seeds": len(seed_times),
                    "seeds_reached": len(reached),
                    "reached_fraction": len(reached) / max(len(seed_times), 1),
                    "estimated_time_to_threshold_s_mean": (
                        statistics.fmean(reached) if reached else None
                    ),
                    "estimated_time_to_threshold_s_std": (
                        statistics.pstdev(reached) if reached else None
                    ),
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results_corrected")
    parser.add_argument(
        "--expected-experiments",
        default="mat_quad_reg,mat_log_reg,low_rank_mat_comp",
    )
    parser.add_argument("--expected-seeds", default="42,142,242")
    parser.add_argument("--expected-repeats", type=int, default=3)
    parser.add_argument(
        "--threshold",
        action="append",
        type=parse_threshold,
        default=[],
        help="Optional experiment:metric:value threshold; may be repeated.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    manifest_path = results_dir / "benchmark_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"Missing benchmark manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = sorted(results_dir.glob("*_benchmark.json"))
    if not paths:
        raise RuntimeError(f"No benchmark JSON files found in {results_dir}.")
    runs = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        validate_identity(data, manifest, path)
        runs.append(data)

    expected_seeds = sorted(int(item) for item in parse_csv(args.expected_seeds))
    expected_experiments = set(parse_csv(args.expected_experiments))
    observed_experiments = {str(run["experiment"]) for run in runs}
    missing_experiments = sorted(expected_experiments - observed_experiments)
    if missing_experiments:
        raise RuntimeError(
            f"Missing expected experiments: {', '.join(missing_experiments)}."
        )
    run_ids = [(str(run["experiment"]), int(run["seed"])) for run in runs]
    if len(run_ids) != len(set(run_ids)):
        duplicates = sorted(
            run_id for run_id in set(run_ids) if run_ids.count(run_id) > 1
        )
        raise RuntimeError(f"Duplicate experiment/seed runs found: {duplicates}.")
    for experiment in expected_experiments:
        observed_seeds = sorted(
            int(run["seed"])
            for run in runs
            if run["experiment"] == experiment
        )
        if expected_seeds and observed_seeds != expected_seeds:
            raise RuntimeError(
                f"{experiment} has seeds {observed_seeds}; expected "
                f"{expected_seeds}."
            )
    summaries = aggregate_records(
        runs, expected_seeds, args.expected_repeats
    )
    summary_csv = results_dir / "combined_benchmark_summary.csv"
    summary_json = results_dir / "combined_benchmark_summary.json"
    write_csv(summary_csv, summaries)
    summary_json.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    threshold_rows = aggregate_thresholds(runs, args.threshold)
    if args.threshold and not threshold_rows:
        raise RuntimeError(
            "No convergence traces matched the requested threshold settings."
        )
    threshold_csv = results_dir / "combined_time_to_threshold.csv"
    if threshold_rows:
        write_csv(threshold_csv, threshold_rows)
    print(f"Saved {summary_csv} and {summary_json}.")
    if threshold_rows:
        print(f"Saved {threshold_csv}.")


if __name__ == "__main__":
    main()
