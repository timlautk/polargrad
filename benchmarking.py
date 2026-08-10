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

"""Utilities for reproducible optimizer and polar-oracle benchmarks.

The figure-producing experiment loops compute expensive diagnostics such as
matrix condition numbers at every iteration.  Timing those loops would measure
the diagnostics rather than the optimizer.  The helpers in this module run a
fresh, diagnostics-free copy of each workload, exclude compilation warmup, and
record wall-clock time, device time, and peak CUDA memory separately.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

import torch


MIB = 1024**2
BENCHMARK_SCHEMA_VERSION = "2.0"


def _repository_root() -> Path:
    return Path(__file__).resolve().parent


def _git_command(*args: str) -> Optional[str]:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=_repository_root(),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_tree_sha256() -> str:
    """Hash the Python source used by a run, including uncommitted files."""
    digest = hashlib.sha256()
    repository = _repository_root()
    excluded_parts = {".git", "__pycache__", "fig", "output", "results"}
    for path in sorted(repository.rglob("*.py")):
        relative = path.relative_to(repository)
        if any(part in excluded_parts for part in relative.parts):
            continue
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def ensure_clean_git(*, allow_dirty: bool = False) -> None:
    """Require an immutable source state for publication benchmark runs."""
    status = _git_command("status", "--porcelain", "--untracked-files=all")
    if status is None:
        if not allow_dirty:
            raise RuntimeError(
                "Could not determine the Git worktree state. Commit the benchmark "
                "source or rerun with the explicit dirty-source override."
            )
        return
    if status and not allow_dirty:
        changed = ", ".join(line[3:] for line in status.splitlines()[:8])
        raise RuntimeError(
            "Publication benchmarks require a clean Git worktree. Commit the "
            f"benchmark source first. Changed paths include: {changed}"
        )


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve ``auto`` and fail explicitly for unavailable accelerators."""
    requested = str(device).lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(requested)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested, but torch.backends.mps.is_available() is False.")
    return resolved


def seed_everything(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def tensor_bytes(tensors: Iterable[torch.Tensor]) -> int:
    """Count unique tensor storages approximately, without double-counting views."""
    total = 0
    seen = set()
    for tensor in tensors:
        if not torch.is_tensor(tensor):
            continue
        try:
            key = (tensor.untyped_storage().data_ptr(), tensor.untyped_storage().nbytes())
            nbytes = tensor.untyped_storage().nbytes()
        except (AttributeError, RuntimeError):
            key = (tensor.data_ptr(), tensor.numel(), tensor.element_size())
            nbytes = tensor.numel() * tensor.element_size()
        if key not in seen:
            seen.add(key)
            total += nbytes
    return total


def model_parameter_bytes(model: Optional[torch.nn.Module]) -> int:
    if model is None:
        return 0
    return tensor_bytes(list(model.parameters()) + list(model.buffers()))


def optimizer_state_bytes(optimizer: Optional[torch.optim.Optimizer]) -> int:
    if optimizer is None:
        return 0
    tensors = []
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                tensors.append(value)
    return tensor_bytes(tensors)


class StepBenchmark:
    """Measure a diagnostics-free training loop.

    On CUDA, events measure the forward/backward and optimizer portions without
    synchronizing every iteration.  A single synchronization at the end gives
    end-to-end wall-clock latency while avoiding per-step synchronization bias.
    """

    def __init__(
        self,
        *,
        name: str,
        repeat: int,
        steps: int,
        warmup_steps: int,
        device: torch.device,
        metadata: Optional[Mapping[str, Any]] = None,
        component_sample_every: int = 10,
    ) -> None:
        self.name = name
        self.repeat = repeat
        self.steps = steps
        self.warmup_steps = warmup_steps
        self.device = device
        self.metadata = dict(metadata or {})
        self.component_sample_every = max(int(component_sample_every), 1)
        self._wall_start = 0.0
        self._wall_time_s = 0.0
        self._step_start_time = 0.0
        self._optimizer_start_time = 0.0
        self._forward_backward_ms: List[float] = []
        self._optimizer_step_ms: List[float] = []
        self._step_events = []
        self._step_index = 0
        self._step_number = 0
        self._record_current_step = False
        self._cuda_stream = None
        self._region_start_event = None
        self._region_end_event = None
        self._baseline_allocated = 0
        self._peak_allocated = 0
        self._peak_reserved = 0
        self._final_allocated = 0

    def start(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._baseline_allocated = torch.cuda.memory_allocated(self.device)
            self._cuda_stream = torch.cuda.current_stream(self.device)
            # Construct event objects outside the timed region. Their record
            # calls remain in-region, but Python allocation does not pollute
            # the wall-clock measurement for small matrix workloads.
            sample_offset = min(self.component_sample_every - 1, self.steps - 1)
            sample_count = 1 + (
                self.steps - 1 - sample_offset
            ) // self.component_sample_every
            self._sample_offset = sample_offset
            self._step_events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(sample_count)
            ]
            self._region_start_event = torch.cuda.Event(enable_timing=True)
            self._region_end_event = torch.cuda.Event(enable_timing=True)
        else:
            synchronize(self.device)
        self._wall_start = time.perf_counter()
        if self._region_start_event is not None:
            self._region_start_event.record(self._cuda_stream)

    def start_step(self) -> None:
        if self.device.type == "cuda":
            self._record_current_step = (
                self._step_number >= self._sample_offset
                and (self._step_number - self._sample_offset)
                % self.component_sample_every
                == 0
            )
            if self._record_current_step:
                start, _, _ = self._step_events[self._step_index]
                start.record(self._cuda_stream)
                self._step_index += 1
            self._step_number += 1
        else:
            self._step_start_time = time.perf_counter()

    def start_optimizer(self) -> None:
        if self.device.type == "cuda":
            if self._record_current_step:
                self._step_events[self._step_index - 1][1].record(self._cuda_stream)
        else:
            now = time.perf_counter()
            self._forward_backward_ms.append((now - self._step_start_time) * 1000.0)
            self._optimizer_start_time = now

    def end_step(self) -> None:
        if self.device.type == "cuda":
            if self._record_current_step:
                self._step_events[self._step_index - 1][2].record(self._cuda_stream)
        else:
            self._optimizer_step_ms.append(
                (time.perf_counter() - self._optimizer_start_time) * 1000.0
            )

    def finish(
        self,
        *,
        model: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, Any]:
        if self._region_end_event is not None:
            self._region_end_event.record(self._cuda_stream)
        synchronize(self.device)
        self._wall_time_s = time.perf_counter() - self._wall_start

        cuda_stream_span_ms = None
        if self.device.type == "cuda":
            measured_events = self._step_events[:self._step_index]
            self._forward_backward_ms = [
                start.elapsed_time(optimizer_start)
                for start, optimizer_start, _ in measured_events
            ]
            self._optimizer_step_ms = [
                optimizer_start.elapsed_time(end)
                for _, optimizer_start, end in measured_events
            ]
            self._peak_allocated = torch.cuda.max_memory_allocated(self.device)
            self._peak_reserved = torch.cuda.max_memory_reserved(self.device)
            self._final_allocated = torch.cuda.memory_allocated(self.device)
            cuda_stream_span_ms = self._region_start_event.elapsed_time(
                self._region_end_event
            )

        component_samples = len(self._forward_backward_ms)
        forward_backward_per_step_ms = (
            statistics.median(self._forward_backward_ms)
            if self._forward_backward_ms
            else 0.0
        )
        optimizer_per_step_ms = (
            statistics.median(self._optimizer_step_ms)
            if self._optimizer_step_ms
            else 0.0
        )
        step_time_ms = 1000.0 * self._wall_time_s / max(self.steps, 1)
        optimizer_state_mib = optimizer_state_bytes(optimizer) / MIB
        incremental_peak_mib = max(
            self._peak_allocated - self._baseline_allocated, 0
        ) / MIB

        record: Dict[str, Any] = {
            "name": self.name,
            "repeat": self.repeat,
            "device": str(self.device),
            "steps": self.steps,
            "warmup_steps": self.warmup_steps,
            "component_sample_every": (
                self.component_sample_every if self.device.type == "cuda" else 1
            ),
            "component_sample_offset": (
                self._sample_offset if self.device.type == "cuda" else 0
            ),
            "component_samples": component_samples,
            "wall_time_s": self._wall_time_s,
            "step_time_ms": step_time_ms,
            "steps_per_s": self.steps / max(self._wall_time_s, 1e-12),
            "forward_backward_ms": forward_backward_per_step_ms,
            "forward_backward_ms_mean": (
                statistics.fmean(self._forward_backward_ms)
                if self._forward_backward_ms
                else 0.0
            ),
            "optimizer_step_ms": optimizer_per_step_ms,
            "optimizer_step_ms_mean": (
                statistics.fmean(self._optimizer_step_ms)
                if self._optimizer_step_ms
                else 0.0
            ),
            "optimizer_fraction_pct": 100.0
            * optimizer_per_step_ms
            / max(
                forward_backward_per_step_ms + optimizer_per_step_ms, 1e-12
            ),
            # This is the duration of the current CUDA stream's measured span,
            # not GPU utilization, SM occupancy, or achieved FLOP efficiency.
            "cuda_stream_span_ms": cuda_stream_span_ms,
            "cuda_stream_span_fraction_pct": (
                100.0
                * cuda_stream_span_ms
                / max(1000.0 * self._wall_time_s, 1e-12)
                if cuda_stream_span_ms is not None
                else None
            ),
            "model_parameter_mib": model_parameter_bytes(model) / MIB,
            "optimizer_state_mib": optimizer_state_mib,
            "baseline_allocated_mib": self._baseline_allocated / MIB,
            "final_allocated_mib": self._final_allocated / MIB,
            "peak_allocated_mib": self._peak_allocated / MIB,
            "incremental_peak_allocated_mib": incremental_peak_mib,
            "estimated_temporary_workspace_mib": max(
                incremental_peak_mib - optimizer_state_mib, 0.0
            ),
            "peak_reserved_mib": self._peak_reserved / MIB,
        }
        record.update(self.metadata)
        return record


def _cleanup_state(state: Mapping[str, Any], device: torch.device) -> None:
    if hasattr(state, "clear"):
        state.clear()
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def run_training_benchmark(
    *,
    name: str,
    setup_fn: Callable[[int], Dict[str, Any]],
    step_fn: Callable[[Dict[str, Any], Optional[StepBenchmark]], torch.Tensor],
    seed: int,
    steps: int,
    warmup_steps: int,
    repeats: int,
    device: torch.device,
    metadata: Optional[Mapping[str, Any]] = None,
    final_metrics_fn: Optional[
        Callable[[Mapping[str, Any]], Mapping[str, Any]]
    ] = None,
) -> List[Dict[str, Any]]:
    """Run fresh warmup and measured workloads with identical initialization."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be nonnegative")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    records = []
    for repeat in range(repeats):
        repeat_seed = seed
        if warmup_steps:
            warm_state = setup_fn(repeat_seed)
            for _ in range(warmup_steps):
                step_fn(warm_state, None)
            synchronize(device)
            _cleanup_state(warm_state, device)

        state = setup_fn(repeat_seed)
        benchmark_metadata = {"seed": seed}
        benchmark_metadata.update(dict(metadata or {}))
        recorder = StepBenchmark(
            name=name,
            repeat=repeat,
            steps=steps,
            warmup_steps=warmup_steps,
            device=device,
            metadata=benchmark_metadata,
        )
        recorder.start()
        last_output = None
        for _ in range(steps):
            last_output = step_fn(state, recorder)
        if last_output is None:
            raise RuntimeError("The benchmark workload did not execute any steps.")
        record = recorder.finish(
            model=state.get("model"),
            optimizer=state.get("optimizer"),
        )
        if final_metrics_fn is not None:
            with torch.no_grad():
                final_metrics = final_metrics_fn(state)
        else:
            final_metrics = {"final_loss": last_output.detach()}
        synchronize(device)
        for key, value in final_metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(f"Final metric {key!r} must be scalar.")
                value = value.detach().cpu().item()
            record[key] = float(value)
        records.append(record)
        _cleanup_state(state, device)
    return records


def run_convergence_trace(
    *,
    name: str,
    setup_fn: Callable[[int], Dict[str, Any]],
    step_fn: Callable[[Dict[str, Any], Optional[StepBenchmark]], torch.Tensor],
    metrics_fn: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    seed: int,
    steps: int,
    checkpoint_every: int,
    warmup_steps: int,
    device: torch.device,
    estimated_step_time_s: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Collect a separate convergence trace without biasing timed repeats.

    CUDA checkpoint times use stream events. ``estimated_wall_time_s`` uses the
    median per-step wall time from independent timing repetitions when supplied,
    so checkpoint evaluation does not contaminate the estimate.
    """
    if checkpoint_every <= 0:
        return []
    if warmup_steps:
        warm_state = setup_fn(seed)
        for _ in range(warmup_steps):
            step_fn(warm_state, None)
        synchronize(device)
        _cleanup_state(warm_state, device)

    state = setup_fn(seed)
    checkpoint_steps = list(range(checkpoint_every, steps + 1, checkpoint_every))
    if not checkpoint_steps or checkpoint_steps[-1] != steps:
        checkpoint_steps.append(steps)
    checkpoint_set = set(checkpoint_steps)
    metric_values = []
    events = []
    stream = None
    start_event = None
    if device.type == "cuda":
        stream = torch.cuda.current_stream(device)
        start_event = torch.cuda.Event(enable_timing=True)
        events = [torch.cuda.Event(enable_timing=True) for _ in checkpoint_steps]

    synchronize(device)
    with torch.no_grad():
        initial_metrics = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in metrics_fn(state).items()
        }
    synchronize(device)
    converted_initial_metrics = {}
    for key, value in initial_metrics.items():
        if torch.is_tensor(value):
            if value.numel() != 1:
                raise ValueError(f"Trace metric {key!r} must be scalar.")
            value = value.cpu().item()
        converted_initial_metrics[key] = float(value)
    trace = [
        {
            "name": name,
            "seed": seed,
            "step": 0,
            "cuda_stream_elapsed_s": 0.0 if device.type == "cuda" else None,
            "estimated_wall_time_s": 0.0,
            **converted_initial_metrics,
        }
    ]
    wall_start = time.perf_counter()
    if start_event is not None:
        start_event.record(stream)
    event_index = 0
    for step in range(1, steps + 1):
        step_fn(state, None)
        if step in checkpoint_set:
            with torch.no_grad():
                metric_values.append(
                    {
                        key: value.detach() if torch.is_tensor(value) else value
                        for key, value in metrics_fn(state).items()
                    }
                )
            if device.type == "cuda":
                events[event_index].record(stream)
            else:
                events.append(time.perf_counter() - wall_start)
            event_index += 1
    synchronize(device)
    total_wall_time_s = time.perf_counter() - wall_start
    total_cuda_s = (
        start_event.elapsed_time(events[-1]) / 1000.0
        if device.type == "cuda"
        else None
    )

    for index, (step, metrics) in enumerate(zip(checkpoint_steps, metric_values)):
        converted = {}
        for key, value in metrics.items():
            if torch.is_tensor(value):
                if value.numel() != 1:
                    raise ValueError(f"Trace metric {key!r} must be scalar.")
                value = value.detach().cpu().item()
            converted[key] = float(value)
        if device.type == "cuda":
            cuda_elapsed_s = start_event.elapsed_time(events[index]) / 1000.0
        else:
            cuda_elapsed_s = None
        if estimated_step_time_s is not None:
            estimated_wall_time_s = step * estimated_step_time_s
        elif device.type == "cuda":
            estimated_wall_time_s = total_wall_time_s * cuda_elapsed_s / max(
                total_cuda_s, 1e-12
            )
        else:
            estimated_wall_time_s = events[index]
        trace.append(
            {
                "name": name,
                "seed": seed,
                "step": step,
                "cuda_stream_elapsed_s": cuda_elapsed_s,
                "estimated_wall_time_s": estimated_wall_time_s,
                **converted,
            }
        )
    _cleanup_state(state, device)
    return trace


SUMMARY_METRICS = (
    "wall_time_s",
    "step_time_ms",
    "steps_per_s",
    "forward_backward_ms",
    "optimizer_step_ms",
    "forward_backward_ms_mean",
    "optimizer_step_ms_mean",
    "optimizer_fraction_pct",
    "cuda_stream_span_ms",
    "cuda_stream_span_fraction_pct",
    "final_loss",
    "model_parameter_mib",
    "optimizer_state_mib",
    "baseline_allocated_mib",
    "final_allocated_mib",
    "peak_allocated_mib",
    "incremental_peak_allocated_mib",
    "estimated_temporary_workspace_mib",
    "peak_reserved_mib",
)


def summarize_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record["name"])].append(record)

    summaries = []
    for name, group in groups.items():
        summary: Dict[str, Any] = {
            "name": name,
            "repeats": len(group),
            "device": group[0]["device"],
            "steps": group[0]["steps"],
            "warmup_steps": group[0]["warmup_steps"],
        }
        for metric in SUMMARY_METRICS:
            values = [
                float(record[metric])
                for record in group
                if record.get(metric) is not None
            ]
            if values:
                summary[f"{metric}_mean"] = statistics.fmean(values)
                summary[f"{metric}_median"] = statistics.median(values)
                summary[f"{metric}_std"] = statistics.pstdev(values)
        for metric in ("final_loss_per_entry", "objective_gap"):
            values = [
                float(record[metric])
                for record in group
                if record.get(metric) is not None
            ]
            if values:
                summary[f"{metric}_mean"] = statistics.fmean(values)
                summary[f"{metric}_median"] = statistics.median(values)
                summary[f"{metric}_std"] = statistics.pstdev(values)
        summaries.append(summary)
    return summaries


def system_metadata(device: torch.device) -> Dict[str, Any]:
    git_commit = os.environ.get("POLARGRAD_GIT_COMMIT")
    if git_commit is None:
        git_commit = _git_command("rev-parse", "HEAD")
    git_status = _git_command("status", "--porcelain", "--untracked-files=all")
    git_dirty = bool(git_status) if git_status is not None else None
    metadata: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_dirty_paths": (
            [line[3:] for line in git_status.splitlines()]
            if git_status
            else []
        ),
        "source_tree_sha256": source_tree_sha256(),
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "gpu_name": props.name,
                "gpu_total_memory_mib": props.total_memory / MIB,
                "gpu_compute_capability": f"{props.major}.{props.minor}",
            }
        )
    return metadata


def prepare_benchmark_output(
    output_dir: str,
    *,
    device: torch.device,
    allow_mixed_runs: bool = False,
) -> Dict[str, Any]:
    """Validate an output directory before an expensive benchmark starts."""
    system = system_metadata(device)
    write_or_validate_manifest(
        Path(output_dir),
        system=system,
        allow_mixed_runs=allow_mixed_runs,
    )
    return system


def _write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_benchmark_results(
    *,
    records: List[Mapping[str, Any]],
    experiment: str,
    seed: int,
    output_dir: str,
    device: torch.device,
    traces: Optional[List[Mapping[str, Any]]] = None,
    allow_mixed_runs: bool = False,
) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    system = prepare_benchmark_output(
        str(output),
        device=device,
        allow_mixed_runs=allow_mixed_runs,
    )
    prefix = output / f"{experiment}_seed{seed}_benchmark"
    summaries = summarize_records(records)
    raw_csv = prefix.with_name(prefix.name + "_raw.csv")
    summary_csv = prefix.with_name(prefix.name + "_summary.csv")
    json_path = prefix.with_suffix(".json")
    trace_csv = prefix.with_name(prefix.name + "_trace.csv")
    _write_csv(raw_csv, list(records))
    _write_csv(summary_csv, summaries)
    if traces:
        _write_csv(trace_csv, list(traces))
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": experiment,
                "seed": seed,
                "system": system,
                "records": records,
                "summary": summaries,
                "trace": list(traces or []),
            },
            handle,
            indent=2,
        )
    return {
        "json": str(json_path),
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
        "trace_csv": str(trace_csv) if traces else None,
    }


def write_or_validate_manifest(
    output_dir: Path,
    *,
    system: Mapping[str, Any],
    allow_mixed_runs: bool = False,
) -> Path:
    """Prevent accidental aggregation of runs from different source/hardware."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "benchmark_manifest.json"
    identity_keys = (
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
    identity = {key: system.get(key) for key in identity_keys}
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = {
            key: (existing.get(key), identity.get(key))
            for key in identity_keys
            if existing.get(key) != identity.get(key)
        }
        if mismatches and not allow_mixed_runs:
            details = "; ".join(
                f"{key}: {old!r} != {new!r}"
                for key, (old, new) in mismatches.items()
            )
            raise RuntimeError(
                "The output directory contains benchmarks from an incompatible "
                f"run ({details}). Use a new output directory."
            )
    else:
        stale_json = [
            path.name
            for path in output_dir.glob("*_benchmark.json")
            if path.name != manifest_path.name
        ]
        if stale_json and not allow_mixed_runs:
            raise RuntimeError(
                "The output directory already contains benchmark JSON without a "
                "version manifest. Use a new output directory rather than mixing "
                f"runs: {', '.join(stale_json[:5])}"
            )
        manifest_path.write_text(
            json.dumps(identity, indent=2) + "\n", encoding="utf-8"
        )
    return manifest_path


def print_benchmark_summary(records: Iterable[Mapping[str, Any]]) -> None:
    summaries = summarize_records(records)
    print("\nBenchmark summary (median across repeats)")
    print(
        f"{'method':36s} {'wall (s)':>10s} {'step (ms)':>11s} "
        f"{'optim (ms)':>11s} {'peak MiB':>10s}"
    )
    for row in summaries:
        print(
            f"{row['name'][:36]:36s} "
            f"{row['wall_time_s_median']:10.4f} "
            f"{row['step_time_ms_median']:11.4f} "
            f"{row['optimizer_step_ms_median']:11.4f} "
            f"{row['peak_allocated_mib_median']:10.2f}"
        )
