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
            self._step_events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(
                    (self.steps + self.component_sample_every - 1)
                    // self.component_sample_every
                )
            ]
        else:
            synchronize(self.device)
        self._wall_start = time.perf_counter()

    def start_step(self) -> None:
        if self.device.type == "cuda":
            self._record_current_step = (
                self._step_number % self.component_sample_every == 0
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
        final_loss: torch.Tensor,
        model: Optional[torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer],
    ) -> Dict[str, Any]:
        synchronize(self.device)
        self._wall_time_s = time.perf_counter() - self._wall_start

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

        forward_backward_ms = sum(self._forward_backward_ms)
        optimizer_step_ms = sum(self._optimizer_step_ms)
        component_samples = max(len(self._forward_backward_ms), 1)
        forward_backward_per_step_ms = forward_backward_ms / component_samples
        optimizer_per_step_ms = optimizer_step_ms / component_samples
        estimated_device_activity_ms = (
            forward_backward_per_step_ms + optimizer_per_step_ms
        ) * self.steps
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
            "component_samples": component_samples,
            "wall_time_s": self._wall_time_s,
            "step_time_ms": step_time_ms,
            "steps_per_s": self.steps / max(self._wall_time_s, 1e-12),
            "forward_backward_ms": forward_backward_per_step_ms,
            "optimizer_step_ms": optimizer_per_step_ms,
            "optimizer_fraction_pct": 100.0
            * optimizer_step_ms
            / max(forward_backward_ms + optimizer_step_ms, 1e-12),
            # This is a timing-based activity proxy, not hardware SM occupancy.
            "device_activity_pct": 100.0
            * estimated_device_activity_ms
            / max(1000.0 * self._wall_time_s, 1e-12),
            "final_loss": float(final_loss.detach().cpu().item()),
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
        final_loss = None
        for _ in range(steps):
            final_loss = step_fn(state, recorder)
        if final_loss is None:
            raise RuntimeError("The benchmark workload did not execute any steps.")
        records.append(
            recorder.finish(
                final_loss=final_loss,
                model=state.get("model"),
                optimizer=state.get("optimizer"),
            )
        )
        _cleanup_state(state, device)
    return records


SUMMARY_METRICS = (
    "wall_time_s",
    "step_time_ms",
    "steps_per_s",
    "forward_backward_ms",
    "optimizer_step_ms",
    "optimizer_fraction_pct",
    "device_activity_pct",
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
            values = [float(record[metric]) for record in group]
            summary[f"{metric}_mean"] = statistics.fmean(values)
            summary[f"{metric}_median"] = statistics.median(values)
            summary[f"{metric}_std"] = statistics.pstdev(values)
        summaries.append(summary)
    return summaries


def system_metadata(device: torch.device) -> Dict[str, Any]:
    git_commit = os.environ.get("POLARGRAD_GIT_COMMIT")
    git_dirty = None
    if git_commit is None:
        try:
            repository = Path(__file__).resolve().parent
            git_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            git_dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=repository,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()
            )
        except (OSError, subprocess.CalledProcessError):
            git_commit = None
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
) -> Dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = output / f"{experiment}_seed{seed}_benchmark"
    summaries = summarize_records(records)
    raw_csv = prefix.with_name(prefix.name + "_raw.csv")
    summary_csv = prefix.with_name(prefix.name + "_summary.csv")
    json_path = prefix.with_suffix(".json")
    _write_csv(raw_csv, list(records))
    _write_csv(summary_csv, summaries)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "experiment": experiment,
                "seed": seed,
                "system": system_metadata(device),
                "records": records,
                "summary": summaries,
            },
            handle,
            indent=2,
        )
    return {
        "json": str(json_path),
        "raw_csv": str(raw_csv),
        "summary_csv": str(summary_csv),
    }


def print_benchmark_summary(records: Iterable[Mapping[str, Any]]) -> None:
    summaries = summarize_records(records)
    print("\nBenchmark summary (median across repeats)")
    print(
        f"{'method':36s} {'wall (s)':>10s} {'step (ms)':>11s} "
        f"{'optim (ms)':>11s} {'peak MiB':>10s} {'activity %':>11s}"
    )
    for row in summaries:
        print(
            f"{row['name'][:36]:36s} "
            f"{row['wall_time_s_median']:10.4f} "
            f"{row['step_time_ms_median']:11.4f} "
            f"{row['optimizer_step_ms_median']:11.4f} "
            f"{row['peak_allocated_mib_median']:10.2f} "
            f"{row['device_activity_pct_median']:11.2f}"
        )
