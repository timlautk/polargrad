# Copyright 2025 Tim Tsz-Kit Lau.
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
# limitations under the License

import os
import statistics
import fire
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

import torch
import torch.nn as nn
import torch.nn.functional as F

from polar_grad import PolarGrad
from muon import Muon_polar
from benchmarking import (
    ensure_clean_git,
    prepare_benchmark_output,
    print_benchmark_summary,
    resolve_device,
    run_convergence_trace,
    run_training_benchmark,
    save_benchmark_results,
    seed_everything,
)


def smooth(scalars: np.array, weight: float = 0.8) -> np.array:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.array(smoothed)

class MatrixLogisticRegression(nn.Module):
    def __init__(self, m=1000, n=100):
        super().__init__()
        self.X = nn.Parameter(torch.empty(m, n).uniform_(-1., 1.))

    def forward(self, A_batch, B, C_batch):
        logits = A_batch @ self.X @ B
        return torch.sum(F.softplus(-C_batch * logits))


def main(
    seed=42,
    steps=1500,
    device="cpu",
    benchmark=False,
    benchmark_only=False,
    benchmark_steps=None,
    benchmark_warmup=10,
    benchmark_repeats=3,
    benchmark_trace_every=10,
    results_dir="results",
    matmul_precision="high",
    allow_dirty_git=False,
    allow_mixed_runs=False,
):
    if str(matmul_precision) not in {"highest", "high", "medium"}:
        raise ValueError("matmul_precision must be highest, high, or medium.")
    torch.set_float32_matmul_precision(str(matmul_precision))
    device = resolve_device(device)
    print(f"Using device: {device}")

    # Problem setup
    m, n = 1000, 100
    N, q = 10000, 400
    torch.manual_seed(seed)
    A = torch.randn(N, m, device=device)
    B = torch.randn(n, q, device=device)
    C = 2.0 * (torch.randn(N, q, device=device) > 0.5).to(A.dtype) - 1.0

    # Subsampling utility for mini-batch rows of A and C
    def sample_batch(batch_size=1000):
        idx = torch.randint(0, N, (batch_size,), device=device)
        return A[idx], B, C[idx]

    def make_optimizer(model, optimizer_cls, method, lr, use_scheduler):
        inner_steps = 5 if method == "ns" else 2
        if optimizer_cls == torch.optim.Adam:
            optimizer = optimizer_cls(model.parameters(), lr=lr)
        elif optimizer_cls == PolarGrad:
            optimizer = optimizer_cls(
                model.parameters(), method=method, lr=lr, momentum=0.,
                inner_steps=inner_steps
            )
        else:
            optimizer = optimizer_cls(
                model.parameters(), method=method, lr=lr,
                inner_steps=inner_steps
            )
        scheduler = None
        if use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=25, gamma=0.95
            )
        return optimizer, scheduler

    def run_benchmarks():
        ensure_clean_git(allow_dirty=bool(allow_dirty_git))
        prepare_benchmark_output(
            results_dir,
            device=device,
            allow_mixed_runs=bool(allow_mixed_runs),
        )
        measured_steps = int(benchmark_steps or steps)
        batch_size = 1000
        configurations = [
            ("PolarSGD (QDWH)", PolarGrad, "qdwh", 1.5e-7, False),
            ("PolarSGD (QDWH; lr decay)", PolarGrad, "qdwh", 2.5e-7, True),
            ("Muon (NS)", Muon_polar, "ns", 1e-1, False),
            ("Muon (QDWH)", Muon_polar, "qdwh", 1e-1, False),
            ("Muon (QDWH; lr decay)", Muon_polar, "qdwh", 2e-1, True),
            ("Adam", torch.optim.Adam, None, 5e-3, False),
            ("Adam (lr decay)", torch.optim.Adam, None, 1e-2, True),
        ]
        records = []
        traces = []
        for name, optimizer_cls, method, lr, use_scheduler in configurations:
            def setup_fn(run_seed, optimizer_cls=optimizer_cls, method=method,
                         lr=lr, use_scheduler=use_scheduler):
                seed_everything(run_seed, device)
                model = MatrixLogisticRegression(m, n).to(device)
                optimizer, scheduler = make_optimizer(
                    model, optimizer_cls, method, lr, use_scheduler
                )
                return {
                    "model": model,
                    "optimizer": optimizer,
                    "scheduler": scheduler,
                }

            def step_fn(state, recorder):
                if recorder is not None:
                    recorder.start_step()
                A_batch, B_batch, C_batch = sample_batch(batch_size)
                model = state["model"]
                optimizer = state["optimizer"]
                optimizer.zero_grad(set_to_none=True)
                loss = model(A_batch, B_batch, C_batch)
                loss.backward()
                if recorder is not None:
                    recorder.start_optimizer()
                optimizer.step()
                if state["scheduler"] is not None:
                    state["scheduler"].step()
                if recorder is not None:
                    recorder.end_step()
                return loss

            def final_metrics_fn(state):
                final_loss = state["model"](A, B, C)
                return {
                    "final_loss": final_loss,
                    "final_loss_per_entry": final_loss / C.numel(),
                }

            configuration_records = run_training_benchmark(
                    name=name,
                    setup_fn=setup_fn,
                    step_fn=step_fn,
                    seed=seed,
                    steps=measured_steps,
                    warmup_steps=int(benchmark_warmup),
                    repeats=int(benchmark_repeats),
                    device=device,
                    final_metrics_fn=final_metrics_fn,
                    metadata={
                        "optimizer": optimizer_cls.__name__,
                        "polar_method": method,
                        "inner_steps": (
                            5 if method == "ns" else
                            2 if method == "qdwh" else None
                        ),
                        "learning_rate": lr,
                        "lr_decay": use_scheduler,
                        "batch_size": batch_size,
                        "matrix_shape": f"{m}x{n}",
                        "data_shapes": f"A:{N}x{m};B:{n}x{q};C:{N}x{q}",
                        "dtype": str(A.dtype),
                        "label_encoding": "{-1,+1}",
                    },
                )
            records.extend(configuration_records)
            traces.extend(
                run_convergence_trace(
                    name=name,
                    setup_fn=setup_fn,
                    step_fn=step_fn,
                    metrics_fn=final_metrics_fn,
                    seed=seed,
                    steps=measured_steps,
                    checkpoint_every=int(benchmark_trace_every),
                    warmup_steps=int(benchmark_warmup),
                    device=device,
                    estimated_step_time_s=statistics.median(
                        record["step_time_ms"] for record in configuration_records
                    )
                    / 1000.0,
                )
            )
        print_benchmark_summary(records)
        paths = save_benchmark_results(
            records=records,
            experiment="mat_log_reg",
            seed=seed,
            output_dir=results_dir,
            device=device,
            traces=traces,
            allow_mixed_runs=bool(allow_mixed_runs),
        )
        print(f"Saved benchmark results: {paths}")
        return records

    def run_stochastic_optimizer(optimizer_cls, method='qdwh', lr=5e-2, steps=steps, batch_size=1000, scheduler=False):
        torch.manual_seed(seed)
        model = MatrixLogisticRegression(m, n).to(device)
        if optimizer_cls == torch.optim.Adam:
            optimizer = optimizer_cls(model.parameters(), lr=lr)
        elif optimizer_cls == PolarGrad:
            optimizer = optimizer_cls(
                model.parameters(), method=method, lr=lr, momentum=0.,
                inner_steps=5 if method == 'ns' else 2
            )
        else:
            optimizer = optimizer_cls(
                model.parameters(), method=method, lr=lr,
                inner_steps=5 if method == 'ns' else 2
            )
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
        losses = []
        condition_numbers_grad = []
        nuc_norms_grad = []
        for _ in tqdm(range(steps), desc=f"optimizer = {optimizer_cls.__name__}, polar decomp method = {method if optimizer_cls != torch.optim.Adam else None}, lr decay = {scheduler if isinstance(scheduler, bool) else scheduler.__class__.__name__}"):
            A_batch, B_batch, C_batch = sample_batch(batch_size)
            optimizer.zero_grad()
            loss = model(A_batch, B_batch, C_batch)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            losses.append(loss.item())
            condition_numbers_grad.append(torch.linalg.cond(model.X.grad).item())
            nuc_norms_grad.append(torch.linalg.matrix_norm(model.X.grad, ord='nuc').item())
        condition_numbers_grad = smooth(condition_numbers_grad, weight=0.8)
        nuc_norms_grad = smooth(nuc_norms_grad, weight=0.8)
        return losses, condition_numbers_grad, nuc_norms_grad

    if benchmark_only:
        run_benchmarks()
        return

    # Compare optimizers
    loss_polar_grad, cond_grad_polar_grad, nuc_polar_grad = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=1.5e-7)
    loss_polar_grad_decay, cond_grad_polar_grad_decay, nuc_polar_grad_decay = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=2.5e-7, scheduler=True)
    loss_muon, cond_grad_muon, nuc_muon = run_stochastic_optimizer(Muon_polar, method='ns', lr=1e-1)
    loss_muon_qdwh, cond_grad_muon_qdwh, nuc_muon_qdwh = run_stochastic_optimizer(Muon_polar, method='qdwh', lr=1e-1)
    loss_muon_qdwh_decay, cond_grad_muon_qdwh_decay, nuc_muon_qdwh_decay = run_stochastic_optimizer(Muon_polar, method='qdwh', lr=2e-1, scheduler=True)
    loss_adam, cond_grad_adam, nuc_adam = run_stochastic_optimizer(torch.optim.Adam, lr=5e-3)
    loss_adam_decay, cond_grad_adam_decay, nuc_adam_decay = run_stochastic_optimizer(torch.optim.Adam, lr=1e-2, scheduler=True)


    ## Plots
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].semilogy(loss_polar_grad, label="PolarSGD (QDWH)", linestyle='-')
    axes[0].semilogy(loss_polar_grad_decay, label=r"PolarSGD (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_muon, label="Muon (NS)", linestyle='-.')
    axes[0].semilogy(loss_muon_qdwh, label="Muon (QDWH)", linestyle='-')
    axes[0].semilogy(loss_muon_qdwh_decay, label=r"Muon (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_adam, label="Adam", linestyle='-')
    axes[0].semilogy(loss_adam_decay, label=r"Adam (lr $\downarrow$)", linestyle='--')
    axes[0].set_xlabel(r"iteration $k$")
    axes[0].set_ylabel(r"$\mathsf{f}(X_k)$")

    # Plot condition numbers of gradients
    axes[1].plot(cond_grad_polar_grad, linestyle='-')
    axes[1].plot(cond_grad_polar_grad_decay, linestyle='--')
    axes[1].plot(cond_grad_muon, linestyle='-.')
    axes[1].plot(cond_grad_muon_qdwh, linestyle='-')
    axes[1].plot(cond_grad_muon_qdwh_decay, linestyle='--')
    axes[1].plot(cond_grad_adam, linestyle='-')
    axes[1].plot(cond_grad_adam_decay, linestyle='--')
    axes[1].set_xlabel(r"iteration $k$")
    axes[1].set_ylabel(r"$\kappa_2(\nabla\mathsf{f}(X_k, \xi_k))$")
    
    # Plot nuclear norms of gradients
    axes[2].plot(nuc_polar_grad, linestyle='-')
    axes[2].plot(nuc_polar_grad_decay, linestyle='--')
    axes[2].plot(nuc_muon, linestyle='-.')
    axes[2].plot(nuc_muon_qdwh, linestyle='-')
    axes[2].plot(nuc_muon_qdwh_decay, linestyle='--')
    axes[2].plot(nuc_adam, linestyle='-')
    axes[2].plot(nuc_adam_decay, linestyle='--')
    axes[2].set_xlabel(r"iteration $k$")
    axes[2].set_ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla\mathsf{f}(X_k, \xi_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")

    fig.legend(loc='outside lower center', ncol=7, bbox_to_anchor=(0.5, -0.05), borderaxespad=0., fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f'fig/mat_log_reg_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig)


    # Plot nuclear norms of gradients separately
    fig2 = plt.figure(figsize=(7, 5))
    plt.plot(nuc_polar_grad, linestyle='--')
    plt.plot(nuc_polar_grad_decay, linestyle='-.')
    plt.plot(nuc_muon, linestyle='-')
    plt.plot(nuc_muon_qdwh, linestyle='--')
    plt.plot(nuc_muon_qdwh_decay, linestyle='-.')
    plt.plot(nuc_adam, linestyle='-')
    plt.plot(nuc_adam_decay, linestyle='-.')
    plt.xlabel(r"iteration $k$")
    plt.ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla\mathsf{f}(X_k, \xi_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")
    fig2.savefig(f'fig/mat_log_reg_nuc_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig2)

    if benchmark:
        run_benchmarks()


if __name__ == "__main__":
    if not os.path.exists('fig'):
        os.makedirs('fig')
    # Default settings
    mpl.rcParams.update(mpl.rcParamsDefault)
    plt.style.use(['science', 'grid', 'notebook'])
    
    # These are the colors that will be used in the plot
    tab10_colors = list(plt.get_cmap('tab10').colors)     # 10 colors
    dark2_colors = plt.get_cmap('Dark2').colors           # 8 colors

    # Pick two distinct additions
    additional_colors = [dark2_colors[3], dark2_colors[5]]

    # Combine to make 12-color palette
    color_sequence = tab10_colors + additional_colors

    plt.rcParams.update({
        "text.usetex": True,
        "axes.prop_cycle": plt.cycler(color=color_sequence),
        } 
        )
    
    torch.set_float32_matmul_precision("high")
    torch.set_printoptions(precision=8)
    
    fire.Fire(main)
