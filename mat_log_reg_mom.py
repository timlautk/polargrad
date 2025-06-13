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


def main(seed=42, steps=1500):
    # Check device
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Problem setup
    m, n = 1000, 100
    N, q = 10000, 400
    torch.manual_seed(seed)
    A = torch.randn(N, m, device=device)
    B = torch.randn(n, q, device=device)
    C = (torch.rand(N, q, device=device) > 0.5).float()

    # Subsampling utility for mini-batch rows of A and C
    def sample_batch(batch_size=1000):
        idx = torch.randint(0, N, (batch_size,), device=device)
        return A[idx], B, C[idx]

    def run_stochastic_optimizer(optimizer_cls, method='qdwh', steps=steps, lr=5e-2, batch_size=1000, scheduler=False, momentum=False, polar_first=False):
        torch.manual_seed(seed)
        model = MatrixLogisticRegression(m, n).to(device)
        if optimizer_cls == torch.optim.Adam:
            optimizer = optimizer_cls(model.parameters(), lr=lr)
        elif optimizer_cls == PolarGrad:
            if not momentum:
                optimizer = optimizer_cls(model.parameters(), method=method, lr=lr, momentum=0.)
            else:
                if polar_first:
                    optimizer = optimizer_cls(model.parameters(), method=method, lr=lr, momentum=0.95, polar_first=True)
                else:
                    optimizer = optimizer_cls(model.parameters(), method=method, lr=lr, momentum=0.9)
        else:
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr)
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
        losses = []
        condition_numbers_grad = []
        nuc_norms_grad = []
        for _ in tqdm(range(steps), desc=f"optimizer = {optimizer_cls.__name__}, polar decomp method = {method if optimizer_cls != torch.optim.Adam else None}, lr = {lr}, momentum = {momentum}, polar-first = {polar_first}, lr decay = {scheduler if isinstance(scheduler, bool) else scheduler.__class__.__name__}"):
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
    

    # Compare optimizers
    loss_polar_grad, cond_grad_polar_grad, nuc_polar_grad = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=2.5e-7)
    loss_polar_grad_decay, cond_grad_polar_grad_decay, nuc_polar_grad_decay = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=5e-7, scheduler=True)
    loss_polar_grad_polar_first, cond_grad_polar_grad_polar_first, nuc_polar_grad_polar_first = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=5e-7, momentum=True, polar_first=True)
    loss_polar_grad_polar_first_decay, cond_grad_polar_grad_polar_first_decay, nuc_polar_grad_polar_first_decay = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=5e-7, momentum=True, polar_first=True, scheduler=True)
    loss_polar_grad_momentum_first, cond_grad_polar_grad_momentum_first, nuc_polar_grad_momentum_first = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=5e-7, momentum=True)
    loss_polar_grad_momentum_first_decay, cond_grad_polar_grad_momentum_first_decay, nuc_polar_grad_momentum_first_decay = run_stochastic_optimizer(PolarGrad, method='qdwh', lr=5e-7, momentum=True, scheduler=True)
    loss_muon_qdwh, cond_grad_muon_qdwh, nuc_muon_qdwh = run_stochastic_optimizer(Muon_polar, method='qdwh', lr=7.5e-2)
    loss_muon_qdwh_decay, cond_grad_muon_qdwh_decay, nuc_muon_qdwh_decay = run_stochastic_optimizer(Muon_polar, method='qdwh', lr=1.5e-1, scheduler=True)


    ## Plots
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].semilogy(loss_polar_grad, label="PolarSGD", linestyle='-')
    axes[0].semilogy(loss_polar_grad_decay, label=r"PolarSGD (lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_polar_grad_polar_first, label="PolarSGDM (polar-first)", linestyle='-.')
    axes[0].semilogy(loss_polar_grad_polar_first_decay, label=r"PolarSGDM (polar-first; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_polar_grad_momentum_first, label="PolarSGDM (momentum-first)", linestyle='-.')
    axes[0].semilogy(loss_polar_grad_momentum_first_decay, label=r"PolarSGDM (momentum-first; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_muon_qdwh, label="Muon", linestyle='-')
    axes[0].semilogy(loss_muon_qdwh_decay, label=r"Muon (lr $\downarrow$)", linestyle='--')
    axes[0].set_ylabel(r"$\mathsf{f}(X_k)$")

    # Plot condition numbers of gradients
    axes[1].plot(cond_grad_polar_grad, linestyle='-')
    axes[1].plot(cond_grad_polar_grad_decay, linestyle='--')
    axes[1].plot(cond_grad_polar_grad_polar_first, linestyle='-.')
    axes[1].plot(cond_grad_polar_grad_polar_first_decay, linestyle='--')
    axes[1].plot(cond_grad_polar_grad_momentum_first, linestyle='-.')
    axes[1].plot(cond_grad_polar_grad_momentum_first_decay, linestyle='--')
    axes[1].plot(cond_grad_muon_qdwh, linestyle='-')
    axes[1].plot(cond_grad_muon_qdwh_decay, linestyle='--')
    axes[1].set_xlabel(r"iteration $k$")
    axes[1].set_ylabel(r"$\kappa_2(\nabla\mathsf{f}(X_k, \xi_k))$")
    
    # Plot nuclear norms of gradients
    axes[2].plot(nuc_polar_grad, linestyle='-')
    axes[2].plot(nuc_polar_grad_decay, linestyle='--')
    axes[2].plot(nuc_polar_grad_polar_first, linestyle='-.')
    axes[2].plot(nuc_polar_grad_polar_first_decay, linestyle='--')
    axes[2].plot(nuc_polar_grad_momentum_first, linestyle='-.')
    axes[2].plot(nuc_polar_grad_momentum_first_decay, linestyle='--')
    axes[2].plot(nuc_muon_qdwh, linestyle='-')
    axes[2].plot(nuc_muon_qdwh_decay, linestyle='--')
    axes[2].set_xlabel(r"iteration $k$")
    axes[2].set_ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla\mathsf{f}(X_k, \xi_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")

    fig.legend(loc='outside lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), borderaxespad=0., fontsize=16)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(f'fig/mat_log_reg_mom_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig)


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