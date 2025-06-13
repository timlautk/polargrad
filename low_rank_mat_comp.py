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

from polar_grad import PolarGrad
from muon import Muon_polar


def smooth(scalars: np.array, weight: float = 0.9) -> np.array:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.array(smoothed)

# Factorization model: X = A @ B^T
class LowRankModel(nn.Module):
    def __init__(self, m=500, n=250, r=5):
        super().__init__()
        self.X = nn.Parameter(torch.empty(m, r).uniform_(-1., 1.))
        self.Y = nn.Parameter(torch.empty(n, r).uniform_(-1., 1.))

    def forward(self, target, mask):
        mse_loss = torch.sum((self.X @ self.Y.T - target) ** 2) / mask.sum()
        return mse_loss

class LowRankModelAltGD(nn.Module):
    def __init__(self, m=500, n=250, r=5, lr=1e-2):
        """
        Low-Rank Matrix Factorization Model: M ≈ U V^T

        Args:
            m: number of rows of M
            n: number of columns of M
            r: target rank
            weight_decay: regularization strength (default 0)
        """
        super().__init__()
        self.m = m
        self.n = n
        self.r = r
        self.lr = lr

        # Initialize X and Y
        self.X = nn.Parameter(torch.empty(m, r).uniform_(-1., 1.))
        self.Y = nn.Parameter(torch.empty(n, r).uniform_(-1., 1.))

    def loss(self, target, mask):
        """
        Computes the masked loss: squared error only over observed entries.

        Args:
            target: observed matrix (m, n)
            mask: binary mask (m, n), 1 if observed, 0 if missing
        Returns:
            scalar loss        """
        mse_loss = torch.sum((self.X @ self.Y.T - target) ** 2) / mask.sum()
        return mse_loss

    def alternating_gradient_step(self, target, mask, num_inner_steps=1):
        """Performs one step of masked alternating gradient descent."""
        # Update X while fixing Y
        for _ in range(num_inner_steps):
            loss_X = self.loss(target, mask)
            grad_X = torch.autograd.grad(loss_X, self.X, retain_graph=True)[0]
            self.X.data = self.X.data - self.lr * grad_X

        # Update Y while fixing X
        for _ in range(num_inner_steps):
            loss_Y = self.loss(target, mask)
            grad_Y = torch.autograd.grad(loss_Y, self.Y, retain_graph=False)[0]
            self.Y.data = self.Y.data - self.lr * grad_Y
        
        return torch.linalg.cond(grad_X), torch.linalg.cond(grad_Y), torch.linalg.matrix_norm(grad_X, ord='nuc'), torch.linalg.matrix_norm(grad_Y, ord='nuc')

    def fit(self, target, mask, steps=1000, num_inner_steps=1):
        """
        Fit the model to observed entries using masked alternating minimization.

        Args:
            target: observed matrix (m, n)
            mask: binary mask (m, n)
            steps: number of alternating minimization steps
            num_inner_steps: number of least-squares solves per U/V update
        """
        losses = []
        condition_numbers_grad_X = []
        condition_numbers_grad_Y = []
        nuc_norms_grad_X = []
        nuc_norms_grad_Y = []
        for i in tqdm(range(steps), desc=f"optimizer = AltGD"):
            cond_grad_X, cond_grad_Y, nuc_grad_X, nuc_grad_Y = self.alternating_gradient_step(target, mask, num_inner_steps=num_inner_steps)
            current_loss = self.loss(target, mask)
            losses.append(current_loss.item())
            condition_numbers_grad_X.append(cond_grad_X.item())
            condition_numbers_grad_Y.append(cond_grad_Y.item())
            nuc_norms_grad_X.append(nuc_grad_X.item())
            nuc_norms_grad_Y.append(nuc_grad_Y.item())
        condition_numbers_grad_X = smooth(condition_numbers_grad_X)
        condition_numbers_grad_Y = smooth(condition_numbers_grad_Y)
        return losses, condition_numbers_grad_X, condition_numbers_grad_Y, nuc_norms_grad_X, nuc_norms_grad_Y


def main(seed=42, steps=1000):
    # Check device
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    m, n, r = 500, 250, 5
    U_true = torch.randn(m, r, device=device)
    V_true = torch.randn(n, r, device=device)
    M = U_true @ V_true.T  # Ground truth low-rank matrix

    # Observed entries mask (simulate missing data)
    mask = (torch.rand(m, n, device=device) < 0.3).float()

    # Training loop
    def train_lowrank(optimizer_cls, method='qdwh', lr=0.1, steps=steps, scheduler=False):
        torch.manual_seed(seed)
        model = LowRankModel()
        model = model.to(device)
        if optimizer_cls == torch.optim.Adam:
            optimizer = optimizer_cls(model.parameters(), lr=lr)
        elif optimizer_cls == PolarGrad:
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr, momentum=0.)
        else:
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr)
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.95)
        losses = []
        condition_numbers_grad_X = []
        condition_numbers_grad_Y = []
        nuc_norms_grad_X = []
        nuc_norms_grad_Y = []
        for _ in tqdm(range(steps), desc=f"optimizer = {optimizer_cls.__name__}, polar decomp method = {method if optimizer_cls != torch.optim.Adam else None}, lr decay = {scheduler if isinstance(scheduler, bool) else scheduler.__class__.__name__}"):
            loss = model(M, mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            losses.append(loss.item())
            condition_numbers_grad_X.append(torch.linalg.cond(model.X.grad).item())
            condition_numbers_grad_Y.append(torch.linalg.cond(model.Y.grad).item())
            nuc_norms_grad_X.append(torch.linalg.matrix_norm(model.X.grad, ord='nuc').item())
            nuc_norms_grad_Y.append(torch.linalg.matrix_norm(model.Y.grad, ord='nuc').item())
        condition_numbers_grad_X = smooth(condition_numbers_grad_X)
        condition_numbers_grad_Y = smooth(condition_numbers_grad_Y)
        return losses, condition_numbers_grad_X, condition_numbers_grad_Y, nuc_norms_grad_X, nuc_norms_grad_Y

    # Compare optimizers
    loss_polar_grad_lr, cond_X_polar_grad_lr, cond_Y_polar_grad_lr, nuc_X_polar_grad_lr, nuc_Y_polar_grad_lr = train_lowrank(PolarGrad, method='qdwh', lr=1.5e1)
    loss_polar_grad_lr_decay, cond_X_polar_grad_lr_decay, cond_Y_polar_grad_lr_decay, nuc_X_polar_grad_lr_decay, nuc_Y_polar_grad_lr_decay = train_lowrank(PolarGrad, method='qdwh', lr=1.5e1, scheduler=True)
    loss_muon_qdwh_lr, cond_X_muon_qdwh_lr, cond_Y_muon_qdwh_lr, nuc_X_muon_qdwh_lr, nuc_Y_muon_qdwh_lr  = train_lowrank(Muon_polar, method='qdwh', lr=2.5e-1) # 5e-1
    loss_muon_qdwh_lr_decay, cond_X_muon_qdwh_lr_decay, cond_Y_muon_qdwh_lr_decay, nuc_X_muon_qdwh_lr_decay, nuc_Y_muon_qdwh_lr_decay = train_lowrank(Muon_polar, method='qdwh', lr=2.5e-1, scheduler=True)
    loss_muon_ns_lr, cond_X_muon_ns_lr, cond_Y_muon_ns_lr, nuc_X_muon_ns_lr, nuc_Y_muon_ns_lr  = train_lowrank(Muon_polar, method='ns', lr=2.5e-1)
    loss_adam_lr, cond_X_adam_lr, cond_Y_adam_lr, nuc_X_adam_lr, nuc_Y_adam_lr = train_lowrank(torch.optim.Adam, lr=5e-2) # 5e-3
    loss_adam_lr_decay, cond_X_adam_lr_decay, cond_Y_adam_lr_decay, nuc_X_adam_lr_decay, nuc_Y_adam_lr_decay = train_lowrank(torch.optim.Adam, lr=5e-2, scheduler=True)

    torch.manual_seed(seed)
    loss_altgd, cond_X_altgd, cond_Y_altgd, nuc_X_altgd, nuc_Y_altgd = LowRankModelAltGD(lr=5e1).to(device).fit(M, mask, steps=steps)

    ## Plots
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].semilogy(loss_polar_grad_lr, label="PolarGrad (QDWH)", linestyle='-')
    axes[0].semilogy(loss_polar_grad_lr_decay, label=r"PolarGrad (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_muon_ns_lr, label="Muon (NS)", linestyle='-.')
    axes[0].semilogy(loss_muon_qdwh_lr, label="Muon (QDWH)", linestyle='-')
    axes[0].semilogy(loss_muon_qdwh_lr_decay, label=r"Muon (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_adam_lr, label="Adam", linestyle='-')
    axes[0].semilogy(loss_adam_lr_decay, label=r"Adam (lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_altgd, label="AltGD", linestyle=':')
    axes[0].set_xlabel(r"iteration $k$")
    axes[0].set_ylabel(r"$\mathsf{f}(X_k,Y_k)$")

    # Plot condition numbers of gradients of X
    axes[1].plot(cond_X_polar_grad_lr, linestyle='-')
    axes[1].plot(cond_X_polar_grad_lr_decay, linestyle='--')
    axes[1].plot(cond_X_muon_ns_lr, linestyle='-.')
    axes[1].plot(cond_X_muon_qdwh_lr, linestyle='-')
    axes[1].plot(cond_X_muon_qdwh_lr_decay, linestyle='--')
    axes[1].plot(cond_X_adam_lr, linestyle='-')
    axes[1].plot(cond_X_adam_lr_decay, linestyle='--')
    # axes[1].plot(cond_X_altgd, linestyle=':')
    axes[1].set_xlabel(r"iteration $k$")
    axes[1].set_ylabel(r"$\kappa_2(\nabla_X \mathsf{f}(X_k, Y_k))$")

    # Plot condition numbers of gradients of Y
    axes[2].plot(cond_Y_polar_grad_lr, linestyle='-')
    axes[2].plot(cond_Y_polar_grad_lr_decay, linestyle='--')
    axes[2].plot(cond_Y_muon_ns_lr, linestyle='-.')
    axes[2].plot(cond_Y_muon_qdwh_lr, linestyle='-')
    axes[2].plot(cond_Y_muon_qdwh_lr_decay, linestyle='--')
    axes[2].plot(cond_Y_adam_lr, linestyle='-')
    axes[2].plot(cond_Y_adam_lr_decay, linestyle='--')
    # axes[2].plot(cond_Y_altgd, linestyle=':')
    axes[2].set_xlabel(r"iteration $k$")
    axes[2].set_ylabel(r"$\kappa_2(\nabla_Y \mathsf{f}(X_k, Y_k))$")
    
    fig.legend(loc='outside lower center', ncol=8, bbox_to_anchor=(0.5, -0.05), borderaxespad=0., fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f'fig/low_rank_mat_comp_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig)

    # Plots without Adam
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].semilogy(loss_polar_grad_lr, label="PolarGrad (QDWH)", linestyle='-')
    axes[0].semilogy(loss_polar_grad_lr_decay, label=r"PolarGrad (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_muon_ns_lr, label="Muon (NS)", linestyle='-.')
    axes[0].semilogy(loss_muon_qdwh_lr, label="Muon (QDWH)", linestyle='-')
    axes[0].semilogy(loss_muon_qdwh_lr_decay, label=r"Muon (QDWH; lr $\downarrow$)", linestyle='--')
    axes[0].semilogy(loss_adam_lr_decay, label=r"Adam (lr $\downarrow$)", linestyle='--')
    # axes[0].semilogy(loss_altgd, label="AltGD", linestyle=':')
    axes[0].set_xlabel(r"iteration $k$")
    axes[0].set_ylabel(r"$\mathsf{f}(X_k,Y_k)$")

    # Plot condition numbers of gradients of X
    axes[1].plot(cond_X_polar_grad_lr, linestyle='-')
    axes[1].plot(cond_X_polar_grad_lr_decay, linestyle='--')
    axes[1].plot(cond_X_muon_ns_lr, linestyle='-.')
    axes[1].plot(cond_X_muon_qdwh_lr, linestyle='-')
    axes[1].plot(cond_X_muon_qdwh_lr_decay, linestyle='--')
    axes[1].plot(cond_X_adam_lr_decay, linestyle='--')
    # axes[1].plot(cond_X_altgd, linestyle=':')
    axes[1].set_xlabel(r"iteration $k$")
    axes[1].set_ylabel(r"$\kappa_2(\nabla_X \mathsf{f}(X_k, Y_k))$")

    # Plot condition numbers of gradients of Y
    axes[2].plot(cond_Y_polar_grad_lr, linestyle='-')
    axes[2].plot(cond_Y_polar_grad_lr_decay, linestyle='--')
    axes[2].plot(cond_Y_muon_ns_lr, linestyle='-.')
    axes[2].plot(cond_Y_muon_qdwh_lr, linestyle='-')
    axes[2].plot(cond_Y_muon_qdwh_lr_decay, linestyle='--')
    axes[2].plot(cond_Y_adam_lr_decay, linestyle='--')
    # axes[2].plot(cond_Y_altgd, linestyle=':')
    axes[2].set_xlabel(r"iteration $k$")
    axes[2].set_ylabel(r"$\kappa_2(\nabla_Y \mathsf{f}(X_k, Y_k))$")
    
    fig.legend(loc='outside lower center', ncol=6, bbox_to_anchor=(0.51, -0.05), borderaxespad=0., fontsize=16)
    fig.subplots_adjust(bottom=0.15)
    fig.savefig(f'fig/low_rank_mat_comp_2_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig)


    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    # Plot nuclear norms of gradients of X
    axes2[0].plot(nuc_X_polar_grad_lr[0:200], label="PolarGrad (QDWH)", linestyle='-')
    axes2[0].plot(nuc_X_polar_grad_lr_decay[0:200], label=r"PolarGrad (QDWH; lr $\downarrow$)", linestyle='--')
    # axes2[0].plot(nuc_X_muon_ns_lr[0:200], label="Muon (NS)", linestyle='-.')
    axes2[0].plot(nuc_X_muon_qdwh_lr[0:200], label="Muon (QDWH)", linestyle='-')
    axes2[0].plot(nuc_X_muon_qdwh_lr_decay[0:200], label=r"Muon (QDWH; lr $\downarrow$)", linestyle='--')
    axes2[0].plot(nuc_X_adam_lr[0:200], label="Adam", linestyle='-')
    axes2[0].plot(nuc_X_adam_lr_decay[0:200], label=r"Adam (lr $\downarrow$)", linestyle='--')
    axes2[0].plot(nuc_X_altgd[0:200], label="AltGD", linestyle=':')
    axes2[0].set_xlabel(r"iteration $k$")
    axes2[0].set_ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla_X \mathsf{f}(X_k, Y_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")

    # Plot nuclear norms of gradients of Y
    axes2[1].plot(nuc_Y_polar_grad_lr[0:200], linestyle='-')
    axes2[1].plot(nuc_Y_polar_grad_lr_decay[0:200], linestyle='--')
    # axes2[1].plot(nuc_Y_muon_ns_lr[0:200], linestyle='-.')
    axes2[1].plot(nuc_Y_muon_qdwh_lr[0:200], linestyle='-')
    axes2[1].plot(nuc_Y_muon_qdwh_lr_decay[0:200], linestyle='--')
    axes2[1].plot(nuc_Y_adam_lr[0:200], linestyle='-')
    axes2[1].plot(nuc_Y_adam_lr_decay[0:200], linestyle='--')
    axes2[1].plot(nuc_Y_altgd[0:200], linestyle=':')
    axes2[1].set_xlabel(r"iteration $k$")
    axes2[1].set_ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla_Y \mathsf{f}(X_k, Y_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")

    fig2.legend(loc='outside lower center', ncol=7, bbox_to_anchor=(0.51, -0.05), borderaxespad=0., fontsize=16)
    fig2.subplots_adjust(bottom=0.15)
    fig2.savefig(f'fig/low_rank_mat_comp_3_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig2)


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