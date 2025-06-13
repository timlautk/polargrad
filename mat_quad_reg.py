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


def smooth(scalars: np.array, weight: float = 0.8) -> np.array:  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return np.array(smoothed)

# Objective: minimize 0.5 * ||A X B - C||_F^2
def loss_fn(X, A, B, C):
    return 0.5 * torch.linalg.matrix_norm(A @ X @ B - C) ** 2

# Model: learnable matrix X
class MatrixQuadraticModel(nn.Module):
    def __init__(self, m=500, n=100):
        super().__init__()
        self.X = nn.Parameter(torch.empty(m, n).uniform_(-1., 1.))

    def forward(self, A, B, C):
        return loss_fn(self.X, A, B, C)

# Define the 1/k decay function
def lr_lambda(step):
    if step == 0:
        return 1.0  # Avoid division by zero at the first step
    return 1.0 / (step**0.5)

# Gradient and Hessian-vector product
def grad(X, A, B, C):
    return A.T @ (A @ X @ B - C) @ B.T

def inverse_hessian_preconditioner(A, B):
    AtA = A.T @ A  # m x m
    BtB = B @ B.T  # n x n
    AtA_inv = torch.linalg.inv(AtA)
    BtB_inv = torch.linalg.inv(BtB)
    return AtA_inv, BtB_inv

def inverse_hessian_preconditioner_2(A, B):
    AtA = A.T @ A
    BtB = B @ B.T
    AtA_inv = torch.linalg.inv(AtA)
    BtB_inv = torch.linalg.inv(BtB)
    return AtA_inv, BtB_inv, AtA, BtB


def main(seed=42, steps=4000):
    # Check device
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # elif torch.cuda.is_available():
    #     device = torch.device("cuda")
    # else:
    #     device = torch.device("cpu")
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # Matrix dimensions
    m, n, p, q = 500, 100, 1000, 250
    torch.manual_seed(seed)
    A = torch.randn(p, m, device=device)
    B = torch.randn(n, q, device=device)
    C = torch.randn(p, q, device=device)

    print(f"Matrix A rank: {torch.linalg.matrix_rank(A)}")
    print(f"Matrix B rank: {torch.linalg.matrix_rank(B)}")
    print(f"Matrix A condition number: {torch.linalg.cond(A)}")
    print(f"Matrix B condition number: {torch.linalg.cond(B)}")
    hessian_cond_number = (torch.linalg.cond(A)**2 * torch.linalg.cond(B)**2).item()
    print(f"Hessian condition number: {hessian_cond_number}")
    eigvals_A = torch.linalg.eigvalsh(A.T @ A)
    eigvals_B = torch.linalg.eigvalsh(B @ B.T)
    print(f"Hessian condition number: {eigvals_A[-1] / eigvals_A[0] * eigvals_B[-1] / eigvals_B[0]}")

    AtA_inv = torch.linalg.inv(A.T @ A)     # (m x m)
    BBt_inv = torch.linalg.inv(B @ B.T)
    X_star = AtA_inv @ A.T @ C @ B.T @ BBt_inv
    print(f"Loss at X_star: {loss_fn(X_star, A, B, C).item()}")

    # Optimization loop
    def run_quadratic(optimizer_cls, method='qdwh', steps=steps, lr=0.05, scheduler=False):
        torch.manual_seed(seed)
        model = MatrixQuadraticModel(m, n).to(device)
        if optimizer_cls == torch.optim.Adam:
            optimizer = optimizer_cls(model.parameters(), lr=lr)
        elif optimizer_cls == PolarGrad:
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr, momentum=0.)
        else:            
            optimizer = optimizer_cls(model.parameters(), method=method, lr=lr)
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.99)
        losses = []
        condition_numbers_residual = []
        condition_numbers_grad = []
        nuc_norms_grad = []
        for _ in tqdm(range(steps), desc=f"optimizer = {optimizer_cls.__name__}, polar decomp method = {method if optimizer_cls != torch.optim.Adam else None}, lr decay = {scheduler if isinstance(scheduler, bool) else scheduler.__class__.__name__}"):
            loss = model(A, B, C)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            losses.append(loss.item() - loss_fn(X_star, A, B, C).item())
            condition_numbers_residual.append(torch.linalg.cond(A @ model.X @ B - C).item())
            condition_numbers_grad.append(torch.linalg.cond(grad(model.X, A, B, C)).item())
            nuc_norms_grad.append(torch.linalg.matrix_norm(grad(model.X, A, B, C), ord='nuc').item())
        condition_numbers_residual = smooth(condition_numbers_residual)
        condition_numbers_grad = smooth(condition_numbers_grad)
        nuc_norms_grad = smooth(nuc_norms_grad)
        return losses, condition_numbers_residual, condition_numbers_grad, nuc_norms_grad

    # Optimization using Newton's method (i.e., inverse Hessian preconditioning)
    def run_inverse_hessian(steps=steps, lr=0.05):
        torch.manual_seed(seed)
        model = MatrixQuadraticModel(m, n).to(device)
        AtA_inv, BtB_inv = inverse_hessian_preconditioner(A, B)
        losses = []
        condition_numbers_residual = []
        condition_numbers_grad = []
        nuc_norms_grad = []
        for _ in tqdm(range(steps), desc=f"optimizer = Newton"):
            X = model.X
            G = grad(X, A, B, C)
            precond_grad = AtA_inv @ G @ BtB_inv
            model.X.data -= lr * precond_grad
            losses.append(loss_fn(model.X, A, B, C).item() - loss_fn(X_star, A, B, C).item())
            condition_numbers_residual.append(torch.linalg.cond(A @ model.X @ B - C).item())
            condition_numbers_grad.append(torch.linalg.cond(grad(model.X, A, B, C)).item())
            nuc_norms_grad.append(torch.linalg.matrix_norm(grad(model.X, A, B, C), ord='nuc').item())
        condition_numbers_residual = smooth(condition_numbers_residual)
        condition_numbers_grad = smooth(condition_numbers_grad)
        nuc_norms_grad = smooth(nuc_norms_grad)
        return losses, condition_numbers_residual, condition_numbers_grad, nuc_norms_grad


    # Compare optimizers
    loss_polar_grad_quad, cond_polar_grad_quad, cond_grad_polar_grad_quad, nuc_polar_grad_quad = run_quadratic(PolarGrad, method='qdwh', lr=4e-8)
    loss_polar_grad_quad_zolo_pd, cond_polar_grad_quad_zolo_pd, cond_grad_polar_grad_quad_zolo_pd, nuc_polar_grad_quad_zolo_pd = run_quadratic(PolarGrad, method='zolo-pd', lr=3e-8)
    loss_polar_grad_quad_decay, cond_polar_grad_quad_decay, cond_grad_polar_grad_quad_decay, nuc_polar_grad_quad_decay = run_quadratic(PolarGrad, method='qdwh', lr=4.75e-8, scheduler=True)    
    loss_muon_quad_ns, cond_muon_quad_ns, cond_grad_muon_quad_ns, nuc_muon_quad_ns = run_quadratic(Muon_polar, method='ns', lr=1e-1)
    loss_muon_quad_qdwh, cond_muon_quad_qdwh, cond_grad_muon_quad_qdwh, nuc_muon_quad_qdwh = run_quadratic(Muon_polar, method='qdwh', lr=1e-1)
    loss_muon_quad_qdwh_decay, cond_muon_quad_qdwh_decay, cond_grad_muon_quad_qdwh_decay, nuc_muon_quad_qdwh_decay = run_quadratic(Muon_polar, method='qdwh', lr=5e-2, scheduler=True)
    loss_muon_quad_zolo_pd, cond_muon_quad_zolo_pd, cond_grad_muon_quad_zolo_pd, nuc_muon_quad_zolo_pd = run_quadratic(Muon_polar, method='zolo-pd', lr=1e-1)
    loss_adam_quad, cond_adam_quad, cond_grad_adam_quad, nuc_adam_quad = run_quadratic(torch.optim.Adam, lr=5e-2)
    loss_adam_quad_decay, cond_adam_quad_decay, cond_grad_adam_quad_decay, nuc_adam_quad_decay = run_quadratic(torch.optim.Adam, lr=5e-2, scheduler=True)
    loss_hessian, cond_hessian, cond_grad_hessian, nuc_hessian = run_inverse_hessian(lr=2.5e-1)


    ## Plots
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    axes[0].semilogy(loss_polar_grad_quad, label="PolarGrad (QDWH)", linestyle='--')
    axes[0].semilogy(loss_polar_grad_quad_zolo_pd, label="PolarGrad (ZOLO-PD)", linestyle='-.')
    axes[0].semilogy(loss_polar_grad_quad_decay, label=r"PolarGrad (QDWH; lr $\downarrow$)", linestyle=':')
    axes[0].semilogy(loss_muon_quad_ns, label="Muon (NS)", linestyle='-')
    axes[0].semilogy(loss_muon_quad_qdwh, label="Muon (QDWH)", linestyle='--')
    axes[0].semilogy(loss_muon_quad_zolo_pd, label="Muon (ZOLO-PD)", linestyle='-.')   
    axes[0].semilogy(loss_muon_quad_qdwh_decay, label=r"Muon (QDWH; lr $\downarrow$)", linestyle=':')
    axes[0].semilogy(loss_hessian, label=r"Newton ($\nabla^2 \mathsf{f}(X_k)^{-1}$)", linestyle='-')
    axes[0].semilogy(loss_adam_quad, label="Adam")    
    axes[0].semilogy(loss_adam_quad_decay, label=r"Adam (lr $\downarrow$)", linestyle=':')
    axes[0].set_xlabel(r"iteration $k$")
    axes[0].set_ylabel(r"$\mathsf{f}(X_k) - \mathsf{f}^\star$")

    # Plot condition numbers of residuals
    axes[1].semilogy(cond_polar_grad_quad, linestyle='--')
    axes[1].semilogy(cond_polar_grad_quad_zolo_pd, linestyle='-.')
    axes[1].semilogy(cond_polar_grad_quad_decay, linestyle=':')
    axes[1].semilogy(cond_muon_quad_ns, linestyle='-')
    axes[1].semilogy(cond_muon_quad_qdwh, linestyle='--')
    axes[1].semilogy(cond_muon_quad_zolo_pd, linestyle='-.')
    axes[1].semilogy(cond_muon_quad_qdwh_decay, linestyle=':')
    axes[1].semilogy(cond_hessian, linestyle='-')
    axes[1].semilogy(cond_adam_quad)
    axes[1].semilogy(cond_adam_quad_decay, linestyle=':')    
    axes[1].set_xlabel(r"iteration $k$")
    axes[1].set_ylabel(r"$\kappa_2(A X_k B - C)$")

    # Plot condition numbers of gradients
    axes[2].semilogy(cond_grad_polar_grad_quad, linestyle='--')
    axes[2].semilogy(cond_grad_polar_grad_quad_zolo_pd, linestyle='-.')
    axes[2].semilogy(cond_grad_polar_grad_quad_decay, linestyle=':')
    axes[2].semilogy(cond_grad_muon_quad_ns, linestyle='-')
    axes[2].semilogy(cond_grad_muon_quad_qdwh, linestyle='--')
    axes[2].semilogy(cond_grad_muon_quad_zolo_pd, linestyle='-.')
    axes[2].semilogy(cond_grad_muon_quad_qdwh_decay, linestyle=':')
    axes[2].semilogy(cond_grad_hessian, linestyle='-')
    axes[2].semilogy(cond_grad_adam_quad)
    axes[2].semilogy(cond_grad_adam_quad_decay, linestyle=':')
    axes[2].set_xlabel(r"iteration $k$")
    axes[2].set_ylabel(r"$\kappa_2(\nabla\mathsf{f}(X_k))$")

    fig.legend(loc='outside lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), borderaxespad=0., fontsize=16)
    fig.subplots_adjust(bottom=0.22)
    
    fig.savefig(f'fig/mat_quad_reg_{seed}.pdf', dpi=500, bbox_inches='tight')
    plt.close(fig)


    # Plot nuclear norms
    fig2 = plt.figure(figsize=(7, 5))
    plt.semilogy(nuc_polar_grad_quad, label="PolarGrad (QDWH)", linestyle='--')
    plt.semilogy(nuc_polar_grad_quad_zolo_pd, label="PolarGrad (ZOLO-PD)", linestyle='-.')
    plt.semilogy(nuc_polar_grad_quad_decay, label=r"PolarGrad (QDWH; lr $\downarrow$)", linestyle=':')
    plt.semilogy(nuc_muon_quad_ns, label="Muon (NS)", linestyle='-')
    plt.semilogy(nuc_muon_quad_qdwh, label="Muon (QDWH)", linestyle='--')
    plt.semilogy(nuc_muon_quad_zolo_pd, label="Muon (ZOLO-PD)", linestyle='-.')
    plt.semilogy(nuc_muon_quad_qdwh_decay, label=r"Muon (QDWH; lr $\downarrow$)", linestyle=':')
    plt.semilogy(nuc_hessian, label=r"Newton ($\nabla^2 \mathsf{f}(X_k)^{-1}$)", linestyle='-')
    plt.semilogy(nuc_adam_quad, label="Adam")
    plt.semilogy(nuc_adam_quad_decay, label=r"Adam (lr $\downarrow$)", linestyle=':')
    plt.xlabel(r"iteration $k$")
    plt.ylabel(r"$\lvert\kern-0.25ex\lvert\kern-0.25ex\lvert \nabla\mathsf{f}(X_k, \xi_k) \rvert\kern-0.25ex\rvert\kern-0.25ex\rvert_{\text{nuc}}$")
    fig2.legend(loc="outside lower center", ncol=5, bbox_to_anchor=(0.5, -0.05), borderaxespad=0., fontsize=16)
    fig2.subplots_adjust(bottom=0.22)
    fig2.savefig(f'fig/mat_quad_reg_nuc_{seed}.pdf', dpi=500, bbox_inches='tight')
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