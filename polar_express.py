# Adopted from The Polar Express: Optimal Matrix Sign Methods and Their Application to the Muon Algorithm
# by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower at https://arxiv.org/abs/2505.16932

import torch
from itertools import repeat

coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]

# safety factor for numerical stability ( but exclude last polynomial )
coeffs_list = [(a / 1.01, b / 1.01**3, c / 1.01**5) 
                for (a, b, c) in coeffs_list[:-1]] + [coeffs_list[-1]]

@torch.compile
def PolarExpress(
    G: torch.Tensor, 
    compute_hermitian: bool = False, 
    max_iterations: int = 5, 
    ) -> torch.Tensor:
    if G.ndim < 2:
        raise ValueError(
            f"Polar Express expects a matrix or batch of matrices, but received shape {tuple(G.shape)}."
        )
    if min(G.shape[-2:]) == 0:
        raise ValueError("Polar Express does not support empty matrices.")
    if G.is_complex():
        raise TypeError("Polar Express currently supports real matrices only.")
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative.")
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1): 
        X = X.mT  # this reduces FLOPs
    
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    hs = coeffs_list[:max_iterations] + list(repeat(coeffs_list[-1], max_iterations - len(coeffs_list)))

    for a, b, c in hs:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5
    
    if G.size(-2) > G.size(-1): 
        X = X.mT

    if compute_hermitian:
        G_work = G.type_as(X)
        if G.size(-2) >= G.size(-1):
            H = X.mH @ G_work
        else:
            H = G_work @ X.mH
        H = (H + H.mH) / 2
        return X, H
    else:
        return X
