# Adopted from Muon by Keller Jordan at https://github.com/KellerJordan/Muon

import torch

@torch.compile
def zeropower_via_newtonschulz5(G, compute_hermitian=False, max_iterations=5, a=3.4445, b=-4.7750, c=2.0315):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(max_iterations):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if compute_hermitian:
        H = G.type_as(X).mT @ X.mT
        H = (H + H.mT) / 2
    
    if G.size(-2) > G.size(-1):
        X = X.mT
        if compute_hermitian:
            H = H.mT
    
    if compute_hermitian:
        return X, H
    else:
        return X


@torch.compile
def precond_newtonschulz(G, compute_hermitian=False):
    """
    Preconditioned Newton-Schulz iteration to compute the polar factor of G. 
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    s = torch.finfo(X.dtype).eps
    s_ = 0.1
    a = 1.5 * 3**0.5 - s_
    # Perform the NS iterations
    # for _ in range(steps):
    while s < s_:
        s = a * s * (1 - 4/27 * a**2 * s**2)
        X = a * X - 4/27 * a**3 * X @ X.mT @ X

    # Newton--Schulz postprocessing
    delta = 1.
    while delta > max(G.size(-2), G.size(-1)) * torch.finfo(X.dtype).eps:
        X_new = 3/2 * X - 1/2 * X @ X.mT @ X
        delta = torch.linalg.matrix_norm(X_new - X)
        X = X_new

    if compute_hermitian:
        H = G.type_as(X).mT @ X.mT
        H = (H + H.mT) / 2
    
    if G.size(-2) > G.size(-1):
        X = X.mT
        if compute_hermitian:
            H = H.mT
    
    if compute_hermitian:
        return X, H
    else:
        return X