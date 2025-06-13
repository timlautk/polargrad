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

"""A torch.compile-compatible library for QDWH-based polar decomposition, modified from JAX.
https://github.com/jax-ml/jax/blob/main/jax/_src/lax/qdwh.py

QDWH is short for QR-based dynamically weighted Halley iteration. The Halley
iteration implemented through QR decompositions does not require matrix
inversion. This is desirable for multicore and heterogeneous computing systems.

Reference: Nakatsukasa, Yuji, Zhaojun Bai, and François Gygi.
"Optimizing Halley's iteration for computing the matrix polar decomposition."
SIAM Journal on Matrix Analysis and Applications 31, no. 5 (2010): 2700-2720.
https://epubs.siam.org/doi/abs/10.1137/090774999
"""

import torch
from typing import Optional


# Helpers for working with padded shapes
def _mask(x, dims, alternative=0):
    """
    Masks `x` up to the dynamic shape `dims`.

    Replaces values outside those dimensions with `alternative`.
    `alternative` is broadcast with `x`.

    Args:
        x (torch.Tensor): Input tensor.
        dims (list): A list with length equal to x.ndim. For each axis,
                    if the corresponding element is an int, then only indices
                    less than that int are kept; if None, no masking is applied.
        alternative: Value to use for masked-out positions (default 0).

    Returns:
        torch.Tensor: The masked tensor.
    """
    assert x.ndim == len(dims)
    mask = None
    for i, d in enumerate(dims):
        if d is not None:
            # Create a coordinate tensor along axis i.
            # This tensor has shape [1, 1, ..., x.size(i), ..., 1] so that it can be
            # broadcasted over x.
            shape = [1] * x.ndim
            shape[i] = x.size(i)
            idx = torch.arange(x.size(i), device=x.device, dtype=torch.int32).view(*shape)
            mask_dim_i = idx < d
            mask = mask_dim_i if mask is None else (mask & mask_dim_i)

    if mask is None:
        return x
    else:
        # Ensure alternative is a tensor (it might be a scalar)
        if not torch.is_tensor(alternative):
            alternative_tensor = torch.tensor(alternative, dtype=x.dtype, device=x.device)
        else:
            alternative_tensor = alternative
        return torch.where(mask, x, alternative_tensor)

def _pad_in_dim(x, low=0, high=0, interior=0, fill_value=0, axis=0):
    """
    Pads tensor `x` along the specified `axis` with:
        - `low` zeros (or fill_value) at the beginning,
        - `high` zeros (or fill_value) at the end, and
        - `interior` copies of fill_value between consecutive elements.
    
    This mimics JAX’s lax.pad behavior for one dimension while leaving all other dimensions unpadded.
    
    Args:
        x (torch.Tensor): The input tensor.
        low (int): Number of padding elements before the first element.
        high (int): Number of padding elements after the last element.
        interior (int): Number of padding elements inserted between consecutive elements.
        fill_value: The constant value to pad with.
        axis (int): The dimension along which to apply the padding.
        
    Returns:
        torch.Tensor: The padded tensor.
    """
    # Case 1: No interior padding, so only pad low and high.
    if interior == 0:
        new_shape = list(x.shape)
        new_shape[axis] += low + high  # increase the size along the padded axis
        out = torch.full(new_shape, fill_value, dtype=x.dtype, device=x.device)
        # Build index: for axis, we start at `low` and cover the original length.
        idx = [slice(None)] * x.ndim
        idx[axis] = slice(low, low + x.size(axis))
        out[tuple(idx)] = x
        return out

    # Case 2: With interior padding.
    orig_size = x.size(axis)
    # New size is computed as:
    #   low + (orig_size - 1) * (1 + interior) + 1 + high
    new_size = low + (orig_size - 1) * (interior + 1) + 1 + high
    new_shape = list(x.shape)
    new_shape[axis] = new_size
    out = torch.full(new_shape, fill_value, dtype=x.dtype, device=x.device)
    
    # Build index for assignment:
    # Along the padded axis, starting at position `low`, place each element of `x`
    # with a step of (interior + 1) between positions.
    slices = [slice(None)] * x.ndim
    slices[axis] = slice(low, low + orig_size * (interior + 1), interior + 1)
    out[tuple(slices)] = x
    return out

def _dynamic_concat(a, b, m, axis=0):
    """
    Concatenates padded arrays `a` and `b` where the true size of `a` is `m`.
    
    If m is None, simply concatenates a and b along the given axis.
    Otherwise, pads `a` along `axis` (adding b.size(axis) elements at the end),
    and then updates the slice starting at index m with `b`.
    
    Args:
        a (torch.Tensor): The first tensor.
        b (torch.Tensor): The second tensor.
        m (int or None): The true size of `a` along the axis before padding.
                        If None, no dynamic slicing is performed.
        axis (int): The dimension along which to concatenate.
    
    Returns:
        torch.Tensor: The dynamically concatenated tensor.
    """
    if m is None:
        return torch.cat([a, b], dim=axis)

    # Pad `a` along the specified axis with additional space at the end
    padded_a = _pad_in_dim(a, high=b.size(axis), axis=axis)
    
    # Prepare slicing: we want to replace the slice in `padded_a` starting at index m
    # along `axis` with `b`.
    index = [slice(None)] * padded_a.ndim
    index[axis] = slice(m, m + b.size(axis))
    padded_a[tuple(index)] = b
    return padded_a


def _use_qr(u, m, n, params):
    """
    QDWH iteration using QR decomposition.

    Args:
        u (torch.Tensor): A matrix with static (padded) shape M x N.
        m (int or None): The dynamic number of rows (m <= M).
        n (int): The dynamic number of columns (n <= N).
        params (tuple): The QDWH parameters, expected as 
                        (a_minus_e_by_sqrt_c, sqrt_c, e).

    Returns:
        torch.Tensor: The updated matrix.
    """
    a_minus_e_by_sqrt_c, sqrt_c, e = params
    M, N = u.shape

    y = _dynamic_concat(sqrt_c * u, torch.eye(N, dtype=u.dtype, device=u.device), m)
    # Compute the reduced QR decomposition: q has shape M x N.
    # (torch.linalg.qr with mode='reduced' is equivalent to full_matrices=False.)
    q, _ = torch.linalg.qr(y, mode='reduced')

    # q1 corresponds to the first m rows of q, but we use _mask
    # to only keep the dynamic (m x n) portion.
    # In JAX: q1 = _mask(lax.slice(q, (0, 0), (M, N)), (m, n))
    # Here, since q is already M x N, we mask it with (m, n).
    q1 = _mask(q[:M, :N], (m, n))

    # q2 is formed from the lower N rows of q:
    # In JAX: q2 = (q[m:, :]).T.conj()
    # We slice from row m to m+N (since M is padded as m+N),
    # mask the result (with shape (n, n)), then take the transpose and conjugate.
    q2 = q[m:m + N, :]  # shape: N x N
    q2 = _mask(q2, (n, n)).T.conj()
    return e * u + a_minus_e_by_sqrt_c * (q1 @ q2)


def _use_cholesky(u, m, n, params):
    """
    QDWH iteration using Cholesky decomposition.

    Args:
        u (torch.Tensor): A matrix with static (padded) shape M x N.
        m (int): The dynamic number of rows (m <= M).
        n (int): The dynamic number of columns (n <= N).
        params (tuple): The QDWH parameters, expected as (a_minus_e, c, e).

    Returns:
        torch.Tensor: The updated matrix.
    """
    a_minus_e, c, e = params
    _, N = u.shape

    # Compute x = c * (u.T.conj() @ u) + I.
    # In PyTorch, u.T.conj() is given by u.T.conj()
    x = c * (u.T.conj() @ u) + torch.eye(N, dtype=u.dtype, device=u.device)
    
    # Pad the lower-right corner with an identity matrix to avoid issues
    # with non-PSD matrices due to padding.
    x = _mask(x, (n, n), torch.eye(N, dtype=x.dtype, device=x.device))
    
    # Compute the Cholesky factorization; y is lower-triangular.
    y, _ = torch.linalg.cholesky_ex(x)
    
    # First triangular solve.
    # In JAX:
    #   z = triangular_solve(y, u.T, left_side=True, lower=True, conjugate_a=True).conj()
    #
    # In PyTorch we emulate the effect of conjugating A inside the solver by
    # solving with y.conj() and then taking the conjugate of the solution.
    # We solve:  (conj(y)) * temp = u.T, and then set z = temp.conj()
    z = torch.linalg.solve_triangular(y.conj(), u.T, upper=False, left=True, unitriangular=False).conj()
    
    # Second triangular solve.
    # In JAX:
    #   z = triangular_solve(y, z, left_side=True, lower=True,
    #                          transpose_a=True, conjugate_a=True).T.conj()
    #
    # In PyTorch, we solve:
    #   (conj(y)).T * temp2 = z, and then take z = temp2.T.conj()
    z = torch.linalg.solve_triangular(y.conj().T, z, upper=False, left=True, unitriangular=False).T.conj()
    
    return e * u + a_minus_e * z


def _qdwh(x, m, n, max_iterations, eps):
    """
    QR-based dynamically weighted Halley iteration for polar decomposition.

    Args:
        x (torch.Tensor): Input matrix.
        m (int): The dynamic number of rows (m <= padded number of rows).
        n (int): The dynamic number of columns (n <= padded number of columns).
        max_iterations (int): Maximum number of iterations.
        eps (float or None): Machine epsilon for x.dtype; if None, uses torch.finfo.

    Returns:
        tuple: (u, h, num_iters, is_converged) where
            - u is the computed polar factor,
            - h is the computed Hermitian factor (h = u.T.conj() @ x, symmetrized) if compute_hermitian,
            - num_iters is the total number of iterations performed,
            - is_converged is a Boolean indicating whether convergence was reached.
    """
    # Set eps if not provided.
    if eps is None:
        eps = float(torch.finfo(x.dtype).eps)

    # Estimate alpha_inverse such that alpha_inverse = 1/sqrt(||x||_1) * 1/sqrt(||x||_inf)
    one_norm = torch.linalg.norm(x, ord=1)
    inf_norm = torch.linalg.norm(x, ord=float('inf'))
    alpha_inverse = 1 / (one_norm * inf_norm) ** (1 / 2)
    alpha_inverse = torch.where(one_norm == 0, 1, alpha_inverse)
    u = x * alpha_inverse

    # Initialize l, which approximates the lower bound for the smallest singular value.
    l = eps

    # Set iteration tolerances.
    tol_l = 10.0 * eps / 2.0   # (equals 5*eps)
    tol_norm = tol_l ** (1 / 3)

    # Define functions to compute parameters for the QDWH update.
    def get_qr_params(a, b, c):
        e = b / c
        a_minus_e = a - e
        sqrt_c = c ** (1 / 2)
        return (a_minus_e / sqrt_c, sqrt_c, e)

    def get_chol_params(a, b, c):
        e = b / c
        a_minus_e = a - e
        return (a_minus_e, c, e)

    CHOLESKY_CUTOFF = 100

    # Compute coefficient lists for the two update stages.
    qr_coefs = []
    chol_coefs = []
    k = 0
    while l + tol_l < 1 and k < max_iterations:
        k += 1
        l2 = l * l
        dd = (4 * (1 / l2 - 1) / l2) ** (1 / 3)
        sqd = (1.0 + dd) ** (1 / 2)
        a = sqd + (2 - dd + 2 * (2 - l2) / (l2 * sqd)) ** (1 / 2)
        b = (a - 1) ** 2 / 4
        c = a + b - 1
        l = l * (a + b * l2) / (1 + c * l2)
        if c > CHOLESKY_CUTOFF:
            qr_coefs.append(get_qr_params(a, b, c))
        else:
            chol_coefs.append(get_chol_params(a, b, c))

    # ---------------------------
    # Define a single iteration step.
    # ---------------------------
    def iteration(k_idx, state, update_fn, coefs, test_convergence):
        """
        One iteration of the QDWH update.
        
        Args:
            k_idx (int): The current iteration index.
            state (tuple): (u, _) where u is the current iterate.
            update_fn (callable): Update function (_use_qr or _use_cholesky).
            coefs (list or None): List of coefficient tuples, or None (to use Halley's method).
            test_convergence (bool): Whether to test for convergence.
        
        Returns:
            tuple: (updated u, is_not_converged flag)
        """
        u, _ = state
        if coefs is None:
            # As l → 1, the coefficients a, b, c → 3, 1, 3 (i.e. Halley's method).
            params = get_chol_params(3, 1, 3)
        else:
            params = coefs[k_idx]
        u_prev = u.clone()
        u = update_fn(u, m, n, params)

        is_not_converged = True
        if test_convergence:
            # Check if the change is above the tolerance.
            is_not_converged = torch.linalg.matrix_norm(u - u_prev) > tol_norm
        return u, is_not_converged

    # ---------------------------
    # Define an iteration loop over a list of coefficients.
    # ---------------------------
    def iterate_fn(u, coefs, update_fn, test_convergence):
        if not coefs:  # empty list: no iterations to perform
            return u, True
        is_not_converged = True
        for k_idx in range(len(coefs)):
            u, is_not_converged = iteration(k_idx, (u, True), update_fn, coefs, test_convergence)
        return u, is_not_converged

    # Perform the iterations using the QR-based coefficients first.
    # (Here, we do not check for convergence.)
    u, _ = iterate_fn(u, qr_coefs, update_fn=_use_qr, test_convergence=False)
    # Next, iterate using the Cholesky-based coefficients.
    u, is_not_converged = iterate_fn(u, chol_coefs, update_fn=_use_cholesky, test_convergence=True)

    # Continue with Halley's method (coef = None) until convergence or max iterations reached.
    k_counter = len(qr_coefs) + len(chol_coefs)
    while is_not_converged and k_counter < max_iterations:
        u, is_not_converged = iteration(k_counter, (u, is_not_converged),
                                        update_fn=_use_cholesky,
                                        coefs=None,
                                        test_convergence=True)
        k_counter += 1

    num_iters = k_counter

    # Apply a final Newton-Schulz refinement for improved accuracy.
    u = 1.5 * u - 0.5 * u @ (u.T.conj() @ u)

    is_converged = not is_not_converged

    return u, num_iters, is_converged


# TODO: Add pivoting.
# @torch.compile
def qdwh(
    x,
    *,
    is_hermitian: bool = False,
    compute_hermitian: bool = False,
    max_iterations: Optional[int] = None,
    eps: Optional[float] = None,
    dynamic_shape: Optional[tuple[int, int]] = None,
):
    """
    QR-based dynamically weighted Halley iteration for polar decomposition.

    Args:
        x (torch.Tensor): A full-rank matrix with shape M x N. The matrix may be padded
                        up to that size from a smaller true shape (dynamic_shape).
        is_hermitian (bool): True if x is Hermitian. Default is False.
                            (This argument is currently unused.)
        max_iterations (int or None): Maximum number of iterations. If None, defaults to 10.
        eps (float or None): The final iterate will satisfy
            |x_k - x_k-1| < |x_k| * (4*eps)**(1/3)
            where x_k is the iterate. If None, machine epsilon for x.dtype is used.
        dynamic_shape (tuple[int, int] or None): The unpadded (true) shape as an (m, n) tuple.

    Returns:
        tuple: A four-tuple (u, h, num_iters, is_converged) where
            - u (torch.Tensor): The polar factor.
            - h (torch.Tensor): The Hermitian factor computed as (u^* x + x^* u) / 2.
            - num_iters (int): The number of iterations performed.
            - is_converged (bool): True if convergence was reached within max_iterations.
    """
    # Ensure the is_hermitian flag is a bool.
    if not isinstance(is_hermitian, bool):
        raise ValueError("The `is_hermitian` argument must be a bool")

    if max_iterations is None:
        max_iterations = 10
    elif not isinstance(max_iterations, int):
        raise ValueError("The `max_iterations` argument must be an int")

    M, N = x.shape
    # if M < N:
    #     raise ValueError("The input matrix of shape M x N must have M >= N.")

    if dynamic_shape is not None:
        m, n = dynamic_shape
        x = _mask(x, (m, n))
    else:
        m, n = M, N

    # Optionally, if you wish to enforce float32 matmul precision (similar to JAX’s
    # jax.default_matmul_precision('float32')), you might cast x to float32.
    # x = x.to(torch.float32)

    # Call the core QDWH routine (assumes _qdwh is implemented in PyTorch)
    u, num_iters, is_converged = _qdwh(x, m, n, max_iterations, eps)

    if compute_hermitian:
        h = u.T.conj() @ x
        h = (h + h.T.conj()) / 2
        return u, h, num_iters, is_converged
    else:
        return u, num_iters, is_converged


def subspaceit(U, use_rand=False, U1_init=None):
    """
    Subspace iteration for computing an invariant subspace.
    
    Args:
        U (torch.Tensor): An n×n matrix.
        use_rand (bool): If True, use a random starting matrix.
        U1_init (torch.Tensor or None): An initial guess.
    
    Returns:
        U0 (torch.Tensor): An orthonormal basis for one subspace.
        U1 (torch.Tensor): An orthonormal basis for the complementary subspace.
    """
    n = U.shape[0]
    # Estimate size: in MATLAB, xsize = round(norm(U,'fro')^2).
    # For an orthogonal U (n×n), norm(U,'fro')^2 = n.
    xsize = torch.round(torch.linalg.matrix_norm(U)**2).int()
    k = min(xsize + 3, n)
    
    if U1_init is not None:
        UU = U @ U1_init
    elif use_rand:
        UU = U @ torch.randn(n, k, device=U.device, dtype=U.dtype)
    else:
        UU = U[:, :k]
    
    # Economy QR factorization
    Q, _ = torch.linalg.qr(UU, mode='reduced')
    UU = U @ Q
    Q2, _ = torch.linalg.qr(UU, mode='reduced')
    
    # U0: first xsize columns; U1: remaining columns (if any)
    U0 = Q2[:, :xsize]
    U1 = Q2[:, xsize:] if xsize < Q2.shape[1] else torch.empty(U.shape[0], 0, device=U.device, dtype=U.dtype)
    return U0, U1


@torch.compile
def qdwh_eigh(H, normH=None, minlen=1, NS=True):
    """
    Eigendecomposition of a symmetric matrix via a QDWH-based procedure.
    
    Args:
        H (torch.Tensor): A symmetric matrix.
        normH (float or None): Frobenius norm of H (if None, computed internally).
        minlen (int): Minimal block size at which to stop recursion.
        NS (bool): Whether to perform Newton–Schulz postprocessing.
    
    Returns:
        Uout (torch.Tensor): An orthogonal matrix whose columns are eigenvectors.
        eigvals (torch.Tensor): A 1D tensor of eigenvalues (in ascending order).
    """
    # Tolerance for relative backward error.
    eps = torch.finfo(H.dtype).eps
    backtol = 10 * eps / 2
    n = H.shape[0]
    if normH is None:
        normH = torch.linalg.matrix_norm(H)
    
    Uout, eigvals = qdwh_eigh_rep(H, normH, minlen, backtol)
    
    if NS:
        # Newton–Schulz postprocessing: Uout = 3/2*Uout - 1/2*Uout*(UoutᵀUout)
        Uout = 1.5 * Uout - 0.5 * (Uout @ (Uout.T @ Uout))
    
    # Sort eigenvalues in ascending order and adjust eigenvector order.
    sorted_eigvals, _ = torch.sort(eigvals, descending=False)
    Uout = torch.flip(Uout, dims=[1])
    
    return Uout, sorted_eigvals


def qdwh_eigh_rep(H, normH, minlen, backtol, a=None, b=None, shift=None):
    """
    Internal recursive function for QDWHEIG.
    
    Args:
        H (torch.Tensor): Symmetric matrix.
        normH (float): Frobenius norm of H.
        minlen (int): Minimal block size.
        backtol (float): Tolerance for relative backward error.
        a, b (float or None): Parameters for the polar decomposition.
        shift (float or None): Shift parameter.
    
    Returns:
        Uout (torch.Tensor): Orthogonal matrix of eigenvectors.
        eigvals (torch.Tensor): 1D tensor of eigenvalues.
    """
    n = H.shape[0]
    I = torch.eye(n, device=H.device, dtype=H.dtype)
    
    # If H is almost diagonal, return trivial solution.
    H_diag = torch.diag(torch.diag(H))
    if torch.linalg.matrix_norm(H - H_diag) / normH < backtol:
        diagH = torch.diag(H)
        sorted_diag, IX = torch.sort(diagH, descending=True)
        Uout = torch.eye(n, device=H.device, dtype=H.dtype)[:, IX]
        return Uout, sorted_diag

    # Symmetrize H to counter roundoff.
    H = 0.5 * (H + H.T)
    
    # Determine shift: use the median of the diagonal entries.
    if shift is None:
        shift = torch.median(torch.diag(H))
    
    # Estimate a and b.
    if a is None:
        # Approximate 2-norm of (H - shift*I)
        a = torch.linalg.matrix_norm(H - shift * I, ord=2)
    if b is None:
        cond_val = torch.linalg.cond(H - shift * I)
        b = 0.9 / cond_val if cond_val != 0 else 0.9
    
    # Compute polar decomposition via qdwh on (H - shift*I).
    U, _, _ = qdwh(H - shift * I)
    # Form the orthogonal projection matrix.
    U = 0.5 * (U + I)
    
    # Subspace iteration.
    U1, U2 = subspaceit(U)
    # Here, U1 approximates an invariant subspace and U2 its complement.
    minoff = torch.linalg.matrix_norm(U2.T @ H @ U1) / normH
    if minoff > backtol:
        U1, U2 = subspaceit(U, use_rand=False, U1_init=U1)
        minoff = torch.linalg.matrix_norm(U2.T @ H @ U1) / normH
    if minoff > backtol:
        for _ in range(2):
            U1b, U2b = subspaceit(U, use_rand=True)
            minoff2 = torch.linalg.matrix_norm(U2b.T @ H @ U1b) / normH
            if minoff > minoff2:
                U1, U2 = U1b, U2b
                minoff = minoff2
    
    # Initialize list for eigenvalues.
    eigvals_list = []
    if U1.shape[1] == 1:
        eigvals_list.append((U1.T @ H @ U1).squeeze())
    if U2.shape[1] == 1:
        eigvals_list.append((U2.T @ H @ U2).squeeze())
    
    eigvals1 = None
    eigvals2 = None
    
    # Process the U1 block.
    if U1.shape[1] > minlen:
        H_sub = U1.T @ H @ U1
        Ua, eigvals1 = qdwh_eigh_rep(H_sub, normH, minlen, backtol)
        U1 = U1 @ Ua
    elif U1.shape[1] > 1:
        H_sub = U1.T @ H @ U1
        # For symmetric matrices, use torch.linalg.eigh.
        eig_temp, Ua = torch.linalg.eigh(H_sub)
        sorted_eig_temp, indices = torch.sort(eig_temp, descending=True)
        Ua = Ua[:, indices]
        U1 = U1 @ Ua
        eigvals1 = torch.flip(sorted_eig_temp, dims=[0])
    
    # Process the U2 block.
    if U2.shape[1] > minlen:
        H_sub = U2.T @ H @ U2
        Ua, eigvals2 = qdwh_eigh_rep(H_sub, normH, minlen, backtol)
        U2 = U2 @ Ua
    elif U2.shape[1] > 1:
        H_sub = U2.T @ H @ U2
        eig_temp, Ua = torch.linalg.eigh(H_sub)
        sorted_eig_temp, indices = torch.sort(eig_temp, descending=True)
        Ua = Ua[:, indices]
        U2 = U2 @ Ua
        eigvals2 = torch.flip(sorted_eig_temp, dims=[0])
    
    # Concatenate the eigenvector blocks.
    Uout = torch.cat((U1, U2), dim=1)
    
    # Concatenate the eigenvalues from all parts.
    # (If the trivial cases above produced scalars, stack them first.)
    if eigvals1 is None:
        eigvals1 = torch.tensor([], device=H.device, dtype=H.dtype)
    if eigvals2 is None:
        eigvals2 = torch.tensor([], device=H.device, dtype=H.dtype)
    if len(eigvals_list) > 0:
        eigvals_tensor = torch.cat((torch.stack(eigvals_list), eigvals1, eigvals2))
    else:
        eigvals_tensor = torch.cat((eigvals1, eigvals2))
    
    return Uout, eigvals_tensor


@torch.compile
def qdwh_svd(A, minlen=1, NS=True):
    """
    Singular value decomposition (SVD) via QDWH and QDWHEIG.
    Computes an SVD of the rectangular matrix A by first computing its
    polar decomposition A = Uini * H and then computing an eigendecomposition
    of H.
    
    Args:
        A (torch.Tensor): Input 2D tensor.
        minlen (int, optional): Minimum matrix size for the recursive eigen-solver (default 1).
        NS (bool, optional): Newton-Schulz postprocessing flag (default True).
    
    Returns:
        Uout (torch.Tensor): Left singular vectors.
        singvals (torch.Tensor): Diagonal matrix of singular values.
        Vout (torch.Tensor): Right singular vectors.
    
    Such that (possibly up to sign corrections) A = Uout * singvals * Voutᵀ.
    """
    m, n = A.shape
    flip = False
    # If A is a "fat" matrix, flip it.
    if m < n:
        flip = True
        A = A.T
        m, n = A.shape

    normA = torch.linalg.matrix_norm(A)
    
    Qini = None
    if m > 1.15 * n:
        # Initial QR to reduce to square case.
        Qini, A = torch.linalg.qr(A, mode='reduced')
        m = n  # now A is square


    # Compute polar decomposition A = Uini * HH using QDWH.
    Uini, HH, _, _ = qdwh(A, compute_hermitian=True)
    
    # Check if the computed polar factor is nearly unitary.
    if torch.linalg.matrix_norm(Uini)**2 < n - 0.5:
        rankdef = 1
    else:
        rankdef = 0

    # Compute eigendecomposition of HH: HH = Vout * D * Voutᵀ.
    Vout, singvals = qdwh_eigh(HH, normA, minlen, NS)
    # MATLAB code sorts singular values in descending order.
    singvals, idx = torch.sort(singvals, descending=True)
    Vout = Vout[:, idx]

    # Accumulate Uini and Vout to get SVD: Uout = Uini * Vout.
    Uout = Uini @ Vout
    if Qini is not None:
        Uout = Qini @ Uout

    # If the polar factor was not unitary (partial isometry), correct via QR.
    if rankdef == 1:
        Uout, R = torch.linalg.qr(Uout, mode='reduced')
        # Multiply each column by the sign of the corresponding diagonal element of R.
        signs = torch.sign(torch.diag(R))
        Uout = Uout * signs.unsqueeze(0)
    
    if NS:
        # Newton-Schulz postprocessing.
        Uout = 1.5 * Uout - 0.5 * (Uout @ (Uout.T @ Uout))
    
    if flip:
        Uout, Vout = Vout, Uout

    return Uout, singvals, Vout
