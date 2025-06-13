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

import torch
from typing import Tuple, Optional

# Assume that a PyTorch version of `qdwh` is defined elsewhere with the signature:
# def qdwh(x: torch.Tensor, *, is_hermitian: bool = False, max_iterations: int = 10, eps: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, int, bool]:
#     ...
# and that any helper functions like `_mask` are also available.

def polar(a: torch.Tensor,
        *,
        method: str = 'qdwh',
        compute_hermitian: bool = False,
        eps: Optional[float] = None,
        max_iterations: Optional[int] = None, 
        ns_coeffs: tuple = (3.4445, -4.7750, 2.0315)) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""
    Computes the polar decomposition.

    Given the m x n matrix `a`, returns the factors of the polar
    decomposition ``u`` (also m x n) and ``h`` such that
    ``a = u h`` (if m >= n; ``h`` is n x n) or
    ``a = h u`` (if m < n; ``h`` is m x m), where ``h`` is symmetric (Hermitian) positive semidefinite.
    If `a` is nonsingular, ``h`` is positive definite and the decomposition is unique.
    The unitary factor ``u`` has orthonormal columns unless n > m, in which case it has orthonormal rows.

    Three methods are supported:

      * ``method="ns"``:
        Applies the Newton-Schulz iteration to compute the polar decomposition.
        This method requires the choice of three coefficients ``a``, ``b``, and ``c``
        of the iterating matrix polynomial
      * ``method="qdwh"``:
        Applies the QDWH (QR-based Dynamically Weighted Halley) algorithm.
        This method is more efficient for large matrices and is numerically stable
      * ``method="zolo-pd"``:
        Applies the ZOLO-PD (Zolotarev-based Polar Decomposition) algorithm.
        This method is more efficient for large matrices and is numerically stable
    Args:
        a: A full-rank input matrix of shape (m, n). The matrix may be padded if it
        represents a smaller true shape.
        method: Either "qdwh" or "svd" (default "qdwh").
        compute_hermitian: If True, the Hermitian positive-semidefinite factor is computed.
        eps: The precision tolerance; if None, the machine epsilon for `a.dtype` is used.
        max_iterations: Maximum iterations for QDWH. Ignored if ``method != "qdwh"``.
                    If None, a default (e.g. 10) is used.

    Returns:
        A tuple ``(unitary, posdef)`` where:
        - ``unitary`` is the computed unitary factor (m x n),
        - ``posdef`` is the Hermitian positive-semidefinite factor (n x n if m >= n,
            or m x m if m < n) if compute_Hermitian is True.

    Raises:
        ValueError: If the input `a` is not 2-D or if an invalid side or method is provided.
        NotImplementedError: If the combination of matrix shape and `side` is not supported by QDWH.

    Examples:

        >>> a = torch.tensor([[1., 2., 3.],
        ...                    [5., 4., 2.],
        ...                    [3., 2., 1.]])
        >>> U, H = polar(a, compute_hermitian=True)
        >>> torch.allclose(U.T @ U, torch.eye(U.shape[1]))
        True
        >>> a_reconstructed = U @ H
        >>> torch.allclose(a, a_reconstructed)
        True
    """
    # Convert input to tensor.
    arr = torch.as_tensor(a)
    if arr.ndim != 2:
        raise ValueError("The input `a` must be a 2-D array.")

    m, n = arr.shape
    max_iterations = max_iterations if max_iterations is not None else 5

    if method == "qdwh":
        from qdwh import qdwh
        # For QDWH, we support one of two cases.
        if m >= n:
            # Call the QDWH routine on the original matrix.
            res = qdwh(arr, is_hermitian=False, compute_hermitian=compute_hermitian,
                    max_iterations=max_iterations,
                    eps=eps)
            unitary = res[0]
            if compute_hermitian:
                posdef = res[1]
        else:
            # For a left polar decomposition when m < n, work with the conjugate-transpose.
            arr_t = arr.transpose(0, 1).conj()
            res = qdwh(arr_t, is_hermitian=False, compute_hermitian=compute_hermitian,
                    max_iterations=max_iterations,
                    eps=eps)
            unitary = res[0]
            # Revert the transformation.
            unitary = unitary.transpose(0, 1).conj()
            if compute_hermitian:
                posdef = res[1]
                posdef = posdef.transpose(0, 1).conj()
    elif method == "zolo-pd":
        from zolopd import zolopd
        res = zolopd(arr, compute_hermitian=compute_hermitian)
        unitary = res[0]
        if compute_hermitian:
            posdef = res[1]
    elif method == "ns":
        from newton_schulz import zeropower_via_newtonschulz5
        res = zeropower_via_newtonschulz5(arr, compute_hermitian=compute_hermitian, max_iterations=max_iterations, a=ns_coeffs[0], b=ns_coeffs[1], c=ns_coeffs[2])
        if compute_hermitian:
            unitary, posdef = res
        else:
            unitary = res
    elif method == "precond_ns":
        from newton_schulz import precond_newtonschulz
        res = precond_newtonschulz(arr, compute_hermitian=compute_hermitian)
        unitary = res[0]
        if compute_hermitian:
            posdef = res[1]
    else:
        raise ValueError(f"Unknown polar decomposition method {method}.")
    
    return unitary, posdef if compute_hermitian else unitary