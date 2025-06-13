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

"""A library for ZOLO-PD algorithm, modified from MATLAB.
https://people.maths.ox.ac.uk/nakatsukasa/codes/zolomatlabcodes.zip

ZOLO-PD is short for Zolotarev-based polar decomposition. 
ZOLO-PD implemented through QR and Cholesky decompositions does not require matrix
inversion. This is desirable for multicore and heterogeneous computing systems.

Reference: Yuji Nakatsukasa and Roland W. Freund. 
"Computing Fundamental Matrix Decompositions Accurately via the Matrix Sign Function 
in Two Iterations: The Power of Zolotarev's Functions." 
SIAM Review, vol. 58, no. 3 (2016): 461-493.
https://doi.org/10.1137/140990334
"""

import torch
import math


# Helper for estimating the spectral norm, corresponding to the normest function in MATLAB.
def normest(A, tol=1e-6, max_iter=20):
    """
    Estimate the 2-norm (spectral norm) of matrix A using an iterative power method.
    This approach is similar in spirit to MATLAB's normest.

    Args:
        A (torch.Tensor): A 2D tensor representing the matrix.
        tol (float): Relative tolerance for convergence.
        max_iter (int): Maximum number of iterations.

    Returns:
        float: Estimated 2-norm of the matrix A.
    """
    # Get the number of columns in A
    n = A.shape[1]
    
    # Start with a random vector of appropriate size (normalized)
    x = torch.randn(n, device=A.device, dtype=A.dtype)
    x = x / torch.norm(x)
    
    norm_old = 0.0
    for i in range(max_iter):
        # Compute y = A * x
        y = A @ x
        norm_y = torch.norm(y)
        if norm_y == 0:
            # A is the zero matrix, so its norm is 0.
            return 0.0
        
        # Update x by multiplying by A^T (and normalize)
        x = A.T @ y / norm_y
        
        # Current estimate of the norm is the norm of y
        norm_new = norm_y
        
        # Check for convergence: if the relative change is below tolerance, exit early
        if torch.abs(norm_new - norm_old) < tol * norm_new:
            break
        norm_old = norm_new

    return norm_new

def choosem(con):
    """
    Choose the Zolotarev degree based on the estimated condition number.
    """
    if con < 1.001:
        return 2
    elif con <= 1.01:
        return 3
    elif con <= 1.1:
        return 4
    elif con <= 1.2:
        return 5
    elif con <= 1.5:
        return 6
    elif con <= 2:
        return 8  # one-step convergence till here
    elif con < 6.5:
        return 2
    elif con < 180:
        return 3
    elif con < 1.5e4:
        return 4
    elif con < 2e6:
        return 5
    elif con < 1e9:
        return 6
    elif con < 3e12:
        return 7
    else:
        return 8

def computeAA(A, c, it, howqr='house'):
    """
    Compute the rational correction applied to A.
    
    A : current iterate (m x n tensor)
    c : 1D tensor of length 2*m_zol (must be float64)
    it: current iteration number (integer)
    howqr: method string; by default 'house' uses a QR-based method.
    """
    AA = A.clone()
    m, n = A.shape
    r = len(c) // 2  # m_zol

    for ii in range(r):
        # Compute numerator product:
        enu = torch.tensor(1.0, dtype=c.dtype, device=c.device)
        for jj in range(r):
            # MATLAB: c(2*ii-1)-c(2*jj)  => python indices: c[2*ii] - c[2*jj+1]
            enu *= (c[2 * ii] - c[2 * jj + 1])
        # Compute denominator product (excluding jj==ii):
        den = torch.tensor(1.0, dtype=c.dtype, device=c.device)
        for jj in range(r):
            if ii != jj:
                # MATLAB: c(2*ii-1)-c(2*jj-1) => python: c[2*ii] - c[2*jj]
                den *= (c[2 * ii] - c[2 * jj])
        c_val = c[2 * ii]  # corresponds to c(2*ii-1) in MATLAB

        # Branch based on iteration and size of c.
        if it <= 1 and c[:-1].max() > 1e2:
            if howqr == 'house':
                sqrt_c = torch.sqrt(c_val)
                I_n = torch.eye(n, dtype=A.dtype, device=A.device)
                # Stack A and sqrt(c_val)*I vertically:
                stacked = torch.cat([A, sqrt_c * I_n], dim=0)
                Q, _ = torch.linalg.qr(stacked, mode='reduced')
                AAA_top = Q[:m, :]
                AAA_bottom = Q[m:, :]
                AA = AA - (enu / den / sqrt_c) * (AAA_top @ AAA_bottom.T)
            else:
                # Cholesky-based branch (less recommended)
                sqrt_c = torch.sqrt(c_val)
                I_n = torch.eye(n, dtype=A.dtype, device=A.device)
                R, _ = torch.linalg.cholesky_ex(A.T @ A + c_val * I_n)
                Q = torch.linalg.solve_triangular(R, A.T, upper=True).T
                II = sqrt_c * I_n
                II = torch.linalg.solve_triangular(R, II, upper=True).T
                # (Optionally, one could check torch.linalg.cond(R) here.)
                AA = AA - (enu / den / sqrt_c) * (Q @ II.T)
        else:
            I_n = torch.eye(n, dtype=A.dtype, device=A.device)
            Cinv, _ = torch.linalg.cholesky_ex(A.T @ A + c_val * I_n)
            # Compute A / Cinv: solve R X = A.T, then transpose back.
            Qtmp = torch.linalg.solve_triangular(Cinv, A.T, upper=True).T
            # Then divide by Cinv' (i.e. solve with Cinv again)
            Qtmp = torch.linalg.solve_triangular(Cinv, Qtmp.T, upper=True).T
            AA = AA - (enu / den) * Qtmp
    return AA

def mellipke(alpha, tol=None):
    """
    Compute the complete elliptic integrals K and E.
    
    Parameters:
        alpha : float (angle in radians)
        tol   : tolerance (if None, set to machine epsilon for float64)
    
    Returns:
        K, E : two floats
    """
    if tol is None:
        tol = torch.finfo(torch.float64).eps
    m_val = math.sin(alpha) ** 2
    # (m1 is computed in MATLAB but not used further)
    a0 = 1.0
    b0 = math.cos(alpha)
    s0 = m_val
    i1 = 0
    mm = 1.0
    while mm > tol:
        a1 = 0.5 * (a0 + b0)
        b1 = math.sqrt(a0 * b0)
        c1 = 0.5 * (a0 - b0)
        i1 += 1
        w1 = (2 ** i1) * (c1 ** 2)
        mm = w1
        s0 += w1
        a0, b0 = a1, b1
    K = math.pi / (2 * a1)
    E = K * (1 - s0 / 2)
    return K, E

def mellipj(u, alpha, tol=None):
    """
    Compute the Jacobi elliptic functions sn, cn, dn for a scalar u and parameter alpha.
    
    This implementation is a scalar version based on the arithmetic–geometric mean algorithm.
    Parameters:
        u     : scalar (float)
        alpha : float (angle in radians)
        tol   : tolerance (if None, set to machine epsilon for float64)
    Returns:
        sn, cn, dn : three floats
    """
    if tol is None:
        tol = torch.finfo(torch.float64).eps
    m_val = math.sin(alpha) ** 2

    # Initialization (using the standard AGM algorithm)
    a_vals = [1.0]
    b_vals = [math.cos(alpha)]
    c_vals = [math.sin(alpha)]
    i = 0
    while abs(c_vals[i]) > tol and i < 1000:
        a_next = 0.5 * (a_vals[i] + b_vals[i])
        b_next = math.sqrt(a_vals[i] * b_vals[i])
        c_next = 0.5 * (a_vals[i] - b_vals[i])
        a_vals.append(a_next)
        b_vals.append(b_next)
        c_vals.append(c_next)
        i += 1
    n = i
    # Compute the amplitude phi
    phi = (2 ** n) * a_vals[-1] * u
    # Backward recurrence
    for j in range(n - 1, -1, -1):
        # Use math.asin; note that sin(phi) is computed here
        temp = c_vals[j + 1] * math.sin(phi) / a_vals[j + 1]
        # Clamp to [-1,1] to avoid domain issues due to round-off
        temp = max(-1.0, min(1.0, temp))
        phi = 0.5 * (math.asin(temp) + phi)
    sn = math.sin(phi)
    cn = math.cos(phi)
    dn = math.sqrt(1 - m_val * (sn ** 2))
    return sn, cn, dn


# @torch.compile
def zolopd(A, compute_hermitian=False, alpha=None, L=None):
    """
    Polar decomposition via Zolotarev-approximation.
    
    Parameters:
        A     : (m x n) torch.Tensor (input matrix)
        compute_hermitian : (optional) if True, compute the Hermitian factor
        alpha : (optional) a scalar approximating ||A|| (if None, estimated)
        L     : (optional) a scalar approximating 1/cond(A)
    
    Returns:
        U    : the computed polar factor (m x n tensor)
        H    : the computed Hermitian factor (n x n tensor)
        m_zol: the Zolotarev degree (an integer)
        it   : the iteration count (usually 1 or 2)
    """
    # Ensure A is float64 for accuracy.
    if A.dtype != torch.float64:
        A = A.to(torch.float64)
    m, n = A.shape

    # Check if A is (numerically) symmetric.
    if m == n and torch.linalg.matrix_norm(A - A.T) / torch.linalg.matrix_norm(A) < 1e-14:
        symm = True
    else:
        symm = False

    # Estimate the largest singular value if alpha not provided.
    if alpha is None:
        alpha = normest(A, tol=0.1)

    # Scale original matrix to form X0.
    U = A / alpha

    # If L not provided, estimate it.
    if L is None:
        Y = U
        if m > n:
            # Compute a thin QR factorization.
            Q, R = torch.linalg.qr(U, mode='reduced')
            Y = R
        # Estimate smallest singular value approximately:
        cond_Y = torch.linalg.cond(Y, p=1)
        smin_est = torch.linalg.matrix_norm(Y, ord=1) / cond_Y
        L = smin_est / math.sqrt(n)

    U = U / L  # now ||U|| is approximately 1
    con = 1 / L

    it = 0
    m_zol = choosem(con)
    itmax = 1 if con < 2 else 2

    # Iterative loop.
    while it < itmax:
        it += 1
        kp = 1 / con
        # Compute the angle (in radians) for elliptic functions.
        alpha_angle = math.acos(kp)
        K, _ = mellipke(alpha_angle)
        m_zol = choosem(con)  # update degree if desired

        # Build the vector c (of length 2*m_zol).
        c = torch.zeros(2 * m_zol, dtype=A.dtype, device=A.device)
        for ii in range(2 * m_zol):
            # In MATLAB: u = ii*K/(2*m_zol+1) for ii=1,...,2*m_zol.
            u_val = (ii + 1) * K / (2 * m_zol + 1)
            sn_val, cn_val, _ = mellipj(u_val, alpha_angle)
            c[ii] = (sn_val ** 2) / (cn_val ** 2)

        # Define the rational function ff by composing factors.
        def ff(x):
            val = x
            for j in range(m_zol):
                # MATLAB: factor = (x^2 + c(2*ii))/(x^2 + c(2*ii-1))
                # Python indices: c[2*j+1] and c[2*j]
                val = val * ((x ** 2 + c[2 * j + 1]) / (x ** 2 + c[2 * j]))
            return val

        U = computeAA(U, c, it)
        U = U / ff(1)  # normalize so that (roughly) min(svd(A)) = 1

        if con < 2:
            con = max(ff(con) / ff(1), 1)
            break
        else:
            con = max(ff(con) / ff(1), 1)
        if symm:
            U = 0.5 * (U.T + U)  # force symmetry if A is symmetric

    # One step of Newton–Schulz postprocessing.
    U = 1.5 * U - 0.5 * U @ (U.T @ U)

    # Compute the Hermitian factor H.
    if compute_hermitian:
        H = U.T @ A
        H = (H + H.T) / 2
        return U, H, m_zol, it 
    else:
        return U, m_zol, it