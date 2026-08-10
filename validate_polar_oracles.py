# Copyright 2026 Tim Tsz-Kit Lau.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Regression checks for the numerical polar-oracle implementations.

Run this script on CPU before collecting benchmark results. It checks the
Cholesky right solves directly, compares QDWH and ZOLO-PD with a float64 SVD,
tests tall and wide matrices, and verifies that the benchmark residuals detect
an orthogonal factor with the wrong polar orientation.
"""

from __future__ import annotations

import argparse

import torch

from benchmark_polar_oracles import exact_polar_reference, polar_residuals
from polar import polar
from zolopd import _right_solve_spd


def require_below(name, value, tolerance):
    if not value < tolerance:
        raise AssertionError(
            f"{name}={value:.6e} does not satisfy tolerance {tolerance:.6e}."
        )


def validate_right_solve(device):
    torch.manual_seed(11)
    left = torch.randn(13, 5, dtype=torch.float64, device=device)
    base = torch.randn(5, 5, dtype=torch.float64, device=device)
    matrix = base.mH @ base + 0.5 * torch.eye(
        5, dtype=torch.float64, device=device
    )
    computed = _right_solve_spd(left, matrix)
    reference = torch.linalg.solve(matrix, left.mH).mH
    relative_error = torch.linalg.matrix_norm(computed - reference) / torch.linalg.matrix_norm(
        reference
    )
    require_below("Hermitian right-solve relative error", relative_error.item(), 1e-12)


def validate_oracle(method, matrix, max_iterations, direction_tolerance):
    reference, nuclear_norm = exact_polar_reference(matrix)
    unitary = polar(
        matrix, method=method, max_iterations=max_iterations
    )[0]
    residuals = polar_residuals(
        matrix, unitary, reference, nuclear_norm
    )
    require_below(
        f"{method} orthogonality residual",
        residuals["orthogonality_residual"],
        direction_tolerance,
    )
    require_below(
        f"{method} reconstruction residual",
        residuals["reconstruction_residual"],
        direction_tolerance,
    )
    require_below(
        f"{method} direction error",
        residuals["relative_direction_error"],
        direction_tolerance,
    )
    require_below(
        f"{method} PSD violation",
        residuals["relative_psd_violation"],
        direction_tolerance,
    )


def validate_qdwh(device):
    torch.manual_seed(12)
    tall = torch.randn(64, 16, dtype=torch.float64, device=device)
    validate_oracle("qdwh", tall, max_iterations=5, direction_tolerance=1e-9)
    validate_oracle("qdwh", tall.mH, max_iterations=5, direction_tolerance=1e-9)

    left = torch.linalg.qr(
        torch.randn(48, 12, dtype=torch.float64, device=device), mode="reduced"
    ).Q
    right = torch.linalg.qr(
        torch.randn(12, 12, dtype=torch.float64, device=device), mode="reduced"
    ).Q
    singular_values = torch.logspace(
        0, -6, 12, dtype=torch.float64, device=device
    )
    ill_conditioned = (left * singular_values.unsqueeze(0)) @ right.mH
    validate_oracle(
        "qdwh",
        ill_conditioned,
        max_iterations=5,
        direction_tolerance=1e-8,
    )


def validate_zolo(device):
    torch.manual_seed(13)
    tall = torch.randn(32, 8, dtype=torch.float64, device=device)
    validate_oracle("zolo-pd", tall, max_iterations=5, direction_tolerance=1e-8)
    validate_oracle("zolo-pd", tall.mH, max_iterations=5, direction_tolerance=1e-8)


def validate_residual_diagnostics(device):
    torch.manual_seed(14)
    matrix = torch.randn(20, 5, dtype=torch.float64, device=device)
    reference, nuclear_norm = exact_polar_reference(matrix)
    rotation = torch.linalg.qr(
        torch.randn(5, 5, dtype=torch.float64, device=device)
    ).Q
    wrong_direction = reference @ rotation
    residuals = polar_residuals(
        matrix, wrong_direction, reference, nuclear_norm
    )
    require_below(
        "rotated factor orthogonality residual",
        residuals["orthogonality_residual"],
        1e-12,
    )
    if max(
        residuals["hermitian_symmetry_residual"],
        residuals["reconstruction_residual"],
        abs(residuals["polar_objective_relative_gap"]),
    ) <= 1e-3:
        raise AssertionError(
            "The polar diagnostics failed to detect an orthogonal factor with "
            "the wrong orientation."
        )


def validate_input_checks(device):
    try:
        polar(torch.randn(8, device=device), method="polar_express")
    except ValueError:
        return
    raise AssertionError("A one-dimensional oracle input did not raise ValueError.")


def validate_wrapper_shapes(device):
    torch.manual_seed(15)
    for shape in ((8, 3), (3, 8)):
        matrix = torch.randn(*shape, dtype=torch.float32, device=device)
        for method in ("ns", "polar_express", "precond_ns"):
            unitary, _ = polar(matrix, method=method, max_iterations=2)
            if unitary.shape != matrix.shape:
                raise AssertionError(
                    f"polar(..., method={method!r}) returned shape "
                    f"{tuple(unitary.shape)} for input {shape}."
                )
        for method in ("ns", "polar_express"):
            unitary, hermitian = polar(
                matrix,
                method=method,
                max_iterations=2,
                compute_hermitian=True,
            )
            expected_hermitian_shape = (
                (shape[1], shape[1])
                if shape[0] >= shape[1]
                else (shape[0], shape[0])
            )
            if unitary.shape != matrix.shape or hermitian.shape != expected_hermitian_shape:
                raise AssertionError(
                    f"Incorrect {method} polar-factor shapes for input {shape}: "
                    f"U={tuple(unitary.shape)}, H={tuple(hermitian.shape)}."
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--skip-zolo",
        action="store_true",
        help="Skip the slower end-to-end ZOLO-PD checks.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable.")

    torch.set_float32_matmul_precision("highest")
    validate_right_solve(device)
    validate_qdwh(device)
    if not args.skip_zolo:
        validate_zolo(device)
    validate_residual_diagnostics(device)
    validate_input_checks(device)
    validate_wrapper_shapes(device)
    print("All polar-oracle regression checks passed.")


if __name__ == "__main__":
    main()
