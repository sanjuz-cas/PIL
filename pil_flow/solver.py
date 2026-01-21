"""
PIL Solvers: Linear Algebra Operations for Pseudoinverse Learning

This module provides numerically stable implementations of ridge regression
and pseudoinverse computation used throughout PIL-Flow.

Mathematical Foundation:
    Given hidden activations H and targets Y, we solve:

    W = argmin_W ||HW - Y||_2^2 + λ||W||_2^2

    Closed-form solution:
    W = (H^T H + λI)^{-1} H^T Y

    Or equivalently using the pseudoinverse:
    W = H^† Y  (where H^† is the Moore-Penrose pseudoinverse)
"""

import numpy as np
from typing import Optional, Tuple
import warnings


def ridge_solve(
    H: np.ndarray,
    Y: np.ndarray,
    lambda_reg: float = 1e-5,
    rcond: Optional[float] = None,
) -> np.ndarray:
    """
    Solve ridge regression: W = (H^T H + λI)^{-1} H^T Y

    This is the core PIL operation that replaces gradient descent.

    Args:
        H: Hidden activations matrix (N, hidden_dim)
        Y: Target matrix (N, output_dim)
        lambda_reg: Regularization parameter λ (ridge penalty)
        rcond: Cutoff for small singular values (for pinv fallback)

    Returns:
        W: Solved weight matrix (hidden_dim, output_dim)

    Mathematical Notes:
        - For N > hidden_dim: Use normal equations (H^T H + λI)^{-1} H^T Y
        - For N < hidden_dim: Use dual form H^T (H H^T + λI)^{-1} Y
        - Regularization λ ensures numerical stability
    """
    N, hidden_dim = H.shape

    if N >= hidden_dim:
        # Standard form: W = (H^T H + λI)^{-1} H^T Y
        HtH = H.T @ H  # (hidden_dim, hidden_dim)
        regularized = HtH + lambda_reg * np.eye(hidden_dim)
        HtY = H.T @ Y  # (hidden_dim, output_dim)

        try:
            # Try Cholesky decomposition (fastest for positive definite)
            L = np.linalg.cholesky(regularized)
            W = np.linalg.solve(L.T, np.linalg.solve(L, HtY))
        except np.linalg.LinAlgError:
            # Fallback to general solve
            try:
                W = np.linalg.solve(regularized, HtY)
            except np.linalg.LinAlgError:
                # Final fallback to pseudoinverse
                warnings.warn("Matrix ill-conditioned, using pseudoinverse")
                W = np.linalg.pinv(regularized, rcond=rcond) @ HtY
    else:
        # Dual form (more efficient when N < hidden_dim):
        # W = H^T (H H^T + λI)^{-1} Y
        HHt = H @ H.T  # (N, N)
        regularized = HHt + lambda_reg * np.eye(N)

        try:
            L = np.linalg.cholesky(regularized)
            temp = np.linalg.solve(L.T, np.linalg.solve(L, Y))
        except np.linalg.LinAlgError:
            try:
                temp = np.linalg.solve(regularized, Y)
            except np.linalg.LinAlgError:
                warnings.warn("Matrix ill-conditioned, using pseudoinverse")
                temp = np.linalg.pinv(regularized, rcond=rcond) @ Y

        W = H.T @ temp

    return W


def low_rank_ridge_solve(
    H: np.ndarray,
    Y: np.ndarray,
    lambda_reg: float = 1e-5,
    rank: Optional[int] = None,
    energy_threshold: float = 0.99,
) -> np.ndarray:
    """
    Low-rank approximation of ridge regression using truncated SVD.

    For large matrices, computing (H^T H + λI)^{-1} is O(d^3).
    Using truncated SVD, we can approximate with O(N * d * r) where r << d.

    Mathematical Derivation:
        H ≈ U_r Σ_r V_r^T  (truncated SVD)

        W = V_r (Σ_r^2 + λI)^{-1} Σ_r U_r^T Y

    This is exact when rank(H) <= r.

    Args:
        H: Hidden activations matrix (N, hidden_dim)
        Y: Target matrix (N, output_dim)
        lambda_reg: Regularization parameter
        rank: Number of singular values to keep (None = auto)
        energy_threshold: Keep singular values capturing this fraction of energy

    Returns:
        W: Solved weight matrix (hidden_dim, output_dim)
    """
    N, hidden_dim = H.shape

    # Compute full SVD
    U, S, Vt = np.linalg.svd(H, full_matrices=False)

    # Determine rank
    if rank is None:
        # Auto-determine rank based on energy threshold
        total_energy = np.sum(S**2)
        cumulative_energy = np.cumsum(S**2) / total_energy
        rank = np.searchsorted(cumulative_energy, energy_threshold) + 1
        rank = min(rank, len(S))

    # Truncate
    U_r = U[:, :rank]  # (N, rank)
    S_r = S[:rank]  # (rank,)
    Vt_r = Vt[:rank, :]  # (rank, hidden_dim)

    # Compute regularized inverse of singular values
    # (Σ_r^2 + λI)^{-1} Σ_r = Σ_r / (Σ_r^2 + λ)
    S_reg_inv = S_r / (S_r**2 + lambda_reg)  # (rank,)

    # W = V_r^T @ diag(S_reg_inv) @ U_r^T @ Y
    # Efficiently compute step by step
    temp = U_r.T @ Y  # (rank, output_dim)
    temp = S_reg_inv[:, np.newaxis] * temp  # (rank, output_dim)
    W = Vt_r.T @ temp  # (hidden_dim, output_dim)

    return W


def safe_inverse(
    matrix: np.ndarray,
    lambda_reg: float = 1e-5,
    max_condition: float = 1e10,
) -> Tuple[np.ndarray, float]:
    """
    Safe matrix inversion with regularization and condition monitoring.

    Args:
        matrix: Square matrix to invert
        lambda_reg: Regularization parameter
        max_condition: Maximum allowed condition number

    Returns:
        Tuple of (inverted matrix, condition number)
    """
    n = matrix.shape[0]
    regularized = matrix + lambda_reg * np.eye(n)

    # Compute condition number
    cond = np.linalg.cond(regularized)

    if cond > max_condition:
        warnings.warn(f"Matrix ill-conditioned (κ={cond:.2e}), using pseudoinverse")
        return np.linalg.pinv(regularized), cond

    try:
        return np.linalg.inv(regularized), cond
    except np.linalg.LinAlgError:
        return np.linalg.pinv(regularized), cond


def iterative_ridge_update(
    W_old: np.ndarray,
    H_new: np.ndarray,
    Y_new: np.ndarray,
    lambda_reg: float = 1e-5,
    forget_factor: float = 0.99,
) -> np.ndarray:
    """
    Sherman-Morrison-Woodbury update for incremental ridge regression.

    Instead of re-solving from scratch, update weights incrementally
    when new data arrives. Useful for online/streaming learning.

    Mathematical Foundation:
        If we have A^{-1} and want (A + UV^T)^{-1}:

        (A + UV^T)^{-1} = A^{-1} - A^{-1} U (I + V^T A^{-1} U)^{-1} V^T A^{-1}

    Args:
        W_old: Previous weight matrix
        H_new: New hidden activations (N_new, hidden_dim)
        Y_new: New targets (N_new, output_dim)
        lambda_reg: Regularization parameter
        forget_factor: Exponential forgetting (1.0 = no forgetting)

    Returns:
        W_new: Updated weight matrix
    """
    # This is a simplified version - full implementation would
    # maintain the inverse covariance matrix

    # For now, combine old prediction residual with new data
    N_new = H_new.shape[0]

    # Compute residual from old weights
    residual = Y_new - H_new @ W_old

    # Solve for delta W
    delta_W = ridge_solve(H_new, residual, lambda_reg)

    # Update with forgetting
    W_new = forget_factor * W_old + (1 - forget_factor) * delta_W

    return W_new


def batch_ridge_solve(
    H_batches: list,
    Y_batches: list,
    lambda_reg: float = 1e-5,
) -> np.ndarray:
    """
    Solve ridge regression over multiple batches without storing all data.

    Uses the fact that:
        (Σ H_i^T H_i + λI)^{-1} (Σ H_i^T Y_i)

    Can be computed incrementally.

    Args:
        H_batches: List of hidden activation matrices
        Y_batches: List of target matrices
        lambda_reg: Regularization parameter

    Returns:
        W: Solved weight matrix
    """
    if not H_batches:
        raise ValueError("No batches provided")

    hidden_dim = H_batches[0].shape[1]
    output_dim = Y_batches[0].shape[1]

    # Accumulate H^T H and H^T Y
    HtH_sum = np.zeros((hidden_dim, hidden_dim))
    HtY_sum = np.zeros((hidden_dim, output_dim))

    for H, Y in zip(H_batches, Y_batches):
        HtH_sum += H.T @ H
        HtY_sum += H.T @ Y

    # Add regularization and solve
    regularized = HtH_sum + lambda_reg * np.eye(hidden_dim)

    try:
        W = np.linalg.solve(regularized, HtY_sum)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(regularized) @ HtY_sum

    return W
