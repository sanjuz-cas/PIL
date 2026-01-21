"""
PIL-Flow Utilities

Helper functions for initialization, time scheduling, and numerical monitoring.
"""

import numpy as np
from typing import Optional, Callable, Tuple
from dataclasses import dataclass


def orthogonal_init(
    shape: Tuple[int, ...],
    gain: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Initialize a matrix with orthogonal columns.
    
    Uses QR decomposition of a random Gaussian matrix to ensure
    orthogonality, which provides better conditioning for PIL.
    
    Args:
        shape: Shape of the matrix (rows, cols)
        gain: Scaling factor for the orthogonal matrix
        seed: Random seed for reproducibility
        
    Returns:
        Orthogonal matrix of given shape
        
    Mathematical Note:
        For an m×n matrix with m >= n:
        - Q from QR decomposition has orthonormal columns
        - Q^T Q = I_n (columns are orthonormal)
        
        This ensures W_random has good spectral properties.
    """
    rng = np.random.default_rng(seed)
    
    rows, cols = shape
    
    if rows < cols:
        # More columns than rows: create orthogonal rows
        M = rng.standard_normal((cols, rows))
        Q, R = np.linalg.qr(M)
        Q = Q.T[:rows, :]
    else:
        # More rows than columns: create orthogonal columns
        M = rng.standard_normal((rows, cols))
        Q, R = np.linalg.qr(M)
        Q = Q[:, :cols]
    
    # Fix sign ambiguity in QR decomposition
    d = np.diag(R)
    ph = np.sign(d)
    ph[ph == 0] = 1
    
    if rows < cols:
        Q = (ph[:, np.newaxis] * Q)
    else:
        Q = (Q * ph)
    
    return gain * Q.astype(np.float32)


def kaiming_init(
    shape: Tuple[int, ...],
    nonlinearity: str = "leaky_relu",
    a: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Kaiming (He) initialization for layers with ReLU-family activations.
    
    Args:
        shape: Shape of the matrix
        nonlinearity: Type of nonlinearity ('relu', 'leaky_relu', 'gelu')
        a: Negative slope for leaky_relu
        seed: Random seed
        
    Returns:
        Initialized weight matrix
    """
    rng = np.random.default_rng(seed)
    
    fan_in = shape[0]
    
    if nonlinearity == "relu":
        gain = np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a ** 2))
    elif nonlinearity in ["gelu", "silu", "tanh"]:
        gain = 1.0
    else:
        gain = 1.0
    
    std = gain / np.sqrt(fan_in)
    return rng.normal(0, std, shape).astype(np.float32)


# =============================================================================
# Time Schedules for Flow Matching
# =============================================================================

def linear_time_schedule(n_steps: int) -> np.ndarray:
    """
    Linear time schedule: t ∈ [0, 1] with uniform spacing.
    
    Args:
        n_steps: Number of time steps
        
    Returns:
        Array of time values
    """
    return np.linspace(0, 1, n_steps + 1)[:-1].astype(np.float32)


def cosine_time_schedule(n_steps: int, s: float = 0.008) -> np.ndarray:
    """
    Cosine time schedule for smoother interpolation.
    
    Based on "Improved Denoising Diffusion Probabilistic Models"
    
    Args:
        n_steps: Number of time steps
        s: Small offset to prevent singularity at t=0
        
    Returns:
        Array of time values
    """
    steps = np.arange(n_steps + 1)
    alphas_cumprod = np.cos(((steps / n_steps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    # Convert cumulative alphas to time values
    t = 1 - alphas_cumprod[:-1]
    return t.astype(np.float32)


def quadratic_time_schedule(n_steps: int) -> np.ndarray:
    """
    Quadratic time schedule: more steps near t=0.
    
    Args:
        n_steps: Number of time steps
        
    Returns:
        Array of time values
    """
    t = np.linspace(0, 1, n_steps + 1)[:-1]
    return (t ** 2).astype(np.float32)


def sigmoid_time_schedule(
    n_steps: int, 
    start: float = -3, 
    end: float = 3
) -> np.ndarray:
    """
    Sigmoid time schedule for smoother transitions.
    
    Args:
        n_steps: Number of time steps
        start: Start value for sigmoid input
        end: End value for sigmoid input
        
    Returns:
        Array of time values
    """
    t_raw = np.linspace(start, end, n_steps)
    t = 1 / (1 + np.exp(-t_raw))
    # Normalize to [0, 1]
    t = (t - t.min()) / (t.max() - t.min())
    return t.astype(np.float32)


# =============================================================================
# Numerical Monitoring
# =============================================================================

@dataclass
class NumericalStats:
    """Statistics for numerical stability monitoring."""
    condition_number: float
    max_singular_value: float
    min_singular_value: float
    rank: int
    has_nan: bool
    has_inf: bool


def compute_condition_number(matrix: np.ndarray) -> float:
    """
    Compute the condition number of a matrix.
    
    κ(A) = σ_max / σ_min
    
    A well-conditioned matrix has κ close to 1.
    κ > 10^10 indicates severe ill-conditioning.
    
    Args:
        matrix: Input matrix
        
    Returns:
        Condition number
    """
    return float(np.linalg.cond(matrix))


def analyze_matrix(matrix: np.ndarray) -> NumericalStats:
    """
    Comprehensive numerical analysis of a matrix.
    
    Args:
        matrix: Input matrix
        
    Returns:
        NumericalStats with various metrics
    """
    has_nan = np.any(np.isnan(matrix))
    has_inf = np.any(np.isinf(matrix))
    
    if has_nan or has_inf:
        return NumericalStats(
            condition_number=np.inf,
            max_singular_value=np.nan,
            min_singular_value=np.nan,
            rank=0,
            has_nan=has_nan,
            has_inf=has_inf,
        )
    
    try:
        S = np.linalg.svd(matrix, compute_uv=False)
        cond = S[0] / S[-1] if S[-1] > 0 else np.inf
        rank = np.sum(S > 1e-10)
        
        return NumericalStats(
            condition_number=float(cond),
            max_singular_value=float(S[0]),
            min_singular_value=float(S[-1]),
            rank=int(rank),
            has_nan=False,
            has_inf=False,
        )
    except Exception:
        return NumericalStats(
            condition_number=np.inf,
            max_singular_value=np.nan,
            min_singular_value=np.nan,
            rank=0,
            has_nan=has_nan,
            has_inf=has_inf,
        )


def check_numerical_stability(
    matrix: np.ndarray,
    name: str = "matrix",
    warn_threshold: float = 1e8,
    error_threshold: float = 1e12,
) -> bool:
    """
    Check if a matrix is numerically stable.
    
    Args:
        matrix: Matrix to check
        name: Name for logging
        warn_threshold: Condition number threshold for warning
        error_threshold: Condition number threshold for error
        
    Returns:
        True if stable, False otherwise
    """
    stats = analyze_matrix(matrix)
    
    if stats.has_nan:
        print(f"ERROR: {name} contains NaN values")
        return False
    
    if stats.has_inf:
        print(f"ERROR: {name} contains Inf values")
        return False
    
    if stats.condition_number > error_threshold:
        print(f"ERROR: {name} is severely ill-conditioned (κ={stats.condition_number:.2e})")
        return False
    
    if stats.condition_number > warn_threshold:
        print(f"WARNING: {name} has high condition number (κ={stats.condition_number:.2e})")
    
    return True


# =============================================================================
# Interpolation Functions for Flow Matching
# =============================================================================

def linear_interpolation(
    x_0: np.ndarray,
    x_1: np.ndarray,
    t: float,
) -> np.ndarray:
    """
    Linear interpolation between noise (x_0) and data (x_1).
    
    x_t = (1 - t) * x_0 + t * x_1
    
    Args:
        x_0: Noise samples
        x_1: Data samples
        t: Time in [0, 1]
        
    Returns:
        Interpolated samples
    """
    return (1 - t) * x_0 + t * x_1


def optimal_transport_target(
    x_0: np.ndarray,
    x_1: np.ndarray,
) -> np.ndarray:
    """
    Compute the optimal transport velocity target.
    
    For linear interpolation, the optimal velocity is:
    v* = x_1 - x_0
    
    This is constant regardless of t.
    
    Args:
        x_0: Noise samples
        x_1: Data samples
        
    Returns:
        Target velocity
    """
    return x_1 - x_0


def conditional_velocity_target(
    x_0: np.ndarray,
    x_1: np.ndarray,
    x_t: np.ndarray,
    t: float,
    sigma_min: float = 1e-4,
) -> np.ndarray:
    """
    Conditional velocity target for flow matching.
    
    v*(x_t | x_0, x_1) = (x_1 - (1-σ_min)*x_t) / (1 - (1-σ_min)*t)
    
    Args:
        x_0: Noise samples
        x_1: Data samples  
        x_t: Current interpolated samples
        t: Current time
        sigma_min: Minimum noise level
        
    Returns:
        Conditional velocity target
    """
    if t >= 1 - 1e-6:
        return np.zeros_like(x_t)
    
    denom = 1 - (1 - sigma_min) * t
    return (x_1 - (1 - sigma_min) * x_t) / denom


# =============================================================================
# Activation Functions
# =============================================================================

def gelu(x: np.ndarray) -> np.ndarray:
    """
    Gaussian Error Linear Unit activation.
    
    GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
    
    Approximation: GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def leaky_relu(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Leaky ReLU activation."""
    return np.maximum(alpha * x, x)


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU (Swish) activation: x * sigmoid(x)"""
    return x / (1 + np.exp(-x))


def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """Softplus activation: (1/β) * log(1 + exp(β*x))"""
    return np.where(
        x * beta > 20,
        x,
        np.log(1 + np.exp(beta * x)) / beta
    )


# =============================================================================
# Metrics
# =============================================================================

def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(pred - target)))


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm.flatten(), b_norm.flatten()))
