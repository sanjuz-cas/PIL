"""
PIL Velocity Field Networks

Neural network architectures for learning the velocity field v(x, t)
in flow matching, trained via pseudoinverse learning instead of backpropagation.

Mathematical Background:
    Flow matching learns a velocity field v_θ(x, t) such that:

    dx/dt = v_θ(x, t)

    Integrating from t=0 (noise) to t=1 (data) generates samples.

    Standard training: minimize E[||v_θ(x_t, t) - v*(x_t, t)||²] via gradient descent
    PIL training: solve v_θ directly via pseudoinverse in closed form
"""

import numpy as np
from typing import Optional, Callable, Dict, Tuple, Literal
from dataclasses import dataclass

from .solver import ridge_solve, low_rank_ridge_solve
from .utils import (
    orthogonal_init,
    gelu,
    leaky_relu,
    silu,
    check_numerical_stability,
    analyze_matrix,
)


@dataclass
class VelocityFieldConfig:
    """Configuration for velocity field network."""

    input_dim: int
    hidden_dim: int = 256
    output_dim: Optional[int] = None  # Default: same as input_dim
    n_hidden_layers: int = 1
    activation: str = "gelu"
    lambda_reg: float = 1e-5
    use_time_embedding: bool = True
    time_embed_dim: int = 32
    use_low_rank: bool = False
    low_rank_threshold: int = 5000
    seed: Optional[int] = 42


class PILVelocityField:
    """
    Velocity Field Network trained via Pseudoinverse Learning.

    Architecture:
        Input: [x_t, t_embed] → Random Projection → Activation → Solved Output

        v(x, t) = W_out @ σ(W_random @ [x; embed(t)])

    Where:
        - W_random: Fixed random orthogonal matrix
        - σ: Nonlinear activation (GELU, ReLU, etc.)
        - W_out: Solved via ridge regression (PIL)
    """

    def __init__(self, config: VelocityFieldConfig):
        """
        Initialize velocity field network.

        Args:
            config: VelocityFieldConfig with hyperparameters
        """
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim or config.input_dim
        self.lambda_reg = config.lambda_reg

        # Time embedding dimension
        if config.use_time_embedding:
            self.time_embed_dim = config.time_embed_dim
            self.total_input_dim = self.input_dim + self.time_embed_dim
        else:
            self.time_embed_dim = 1  # Just scalar t
            self.total_input_dim = self.input_dim + 1

        # Activation function
        self.activation = self._get_activation(config.activation)

        # Initialize random projection (FIXED, not learned)
        self.W_random = orthogonal_init(
            (self.total_input_dim, self.hidden_dim), seed=config.seed
        )

        # Output weights (SOLVED via PIL)
        self.W_out = np.zeros((self.hidden_dim, self.output_dim), dtype=np.float32)

        # Bias (optional, also solved)
        self.bias = np.zeros(self.output_dim, dtype=np.float32)

        # State tracking
        self._is_fitted = False
        self._fit_stats: Dict = {}

    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        activations = {
            "gelu": gelu,
            "relu": lambda x: np.maximum(0, x),
            "leaky_relu": leaky_relu,
            "silu": silu,
            "tanh": np.tanh,
        }
        return activations.get(name, gelu)

    def _time_embedding(self, t: np.ndarray) -> np.ndarray:
        """
        Sinusoidal time embedding (similar to positional encoding).

        embed(t) = [sin(ω_1 t), cos(ω_1 t), sin(ω_2 t), cos(ω_2 t), ...]

        Args:
            t: Time values (N,) or scalar

        Returns:
            Time embeddings (N, time_embed_dim)
        """
        t = np.atleast_1d(t).astype(np.float32)

        if not self.config.use_time_embedding:
            return t.reshape(-1, 1)

        # Frequency bands (logarithmically spaced)
        half_dim = self.time_embed_dim // 2
        freqs = np.exp(-np.log(10000) * np.arange(half_dim) / half_dim).astype(
            np.float32
        )

        # Compute embeddings
        args = t[:, np.newaxis] * freqs[np.newaxis, :]  # (N, half_dim)
        embedding = np.concatenate([np.sin(args), np.cos(args)], axis=-1)

        return embedding  # (N, time_embed_dim)

    def _expand(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Feature expansion: H = σ(W_random @ [x; embed(t)])

        Args:
            x: Input samples (N, input_dim)
            t: Time values (N,) or scalar

        Returns:
            Hidden activations (N, hidden_dim)
        """
        # Time embedding
        t_embed = self._time_embedding(t)  # (N, time_embed_dim)

        # Concatenate input and time
        x_concat = np.concatenate([x, t_embed], axis=-1)  # (N, total_input_dim)

        # Random projection + activation
        H = self.activation(x_concat @ self.W_random)  # (N, hidden_dim)

        return H

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Forward pass: compute velocity v(x, t).

        Args:
            x: Input samples (N, input_dim) or (input_dim,)
            t: Time values (N,) or scalar

        Returns:
            Velocity vectors (N, output_dim)
        """
        # Handle single sample
        single_sample = x.ndim == 1
        if single_sample:
            x = x.reshape(1, -1)

        t = np.atleast_1d(t)
        if len(t) == 1 and len(x) > 1:
            t = np.full(len(x), t[0])

        # Feature expansion
        H = self._expand(x, t)

        # Output projection
        v = H @ self.W_out + self.bias

        if single_sample:
            v = v.squeeze(0)

        return v

    def fit(
        self,
        x_t: np.ndarray,
        t: np.ndarray,
        v_target: np.ndarray,
        use_low_rank: Optional[bool] = None,
    ) -> Dict:
        """
        Fit velocity field via pseudoinverse learning.

        Solves: W_out = argmin ||H @ W_out - v_target||² + λ||W_out||²

        Closed-form solution:
            W_out = (H^T H + λI)^{-1} H^T v_target

        Args:
            x_t: Interpolated samples at time t (N, input_dim)
            t: Time values (N,)
            v_target: Target velocities (N, output_dim)
            use_low_rank: Use low-rank approximation (auto if None)

        Returns:
            Dictionary with fit statistics
        """
        N = len(x_t)

        # Feature expansion
        H = self._expand(x_t, t)  # (N, hidden_dim)

        # Check numerical stability
        if not check_numerical_stability(H, "hidden_activations"):
            return {"success": False, "reason": "numerical_instability_H"}

        # Determine solver
        if use_low_rank is None:
            use_low_rank = (
                self.config.use_low_rank or N > self.config.low_rank_threshold
            )

        # Solve for W_out
        if use_low_rank:
            W_new = low_rank_ridge_solve(H, v_target, self.lambda_reg)
            method = "low_rank_svd"
        else:
            W_new = ridge_solve(H, v_target, self.lambda_reg)
            method = "ridge_solve"

        # Check solution stability
        if not check_numerical_stability(W_new, "W_out"):
            return {"success": False, "reason": "numerical_instability_W"}

        # Update weights
        self.W_out = W_new.astype(np.float32)

        # Compute bias as mean residual
        v_pred = H @ self.W_out
        residual = v_target - v_pred
        self.bias = residual.mean(axis=0).astype(np.float32)

        self._is_fitted = True

        # Compute statistics
        v_pred_final = v_pred + self.bias
        mse = np.mean((v_target - v_pred_final) ** 2)

        stats = analyze_matrix(H)

        self._fit_stats = {
            "success": True,
            "method": method,
            "n_samples": N,
            "mse": float(mse),
            "condition_number": stats.condition_number,
        }

        return self._fit_stats

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Callable interface for forward pass."""
        return self.forward(x, t)


class BiPILVelocityField:
    """
    Bidirectional PIL Velocity Field.

    Uses two parallel random projections (forward and backward) for
    richer feature representation, following the Bi-PIL architecture.

    Architecture:
        H_fwd = σ(W_fwd @ [x; t])
        H_bwd = σ(W_bwd @ [x; t])
        H = concat(H_fwd, H_bwd)  or  H = H_fwd + H_bwd
        v = W_out @ H
    """

    def __init__(
        self,
        config: VelocityFieldConfig,
        fusion: Literal["concat", "add", "gate"] = "concat",
    ):
        """
        Initialize bidirectional velocity field.

        Args:
            config: VelocityFieldConfig
            fusion: How to combine forward/backward features
        """
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim or config.input_dim
        self.fusion = fusion
        self.lambda_reg = config.lambda_reg

        # Time embedding
        if config.use_time_embedding:
            self.time_embed_dim = config.time_embed_dim
            self.total_input_dim = self.input_dim + self.time_embed_dim
        else:
            self.time_embed_dim = 1
            self.total_input_dim = self.input_dim + 1

        # Fused dimension
        if fusion == "concat":
            self.fused_dim = self.hidden_dim * 2
        else:
            self.fused_dim = self.hidden_dim

        # Activation
        self.activation = self._get_activation(config.activation)

        # Forward random projection (FIXED)
        self.W_fwd = orthogonal_init(
            (self.total_input_dim, self.hidden_dim), seed=config.seed
        )

        # Backward random projection (FIXED, different seed)
        self.W_bwd = orthogonal_init(
            (self.total_input_dim, self.hidden_dim),
            seed=config.seed + 1 if config.seed else None,
        )

        # Output weights (SOLVED)
        self.W_out = np.zeros((self.fused_dim, self.output_dim), dtype=np.float32)
        self.bias = np.zeros(self.output_dim, dtype=np.float32)

        # Gate weights (for gate fusion)
        if fusion == "gate":
            self.gate = np.ones(self.hidden_dim, dtype=np.float32) * 0.5

        self._is_fitted = False
        self._fit_stats: Dict = {}

    def _get_activation(self, name: str) -> Callable:
        activations = {
            "gelu": gelu,
            "relu": lambda x: np.maximum(0, x),
            "leaky_relu": leaky_relu,
            "silu": silu,
            "tanh": np.tanh,
        }
        return activations.get(name, gelu)

    def _time_embedding(self, t: np.ndarray) -> np.ndarray:
        """Sinusoidal time embedding."""
        t = np.atleast_1d(t).astype(np.float32)

        if not self.config.use_time_embedding:
            return t.reshape(-1, 1)

        half_dim = self.time_embed_dim // 2
        freqs = np.exp(-np.log(10000) * np.arange(half_dim) / half_dim).astype(
            np.float32
        )
        args = t[:, np.newaxis] * freqs[np.newaxis, :]

        return np.concatenate([np.sin(args), np.cos(args)], axis=-1)

    def _expand_bidirectional(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Bidirectional feature expansion.

        Args:
            x: Input samples (N, input_dim)
            t: Time values (N,)

        Returns:
            Fused hidden features (N, fused_dim)
        """
        t_embed = self._time_embedding(t)
        x_concat = np.concatenate([x, t_embed], axis=-1)

        # Forward expansion
        H_fwd = self.activation(x_concat @ self.W_fwd)

        # Backward expansion
        H_bwd = self.activation(x_concat @ self.W_bwd)

        # Fusion
        if self.fusion == "concat":
            return np.concatenate([H_fwd, H_bwd], axis=-1)
        elif self.fusion == "add":
            return H_fwd + H_bwd
        elif self.fusion == "gate":
            # Sigmoid gating
            g = 1 / (1 + np.exp(-self.gate))
            return g * H_fwd + (1 - g) * H_bwd
        else:
            return H_fwd + H_bwd

    def forward(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Forward pass: compute velocity v(x, t)."""
        single_sample = x.ndim == 1
        if single_sample:
            x = x.reshape(1, -1)

        t = np.atleast_1d(t)
        if len(t) == 1 and len(x) > 1:
            t = np.full(len(x), t[0])

        H = self._expand_bidirectional(x, t)
        v = H @ self.W_out + self.bias

        if single_sample:
            v = v.squeeze(0)

        return v

    def fit(
        self,
        x_t: np.ndarray,
        t: np.ndarray,
        v_target: np.ndarray,
        use_low_rank: Optional[bool] = None,
    ) -> Dict:
        """Fit velocity field via PIL."""
        N = len(x_t)

        H = self._expand_bidirectional(x_t, t)

        if not check_numerical_stability(H, "hidden_activations"):
            return {"success": False, "reason": "numerical_instability_H"}

        if use_low_rank is None:
            use_low_rank = (
                self.config.use_low_rank or N > self.config.low_rank_threshold
            )

        if use_low_rank:
            W_new = low_rank_ridge_solve(H, v_target, self.lambda_reg)
            method = "low_rank_svd"
        else:
            W_new = ridge_solve(H, v_target, self.lambda_reg)
            method = "ridge_solve"

        if not check_numerical_stability(W_new, "W_out"):
            return {"success": False, "reason": "numerical_instability_W"}

        self.W_out = W_new.astype(np.float32)

        v_pred = H @ self.W_out
        residual = v_target - v_pred
        self.bias = residual.mean(axis=0).astype(np.float32)

        self._is_fitted = True

        v_pred_final = v_pred + self.bias
        mse = np.mean((v_target - v_pred_final) ** 2)

        stats = analyze_matrix(H)

        self._fit_stats = {
            "success": True,
            "method": method,
            "n_samples": N,
            "mse": float(mse),
            "condition_number": stats.condition_number,
            "fusion": self.fusion,
        }

        return self._fit_stats

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def __call__(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        return self.forward(x, t)
