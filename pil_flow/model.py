"""
PIL-Flow: Main Flow Matching Model

This module implements the complete PIL-Flow framework for gradient-free
generative modeling via flow matching.

Mathematical Foundation:
    Flow matching learns a time-dependent velocity field v(x, t) that
    defines an ODE:
    
        dx/dt = v(x, t)
    
    Starting from noise x_0 ~ N(0, I) at t=0 and integrating to t=1
    generates samples from the data distribution.
    
    Training Objective:
        min E_{t, x_0, x_1} [||v_θ(x_t, t) - u_t(x_t | x_0, x_1)||²]
        
    Where:
        - x_t = (1-t)x_0 + tx_1 is the linear interpolation
        - u_t = x_1 - x_0 is the conditional velocity (constant for linear)
        
    PIL Innovation:
        Instead of minimizing via gradient descent, we solve for v_θ
        directly using the pseudoinverse:
        
        W = (H^T H + λI)^{-1} H^T u_t
"""

import numpy as np
from typing import Optional, Dict, List, Literal, Tuple, Callable
from dataclasses import dataclass, field
import time

from .velocity_field import (
    PILVelocityField,
    BiPILVelocityField,
    VelocityFieldConfig,
)
from .solver import ridge_solve, batch_ridge_solve
from .utils import (
    linear_time_schedule,
    cosine_time_schedule,
    quadratic_time_schedule,
    linear_interpolation,
    optimal_transport_target,
    compute_mse,
    check_numerical_stability,
)


@dataclass
class PILFlowConfig:
    """Configuration for PIL-Flow model."""
    
    # Data dimensions
    input_dim: int
    
    # Velocity field architecture
    hidden_dim: int = 512
    n_hidden_layers: int = 1
    activation: str = "gelu"
    use_bipil: bool = True
    bipil_fusion: Literal["concat", "add", "gate"] = "concat"
    
    # Time discretization
    n_train_timesteps: int = 100
    n_sample_timesteps: int = 50
    time_schedule: Literal["linear", "cosine", "quadratic"] = "linear"
    
    # PIL parameters
    lambda_reg: float = 1e-5
    use_low_rank: bool = False
    low_rank_threshold: int = 5000
    
    # Time embedding
    use_time_embedding: bool = True
    time_embed_dim: int = 64
    
    # Training
    per_timestep_fit: bool = True  # Fit separate network per timestep
    batch_size: Optional[int] = None  # None = full batch
    
    # Generation
    ode_solver: Literal["euler", "midpoint", "rk4"] = "euler"
    
    # Reproducibility
    seed: Optional[int] = 42


class PILFlowMatching:
    """
    PIL-Flow: Gradient-Free Flow Matching
    
    A complete implementation of flow matching trained via pseudoinverse
    learning instead of backpropagation.
    
    Key Features:
        - One-shot training per timestep (no iterative optimization)
        - Bidirectional PIL for richer representations
        - Multiple ODE solvers for generation
        - Numerical stability monitoring throughout
        
    Usage:
        ```python
        config = PILFlowConfig(input_dim=784)
        model = PILFlowMatching(config)
        
        # Training (one-shot!)
        model.fit(X_data)
        
        # Generation
        samples = model.sample(n_samples=100)
        ```
    """
    
    def __init__(self, config: PILFlowConfig):
        """
        Initialize PIL-Flow model.
        
        Args:
            config: PILFlowConfig with hyperparameters
        """
        self.config = config
        self.input_dim = config.input_dim
        
        # Time schedule
        self.train_timesteps = self._get_time_schedule(
            config.n_train_timesteps, config.time_schedule
        )
        self.sample_timesteps = self._get_time_schedule(
            config.n_sample_timesteps, config.time_schedule
        )
        
        # Velocity field(s)
        # If per_timestep_fit, we create one network per timestep
        # Otherwise, one shared network
        self.velocity_fields: Dict[int, PILVelocityField] = {}
        self._init_velocity_fields()
        
        # State tracking
        self._is_fitted = False
        self._fit_history: List[Dict] = []
        self._rng = np.random.default_rng(config.seed)
    
    def _get_time_schedule(self, n_steps: int, schedule: str) -> np.ndarray:
        """Get time schedule based on config."""
        if schedule == "linear":
            return linear_time_schedule(n_steps)
        elif schedule == "cosine":
            return cosine_time_schedule(n_steps)
        elif schedule == "quadratic":
            return quadratic_time_schedule(n_steps)
        else:
            return linear_time_schedule(n_steps)
    
    def _init_velocity_fields(self):
        """Initialize velocity field networks."""
        vf_config = VelocityFieldConfig(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.input_dim,
            n_hidden_layers=self.config.n_hidden_layers,
            activation=self.config.activation,
            lambda_reg=self.config.lambda_reg,
            use_time_embedding=self.config.use_time_embedding,
            time_embed_dim=self.config.time_embed_dim,
            use_low_rank=self.config.use_low_rank,
            low_rank_threshold=self.config.low_rank_threshold,
            seed=self.config.seed,
        )
        
        if self.config.per_timestep_fit:
            # One network per timestep
            for i in range(len(self.train_timesteps)):
                if self.config.use_bipil:
                    self.velocity_fields[i] = BiPILVelocityField(
                        vf_config,
                        fusion=self.config.bipil_fusion,
                    )
                else:
                    # Different seed per timestep for diversity
                    vf_config_i = VelocityFieldConfig(
                        **{**vf_config.__dict__, "seed": self.config.seed + i}
                    )
                    self.velocity_fields[i] = PILVelocityField(vf_config_i)
        else:
            # Single shared network
            if self.config.use_bipil:
                self.velocity_fields[0] = BiPILVelocityField(
                    vf_config,
                    fusion=self.config.bipil_fusion,
                )
            else:
                self.velocity_fields[0] = PILVelocityField(vf_config)
    
    def _get_velocity_field(self, timestep_idx: int):
        """Get velocity field for given timestep."""
        if self.config.per_timestep_fit:
            return self.velocity_fields[timestep_idx]
        else:
            return self.velocity_fields[0]
    
    def fit(
        self,
        X_data: np.ndarray,
        verbose: bool = True,
    ) -> Dict:
        """
        Fit the flow matching model via PIL.
        
        For each timestep t:
            1. Sample noise x_0 ~ N(0, I)
            2. Interpolate: x_t = (1-t)x_0 + t*x_1
            3. Compute target velocity: v* = x_1 - x_0
            4. Solve via PIL: W = (H^T H + λI)^{-1} H^T v*
        
        Args:
            X_data: Training data (N, input_dim)
            verbose: Print progress
            
        Returns:
            Dictionary with training statistics
        """
        N = len(X_data)
        X_data = X_data.astype(np.float32)
        
        if verbose:
            print(f"PIL-Flow Training: {N} samples, {len(self.train_timesteps)} timesteps")
            print(f"Architecture: {'BiPIL' if self.config.use_bipil else 'PIL'}")
            print("-" * 50)
        
        start_time = time.time()
        total_mse = 0.0
        
        for i, t in enumerate(self.train_timesteps):
            # Sample noise
            x_0 = self._rng.standard_normal((N, self.input_dim)).astype(np.float32)
            
            # Data
            x_1 = X_data
            
            # Interpolate
            x_t = linear_interpolation(x_0, x_1, t)
            
            # Target velocity (optimal transport)
            v_target = optimal_transport_target(x_0, x_1)
            
            # Time values (all same t for this batch)
            t_batch = np.full(N, t, dtype=np.float32)
            
            # Get velocity field for this timestep
            vf = self._get_velocity_field(i)
            
            # Fit via PIL (one-shot!)
            fit_stats = vf.fit(x_t, t_batch, v_target)
            
            if not fit_stats.get("success", False):
                print(f"  Warning: Fit failed at t={t:.3f}: {fit_stats.get('reason')}")
                continue
            
            total_mse += fit_stats.get("mse", 0)
            
            self._fit_history.append({
                "timestep_idx": i,
                "t": float(t),
                **fit_stats
            })
            
            if verbose and (i + 1) % max(1, len(self.train_timesteps) // 10) == 0:
                print(f"  Timestep {i+1}/{len(self.train_timesteps)}: "
                      f"t={t:.3f}, MSE={fit_stats.get('mse', 0):.6f}, "
                      f"κ={fit_stats.get('condition_number', 0):.2e}")
        
        self._is_fitted = True
        elapsed = time.time() - start_time
        avg_mse = total_mse / len(self.train_timesteps)
        
        if verbose:
            print("-" * 50)
            print(f"Training complete in {elapsed:.2f}s")
            print(f"Average MSE: {avg_mse:.6f}")
        
        return {
            "n_samples": N,
            "n_timesteps": len(self.train_timesteps),
            "avg_mse": avg_mse,
            "elapsed_time": elapsed,
            "history": self._fit_history,
        }
    
    def velocity(self, x: np.ndarray, t: float) -> np.ndarray:
        """
        Compute velocity v(x, t) at given point and time.
        
        Args:
            x: Points (N, input_dim) or (input_dim,)
            t: Time value in [0, 1]
            
        Returns:
            Velocity vectors (N, input_dim) or (input_dim,)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Find closest timestep
        timestep_idx = np.argmin(np.abs(self.train_timesteps - t))
        vf = self._get_velocity_field(timestep_idx)
        
        return vf(x, np.array([t]))
    
    def _euler_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Euler integration step: x_{t+dt} = x_t + dt * v(x_t, t)"""
        v = self.velocity(x, t)
        return x + dt * v
    
    def _midpoint_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Midpoint (RK2) integration step."""
        v1 = self.velocity(x, t)
        x_mid = x + 0.5 * dt * v1
        v2 = self.velocity(x_mid, t + 0.5 * dt)
        return x + dt * v2
    
    def _rk4_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Runge-Kutta 4 integration step."""
        k1 = self.velocity(x, t)
        k2 = self.velocity(x + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.velocity(x + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.velocity(x + dt * k3, t + dt)
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def _integration_step(self, x: np.ndarray, t: float, dt: float) -> np.ndarray:
        """Single integration step using configured solver."""
        if self.config.ode_solver == "euler":
            return self._euler_step(x, t, dt)
        elif self.config.ode_solver == "midpoint":
            return self._midpoint_step(x, t, dt)
        elif self.config.ode_solver == "rk4":
            return self._rk4_step(x, t, dt)
        else:
            return self._euler_step(x, t, dt)
    
    def sample(
        self,
        n_samples: int,
        return_trajectory: bool = False,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate samples by integrating the learned ODE.
        
        Solves: dx/dt = v(x, t) from t=0 to t=1
        Starting from x_0 ~ N(0, I)
        
        Args:
            n_samples: Number of samples to generate
            return_trajectory: Return full trajectory (all timesteps)
            seed: Random seed for noise
            
        Returns:
            Generated samples (n_samples, input_dim)
            If return_trajectory: (n_timesteps+1, n_samples, input_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        rng = np.random.default_rng(seed or self.config.seed)
        
        # Start from noise
        x = rng.standard_normal((n_samples, self.input_dim)).astype(np.float32)
        
        trajectory = [x.copy()] if return_trajectory else None
        
        # Integrate from t=0 to t=1
        timesteps = self.sample_timesteps
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            dt = timesteps[i + 1] - t
            
            x = self._integration_step(x, t, dt)
            
            if return_trajectory:
                trajectory.append(x.copy())
        
        # Final step to t=1
        if timesteps[-1] < 1.0:
            t = timesteps[-1]
            dt = 1.0 - t
            x = self._integration_step(x, t, dt)
            if return_trajectory:
                trajectory.append(x.copy())
        
        if return_trajectory:
            return np.stack(trajectory, axis=0)
        
        return x
    
    def reconstruct(self, x_data: np.ndarray, noise_level: float = 0.0) -> np.ndarray:
        """
        Reconstruct data by encoding to noise and decoding back.
        
        This provides a way to measure reconstruction quality.
        
        Args:
            x_data: Data samples (N, input_dim)
            noise_level: Amount of noise to add at t=0 (0 = perfect reconstruction)
            
        Returns:
            Reconstructed samples (N, input_dim)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        N = len(x_data)
        
        # Add noise
        if noise_level > 0:
            noise = self._rng.standard_normal(x_data.shape).astype(np.float32)
            x = noise_level * noise + (1 - noise_level) * x_data
        else:
            x = x_data.copy()
        
        # Integrate from current state
        timesteps = self.sample_timesteps
        t_start = 1.0 - noise_level  # If noise_level=0.5, start from t=0.5
        
        for i in range(len(timesteps) - 1):
            t = timesteps[i]
            if t < t_start:
                continue
            dt = timesteps[i + 1] - t
            x = self._integration_step(x, t, dt)
        
        return x
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def get_stats(self) -> Dict:
        """Get training statistics."""
        return {
            "config": self.config.__dict__,
            "n_velocity_fields": len(self.velocity_fields),
            "is_fitted": self._is_fitted,
            "fit_history": self._fit_history,
        }
    
    def save(self, path: str):
        """Save model to file."""
        import pickle
        
        state = {
            "config": self.config,
            "velocity_fields": {
                k: {
                    "W_random": vf.W_random if hasattr(vf, 'W_random') else None,
                    "W_fwd": vf.W_fwd if hasattr(vf, 'W_fwd') else None,
                    "W_bwd": vf.W_bwd if hasattr(vf, 'W_bwd') else None,
                    "W_out": vf.W_out,
                    "bias": vf.bias,
                }
                for k, vf in self.velocity_fields.items()
            },
            "train_timesteps": self.train_timesteps,
            "sample_timesteps": self.sample_timesteps,
            "is_fitted": self._is_fitted,
            "fit_history": self._fit_history,
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> "PILFlowMatching":
        """Load model from file."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        model = cls(state["config"])
        model.train_timesteps = state["train_timesteps"]
        model.sample_timesteps = state["sample_timesteps"]
        model._is_fitted = state["is_fitted"]
        model._fit_history = state["fit_history"]
        
        # Restore velocity field weights
        for k, vf_state in state["velocity_fields"].items():
            vf = model.velocity_fields[k]
            if vf_state.get("W_random") is not None:
                vf.W_random = vf_state["W_random"]
            if vf_state.get("W_fwd") is not None:
                vf.W_fwd = vf_state["W_fwd"]
            if vf_state.get("W_bwd") is not None:
                vf.W_bwd = vf_state["W_bwd"]
            vf.W_out = vf_state["W_out"]
            vf.bias = vf_state["bias"]
            vf._is_fitted = True
        
        return model
