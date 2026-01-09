"""
Bi-PIL Layer: Bidirectional Pseudoinverse Learning Layer.

This module implements the core FFN replacement for the Attention-PIL Hybrid architecture.
Training uses algebraic weight solving via pseudoinverse, NOT backpropagation.

Reference:
- "Bi-PIL: Bidirectional Gradient-Free Learning Scheme"
- "SONG: Synergetic Learning System Based on Swarm of Non-Gradient Learners"
- "HRA: Householder Reflection Adaptation" (NeurIPS 2024 Spotlight)

HRA Integration:
    We integrate Householder Reflection Adaptation which constructs orthogonal 
    transformation matrices via chains of learnable Householder reflections.
    Formula: W_orth = I - 2 * U @ U^T where U is orthonormalized via Gram-Schmidt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal, List
import structlog

from app.core.pil_utils import (
    ridge_solve,
    low_rank_ridge_solve,
    orthogonal_init,
    NumericalMonitor,
    condition_number,
)

logger = structlog.get_logger(__name__)


# =============================================================================
# Householder Reflection Adaptation (HRA) - NeurIPS 2024 Spotlight
# =============================================================================

class HouseholderReflection(nn.Module):
    """
    Householder Reflection Transformation.
    
    Constructs an orthogonal matrix via chain of Householder reflections:
    Q = (I - 2*u_1*u_1^T) @ (I - 2*u_2*u_2^T) @ ... @ (I - 2*u_r*u_r^T)
    
    Or equivalently with Gram-Schmidt orthogonalization:
    Q = I - 2 * U @ U^T where U is orthonormalized
    
    Reference: "Bridging The Gap between Low-rank and Orthogonal Adaptation 
               via Householder Reflection Adaptation" (NeurIPS 2024)
    """
    
    def __init__(
        self,
        dim: int,
        r: int = 8,
        apply_gs: bool = True,
        eps: float = 1e-6,
    ):
        """
        Initialize Householder Reflection.
        
        Args:
            dim: Input/output dimension
            r: Number of Householder reflections (rank). Should be even.
            apply_gs: Apply Gram-Schmidt orthogonalization for stability
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        self.dim = dim
        self.r = r
        self.apply_gs = apply_gs
        self.eps = eps
        
        # Initialize Householder vectors
        # Each vector is normalized and used to construct H_i = I - 2*u*u^T
        # Use orthogonal initialization for better conditioning
        hra_u = torch.zeros(dim, r)
        nn.init.orthogonal_(hra_u)  # Orthogonal columns for diversity
        
        self.hra_u = nn.Parameter(hra_u, requires_grad=True)
    
    def get_orthogonal_matrix(self) -> torch.Tensor:
        """
        Compute orthogonal transformation matrix via Householder reflections.
        
        Each Householder reflection H_i = I - 2*u_i*u_i^T is orthogonal.
        The product Q = H_1 @ H_2 @ ... @ H_r is also orthogonal.
        
        Returns:
            Orthogonal matrix (dim, dim)
        """
        device = self.hra_u.device
        dtype = self.hra_u.dtype
        
        Q = torch.eye(self.dim, device=device, dtype=dtype)
        
        if self.apply_gs:
            # Gram-Schmidt orthogonalization of Householder vectors first
            # Then chain the reflections
            U_list = []
            U_list.append((self.hra_u[:, 0] / (self.hra_u[:, 0].norm() + self.eps)).view(-1, 1))
            
            for i in range(1, self.r):
                ui = self.hra_u[:, i].view(-1, 1)
                # Orthogonalize against previous vectors
                for j in range(i):
                    ui = ui - (U_list[j].t() @ ui) * U_list[j]
                U_list.append((ui / (ui.norm() + self.eps)).view(-1, 1))
            
            # Chain orthogonal Householder reflections: Q = H_1 @ H_2 @ ... @ H_r
            for ui in U_list:
                # H_i = I - 2 * u_i @ u_i^T
                Q = Q @ (torch.eye(self.dim, device=device, dtype=dtype) - 2 * ui @ ui.t())
        else:
            # Chain of individual Householder reflections without GS
            hra_u_norm = self.hra_u / (self.hra_u.norm(dim=0, keepdim=True) + self.eps)
            
            for i in range(self.r):
                ui = hra_u_norm[:, i].view(-1, 1)
                # H_i = I - 2 * u_i @ u_i^T
                Q = Q @ (torch.eye(self.dim, device=device, dtype=dtype) - 2 * ui @ ui.t())
        
        return Q
    
    def forward(self, W: torch.Tensor) -> torch.Tensor:
        """
        Apply orthogonal transformation to weight matrix.
        
        Args:
            W: Weight matrix (out_features, in_features)
            
        Returns:
            Transformed weight matrix W @ Q
        """
        Q = self.get_orthogonal_matrix()
        return W @ Q


class HRALinear(nn.Module):
    """
    HRA-Enhanced Linear Layer.
    
    Applies Householder Reflection Adaptation to a linear transformation.
    The base weight is frozen, only the HRA parameters are learnable.
    
    Forward: y = x @ (W @ Q)^T + bias
    where Q is the orthogonal matrix from Householder reflections.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        r: int = 8,
        apply_gs: bool = True,
        freeze_base: bool = True,
    ):
        """
        Initialize HRA Linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension  
            bias: Include bias term
            r: Number of Householder reflections
            apply_gs: Apply Gram-Schmidt orthogonalization
            freeze_base: Freeze base weight matrix
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Base linear layer (frozen)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features),
            requires_grad=not freeze_base
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # HRA transformation (learnable)
        self.hra = HouseholderReflection(in_features, r=r, apply_gs=apply_gs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with HRA-transformed weights.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # Get orthogonal transformation
        Q = self.hra.get_orthogonal_matrix()
        
        # Apply transformed weight: W_new = W @ Q
        new_weight = self.weight @ Q
        
        return F.linear(x, new_weight, self.bias)
    
    def get_effective_weight(self) -> torch.Tensor:
        """Get the effective weight matrix after HRA transformation."""
        Q = self.hra.get_orthogonal_matrix()
        return self.weight @ Q


class HRAInjectedLinear(nn.Module):
    """
    HRA Injection wrapper for existing Linear layers.
    
    Wraps an existing nn.Linear and adds HRA adaptation.
    The original weights are frozen, only HRA parameters are trained.
    
    Reference: DaShenZi721/HRA generation/control/hra.py
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        apply_gs: bool = True,
    ):
        """
        Initialize HRA Injection.
        
        Args:
            base_layer: Existing nn.Linear layer to wrap
            r: Number of Householder reflections
            apply_gs: Apply Gram-Schmidt orthogonalization
        """
        super().__init__()
        
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Store frozen base layer
        self.fixed_linear = base_layer
        self.fixed_linear.weight.requires_grad = False
        if self.fixed_linear.bias is not None:
            self.fixed_linear.bias.requires_grad = False
        
        # HRA adaptation
        self.hra = HouseholderReflection(
            self.in_features, r=r, apply_gs=apply_gs
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with HRA transformation."""
        Q = self.hra.get_orthogonal_matrix()
        new_weight = self.fixed_linear.weight @ Q
        return F.linear(x, new_weight, self.fixed_linear.bias)


def inject_hra_into_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    r: int = 8,
    apply_gs: bool = True,
    verbose: bool = False,
) -> Tuple[List[nn.Parameter], List[str]]:
    """
    Inject HRA adapters into a model's Linear layers.
    
    Args:
        model: PyTorch model to inject HRA into
        target_modules: List of module names to target (None = all Linear)
        r: Number of Householder reflections
        apply_gs: Apply Gram-Schmidt
        verbose: Print injection info
        
    Returns:
        Tuple of (trainable_params, injected_names)
    """
    trainable_params = []
    injected_names = []
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            if target_modules is None or any(t in name for t in target_modules):
                # Get parent module and attribute name
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    attr_name = name
                
                # Create HRA-injected version
                hra_module = HRAInjectedLinear(module, r=r, apply_gs=apply_gs)
                setattr(parent, attr_name, hra_module)
                
                trainable_params.extend(hra_module.hra.parameters())
                injected_names.append(name)
                
                if verbose:
                    logger.info(f"HRA injected into: {name} (r={r})")
    
    return trainable_params, injected_names


# =============================================================================
# HRA-Enhanced PIL Layer
# =============================================================================

class HRAPilLayer(nn.Module):
    """
    Householder Reflection Adapted PIL Layer.
    
    Combines PIL (Pseudoinverse Learning) with HRA (Householder Reflection Adaptation):
    - Random expansion weights are transformed via Householder reflections
    - Output weights are still solved via pseudoinverse (not backprop)
    
    This creates learnable orthogonal transformations in the expansion space
    while maintaining the one-shot PIL solving for output weights.
    
    Forward: y = σ(x @ (W_random @ Q)) @ W_out
    where Q is HRA orthogonal matrix, W_out is solved via pseudoinverse
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hra_r: int = 8,
        apply_gs: bool = True,
        activation: str = "gelu",
        reg_lambda: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """
        Initialize HRA-PIL Layer.
        
        Args:
            input_dim: Input dimension
            hidden_dim: Hidden expansion dimension
            output_dim: Output dimension
            hra_r: Number of Householder reflections for input transformation
            apply_gs: Apply Gram-Schmidt in HRA
            activation: Activation function
            reg_lambda: Ridge regularization for pseudoinverse
            seed: Random seed
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reg_lambda = reg_lambda
        
        # Fixed random expansion weights
        self.register_buffer(
            "W_random", orthogonal_init((input_dim, hidden_dim), seed=seed)
        )
        
        # HRA transformation on input space (learnable orthogonal adaptation)
        self.input_hra = HouseholderReflection(input_dim, r=hra_r, apply_gs=apply_gs)
        
        # Output weights (solved via pseudoinverse, NOT backprop)
        self.register_buffer("W_out", torch.zeros(hidden_dim, output_dim))
        self.register_buffer("bias", torch.zeros(output_dim))
        
        # Activation
        self.activation = self._get_activation(activation)
        
        self._is_fitted = False
        self.monitor = NumericalMonitor()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.GELU())
    
    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        """Feature expansion with HRA-transformed weights."""
        Q = self.input_hra.get_orthogonal_matrix()
        W_transformed = Q @ self.W_random  # Apply HRA to random weights
        return self.activation(x @ W_transformed)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: y = σ(x @ (Q @ W_random)) @ W_out + bias"""
        h = self._expand(x)
        return h @ self.W_out + self.bias
    
    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        use_low_rank: bool = False,
        rank: Optional[int] = None,
    ) -> dict:
        """
        Solve for W_out using pseudoinverse with HRA-transformed features.
        
        W_out = (H^T H + λI)^{-1} H^T Y
        where H = σ(X @ (Q @ W_random))
        """
        if x.dim() > 2:
            x = x.reshape(-1, x.shape[-1])
            target = target.reshape(-1, target.shape[-1])
        
        # HRA-transformed expansion
        H = self._expand(x)
        
        if not self.monitor.check_matrix(H, "hra_hidden"):
            return {"success": False, "reason": "numerical_instability"}
        
        N = H.shape[0]
        
        if use_low_rank or N > 10000:
            W_new = low_rank_ridge_solve(H, target, self.reg_lambda, rank)
            method = "low_rank_svd"
        else:
            W_new = ridge_solve(H, target, self.reg_lambda)
            method = "ridge_solve"
        
        if not self.monitor.check_matrix(W_new, "hra_W_out"):
            return {"success": False, "reason": "unstable_weights"}
        
        self.W_out.copy_(W_new)
        residual = target - (H @ self.W_out)
        self.bias.copy_(residual.mean(dim=0))
        
        self._is_fitted = True
        
        y_pred = H @ self.W_out + self.bias
        mse = ((target - y_pred) ** 2).mean().item()
        
        return {
            "success": True,
            "method": method,
            "n_samples": N,
            "mse": mse,
        }
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# =============================================================================
# Original PIL Implementation (unchanged core)
# =============================================================================


class PILLayer(nn.Module):
    """
    Single-direction Pseudoinverse Learning Layer.

    Implements: y = activation(x @ W_random) @ W_out

    W_random: Fixed random orthogonal expansion (requires_grad=False)
    W_out: Solved via pseudoinverse (requires_grad=False)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "gelu",
        reg_lambda: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """
        Initialize PIL Layer.

        Args:
            input_dim: Input dimension (D)
            hidden_dim: Hidden expansion dimension (H) - typically 4x input_dim
            output_dim: Output dimension (D_out)
            activation: Activation function ("gelu", "relu", "leaky_relu", "silu")
            reg_lambda: Ridge regularization parameter (λ)
            seed: Random seed for reproducibility
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.reg_lambda = reg_lambda

        # Fixed random expansion weights (NOT trained via backprop)
        self.register_buffer(
            "W_random", orthogonal_init((input_dim, hidden_dim), seed=seed)
        )

        # Output weights (solved via pseudoinverse, NOT trained via backprop)
        # Using register_buffer instead of nn.Parameter since we don't want gradients
        self.register_buffer("W_out", torch.zeros(hidden_dim, output_dim))

        # Bias (optional, also solved)
        self.register_buffer("bias", torch.zeros(output_dim))

        # Activation function
        self.activation = self._get_activation(activation)

        # State tracking
        self._is_fitted = False
        self.monitor = NumericalMonitor()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }
        return activations.get(name, nn.GELU())

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feature expansion: H = activation(X @ W_random)

        Args:
            x: Input tensor (..., input_dim)

        Returns:
            Hidden activations (..., hidden_dim)
        """
        return self.activation(x @ self.W_random)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = H @ W_out + bias

        Args:
            x: Input tensor (B, S, D) or (N, D)

        Returns:
            Output tensor (B, S, D_out) or (N, D_out)
        """
        h = self._expand(x)
        return h @ self.W_out + self.bias

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        use_low_rank: bool = False,
        rank: Optional[int] = None,
    ) -> dict:
        """
        Solve for W_out using pseudoinverse.

        W_out = (H^T H + λI)^{-1} H^T Y

        Args:
            x: Input tensor (N, D) - flattened batch
            target: Target tensor (N, D_out)
            use_low_rank: Use low-rank SVD approximation for large N
            rank: Rank for low-rank approximation

        Returns:
            Dictionary with fit statistics
        """
        # Ensure 2D tensors
        if x.dim() > 2:
            orig_shape = x.shape
            x = x.reshape(-1, x.shape[-1])
            target = target.reshape(-1, target.shape[-1])

        # Feature expansion
        H = self._expand(x)  # (N, hidden_dim)

        # Check numerical stability
        if not self.monitor.check_matrix(H, "hidden_activations"):
            logger.error("fit_aborted_numerical_instability")
            return {"success": False, "reason": "numerical_instability"}

        # Solve for weights
        N = H.shape[0]

        if use_low_rank or N > 10000:
            W_new = low_rank_ridge_solve(H, target, self.reg_lambda, rank)
            method = "low_rank_svd"
        else:
            W_new = ridge_solve(H, target, self.reg_lambda)
            method = "ridge_solve"

        # Check solution stability
        if not self.monitor.check_matrix(W_new, "W_out"):
            logger.error("fit_produced_unstable_weights")
            return {"success": False, "reason": "unstable_weights"}

        # Update weights in-place
        self.W_out.copy_(W_new)

        # Compute bias as mean residual (optional)
        residual = target - (H @ self.W_out)
        self.bias.copy_(residual.mean(dim=0))

        self._is_fitted = True

        # Compute fit statistics
        y_pred = H @ self.W_out + self.bias
        mse = ((target - y_pred) ** 2).mean().item()

        logger.debug(
            "pil_layer_fitted",
            n_samples=N,
            method=method,
            mse=mse,
        )

        return {
            "success": True,
            "method": method,
            "n_samples": N,
            "mse": mse,
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class BiPILLayer(nn.Module):
    """
    Bidirectional Pseudoinverse Learning Layer (Bi-PIL).

    Implements two parallel expansion flows:
    - Forward: H_fwd = σ(X @ W_fwd)
    - Backward: H_bwd = σ(X @ W_bwd)
    - Fusion: H = concat(H_fwd, H_bwd) or H = H_fwd + H_bwd

    This replaces the standard FFN in Transformer blocks.
    """

    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        activation: str = "gelu",
        fusion: Literal["concat", "add", "gate"] = "concat",
        reg_lambda: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """
        Initialize Bi-PIL Layer.

        Args:
            dim: Model dimension (D)
            expansion_factor: Hidden dim = dim * expansion_factor
            activation: Activation function
            fusion: How to combine forward/backward features
            reg_lambda: Ridge regularization parameter
            seed: Random seed
        """
        super().__init__()

        self.dim = dim
        self.hidden_dim = dim * expansion_factor
        self.fusion = fusion
        self.reg_lambda = reg_lambda

        # Determine output hidden dim based on fusion
        if fusion == "concat":
            self.fused_dim = self.hidden_dim * 2
        else:
            self.fused_dim = self.hidden_dim

        # Forward expansion: W_fwd is FIXED (requires_grad=False)
        self.register_buffer(
            "W_fwd", orthogonal_init((dim, self.hidden_dim), seed=seed)
        )

        # Backward expansion: W_bwd is FIXED (requires_grad=False)
        self.register_buffer(
            "W_bwd",
            orthogonal_init((dim, self.hidden_dim), seed=seed + 1 if seed else None),
        )

        # Output projection: W_out is SOLVED via pseudoinverse
        self.register_buffer("W_out", torch.zeros(self.fused_dim, dim))

        # Optional gating for fusion
        if fusion == "gate":
            self.register_buffer("gate_weights", torch.ones(self.hidden_dim) * 0.5)

        # Bias
        self.register_buffer("bias", torch.zeros(dim))

        # Activation
        self.activation = self._get_activation(activation)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(dim)

        # State
        self._is_fitted = False
        self.monitor = NumericalMonitor()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.GELU())

    def _expand_bidirectional(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional feature expansion.

        Args:
            x: Input (..., dim)

        Returns:
            Fused hidden features (..., fused_dim)
        """
        # Forward expansion: H_fwd = σ(X @ W_fwd)
        H_fwd = self.activation(x @ self.W_fwd)

        # Backward expansion: H_bwd = σ(X @ W_bwd)
        H_bwd = self.activation(x @ self.W_bwd)

        # Fusion
        if self.fusion == "concat":
            return torch.cat([H_fwd, H_bwd], dim=-1)
        elif self.fusion == "add":
            return H_fwd + H_bwd
        elif self.fusion == "gate":
            # Learned gating (but gate_weights are also solved, not gradient-trained)
            gate = torch.sigmoid(self.gate_weights)
            return gate * H_fwd + (1 - gate) * H_bwd
        else:
            return H_fwd + H_bwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor (B, S, D)

        Returns:
            Output tensor (B, S, D)
        """
        # Store residual
        residual = x

        # Bidirectional expansion
        H = self._expand_bidirectional(x)

        # Output projection
        out = H @ self.W_out + self.bias

        # Residual connection + LayerNorm
        out = self.layer_norm(out + residual)

        return out

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_low_rank: bool = False,
        rank: Optional[int] = None,
    ) -> dict:
        """
        Solve for W_out using pseudoinverse.

        If target is None, uses residual learning: target = x (identity mapping).

        Args:
            x: Input tensor (B, S, D) or (N, D)
            target: Target tensor (same shape as x). If None, learns identity.
            use_low_rank: Use low-rank approximation
            rank: Rank for approximation

        Returns:
            Fit statistics dictionary
        """
        # Flatten to 2D
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)

        # Default target is identity (residual learning)
        if target is None:
            target_flat = x_flat.clone()
        else:
            target_flat = target.reshape(-1, self.dim)

        # Bidirectional expansion
        H = self._expand_bidirectional(x_flat)  # (N, fused_dim)

        # Check numerical stability
        if not self.monitor.check_matrix(H, "bi_hidden"):
            return {"success": False, "reason": "numerical_instability"}

        N = H.shape[0]

        # Solve ridge regression: W_out = (H^T H + λI)^{-1} H^T Y
        if use_low_rank or N > 10000:
            W_new = low_rank_ridge_solve(H, target_flat, self.reg_lambda, rank)
            method = "low_rank_svd"
        else:
            W_new = ridge_solve(H, target_flat, self.reg_lambda)
            method = "ridge_solve"

        # Check solution
        if not self.monitor.check_matrix(W_new, "bi_W_out"):
            return {"success": False, "reason": "unstable_weights"}

        # Update weights
        self.W_out.copy_(W_new)

        # Update bias
        residual = target_flat - (H @ self.W_out)
        self.bias.copy_(residual.mean(dim=0))

        self._is_fitted = True

        # Statistics
        y_pred = H @ self.W_out + self.bias
        mse = ((target_flat - y_pred) ** 2).mean().item()

        cond = condition_number(H.T @ H).item()

        logger.debug(
            "bipil_layer_fitted",
            n_samples=N,
            method=method,
            mse=mse,
            condition_number=cond,
        )

        return {
            "success": True,
            "method": method,
            "n_samples": N,
            "mse": mse,
            "condition_number": cond,
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_effective_rank(self) -> int:
        """Compute effective rank of the learned transformation."""
        if not self._is_fitted:
            return 0
        s = torch.linalg.svdvals(self.W_out)
        threshold = s.max() * 1e-5
        return (s > threshold).sum().item()


class SwarmPIL(nn.Module):
    """
    Swarm of Non-Gradient Learners (SONG Implementation).

    Uses a ModuleList of small PIL learners and averages their outputs
    for improved robustness and diversity.

    Reference: "SONG: Synergetic Learning System Based on Swarm of Non-Gradient Learners"
    """

    def __init__(
        self,
        dim: int,
        n_learners: int = 4,
        expansion_factor: int = 2,
        reg_lambda: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """
        Initialize Swarm PIL.

        Args:
            dim: Model dimension
            n_learners: Number of parallel learners in the swarm
            expansion_factor: Expansion factor per learner (smaller than BiPIL)
            reg_lambda: Regularization parameter
            seed: Random seed
        """
        super().__init__()

        self.dim = dim
        self.n_learners = n_learners

        # Create swarm of small PIL learners
        self.learners = nn.ModuleList(
            [
                PILLayer(
                    input_dim=dim,
                    hidden_dim=dim * expansion_factor,
                    output_dim=dim,
                    reg_lambda=reg_lambda,
                    seed=seed + i if seed else None,
                )
                for i in range(n_learners)
            ]
        )

        # Optional learner weights (for weighted averaging)
        self.register_buffer("learner_weights", torch.ones(n_learners) / n_learners)

        # Layer norm
        self.layer_norm = nn.LayerNorm(dim)

        self._is_fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: weighted average of all learners.

        Args:
            x: Input (B, S, D)

        Returns:
            Output (B, S, D)
        """
        residual = x

        # Collect outputs from all learners
        outputs = []
        for i, learner in enumerate(self.learners):
            out = learner(x)
            outputs.append(self.learner_weights[i] * out)

        # Average
        combined = sum(outputs)

        # Residual + LayerNorm
        return self.layer_norm(combined + residual)

    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """
        Fit all learners in the swarm.

        Args:
            x: Input tensor
            target: Target tensor

        Returns:
            Combined fit statistics
        """
        if target is None:
            target = x

        results = []
        for i, learner in enumerate(self.learners):
            # Each learner gets the same input/target
            result = learner.fit(
                x.reshape(-1, self.dim), target.reshape(-1, self.dim), **kwargs
            )
            results.append(result)

        self._is_fitted = all(r.get("success", False) for r in results)

        # Aggregate statistics
        mses = [r.get("mse", float("inf")) for r in results]

        return {
            "success": self._is_fitted,
            "n_learners": self.n_learners,
            "mean_mse": sum(mses) / len(mses),
            "min_mse": min(mses),
            "max_mse": max(mses),
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# =============================================================================
# HRA-Enhanced Bidirectional PIL (HRA-BiPIL)
# =============================================================================

class HRABiPILLayer(nn.Module):
    """
    Householder Reflection Adapted Bidirectional PIL Layer.
    
    Extends BiPIL with HRA orthogonal transformations for enhanced 
    representational capacity while maintaining gradient-free training.
    
    Architecture:
    - Forward flow: H_fwd = σ(X @ Q_fwd @ W_fwd)
    - Backward flow: H_bwd = σ(X @ Q_bwd @ W_bwd)
    - Fusion + Output: y = [H_fwd; H_bwd] @ W_out (solved via pseudoinverse)
    
    HRA adds learnable orthogonal transformations (Q_fwd, Q_bwd) that can be
    trained via backprop while W_out is still solved algebraically.
    """
    
    def __init__(
        self,
        dim: int,
        expansion_factor: int = 4,
        hra_r: int = 8,
        apply_gs: bool = True,
        activation: str = "gelu",
        fusion: Literal["concat", "add", "gate"] = "concat",
        reg_lambda: float = 1e-5,
        seed: Optional[int] = None,
    ):
        """
        Initialize HRA-BiPIL Layer.
        
        Args:
            dim: Model dimension
            expansion_factor: Hidden dim multiplier
            hra_r: Number of Householder reflections per direction
            apply_gs: Apply Gram-Schmidt orthogonalization
            activation: Activation function
            fusion: Feature fusion method
            reg_lambda: Ridge regularization
            seed: Random seed
        """
        super().__init__()
        
        self.dim = dim
        self.hidden_dim = dim * expansion_factor
        self.fusion = fusion
        self.reg_lambda = reg_lambda
        
        if fusion == "concat":
            self.fused_dim = self.hidden_dim * 2
        else:
            self.fused_dim = self.hidden_dim
        
        # Fixed random expansion weights
        self.register_buffer(
            "W_fwd", orthogonal_init((dim, self.hidden_dim), seed=seed)
        )
        self.register_buffer(
            "W_bwd", orthogonal_init((dim, self.hidden_dim), seed=seed + 1 if seed else None)
        )
        
        # HRA transformations (learnable via backprop)
        self.hra_fwd = HouseholderReflection(dim, r=hra_r, apply_gs=apply_gs)
        self.hra_bwd = HouseholderReflection(dim, r=hra_r, apply_gs=apply_gs)
        
        # Output weights (solved via pseudoinverse)
        self.register_buffer("W_out", torch.zeros(self.fused_dim, dim))
        self.register_buffer("bias", torch.zeros(dim))
        
        # Optional gating
        if fusion == "gate":
            self.gate = nn.Parameter(torch.ones(self.hidden_dim) * 0.5)
        
        self.activation = self._get_activation(activation)
        self.layer_norm = nn.LayerNorm(dim)
        
        self._is_fitted = False
        self.monitor = NumericalMonitor()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            "gelu": nn.GELU(),
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.GELU())
    
    def _expand_bidirectional(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional expansion with HRA transformations."""
        # Get orthogonal transformations
        Q_fwd = self.hra_fwd.get_orthogonal_matrix()
        Q_bwd = self.hra_bwd.get_orthogonal_matrix()
        
        # Forward: H_fwd = σ(X @ Q_fwd @ W_fwd)
        H_fwd = self.activation(x @ Q_fwd @ self.W_fwd)
        
        # Backward: H_bwd = σ(X @ Q_bwd @ W_bwd)
        H_bwd = self.activation(x @ Q_bwd @ self.W_bwd)
        
        # Fusion
        if self.fusion == "concat":
            return torch.cat([H_fwd, H_bwd], dim=-1)
        elif self.fusion == "add":
            return H_fwd + H_bwd
        elif self.fusion == "gate":
            gate = torch.sigmoid(self.gate)
            return gate * H_fwd + (1 - gate) * H_bwd
        else:
            return H_fwd + H_bwd
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with HRA-enhanced bidirectional expansion."""
        residual = x
        H = self._expand_bidirectional(x)
        out = H @ self.W_out + self.bias
        return self.layer_norm(out + residual)
    
    @torch.no_grad()
    def fit(
        self,
        x: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        use_low_rank: bool = False,
        rank: Optional[int] = None,
    ) -> dict:
        """Solve for W_out using pseudoinverse with HRA-transformed features."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.dim)
        
        if target is None:
            target_flat = x_flat.clone()
        else:
            target_flat = target.reshape(-1, self.dim)
        
        H = self._expand_bidirectional(x_flat)
        
        if not self.monitor.check_matrix(H, "hra_bi_hidden"):
            return {"success": False, "reason": "numerical_instability"}
        
        N = H.shape[0]
        
        if use_low_rank or N > 10000:
            W_new = low_rank_ridge_solve(H, target_flat, self.reg_lambda, rank)
            method = "low_rank_svd"
        else:
            W_new = ridge_solve(H, target_flat, self.reg_lambda)
            method = "ridge_solve"
        
        if not self.monitor.check_matrix(W_new, "hra_bi_W_out"):
            return {"success": False, "reason": "unstable_weights"}
        
        self.W_out.copy_(W_new)
        residual = target_flat - (H @ self.W_out)
        self.bias.copy_(residual.mean(dim=0))
        
        self._is_fitted = True
        
        y_pred = H @ self.W_out + self.bias
        mse = ((target_flat - y_pred) ** 2).mean().item()
        cond = condition_number(H.T @ H).item()
        
        return {
            "success": True,
            "method": method,
            "n_samples": N,
            "mse": mse,
            "condition_number": cond,
        }
    
    def get_hra_params(self) -> List[nn.Parameter]:
        """Get HRA parameters for optimizer (if hybrid training)."""
        return list(self.hra_fwd.parameters()) + list(self.hra_bwd.parameters())
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


# =============================================================================
# Orthogonality Regularization Loss (from HRA paper)
# =============================================================================

def hra_orthogonality_loss(hra_module: HouseholderReflection) -> torch.Tensor:
    """
    Compute orthogonality regularization loss for HRA.
    
    Ensures the Householder vectors remain well-conditioned.
    Loss = ||I - U_norm^T @ U_norm||_F
    
    Args:
        hra_module: HouseholderReflection module
        
    Returns:
        Orthogonality loss scalar
    """
    U = hra_module.hra_u
    U_norm = U / (U.norm(dim=0, keepdim=True) + hra_module.eps)
    
    # Gram matrix should be close to identity
    gram = U_norm.t() @ U_norm
    identity = torch.eye(hra_module.r, device=U.device, dtype=U.dtype)
    
    return torch.norm(identity - gram, p='fro')


def compute_model_hra_loss(model: nn.Module, weight: float = 1e-4) -> torch.Tensor:
    """
    Compute total HRA orthogonality loss for all HRA modules in a model.
    
    Args:
        model: Model containing HRA modules
        weight: Loss weight
        
    Returns:
        Weighted orthogonality loss
    """
    total_loss = torch.tensor(0.0)
    
    for module in model.modules():
        if isinstance(module, HouseholderReflection):
            total_loss = total_loss + hra_orthogonality_loss(module)
    
    return weight * total_loss