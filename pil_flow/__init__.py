"""
PIL-Flow: Gradient-Free Flow Matching via Pseudoinverse Learning

A novel framework for training continuous normalizing flows without backpropagation.
Instead of gradient descent, we use closed-form pseudoinverse solutions to learn
the velocity field that transports noise to data.

Key Components:
- PILFlowMatching: Core flow matching model with PIL training
- PILVelocityField: Velocity field network solved via pseudoinverse
- PILFlowConfig: Configuration dataclass for hyperparameters

Reference:
- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- "Bi-PIL: Bidirectional Gradient-Free Learning Scheme"
- "SONG: Synergetic Learning System Based on Swarm of Non-Gradient Learners"

Author: Project Emergent-1
"""

from .model import PILFlowMatching, PILFlowConfig
from .velocity_field import PILVelocityField, BiPILVelocityField
from .solver import ridge_solve, low_rank_ridge_solve, safe_inverse
from .utils import (
    orthogonal_init,
    cosine_time_schedule,
    linear_time_schedule,
    compute_condition_number,
)

__version__ = "0.1.0"
__all__ = [
    "PILFlowMatching",
    "PILFlowConfig",
    "PILVelocityField",
    "BiPILVelocityField",
    "ridge_solve",
    "low_rank_ridge_solve",
    "safe_inverse",
    "orthogonal_init",
    "cosine_time_schedule",
    "linear_time_schedule",
    "compute_condition_number",
]
