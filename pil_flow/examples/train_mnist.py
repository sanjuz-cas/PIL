"""
PIL-Flow MNIST Example

Demonstrates training PIL-Flow on MNIST digits and generating new samples.
This showcases gradient-free generative modeling via pseudoinverse learning.

Usage:
    python train_mnist.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pil_flow import PILFlowMatching, PILFlowConfig


def load_mnist_subset(n_samples: int = 5000, digit: int = None):
    """
    Load MNIST data (simplified - uses random data for demo).
    
    In practice, replace with actual MNIST loading:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784')
        X = mnist.data / 255.0
    
    Args:
        n_samples: Number of samples to load
        digit: Specific digit to filter (None = all)
        
    Returns:
        X: Flattened images (N, 784)
    """
    print("Loading MNIST-like data...")
    
    # For demo: create synthetic "digit-like" data
    # Replace with actual MNIST in production
    rng = np.random.default_rng(42)
    
    # Simulate digit patterns (simplified)
    X = []
    for _ in range(n_samples):
        # Create a simple pattern
        img = np.zeros((28, 28))
        
        # Random stroke-like patterns
        cx, cy = rng.integers(8, 20, 2)
        for _ in range(rng.integers(3, 8)):
            dx, dy = rng.integers(-5, 6, 2)
            for t in np.linspace(0, 1, 20):
                x = int(cx + t * dx)
                y = int(cy + t * dy)
                if 0 <= x < 28 and 0 <= y < 28:
                    img[y, x] = 1.0
                    # Add some thickness
                    for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + ox, y + oy
                        if 0 <= nx < 28 and 0 <= ny < 28:
                            img[ny, nx] = max(img[ny, nx], 0.5)
        
        X.append(img.flatten())
    
    X = np.array(X, dtype=np.float32)
    
    # Normalize to [-1, 1] (common for flow matching)
    X = 2 * X - 1
    
    print(f"Loaded {len(X)} samples, shape: {X.shape}")
    return X


def visualize_samples(
    samples: np.ndarray,
    title: str = "Generated Samples",
    n_show: int = 16,
    save_path: str = None,
):
    """Visualize generated samples as a grid."""
    n_show = min(n_show, len(samples))
    n_cols = int(np.ceil(np.sqrt(n_show)))
    n_rows = int(np.ceil(n_show / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    axes = np.atleast_2d(axes)
    
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            ax = axes[i, j]
            
            if idx < n_show:
                # Reshape to 28x28 and denormalize
                img = samples[idx].reshape(28, 28)
                img = (img + 1) / 2  # [-1, 1] -> [0, 1]
                img = np.clip(img, 0, 1)
                
                ax.imshow(img, cmap='gray')
            
            ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_trajectory(trajectory: np.ndarray, sample_idx: int = 0):
    """Visualize generation trajectory for a single sample."""
    n_steps = len(trajectory)
    n_show = min(10, n_steps)
    step_indices = np.linspace(0, n_steps - 1, n_show, dtype=int)
    
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 1.5, 2))
    
    for i, step_idx in enumerate(step_indices):
        img = trajectory[step_idx, sample_idx].reshape(28, 28)
        img = (img + 1) / 2
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"t={step_idx/(n_steps-1):.2f}")
        axes[i].axis('off')
    
    plt.suptitle("Generation Trajectory: Noise → Sample")
    plt.tight_layout()
    plt.show()


def main():
    print("=" * 60)
    print("PIL-Flow: Gradient-Free Flow Matching on MNIST")
    print("=" * 60)
    
    # Configuration
    config = PILFlowConfig(
        input_dim=784,  # 28x28 flattened
        hidden_dim=512,
        activation="gelu",
        use_bipil=True,
        bipil_fusion="concat",
        n_train_timesteps=50,  # Number of timesteps for training
        n_sample_timesteps=25,  # Number of steps for generation
        time_schedule="linear",
        lambda_reg=1e-4,
        use_time_embedding=True,
        time_embed_dim=64,
        per_timestep_fit=True,  # Separate network per timestep
        ode_solver="euler",
        seed=42,
    )
    
    print("\nConfiguration:")
    print(f"  Input dim: {config.input_dim}")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  BiPIL: {config.use_bipil} (fusion: {config.bipil_fusion})")
    print(f"  Train timesteps: {config.n_train_timesteps}")
    print(f"  Sample timesteps: {config.n_sample_timesteps}")
    print(f"  λ (regularization): {config.lambda_reg}")
    
    # Load data
    X_train = load_mnist_subset(n_samples=3000)
    
    # Create model
    print("\nInitializing PIL-Flow model...")
    model = PILFlowMatching(config)
    
    # Train (one-shot per timestep!)
    print("\nTraining (gradient-free!)...")
    print("-" * 60)
    
    start_time = time.time()
    train_stats = model.fit(X_train, verbose=True)
    train_time = time.time() - start_time
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"This is ONE-SHOT learning - no gradient descent iterations!")
    
    # Generate samples
    print("\nGenerating samples...")
    start_time = time.time()
    
    samples = model.sample(n_samples=25, seed=123)
    
    gen_time = time.time() - start_time
    print(f"Generated 25 samples in {gen_time:.2f} seconds")
    
    # Visualize
    visualize_samples(
        samples,
        title="PIL-Flow Generated Samples (Gradient-Free!)",
        n_show=25,
    )
    
    # Generate with trajectory
    print("\nGenerating with trajectory visualization...")
    trajectory = model.sample(n_samples=4, return_trajectory=True, seed=456)
    print(f"Trajectory shape: {trajectory.shape}")
    
    visualize_trajectory(trajectory, sample_idx=0)
    
    # Save model
    save_path = Path(__file__).parent / "pil_flow_mnist.pkl"
    model.save(str(save_path))
    print(f"\nModel saved to {save_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Training time: {train_time:.2f}s")
    print(f"Average MSE: {train_stats['avg_mse']:.6f}")
    print(f"Method: PIL (Pseudoinverse Learning) - NO BACKPROPAGATION")
    print("=" * 60)


if __name__ == "__main__":
    main()
