"""
Quick test script for PIL-Flow
"""
import sys
sys.path.insert(0, ".")

from pil_flow import PILFlowMatching, PILFlowConfig
import numpy as np

print("=" * 60)
print("PIL-Flow Quick Test")
print("=" * 60)

# Create config
config = PILFlowConfig(
    input_dim=64,
    hidden_dim=128,
    n_train_timesteps=10,
    n_sample_timesteps=5,
    use_bipil=True,
)

print(f"\nConfig: input_dim={config.input_dim}, hidden_dim={config.hidden_dim}")
print(f"BiPIL: {config.use_bipil}, timesteps: {config.n_train_timesteps}")

# Create model
model = PILFlowMatching(config)
print("\nModel created successfully!")

# Create synthetic data
X = np.random.randn(100, 64).astype(np.float32)
print(f"Training data shape: {X.shape}")

# Train
print("\nTraining (gradient-free!)...")
stats = model.fit(X, verbose=True)

print(f"\nTraining complete!")
print(f"Average MSE: {stats['avg_mse']:.6f}")
print(f"Time: {stats['elapsed_time']:.2f}s")

# Generate samples
print("\nGenerating samples...")
samples = model.sample(n_samples=5)
print(f"Generated samples shape: {samples.shape}")

print("\n" + "=" * 60)
print("SUCCESS! PIL-Flow is working correctly.")
print("=" * 60)
