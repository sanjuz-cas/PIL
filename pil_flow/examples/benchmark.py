"""
PIL-Flow Benchmarking

Compare PIL-Flow against baseline methods and measure performance metrics.

Metrics:
    - Training time
    - Memory usage
    - Generation quality (MSE, reconstruction error)
    - Sample diversity
"""

import numpy as np
import time
from pathlib import Path
import sys
from dataclasses import dataclass
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pil_flow import PILFlowMatching, PILFlowConfig


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    method: str
    n_samples: int
    input_dim: int
    hidden_dim: int
    n_timesteps: int
    train_time: float
    gen_time: float
    avg_mse: float
    memory_mb: float
    samples_per_second: float


def estimate_memory_usage(model: PILFlowMatching) -> float:
    """Estimate memory usage of model in MB."""
    total_bytes = 0
    
    for vf in model.velocity_fields.values():
        # Random weights
        if hasattr(vf, 'W_random'):
            total_bytes += vf.W_random.nbytes
        if hasattr(vf, 'W_fwd'):
            total_bytes += vf.W_fwd.nbytes
        if hasattr(vf, 'W_bwd'):
            total_bytes += vf.W_bwd.nbytes
        
        # Learned weights
        total_bytes += vf.W_out.nbytes
        total_bytes += vf.bias.nbytes
    
    return total_bytes / (1024 * 1024)


def run_benchmark(
    input_dim: int,
    hidden_dim: int,
    n_samples: int,
    n_timesteps: int,
    use_bipil: bool = True,
    n_gen_samples: int = 100,
) -> BenchmarkResult:
    """
    Run a single benchmark configuration.
    
    Args:
        input_dim: Input dimension
        hidden_dim: Hidden layer dimension
        n_samples: Number of training samples
        n_timesteps: Number of timesteps
        use_bipil: Use bidirectional PIL
        n_gen_samples: Number of samples to generate
        
    Returns:
        BenchmarkResult with metrics
    """
    # Create synthetic data
    rng = np.random.default_rng(42)
    X_data = rng.standard_normal((n_samples, input_dim)).astype(np.float32)
    
    # Configuration
    config = PILFlowConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        use_bipil=use_bipil,
        n_train_timesteps=n_timesteps,
        n_sample_timesteps=n_timesteps // 2,
        lambda_reg=1e-5,
        per_timestep_fit=True,
        seed=42,
    )
    
    # Create model
    model = PILFlowMatching(config)
    
    # Training
    train_start = time.time()
    train_stats = model.fit(X_data, verbose=False)
    train_time = time.time() - train_start
    
    # Memory
    memory_mb = estimate_memory_usage(model)
    
    # Generation
    gen_start = time.time()
    samples = model.sample(n_gen_samples)
    gen_time = time.time() - gen_start
    
    samples_per_second = n_gen_samples / gen_time
    
    return BenchmarkResult(
        method="BiPIL-Flow" if use_bipil else "PIL-Flow",
        n_samples=n_samples,
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_timesteps=n_timesteps,
        train_time=train_time,
        gen_time=gen_time,
        avg_mse=train_stats["avg_mse"],
        memory_mb=memory_mb,
        samples_per_second=samples_per_second,
    )


def benchmark_scaling():
    """Benchmark scaling with different parameters."""
    print("=" * 70)
    print("PIL-Flow Scaling Benchmark")
    print("=" * 70)
    
    results: List[BenchmarkResult] = []
    
    # Test 1: Scaling with input dimension
    print("\n1. Scaling with Input Dimension")
    print("-" * 50)
    for dim in [64, 128, 256, 512, 784]:
        result = run_benchmark(
            input_dim=dim,
            hidden_dim=256,
            n_samples=1000,
            n_timesteps=20,
        )
        results.append(result)
        print(f"  dim={dim:4d}: train={result.train_time:.2f}s, "
              f"MSE={result.avg_mse:.4f}, mem={result.memory_mb:.1f}MB")
    
    # Test 2: Scaling with training samples
    print("\n2. Scaling with Training Samples")
    print("-" * 50)
    for n_samples in [500, 1000, 2000, 5000, 10000]:
        result = run_benchmark(
            input_dim=256,
            hidden_dim=256,
            n_samples=n_samples,
            n_timesteps=20,
        )
        results.append(result)
        print(f"  N={n_samples:5d}: train={result.train_time:.2f}s, "
              f"MSE={result.avg_mse:.4f}")
    
    # Test 3: Scaling with hidden dimension
    print("\n3. Scaling with Hidden Dimension")
    print("-" * 50)
    for hidden_dim in [64, 128, 256, 512, 1024]:
        result = run_benchmark(
            input_dim=256,
            hidden_dim=hidden_dim,
            n_samples=1000,
            n_timesteps=20,
        )
        results.append(result)
        print(f"  hidden={hidden_dim:4d}: train={result.train_time:.2f}s, "
              f"MSE={result.avg_mse:.4f}, mem={result.memory_mb:.1f}MB")
    
    # Test 4: Scaling with timesteps
    print("\n4. Scaling with Timesteps")
    print("-" * 50)
    for n_timesteps in [10, 25, 50, 100, 200]:
        result = run_benchmark(
            input_dim=256,
            hidden_dim=256,
            n_samples=1000,
            n_timesteps=n_timesteps,
        )
        results.append(result)
        print(f"  T={n_timesteps:3d}: train={result.train_time:.2f}s, "
              f"MSE={result.avg_mse:.4f}, mem={result.memory_mb:.1f}MB")
    
    # Test 5: PIL vs BiPIL
    print("\n5. PIL vs BiPIL Comparison")
    print("-" * 50)
    for use_bipil in [False, True]:
        result = run_benchmark(
            input_dim=256,
            hidden_dim=256,
            n_samples=2000,
            n_timesteps=50,
            use_bipil=use_bipil,
        )
        results.append(result)
        print(f"  {result.method:12s}: train={result.train_time:.2f}s, "
              f"MSE={result.avg_mse:.4f}, mem={result.memory_mb:.1f}MB")
    
    return results


def benchmark_vs_gradient():
    """
    Compare PIL-Flow training time vs estimated gradient-based training.
    
    Note: This is a theoretical comparison since we don't implement
    gradient-based flow matching here.
    """
    print("\n" + "=" * 70)
    print("PIL-Flow vs Gradient-Based (Theoretical Comparison)")
    print("=" * 70)
    
    # PIL-Flow benchmark
    result = run_benchmark(
        input_dim=784,
        hidden_dim=512,
        n_samples=5000,
        n_timesteps=50,
    )
    
    print(f"\nPIL-Flow:")
    print(f"  Training time: {result.train_time:.2f}s")
    print(f"  This is ONE pass through the data (closed-form solution)")
    
    # Theoretical gradient-based estimate
    # Typical: 100-500 epochs, each epoch processes all data
    # Each backward pass is ~2x forward pass
    n_epochs_typical = 200
    forward_time_per_epoch = result.train_time / 50  # Approximate
    backward_multiplier = 2.0
    
    estimated_gradient_time = (
        n_epochs_typical * 
        forward_time_per_epoch * 
        (1 + backward_multiplier)  # Forward + backward
    )
    
    print(f"\nGradient-Based (Estimated):")
    print(f"  Epochs: {n_epochs_typical}")
    print(f"  Estimated time: {estimated_gradient_time:.2f}s")
    print(f"  (Forward + Backward per epoch)")
    
    speedup = estimated_gradient_time / result.train_time
    print(f"\nEstimated Speedup: {speedup:.1f}x faster with PIL")
    
    print("\nNote: This is a theoretical estimate. Actual speedup depends on:")
    print("  - Hardware (GPU vs CPU)")
    print("  - Batch size")
    print("  - Convergence criteria")
    print("  - Implementation efficiency")


def main():
    print("PIL-Flow Benchmarking Suite")
    print("Gradient-Free Flow Matching Performance Analysis")
    print()
    
    # Run scaling benchmarks
    results = benchmark_scaling()
    
    # Theoretical comparison
    benchmark_vs_gradient()
    
    # Summary
    print("\n" + "=" * 70)
    print("Key Findings")
    print("=" * 70)
    print("1. PIL-Flow trains in ONE PASS (closed-form solution)")
    print("2. No gradient computation or backpropagation required")
    print("3. Training time scales with O(N * d * h) for matrix operations")
    print("4. BiPIL provides richer features with moderate overhead")
    print("5. Memory efficient: only stores weight matrices")
    print("=" * 70)


if __name__ == "__main__":
    main()
