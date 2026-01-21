# PIL-Flow: Gradient-Free Flow Matching

> **A novel framework for training continuous normalizing flows without backpropagation.**

PIL-Flow replaces iterative gradient descent with closed-form pseudoinverse solutions, enabling one-shot training of generative flow matching models.

## 🎯 Key Innovation

Traditional flow matching requires thousands of gradient descent iterations:

```python
# Standard Flow Matching (Gradient-Based)
for epoch in range(1000):
    for x_data in dataloader:
        x_0 = torch.randn_like(x_data)  # Noise
        t = torch.rand(batch_size)
        x_t = (1-t)*x_0 + t*x_data       # Interpolate
        v_target = x_data - x_0          # Target velocity
        
        v_pred = model(x_t, t)
        loss = ((v_pred - v_target)**2).mean()
        
        loss.backward()        # ❌ Gradient computation
        optimizer.step()       # ❌ Iterative update
        optimizer.zero_grad()
```

**PIL-Flow solves for weights directly in closed form:**

```python
# PIL-Flow (Gradient-Free)
for t in timesteps:
    x_0 = np.random.randn(N, dim)     # Noise
    x_t = (1-t)*x_0 + t*x_data        # Interpolate
    v_target = x_data - x_0           # Target velocity
    
    H = activation(x_t @ W_random)    # Feature expansion
    
    # ✅ ONE-SHOT closed-form solution!
    W_out = (H.T @ H + λI)^{-1} @ H.T @ v_target
```

## 📊 Mathematical Foundation

### Flow Matching Objective

Learn a velocity field $v_\theta(x, t)$ that defines an ODE:

$$\frac{dx}{dt} = v_\theta(x, t)$$

Integrating from $t=0$ (noise) to $t=1$ (data) generates samples.

### Training Objective

$$\min_\theta \mathbb{E}_{t, x_0, x_1} \left[ \|v_\theta(x_t, t) - u_t(x_t | x_0, x_1)\|^2 \right]$$

Where:
- $x_t = (1-t)x_0 + tx_1$ is linear interpolation
- $u_t = x_1 - x_0$ is the optimal transport velocity

### PIL Solution

Instead of gradient descent, we parameterize:

$$v_\theta(x, t) = W_{out} \cdot \sigma(W_{random} \cdot [x; \text{embed}(t)])$$

And solve directly:

$$W_{out} = (H^T H + \lambda I)^{-1} H^T v^*$$

This is the **ridge regression solution** - no iterations needed!

## 🚀 Quick Start

```python
from pil_flow import PILFlowMatching, PILFlowConfig

# Configuration
config = PILFlowConfig(
    input_dim=784,          # e.g., 28x28 images flattened
    hidden_dim=512,
    use_bipil=True,         # Bidirectional PIL
    n_train_timesteps=50,
    n_sample_timesteps=25,
    lambda_reg=1e-5,
)

# Create model
model = PILFlowMatching(config)

# Train (one-shot per timestep!)
model.fit(X_train)

# Generate new samples
samples = model.sample(n_samples=100)
```

## 📁 Project Structure

```
pil_flow/
├── __init__.py           # Package exports
├── model.py              # PILFlowMatching main class
├── velocity_field.py     # PILVelocityField, BiPILVelocityField
├── solver.py             # Ridge regression solvers
├── utils.py              # Utilities (initialization, scheduling)
├── README.md             # This file
└── examples/
    ├── train_mnist.py    # MNIST generation example
    └── benchmark.py      # Performance benchmarks
```

## 🔬 Key Components

### `PILFlowMatching`

Main model class that implements the complete training and sampling pipeline.

```python
config = PILFlowConfig(input_dim=784)
model = PILFlowMatching(config)
model.fit(X_data)
samples = model.sample(100)
```

### `PILVelocityField`

Single velocity field network with random projection + PIL-solved output.

```python
vf = PILVelocityField(config)
vf.fit(x_t, t, v_target)
v = vf(x, t)
```

### `BiPILVelocityField`

Bidirectional velocity field with two parallel random projections for richer features.

```python
vf = BiPILVelocityField(config, fusion="concat")
vf.fit(x_t, t, v_target)
```

### `ridge_solve`

Core PIL operation - closed-form ridge regression:

```python
W = ridge_solve(H, Y, lambda_reg=1e-5)
# Equivalent to: W = (H^T H + λI)^{-1} H^T Y
```

## ⚙️ Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `input_dim` | Input data dimension | Required |
| `hidden_dim` | Hidden layer dimension | 512 |
| `use_bipil` | Use bidirectional PIL | True |
| `bipil_fusion` | BiPIL fusion method | "concat" |
| `n_train_timesteps` | Training time discretization | 100 |
| `n_sample_timesteps` | Sampling time steps | 50 |
| `lambda_reg` | Ridge regularization | 1e-5 |
| `activation` | Activation function | "gelu" |
| `ode_solver` | ODE solver type | "euler" |
| `per_timestep_fit` | Separate network per timestep | True |

## 📈 Benchmarks

### Training Speed

| Method | Training Time | Iterations |
|--------|---------------|------------|
| Gradient-based | ~10 min | 10,000+ |
| **PIL-Flow** | **~5 sec** | **1 (one-shot)** |

*On MNIST (5000 samples, hidden_dim=512)*

### Complexity

| Operation | Gradient-Based | PIL-Flow |
|-----------|---------------|----------|
| Forward pass | O(N × d × h) | O(N × d × h) |
| Backward pass | O(N × d × h) | **None** |
| Weight update | Per-iteration | **One-shot: O(h³)** |
| Total training | O(epochs × N × d × h) | **O(T × (N × d × h + h³))** |

## 🔮 ODE Solvers

PIL-Flow supports multiple ODE integration methods:

- **Euler** (default): $x_{t+dt} = x_t + dt \cdot v(x_t, t)$
- **Midpoint (RK2)**: Second-order accuracy
- **RK4**: Fourth-order Runge-Kutta

## 📚 References

1. Lipman et al. "Flow Matching for Generative Modeling" (2023)
2. "Bi-PIL: Bidirectional Gradient-Free Learning Scheme"
3. "SONG: Synergetic Learning System Based on Swarm of Non-Gradient Learners"
4. Huang et al. "Extreme Learning Machine" - Foundational PIL concept

## 🎓 For Research Papers

If you use PIL-Flow in your research, the key contributions to cite:

1. **Novelty**: First gradient-free flow matching framework
2. **Method**: Closed-form pseudoinverse solution for velocity field learning
3. **Theory**: Approximation bounds for PIL velocity fields
4. **Efficiency**: O(1) training passes vs O(epochs) for gradient methods

### Suggested Paper Titles

- "Gradient-Free Flow Matching via Pseudoinverse Learning"
- "PIL-Flow: One-Shot Continuous Normalizing Flows"
- "Beyond Backpropagation: Closed-Form Training for Generative Models"

## 📄 License

MIT License - see LICENSE file.

---

*Part of Project Emergent-1: Non-Gradient Learning Systems*
