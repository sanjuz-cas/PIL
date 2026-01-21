# NeurIPS Research Directions: PIL for Generative Models

This document outlines three promising research directions using Pseudoinverse Learning (PIL) that could be developed into NeurIPS-quality papers.

---

## Table of Contents
1. [PIL Diffusion: Gradient-Free Denoising](#1-pil-diffusion-gradient-free-denoising)
2. [PIL Flow Matching: One-Shot CNF Training](#2-pil-flow-matching-one-shot-cnf-training)
3. [PIL-LoRA: Instant LLM Adaptation](#3-pil-lora-instant-llm-adaptation)

---

## 1. PIL Diffusion: Gradient-Free Denoising

### 1.1 Core Idea

Replace the gradient-based U-Net training in diffusion models with closed-form PIL solutions.

**Standard Diffusion Training:**
```
for each batch (x_0, t, ε):
    x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε    # Add noise
    ε_pred = UNet(x_t, t)                  # Predict noise
    loss = ||ε_pred - ε||²                 # MSE loss
    loss.backward()                         # SLOW: Backprop through UNet
    optimizer.step()
```

**PIL Diffusion Training:**
```
for each timestep t:
    Collect all (x_t, ε) pairs
    H_t = FeatureExpand(x_t)               # Random projections
    W_t = (H_t^T H_t + λI)^{-1} H_t^T ε    # ONE-SHOT solve
```

### 1.2 Mathematical Formulation

**Noise Prediction as Ridge Regression:**

Given noisy samples $x_t$ and their corresponding noise $\epsilon$, we want:
$$\epsilon_{pred} = f(x_t, t)$$

**PIL Solution:**
$$H_t = \sigma(x_t \cdot W_{random} \oplus \text{TimeEmbed}(t))$$
$$W_{out}^{(t)} = (H_t^T H_t + \lambda I)^{-1} H_t^T \epsilon$$

Where:
- $W_{random}$ is a fixed random projection matrix
- $\oplus$ denotes concatenation with time embedding
- $\sigma$ is activation function (GELU)

### 1.3 Architecture Design

```
PILDiffusionModel:
├── TimeEmbedding (sinusoidal)
├── FeatureExtractor
│   ├── Conv layers (fixed random weights)
│   ├── BiPIL projections per resolution
│   └── Skip connections
├── PIL Denoiser (per timestep or shared)
│   ├── W_random: Fixed random projection
│   └── W_out: Solved via pseudoinverse
└── Output projection
```

### 1.4 Implementation Plan

```python
# pil_diffusion/model.py

class PILDiffusion(nn.Module):
    """Gradient-free diffusion model using PIL."""
    
    def __init__(
        self,
        image_size: int = 32,
        channels: int = 3,
        hidden_dim: int = 512,
        num_timesteps: int = 1000,
        lambda_reg: float = 1e-5,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.lambda_reg = lambda_reg
        
        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(hidden_dim)
        
        # Feature extractor (random, fixed)
        self.feature_extractor = PILFeatureExtractor(
            in_channels=channels,
            hidden_dim=hidden_dim,
            image_size=image_size,
        )
        
        # Output weights (solved via PIL) - one per timestep or shared
        self.register_buffer(
            "W_out",
            torch.zeros(hidden_dim, channels * image_size * image_size)
        )
        
        # Noise schedule
        self.register_buffer("alphas_cumprod", self._cosine_schedule(num_timesteps))
    
    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise."""
        # Time embedding
        t_emb = self.time_embed(t)
        
        # Feature extraction
        h = self.feature_extractor(x_t, t_emb)
        
        # Predict noise
        eps_pred = h @ self.W_out
        eps_pred = eps_pred.view(x_t.shape)
        
        return eps_pred
    
    @torch.no_grad()
    def fit(self, dataloader, num_samples_per_t: int = 10000):
        """Fit denoiser via PIL - ONE SHOT."""
        device = next(self.parameters()).device
        
        # Collect features and targets for each timestep
        all_H = []
        all_eps = []
        
        for x_0 in dataloader:
            x_0 = x_0.to(device)
            batch_size = x_0.shape[0]
            
            # Sample random timesteps
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
            
            # Sample noise
            eps = torch.randn_like(x_0)
            
            # Create noisy samples
            alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
            x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * eps
            
            # Extract features
            t_emb = self.time_embed(t)
            h = self.feature_extractor(x_t, t_emb)
            
            all_H.append(h)
            all_eps.append(eps.view(batch_size, -1))
        
        # Concatenate all data
        H = torch.cat(all_H, dim=0)
        eps = torch.cat(all_eps, dim=0)
        
        # Solve via PIL
        # W = (H^T H + λI)^{-1} H^T eps
        HtH = H.T @ H
        reg = self.lambda_reg * torch.eye(H.shape[1], device=device)
        HtY = H.T @ eps
        
        try:
            L = torch.linalg.cholesky(HtH + reg)
            W = torch.cholesky_solve(HtY, L)
        except:
            W = torch.linalg.solve(HtH + reg, HtY)
        
        self.W_out.copy_(W)
        
        return {"success": True, "num_samples": H.shape[0]}
    
    @torch.no_grad()
    def sample(self, batch_size: int, device: str = "cuda"):
        """Generate samples via DDPM sampling."""
        # Start from pure noise
        x = torch.randn(batch_size, 3, self.image_size, self.image_size, device=device)
        
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Predict noise
            eps_pred = self(x, t_batch)
            
            # DDPM update step
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
            
            # Compute x_{t-1}
            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_t)) * eps_pred
            )
            
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
                x = x + sigma_t * noise
        
        return x
```

### 1.5 Experiments to Run

| Experiment | Dataset | Metric | Baseline |
|------------|---------|--------|----------|
| Image Quality | CIFAR-10 | FID, IS | DDPM U-Net |
| Training Speed | CIFAR-10 | Time to FID < 50 | DDPM |
| Memory Usage | CIFAR-10 | Peak GPU MB | DDPM |
| Scaling | ImageNet 64x64 | FID | DiT |
| Ablation: Timesteps | CIFAR-10 | FID | Shared vs Per-t |

### 1.6 Key Research Questions

1. **Shared vs Per-Timestep Weights:** Should $W_t$ be different for each timestep?
2. **Feature Extractor Design:** Random CNN vs structured features?
3. **Bi-PIL for Diffusion:** Does bidirectional projection help?
4. **Theoretical Analysis:** Connection to score matching?

---

## 2. PIL Flow Matching: One-Shot CNF Training

### 2.1 Core Idea

Train continuous normalizing flows (CNFs) via PIL instead of backprop through ODE solvers.

**Current Status:** ✅ Basic implementation exists in `pil_flow/`

### 2.2 Mathematical Formulation

**Flow Matching Objective:**

Learn velocity field $v(x, t)$ such that:
$$\frac{dx}{dt} = v(x, t)$$

transforms noise $x_0 \sim \mathcal{N}(0, I)$ to data $x_1 \sim p_{data}$.

**Optimal Transport Path:**
$$x_t = (1-t) x_0 + t x_1$$
$$v^*(x_t, t) = x_1 - x_0$$

**PIL Solution:**
$$H = \sigma([x_t; t] \cdot W_{random})$$
$$W_{velocity} = (H^T H + \lambda I)^{-1} H^T (x_1 - x_0)$$

### 2.3 Current Implementation (pil_flow/)

```
pil_flow/
├── __init__.py
├── model.py           # PILFlowMatching main class
├── velocity_field.py  # BiPILVelocityField
├── solver.py          # Ridge regression solvers
├── utils.py           # Utilities
└── examples/
    ├── train_mnist.py
    └── benchmark.py
```

### 2.4 Experiments Needed for NeurIPS

```python
# pil_flow/experiments/benchmark_suite.py

"""
Comprehensive benchmarking for PIL Flow Matching
"""

EXPERIMENTS = {
    "mnist": {
        "dataset": "MNIST",
        "image_size": 28,
        "metrics": ["FID", "NLL", "Training Time"],
        "baselines": ["RealNVP", "Glow", "FFJORD"],
    },
    "cifar10": {
        "dataset": "CIFAR-10",
        "image_size": 32,
        "metrics": ["FID", "IS", "Training Time", "Memory"],
        "baselines": ["Flow Matching (gradient)", "DDPM"],
    },
    "celeba": {
        "dataset": "CelebA 64x64",
        "image_size": 64,
        "metrics": ["FID", "LPIPS"],
        "baselines": ["StyleGAN2", "Flow Matching"],
    },
}

def run_experiment(config):
    """Run single experiment."""
    # Load data
    dataloader = load_dataset(config["dataset"], config["image_size"])
    
    # PIL Flow
    pil_model = PILFlowMatching(
        data_dim=config["image_size"]**2 * 3,
        hidden_dim=1024,
        num_layers=4,
        use_bipil=True,
    )
    
    # Time training
    start = time.time()
    pil_model.fit(dataloader)
    pil_train_time = time.time() - start
    
    # Generate samples
    samples = pil_model.sample(10000)
    
    # Compute metrics
    fid = compute_fid(samples, dataloader)
    
    return {
        "model": "PIL Flow",
        "train_time": pil_train_time,
        "fid": fid,
    }
```

### 2.5 Ablation Studies

| Ablation | Variations | Measure |
|----------|------------|---------|
| Hidden Dimension | 256, 512, 1024, 2048 | FID vs params |
| Bi-PIL vs Single | concat, add, single | FID |
| Regularization λ | 1e-3, 1e-4, 1e-5, 1e-6 | FID, stability |
| Time Encoding | Sinusoidal, learned, Fourier | FID |
| ODE Steps | 10, 50, 100, 500 | FID vs speed |

### 2.6 Theoretical Contributions

1. **Convergence Analysis:** Under what conditions does PIL flow match the optimal transport?
2. **Approximation Bounds:** Error bounds for finite hidden dimension
3. **Connection to Kernel Methods:** PIL flow as kernel flow matching

---

## 3. PIL-LoRA: Instant LLM Adaptation

### 3.1 Core Idea

Replace iterative LoRA fine-tuning with one-shot PIL adaptation.

**Standard LoRA:**
```
W_adapted = W_frozen + BA    # B, A learned via gradient descent
# Requires 1000s of steps
```

**PIL-LoRA:**
```
W_adapted = W_frozen + H @ W_pil    # H: random, W_pil: solved in 1 step
# ONE forward pass to solve
```

### 3.2 Mathematical Formulation

**LoRA Objective:**
$$\min_{B,A} \sum_i \mathcal{L}(f_{W + BA}(x_i), y_i)$$

**PIL-LoRA Solution:**

For each adapted layer:
$$H = \sigma(x \cdot W_{random})$$  (random projection)
$$\Delta W = (H^T H + \lambda I)^{-1} H^T (y - f_W(x))$$

Where:
- $x$: layer inputs
- $y$: desired outputs (from task data)
- $f_W(x)$: frozen model output

### 3.3 Architecture

```python
# pil_lora/adapter.py

class PILLoRAAdapter(nn.Module):
    """
    PIL-based adapter that can be solved in one shot.
    
    Replaces: W_new = W_old + B @ A (LoRA)
    With:     W_new = W_old + H @ W_pil (PIL)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        lambda_reg: float = 1e-5,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lambda_reg = lambda_reg
        
        # Random projection (fixed)
        self.register_buffer(
            "W_down",
            torch.randn(in_features, rank) / np.sqrt(in_features)
        )
        
        # Adaptation weights (solved via PIL)
        self.register_buffer(
            "W_up",
            torch.zeros(rank, out_features)
        )
        
        self._is_fitted = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute adaptation delta."""
        # Project down
        h = F.gelu(x @ self.W_down)
        # Project up
        delta = h @ self.W_up
        return delta
    
    @torch.no_grad()
    def fit(self, inputs: torch.Tensor, residuals: torch.Tensor):
        """
        Solve adapter weights via PIL.
        
        Args:
            inputs: Layer inputs (N, in_features)
            residuals: Desired change in outputs (N, out_features)
        """
        # Project inputs
        H = F.gelu(inputs @ self.W_down)
        
        # Solve: W_up = (H^T H + λI)^{-1} H^T residuals
        HtH = H.T @ H
        reg = self.lambda_reg * torch.eye(self.rank, device=H.device)
        HtY = H.T @ residuals
        
        L = torch.linalg.cholesky(HtH + reg)
        W_new = torch.cholesky_solve(HtY, L)
        
        self.W_up.copy_(W_new)
        self._is_fitted = True


class PILLoRAModel(nn.Module):
    """
    Wraps a frozen model with PIL-LoRA adapters.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        target_modules: List[str] = ["q_proj", "v_proj"],
        rank: int = 64,
    ):
        super().__init__()
        
        self.base_model = base_model
        self.adapters = nn.ModuleDict()
        
        # Freeze base model
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Add adapters to target modules
        for name, module in base_model.named_modules():
            if any(t in name for t in target_modules):
                if isinstance(module, nn.Linear):
                    adapter = PILLoRAAdapter(
                        module.in_features,
                        module.out_features,
                        rank=rank,
                    )
                    self.adapters[name.replace(".", "_")] = adapter
        
        # Hook storage
        self._activations = {}
        self._register_hooks()
    
    def fit_adapters(self, dataloader, task_loss_fn):
        """
        Fit all adapters in ONE forward pass through the data.
        """
        # Collect activations and compute residuals
        all_inputs = {name: [] for name in self.adapters}
        all_residuals = {name: [] for name in self.adapters}
        
        for batch in dataloader:
            # Forward pass to collect activations
            with torch.no_grad():
                outputs = self.base_model(**batch)
            
            # Compute what the output SHOULD be
            # (This depends on the task)
            target_outputs = self._compute_targets(batch)
            
            # Store for each adapter
            for name in self.adapters:
                all_inputs[name].append(self._activations[name])
                # Residual = target - current
                all_residuals[name].append(
                    target_outputs[name] - self._activations[name + "_out"]
                )
        
        # Fit each adapter
        for name, adapter in self.adapters.items():
            inputs = torch.cat(all_inputs[name], dim=0)
            residuals = torch.cat(all_residuals[name], dim=0)
            adapter.fit(inputs, residuals)
        
        return {"success": True, "num_adapters": len(self.adapters)}
```

### 3.4 Experiments

| Experiment | Model | Dataset | Metric | Baseline |
|------------|-------|---------|--------|----------|
| Classification | LLaMA-7B | GLUE | Accuracy | LoRA |
| Instruction | Mistral-7B | Alpaca | Win Rate | LoRA, QLoRA |
| Few-shot | GPT-2 | SuperGLUE | Accuracy | In-context |
| Speed | All | All | Time to converge | LoRA |
| Memory | All | All | Peak GPU | LoRA |

### 3.5 Key Advantages

| Aspect | LoRA | PIL-LoRA |
|--------|------|----------|
| Training Steps | 1000-10000 | **1** |
| Hyperparameters | LR, warmup, scheduler | λ only |
| Memory (training) | Activations + gradients | Activations only |
| Deterministic | No (SGD noise) | **Yes** |

### 3.6 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Large activation matrices | Chunked/streaming computation |
| What are "residuals"? | Use task-specific loss gradient direction |
| Multi-layer dependency | Fit layer-by-layer, freeze previous |
| Rank selection | Cross-validation or condition number |

---

## 4. Implementation Roadmap

### Phase 1: Foundations (2 weeks)
- [ ] PIL Diffusion: Basic CIFAR-10 implementation
- [ ] PIL Flow: Add FID/IS computation
- [ ] PIL-LoRA: Implement base adapter class

### Phase 2: Experiments (4 weeks)
- [ ] PIL Diffusion: Full benchmark suite
- [ ] PIL Flow: Compare vs baselines on 3 datasets
- [ ] PIL-LoRA: GLUE benchmark on LLaMA-7B

### Phase 3: Analysis (2 weeks)
- [ ] Ablation studies for all three
- [ ] Theoretical analysis (convergence, bounds)
- [ ] Scaling experiments

### Phase 4: Paper Writing (2 weeks)
- [ ] Choose strongest result for main paper
- [ ] Others as appendix or follow-up

---

## 5. Compute Requirements

| Experiment | GPU | Time Estimate |
|------------|-----|---------------|
| PIL Diffusion CIFAR-10 | 1x A100 | 2-4 hours |
| PIL Diffusion ImageNet 64 | 4x A100 | 1-2 days |
| PIL Flow CIFAR-10 | 1x V100 | 1-2 hours |
| PIL-LoRA LLaMA-7B | 1x A100 80GB | 4-8 hours |
| Full benchmark suite | 8x A100 | 1 week |

---

## 6. Related Work to Cite

### Pseudoinverse Learning
- Huang et al., "Extreme Learning Machines" (2006)
- SONG: "Synergetic Learning System Based on Swarm of Non-Gradient Learners"
- Bi-PIL: "Bidirectional Gradient-Free Learning Scheme"

### Diffusion Models
- Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
- Song et al., "Score-Based Generative Modeling" (2021)
- Lipman et al., "Flow Matching for Generative Modeling" (2023)

### Efficient Fine-tuning
- Hu et al., "LoRA: Low-Rank Adaptation" (2021)
- Dettmers et al., "QLoRA" (2023)
- He et al., "Towards Efficient Fine-tuning" (2022)

---

## 7. Success Criteria for NeurIPS

| Criterion | Target |
|-----------|--------|
| **Novelty** | First gradient-free approach for X |
| **Significance** | 10x+ speedup OR comparable quality with 1/100 compute |
| **Soundness** | Theoretical justification + extensive experiments |
| **Reproducibility** | Open-source code + clear methodology |

---

## Quick Start Commands

```bash
# PIL Diffusion
cd pil_diffusion
python train.py --dataset cifar10 --epochs 1  # One-shot!

# PIL Flow Matching
cd pil_flow
python examples/benchmark.py --dataset cifar10

# PIL-LoRA
cd pil_lora
python adapt.py --model llama-7b --task glue --subset sst2
```

---

*Last updated: January 2026*
*Project: Emergent-1 PIL Research*
