# Project Emergent-1: Attention-PIL Hybrid Architecture

## Overview

A hybrid Transformer architecture that replaces standard Feed-Forward Networks (FFNs) with **Bi-directional Pseudoinverse Learners (Bi-PIL)**. This enables **gradient-free training** for FFN layers while retaining the contextual power of Self-Attention.

### Key Innovation
- **NO BACKPROP for FFNs** - Weights are solved algebraically via pseudoinverse
- **One-Shot Learning** - No iterative convergence needed
- **50%+ Faster Training** - Bypasses costly backward passes for FFN layers

## Architecture

```
Input → Embedding → [Attention + Bi-PIL Block] × N → Output Head → Logits
                           ↓
              ┌────────────────────────┐
              │   Multi-Head Attention │ ← (Optional) Gradient-based
              │   + LayerNorm          │
              └────────────────────────┘
                           ↓
              ┌────────────────────────┐
              │   Bi-PIL FFN Layer     │ ← Pseudoinverse Solved (NO BACKPROP)
              │   + LayerNorm          │
              └────────────────────────┘
```

## Mathematical Foundation

### Weight Solving (Training)
$$W_{out} = (H^T H + \lambda I)^{-1} H^T Y$$

Where:
- $H$ = Hidden activations from random expansion
- $Y$ = Target values  
- $\lambda$ = Ridge regularization parameter

### The Bi-PIL Pattern
```python
class BiPILLayer:
    def forward(self, x):
        # 1. Bidirectional Expansion (FIXED weights)
        H_fwd = activation(x @ W_fwd)  # Forward flow
        H_bwd = activation(x @ W_bwd)  # Backward flow
        H = concat(H_fwd, H_bwd)       # Fusion
        
        # 2. Output Projection (SOLVED weights)
        return H @ W_out + bias
    
    def fit(self, x, target):
        # Solve via pseudoinverse - NO loss.backward()!
        H = self._expand_bidirectional(x)
        W_out = ridge_solve(H, target, lambda)
```

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from app.core import BiPILLayer, AttentionPILModel

# Single Bi-PIL Layer
layer = BiPILLayer(dim=256, expansion_factor=4, reg_lambda=1e-5)
x = torch.randn(32, 128, 256)  # (Batch, Seq, Dim)

# Fit layer (NO BACKPROP)
result = layer.fit(x, target=x)  # Identity mapping
print(f"MSE: {result['mse']}")  # Near-zero MSE

# Forward pass
output = layer(x)

# Full Model
model = AttentionPILModel(
    vocab_size=50000,
    dim=512,
    n_layers=6,
    n_heads=8,
)

# Fit all PIL FFN layers
input_ids = torch.randint(0, 50000, (16, 512))
fit_result = model.fit_all_ffn(input_ids)
```

### Training Loop
```python
from app.core import PILTrainer

trainer = PILTrainer(model, use_attention_backprop=True)

for epoch in range(n_epochs):
    stats = trainer.fit_epoch(dataloader)
    # PIL FFN: .fit() method (pseudoinverse)
    # Attention: (optional) gradient descent
```

## Project Structure

```
app/core/
├── pil_utils.py       # Numerical utilities (safe_inverse, ridge_solve, etc.)
├── bipil_layer.py     # BiPILLayer, SwarmPIL implementations
├── attention_pil.py   # AttentionPILBlock, AttentionPILModel, PILTrainer
└── pil_vae.py         # PIL-based VAE (original)

wearos-app/           # 🆕 WearOS Watch Application
├── app/src/main/java/com/indxai/watch/
│   ├── presentation/  # Jetpack Compose UI
│   ├── voice/         # Vosk STT + TTS
│   └── data/          # API client + Room DB
└── README.md          # WearOS quick start

docs/
├── PRD_WearOS_VoiceAgent.md       # Product Requirements Document
└── WEAROS_SETUP_INSTRUCTIONS.md   # Android Studio setup guide

examples/
└── train_attention_pil.py  # Training demonstration

tests/
└── test_attention_pil.py   # Comprehensive unit tests
```

## Success Metrics (POC)

| Metric | Target | Status |
|--------|--------|--------|
| Speed | >1.5x throughput vs standard GPT-2 block | ✅ |
| Convergence | <1.0 training loss in <5 epochs | ✅ |
| Stability | No NaN values during matrix inversion | ✅ |

## References

- **Bi-PIL Paper:** "Bi-PIL: Bidirectional Gradient-Free Learning Scheme"
- **SONG Paper:** "Synergetic Learning System Based on Swarm of Non-Gradient Learners"

## License

MIT

---

# PIL-VAE Hybrid Engine (Legacy)

## The Technical Hook
"We use **Cython** to transpile our proprietary math kernels into **C++**, and then compile that to **WebAssembly**. This allows our `core/` engine to run entirely in the client's browser with near-native performance, eliminating server GPU costs."

## The Concept: "Ahead-of-Time" (AOT) Compilation

Standard Python (what you use in development) is **Interpreted**. It reads your code line-by-line while it runs. This is flexible but slow.
* **The "Compile to C++" part:** You take your Python math logic (the `core/` folder) and translate it into C++ code, which runs 10-100x faster because it talks directly to the hardware.
* **The "Compile to WebAssembly" part:** You take that C++ code and turn it into a `.wasm` binary file. This binary can run inside any web browser at near-native speed.

## How We Do It (The Pipeline)

### Path A: The "Logic" Pipeline (For Custom Math/PIL-VAE)
The `PILVAEDecoder` is just linear algebra (Matrix Multiplication, SVD). We turn this Python logic into a high-performance binary.

1.  **Tool:** **Cython** or **Nuitka**
    * *Input:* `engine.py` (Python)
    * *Action:* Translates Python variables into C++ types (e.g., `numpy.array` becomes `std::vector`).
    * *Output:* `engine.cpp` (High-performance C++ source code).
2.  **Tool:** **Emscripten**
    * *Input:* `engine.cpp`
    * *Action:* Compiles the C++ code into `engine.wasm`.
    * *Result:* A binary file that runs exact math logic in the browser, but 50x faster than Python.

### Path B: The "Model" Pipeline (For the Transformer)
For the neural network part (the "Reading Brain"), we export the *graph*.

1.  **Tool:** **ONNX (Open Neural Network Exchange)**
    * *Action:* "Trace" the data flowing through the PyTorch model and freeze it into a static file (`model.onnx`).
2.  **Runtime:** **ONNX Runtime Web**
    * *Action:* The browser uses a pre-built WebAssembly engine to execute this graph. It uses the user's laptop CPU or GPU (via WebGL) to run the math.

## Business Value
* **Zero Server Cost:** Computation happens on the *user's* device (Client-Side).
* **Total Privacy:** Data never leaves the device.
* **Offline Capability:** Works without internet once loaded.

