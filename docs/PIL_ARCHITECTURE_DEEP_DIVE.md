# PIL (Pseudoinverse Learning) Architecture Deep Dive

> A comprehensive technical document covering the mathematical foundations, architecture, and future generative extensions of the PIL-VAE system.

---

## Table of Contents

1. [PIL vs Standard PyTorch Gradient Descent](#1-pil-vs-standard-pytorch-gradient-descent)
2. [Confidence Score Calculation](#2-confidence-score-calculation)
3. [Embedding Model](#3-embedding-model)
4. [Ridge Regression and Activation Functions](#4-ridge-regression-and-activation-functions)
5. [Bidirectional PIL (Bi-PIL)](#5-bidirectional-pil-bi-pil)
6. [Complete Data Flow](#6-complete-data-flow)
7. [Generative Extensions](#7-generative-extensions)
   - [Option 1: Probabilistic Sampling (Variational)](#option-1-probabilistic-sampling-variational)
   - [Option 2: Diffusion-Style Denoising](#option-2-diffusion-style-denoising-score-matching)
   - [Option 3: Autoregressive Token Generation](#option-3-autoregressive-token-generation)
   - [Option 4: Flow Matching](#option-4-flow-matching)
   - [Hybrid PIL-Transformer](#hybrid-pil-transformer-for-text-generation)

---

## 1. PIL vs Standard PyTorch Gradient Descent

### Standard PyTorch (Gradient Descent)

```python
# Traditional approach
model = nn.Linear(input_dim, output_dim)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    output = model(x)
    loss = criterion(output, target)
    loss.backward()        # ❌ Compute gradients via backprop
    optimizer.step()       # ❌ Update weights iteratively
    optimizer.zero_grad()
```

**How it works:** Slowly nudges weights in the direction that reduces error, one small step at a time over thousands of iterations.

### PIL Approach (One-Shot Pseudoinverse)

```python
# PIL custom approach
class PILLayer(nn.Module):
    def __init__(self):
        self.W_random = nn.Parameter(..., requires_grad=False)  # Fixed random projection
        self.W_out = nn.Parameter(..., requires_grad=False)     # Solved, not trained
    
    def fit(self, x, target):
        with torch.no_grad():
            H = activation(x @ self.W_random)  # Feature expansion
            # ✅ One-shot closed-form solution:
            # W = (H^T H + λI)^{-1} H^T Y
            self.W_out.copy_(torch.linalg.solve(...))
```

**How it works:** Directly solves for the optimal weights using linear algebra (pseudoinverse) in **one step** - no iterations needed!

### Mathematical Formulas

**Standard Gradient Descent:**
$$W_{t+1} = W_t - \alpha \frac{\partial L}{\partial W}$$

**PIL Closed-Form Ridge Regression:**
$$W_{out} = (H^T H + \lambda I)^{-1} H^T Y$$

This finds the **exact optimal weights** that minimize squared error in a single computation.

### Key Differences

| Aspect | Standard PyTorch | PIL |
|--------|------------------|-----|
| **Weight Update** | `optimizer.step()` | `torch.linalg.solve()` / `pinv()` |
| **Iterations** | Thousands | **One** |
| **Gradients** | `loss.backward()` | **None** (no backprop) |
| **Math** | $W = W - \alpha \nabla L$ | $W = H^{\dagger} Y$ |
| **Speed** | Slow convergence | **Instant** (but $O(N^3)$ matrix ops) |

### Why PIL Matters

1. **No vanishing/exploding gradients** - there are no gradients
2. **Deterministic** - same input always gives same result
3. **Fast training** - one matrix solve vs thousands of forward/backward passes
4. **No hyperparameter tuning** - no learning rate, momentum, etc.

---

## 2. Confidence Score Calculation

The confidence score comes from **FAISS cosine similarity** during retrieval, not from the PIL layer directly.

### FAISS Index Setup

```python
# Uses Inner Product on normalized vectors = Cosine Similarity
self.index = faiss.IndexFlatIP(self.dim)  # Inner Product index
```

### Vector Normalization

Before adding or querying, vectors are **L2 normalized**:

```python
norm = np.linalg.norm(vec_normalized)
if norm > 0:
    vec_normalized = vec_normalized / norm
```

### Similarity Search

```python
scores, indices = self.index.search(query_vec, k)
# score = cosine similarity ∈ [-1, 1], typically [0, 1] for embeddings
results.append({"text": self._text_lookup[idx], "score": float(score)})
```

### Confidence Score Formula

Since we use **`IndexFlatIP`** (Inner Product) on **normalized vectors**, the score is:

$$\text{score} = \cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{||\vec{q}|| \cdot ||\vec{d}||} = \vec{q} \cdot \vec{d}$$

Where:
- $\vec{q}$ = normalized query embedding (from PIL-VAE encode → decode)
- $\vec{d}$ = normalized document embedding in memory

### Score Interpretation

| Score | Meaning |
|-------|---------|
| 1.0 | Identical vectors (perfect match) |
| 0.7+ | High similarity (included in response) |
| 0.6-0.7 | Moderate similarity (needs keyword overlap) |
| <0.6 | Low similarity (filtered out) |

### Filtering Logic

```python
if score < 0.6:
    continue  # Skip low-confidence results

# Only include if score is high OR has direct term relevance
if score >= 0.7 or has_relevance:
    top_texts.append(text)
```

### Display Format

```python
response += f"{i}. [{score:.2f}] {display_txt}\n\n"
#              ↑ Shows confidence like [0.85]
```

---

## 3. Embedding Model

### Model: `all-MiniLM-L6-v2`

The system uses **Sentence Transformers** with the **`all-MiniLM-L6-v2`** model.

### Configuration

```python
TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIM: int = 384  # Must match model output
```

### Initialization

```python
from sentence_transformers import SentenceTransformer
self.embedder = SentenceTransformer(settings.TRANSFORMER_MODEL)
```

### Model Specifications

| Property | Value |
|----------|-------|
| **Model** | `all-MiniLM-L6-v2` |
| **Dimensions** | 384 |
| **Max Sequence Length** | 256 tokens |
| **Architecture** | MiniLM (distilled BERT) |
| **Parameters** | ~22M |
| **Speed** | Very fast (~14,000 sentences/sec on GPU) |

### Why This Model?

1. **Lightweight** - Only 22M parameters (vs 110M for BERT-base)
2. **Fast inference** - Ideal for real-time chat applications
3. **Good quality** - Trained on 1B+ sentence pairs
4. **384-dim vectors** - Good balance of quality vs storage/compute

### Usage in Code

```python
# Text → 384-dim embedding
query_vec = self.embedder.encode(query)  # → np.ndarray (384,)

# Then fed into PIL-VAE
z = self.vae.encode(query_vec)           # → Compress to latent (24,)
e_gen = self.vae.decode(z)               # → Reconstruct to (384,)

# FAISS similarity search
docs = self.memory.retrieve(e_gen, top_k=5)
```

### Full Pipeline Diagram

```
Text Input
    ↓
┌─────────────────────────────┐
│  SentenceTransformer        │  ← all-MiniLM-L6-v2
│  "all-MiniLM-L6-v2"         │
└─────────────────────────────┘
    ↓ (384-dim vector)
┌─────────────────────────────┐
│  PIL-VAE Encoder            │  ← Custom PIL layer
│  384 → 128 → 24 (latent)    │
└─────────────────────────────┘
    ↓ (24-dim latent)
┌─────────────────────────────┐
│  PIL-VAE Decoder            │  ← Custom PIL layer
│  24 → 128 → 384             │
└─────────────────────────────┘
    ↓ (384-dim reconstructed)
┌─────────────────────────────┐
│  FAISS IndexFlatIP          │  ← Cosine similarity search
│  (Normalized Inner Product) │
└─────────────────────────────┘
    ↓
Retrieved Knowledge + Confidence Scores
```

---

## 4. Ridge Regression and Activation Functions

### Ridge Regression is Linear, BUT...

Yes, ridge regression is fundamentally a **linear** solution. The formula:

$$W_{out} = (H^T H + \lambda I)^{-1} H^T Y$$

solves for linear weights.

### Non-Linearity Comes from the Hidden Layer!

```python
H = activation(X @ W_random)  # ← Non-linear activation BEFORE solving
W_out = solve(H, Y)           # ← Linear solve on non-linear features
```

The **activation function** (e.g., ReLU, GELU, tanh) transforms the input into a **non-linear feature space** first. Then ridge regression finds the optimal linear mapping from that non-linear space to the output.

This is the **Extreme Learning Machine (ELM)** principle:
> Random projection + Non-linearity + Linear solve = Universal Approximator

### Activation Functions Used

```python
def leaky_relu(self, x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Leaky ReLU activation with numerical stability."""
    return np.maximum(alpha * x, x)
```

```python
def _get_activation(self, name: str) -> nn.Module:
    activations = {
        "gelu": nn.GELU(),      # Default!
        "relu": nn.ReLU(),
        "leaky_relu": nn.LeakyReLU(0.1),
        "silu": nn.SiLU(),
        "tanh": nn.Tanh(),
    }
```

### The Pattern

```
H = activation(X @ W_random)   ← NON-LINEAR (activation applied here!)
W_out = solve(H, Y)            ← LINEAR (ridge regression on non-linear features)
```

This makes the overall function **non-linear** despite using a linear solver!

### Component Summary

| Component | Linear? | Learned? | Method |
|-----------|---------|----------|--------|
| `W_random` / `W_fwd` / `W_bwd` | ✅ | ❌ Fixed | Random orthogonal init |
| `activation()` | ❌ Non-linear | ❌ Fixed | GELU, ReLU, etc. |
| `W_out` | ✅ | ✅ Learned | **Ridge Regression (PIL)** |

---

## 5. Bidirectional PIL (Bi-PIL)

### What "Bidirectional" Means

The "backward" in Bi-PIL is **NOT backpropagation**. It's **two parallel random projections** with different random weights:

```python
def _expand_bidirectional(self, x: torch.Tensor) -> torch.Tensor:
    # Forward expansion: H_fwd = σ(X @ W_fwd)
    H_fwd = self.activation(x @ self.W_fwd)

    # Backward expansion: H_bwd = σ(X @ W_bwd)  
    H_bwd = self.activation(x @ self.W_bwd)

    # Fusion options: concat, add, or gate
    if self.fusion == "concat":
        return torch.cat([H_fwd, H_bwd], dim=-1)
```

### Visual Architecture

```
Input X (dim=384)
    │
    ├──────────────────┬──────────────────┐
    │                  │                  │
    ▼                  ▼                  │
┌─────────┐      ┌─────────┐              │
│ W_fwd   │      │ W_bwd   │              │
│ (fixed) │      │ (fixed) │              │
└────┬────┘      └────┬────┘              │
     │                │                   │
     ▼                ▼                   │
┌─────────┐      ┌─────────┐              │
│ GELU()  │      │ GELU()  │              │
└────┬────┘      └────┬────┘              │
     │                │                   │
     ▼                ▼                   │
   H_fwd           H_bwd                  │
  (hidden)        (hidden)                │
     │                │                   │
     └───────┬────────┘                   │
             │ CONCAT                     │
             ▼                            │
       H_fused (2 × hidden)               │
             │                            │
             ▼                            │
       ┌───────────┐                      │
       │   W_out   │  ← SOLVED via PIL    │
       │ (learned) │                      │
       └─────┬─────┘                      │
             │                            │
             ▼                            │
         Output                           │
             │                            │
             └────────────────────────────┘
                   + RESIDUAL
```

### Why Two Projections?

**Diversity of features!** Two different random projections capture different aspects of the input:

| Single Projection | Bi-Directional |
|-------------------|----------------|
| $H = \sigma(X W_{random})$ | $H = [\sigma(X W_{fwd}), \sigma(X W_{bwd})]$ |
| One "view" of input | Two diverse "views" |
| Limited feature space | Richer feature space |

This is similar to **ensemble methods** - multiple random projections are better than one!

### Complete Training Flow (No Backprop)

```python
# 1. FORWARD PASS - Get non-linear features
H = activation(X @ W_random)   # W_random is FIXED, never updated

# 2. SOLVE (not backprop!) - One-shot weight computation
W_out = (H.T @ H + λI)^{-1} @ H.T @ Y   # Direct algebraic solution

# 3. UPDATE - Copy solution to layer
self.W_out.copy_(W_new)  # In-place update, no optimizer.step()!
```

**What's NOT happening:**
- ❌ No `loss.backward()`
- ❌ No `optimizer.step()`
- ❌ No gradient computation
- ❌ No iterative updates

**What IS happening:**
- ✅ One matrix solve per `.fit()` call
- ✅ Direct closed-form solution
- ✅ Activation provides non-linearity
- ✅ Bidirectional provides feature diversity

---

## 6. Complete Data Flow

### User Input to Model Output

#### Step 1: User Input (Text)

```
User types: "What is machine learning?"
            ↓
        Raw String
```

#### Step 2: Tokenization & Embedding

**PIL System:**
```python
# Single embedding for entire query
query_vec = self.embedder.encode(query)  # SentenceTransformer
# Output: (384,) single vector representing whole sentence
```

```
"What is machine learning?"
            ↓
   SentenceTransformer (all-MiniLM-L6-v2)
            ↓
   [0.023, -0.156, 0.892, ..., 0.045]  ← 384-dim vector
```

**Standard GPT:**
```python
# Token-by-token embedding
tokens = tokenizer.encode(query)  # [1024, 318, 4572, 4673, 30]
embeddings = embed_layer(tokens)  # (seq_len, 768) per-token embeddings
```

| Aspect | PIL | GPT |
|--------|-----|-----|
| Granularity | Whole sentence → 1 vector | Each token → separate vectors |
| Dimension | 384 | 768-12288 |
| Model | SentenceTransformer | Learned embedding layer |

#### Step 3: Core Processing

**PIL-VAE:**

```
Input: query_vec (384,)
           ↓
┌──────────────────────────────────────┐
│         ENCODER (Compress)           │
│                                      │
│  h1 = LeakyReLU(W1 @ x)             │  ← W1: Fixed random orthogonal (128×384)
│  z = W_proj.T @ (h1 - μ)            │  ← W_proj: Learned via PCA/SVD
│                                      │
│  Output: z (24,) latent vector       │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│         DECODER (Reconstruct)        │
│                                      │
│  h_rec = W4 @ z                     │  ← W4: Solved via PIL (128×24)
│  x_rec = W6 @ h_rec                 │  ← W6: Solved via PIL (384×128)
│                                      │
│  Output: e_gen (384,) reconstructed  │
└──────────────────────────────────────┘
```

**Mathematical formulas:**
```
ENCODE:  z = W_proj.T @ (LeakyReLU(W1 @ x) - μ)
DECODE:  x̂ = W6 @ (W4 @ z)
```

**Standard GPT:**
```
Input: token embeddings (seq_len, 768)
           ↓
┌──────────────────────────────────────┐
│    ATTENTION (Context Mixing)        │
│    Q, K, V = x @ W_q, x @ W_k, x @ W_v
│    attn = softmax(Q @ K.T / √d) @ V │
└──────────────────────────────────────┘
           ↓
┌──────────────────────────────────────┐
│    FFN (Feature Transform)           │
│    h = GELU(x @ W1 + b1)            │  ← Learned via backprop
│    out = h @ W2 + b2                │  ← Learned via backprop
└──────────────────────────────────────┘
           ↓
      × 12-96 layers
```

#### Step 4: Knowledge Retrieval

**PIL System:**
```python
# 1. VAE processes query
z = self.vae.encode(query_vec)      # Compress to latent
e_gen = self.vae.decode(z)          # Reconstruct (refined embedding)

# 2. FAISS similarity search
docs = self.memory.retrieve(e_gen, top_k=5)  # Find similar stored knowledge
```

```
query_vec (384)
      ↓
   ENCODE
      ↓
   z (24) ← Latent "concept" representation
      ↓
   DECODE  
      ↓
   e_gen (384) ← "Refined" embedding
      ↓
   FAISS Search
      ↓
   [doc1: 0.89, doc2: 0.76, doc3: 0.71, ...]
```

**Standard GPT:**
```python
# GPT doesn't retrieve - it generates from learned parameters
# All "knowledge" is baked into the 175B+ parameters
next_token_probs = softmax(hidden @ vocab_embeddings.T)
```

#### Step 5: Output Generation

**PIL System:**
```python
# Compose response from retrieved knowledge
response = f"**Analysis**: Key topics: {concepts}\n\n"
response += "**Findings**:\n"

for i, (txt, score) in enumerate(zip(top_texts, top_scores)):
    response += f"{i}. [{score:.2f}] {txt}\n\n"

response += "**Summary**: " + best_sentence
```

**Output is COMPOSED, not generated token-by-token!**

**Standard GPT:**
```python
# Autoregressive generation - one token at a time
for _ in range(max_tokens):
    logits = model(input_ids)
    next_token = sample(logits[:, -1, :])
    input_ids = concat(input_ids, next_token)
```

### Complete Visual: PIL System

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                                │
│                  "What is machine learning?"                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              SENTENCE TRANSFORMER (all-MiniLM-L6-v2)            │
│                    Entire sentence → 384-dim vector              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    query_vec: [0.02, -0.15, ..., 0.04] (384,)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PIL-VAE ENCODER                             │
│                                                                  │
│   h = LeakyReLU(W1 @ query_vec)     W1: FIXED (128×384)         │
│   z = W_proj.T @ (h - μ)            W_proj: SOLVED via SVD       │
│                                                                  │
│   Output: z (24-dim latent)                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    z: [1.2, -0.8, ..., 0.3] (24,)  ← Compressed concept
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      PIL-VAE DECODER                             │
│                                                                  │
│   h_rec = W4 @ z                    W4: SOLVED via PIL           │
│   e_gen = W6 @ h_rec                W6: SOLVED via PIL           │
│                                                                  │
│   Output: e_gen (384-dim refined embedding)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    e_gen: [0.03, -0.12, ..., 0.05] (384,)
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FAISS SIMILARITY SEARCH                       │
│                                                                  │
│   scores, indices = index.search(e_gen, k=5)                    │
│   Cosine similarity against stored knowledge vectors             │
│                                                                  │
│   Results:                                                       │
│     [0.89] "ML is a subset of AI that learns from data..."      │
│     [0.76] "Algorithms improve through experience..."            │
│     [0.71] "Training involves optimization of parameters..."     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RESPONSE COMPOSITION                           │
│                                                                  │
│   **Analysis**: Key topics: machine, learning, AI                │
│   **Findings**:                                                  │
│   1. [0.89] ML is a subset of AI that learns from data...       │
│   2. [0.76] Algorithms improve through experience...             │
│   **Summary**: Machine learning is a subset of AI...             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                        FINAL OUTPUT
```

### Key Differences Summary

| Aspect | PIL System | GPT |
|--------|------------|-----|
| **Input Processing** | Whole sentence → 1 vector | Token-by-token |
| **Core Computation** | Encode → Decode → Retrieve | Attention → FFN × N layers |
| **Training Method** | **PIL: One-shot matrix solve** | Backprop (thousands of iterations) |
| **Knowledge Storage** | External FAISS index | Implicit in 175B parameters |
| **Output Generation** | **Compose from retrieved docs** | Autoregressive token sampling |
| **Confidence** | Explicit scores [0.89] | Implicit (token probabilities) |
| **Determinism** | More deterministic | Stochastic (sampling) |

---

## 7. Generative Extensions

The current PIL system is **retrieval-based**. Here are mathematical modifications to make it **truly generative**.

### Current Limitation

The current decoder simply reconstructs:
$$\hat{x} = W_6 (W_4 z)$$

This is a **linear mapping** from latent to output. It can only produce vectors that lie in the span of training data.

---

### Option 1: Probabilistic Sampling (Variational)

#### Current (Deterministic)

```python
z = W_proj.T @ (h - μ)  # Deterministic latent
```

#### Change to Probabilistic

$$z \sim \mathcal{N}(\mu_z, \sigma_z^2 I)$$

Where:
$$\mu_z = W_\mu^T h, \quad \log \sigma_z^2 = W_\sigma^T h$$

#### PIL-Compatible Solution

Solve both $W_\mu$ and $W_\sigma$ via ridge regression on the training data statistics:

```python
# In fit():
H = leaky_relu(W1 @ X)  # Hidden activations

# Compute empirical mean and variance per latent dimension
Z_target_mu = empirical_mean(H)      # What we want μ_z to predict
Z_target_var = empirical_variance(H)  # What we want σ_z to predict

# Solve via PIL
W_mu = ridge_solve(H, Z_target_mu)
W_sigma = ridge_solve(H, log(Z_target_var))

# In generate():
mu_z = W_mu.T @ h
sigma_z = exp(0.5 * W_sigma.T @ h)
z = mu_z + sigma_z * epsilon  # epsilon ~ N(0, I)
```

#### Benefits

- Adds **stochasticity** for diverse outputs
- Still uses PIL (no backprop)
- Enables sampling from learned distribution

---

### Option 2: Diffusion-Style Denoising (Score Matching)

#### Core Idea

Add a **denoising score function** solved via PIL:

$$s_\theta(x_t, t) \approx -\nabla_{x_t} \log p(x_t)$$

#### Training (PIL-style)

For noise level $t$, corrupt data:
$$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1-\alpha_t} \epsilon$$

Solve for denoising weights:
$$W_{denoise}^{(t)} = (H_t^T H_t + \lambda I)^{-1} H_t^T \epsilon$$

Where $H_t = \sigma(x_t W_{random})$

#### Generation

Iteratively denoise from pure noise:
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} s_\theta(x_t, t)\right) + \sigma_t z$$

#### Implementation

```python
class PILDiffusion:
    def __init__(self, timesteps=100):
        self.T = timesteps
        self.W_denoise = {}  # One weight matrix per timestep
        
    def fit(self, X):
        for t in range(self.T):
            alpha_t = self.alpha_schedule(t)
            
            # Add noise
            epsilon = np.random.randn(*X.shape)
            X_t = np.sqrt(alpha_t) * X + np.sqrt(1 - alpha_t) * epsilon
            
            # Feature expansion
            H_t = leaky_relu(X_t @ self.W_random)
            
            # PIL solve: predict the noise
            self.W_denoise[t] = ridge_solve(H_t, epsilon)
    
    def generate(self, n_samples):
        x = np.random.randn(n_samples, self.d)  # Pure noise
        
        for t in reversed(range(self.T)):
            H_t = leaky_relu(x @ self.W_random)
            epsilon_pred = H_t @ self.W_denoise[t]
            
            # Denoise step
            x = self.denoise_step(x, epsilon_pred, t)
        
        return x
```

#### Benefits

- State-of-the-art generative quality
- Each timestep uses PIL (no backprop)
- Can generate completely new samples

---

### Option 3: Autoregressive Token Generation

#### Core Idea

Transform to **next-token prediction** with PIL:

$$P(x_{t+1} | x_{1:t}) = \text{softmax}(W_{out} \cdot \sigma(x_{1:t} W_{random}))$$

#### Key Insight

We can solve $W_{out}$ via PIL to predict the next token!

$$W_{out} = (H^T H + \lambda I)^{-1} H^T Y_{onehot}$$

Where:
- $H$ = hidden activations from context
- $Y_{onehot}$ = one-hot encoded next tokens

#### Implementation

```python
class PILLanguageModel:
    def __init__(self, vocab_size, hidden_dim, context_length):
        self.vocab_size = vocab_size
        self.W_random = orthogonal_init((context_length * embed_dim, hidden_dim))
        self.W_out = None  # Solved via PIL
        
    def fit(self, sequences):
        """
        sequences: (N, seq_len) token IDs
        """
        contexts = []
        next_tokens = []
        
        for seq in sequences:
            for t in range(len(seq) - 1):
                # Context: all tokens up to t
                ctx = self.embed(seq[:t+1])  # (t+1, embed_dim) -> flatten
                contexts.append(ctx.flatten())
                
                # Target: next token (one-hot)
                next_tokens.append(one_hot(seq[t+1], self.vocab_size))
        
        X = np.stack(contexts)  # (N_samples, context_dim)
        Y = np.stack(next_tokens)  # (N_samples, vocab_size)
        
        # Feature expansion
        H = gelu(X @ self.W_random)  # (N_samples, hidden_dim)
        
        # PIL SOLVE for next-token prediction!
        self.W_out = ridge_solve(H, Y)
    
    def generate(self, prompt, max_tokens=100):
        tokens = prompt.copy()
        
        for _ in range(max_tokens):
            ctx = self.embed(tokens).flatten()
            h = gelu(ctx @ self.W_random)
            
            # Get logits
            logits = h @ self.W_out
            
            # Sample next token
            probs = softmax(logits / temperature)
            next_token = np.random.choice(self.vocab_size, p=probs)
            
            tokens.append(next_token)
            
            if next_token == EOS_TOKEN:
                break
        
        return tokens
```

#### Benefits

- Most GPT-like approach
- Token-by-token generation
- Still uses PIL for weight solving

---

### Option 4: Flow Matching

#### Core Idea

Use **Continuous Normalizing Flows** with PIL-solved velocity field:

$$\frac{dx}{dt} = v_\theta(x, t)$$

Where $v_\theta$ is solved via PIL to match the optimal transport from noise to data:

$$v^*(x, t) = \frac{x_1 - x_0}{1 - t} \quad \text{(linear interpolation target)}$$

#### Training

$$W_{velocity}^{(t)} = \text{ridge\_solve}(H_t, v^*)$$

#### Generation

Solve ODE from $t=0$ (noise) to $t=1$ (data):
$$x_1 = x_0 + \int_0^1 v_\theta(x_t, t) dt$$

#### Implementation

```python
class PILFlowMatching:
    def __init__(self, n_time_steps=50):
        self.T = n_time_steps
        self.W_velocity = {}
        
    def fit(self, X_data):
        for i, t in enumerate(np.linspace(0, 1, self.T)):
            # Sample noise
            X_noise = np.random.randn(*X_data.shape)
            
            # Interpolate: x_t = (1-t)*x_noise + t*x_data
            X_t = (1 - t) * X_noise + t * X_data
            
            # Target velocity: v* = x_data - x_noise
            V_target = X_data - X_noise
            
            # Feature expansion
            H_t = gelu(X_t @ self.W_random)
            
            # PIL solve for velocity field
            self.W_velocity[i] = ridge_solve(H_t, V_target)
    
    def generate(self, n_samples):
        x = np.random.randn(n_samples, self.d)  # Start from noise
        dt = 1.0 / self.T
        
        for i in range(self.T):
            H = gelu(x @ self.W_random)
            v = H @ self.W_velocity[i]
            x = x + v * dt  # Euler integration
        
        return x
```

#### Benefits

- Modern, elegant approach
- Simpler than diffusion (no noise schedules)
- Continuous-time generation

---

### Hybrid PIL-Transformer for Text Generation

Combine the best of both worlds:

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT TOKENS                                  │
│              [What] [is] [machine] [learning]                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                TOKEN EMBEDDINGS (Learned via backprop)           │
│                     (seq_len, 384)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│        CAUSAL ATTENTION (Learned via backprop - optional)        │
│                                                                  │
│        Q, K, V = x @ W_qkv                                       │
│        attn = softmax(mask(Q @ K.T / √d)) @ V                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              BI-PIL FFN (Solved via pseudoinverse)               │
│                                                                  │
│        H_fwd = GELU(x @ W_fwd)      ← W_fwd: Fixed random       │
│        H_bwd = GELU(x @ W_bwd)      ← W_bwd: Fixed random       │
│        H = concat(H_fwd, H_bwd)                                  │
│        out = H @ W_out              ← W_out: SOLVED via PIL     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                     × N layers
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           OUTPUT HEAD (Solved via PIL!)                          │
│                                                                  │
│        logits = hidden @ W_vocab                                 │
│        W_vocab = ridge_solve(H_all, Y_onehot)                   │
│                                                                  │
│        probs = softmax(logits / τ)                              │
│        next_token = sample(probs)                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    GENERATED TOKEN
```

---

### Mathematical Summary

| Component | Current (Retrieval) | Generative Modification |
|-----------|--------------------|-----------------------|
| **Latent** | $z = W^T(h - \mu)$ deterministic | $z \sim \mathcal{N}(W_\mu^T h, \exp(W_\sigma^T h))$ |
| **Decoder** | $\hat{x} = W_6 W_4 z$ reconstruct | $P(x\|z) = \text{softmax}(W_{out} \sigma(zW_r))$ sample |
| **Output** | Retrieve & compose | Autoregressive: $P(x_t\|x_{<t})$ |
| **Training** | One-shot PIL | Still one-shot PIL per component! |

---

### Simplest Immediate Change

Add **sampling with temperature** to the existing VAE:

```python
def generate_text(self, prompt, temperature=0.8, top_k=50):
    # 1. Encode prompt
    prompt_vec = self.embedder.encode(prompt)
    z = self.vae.encode(prompt_vec)
    
    # 2. Add controlled noise for diversity
    z_noisy = z + temperature * np.random.randn(*z.shape)
    
    # 3. Decode to embedding space
    e_gen = self.vae.decode(z_noisy)
    
    # 4. Find k nearest neighbors
    docs = self.memory.retrieve(e_gen, top_k=top_k)
    
    # 5. SAMPLE from retrieved docs (weighted by score)
    scores = np.array([d['score'] for d in docs])
    probs = softmax(scores / temperature)
    selected_idx = np.random.choice(len(docs), p=probs)
    
    # 6. Use selected doc as seed for response
    return self._compose_response(docs, selected_idx)
```

This adds **stochasticity** while keeping the PIL architecture intact!

---

## References

1. **SONG Paper:** "Synergetic Learning System Based on Swarm of Non-Gradient Learners"
2. **Bi-PIL Paper:** "Bi-PIL: Bidirectional Gradient-Free Learning Scheme"
3. **HRA Paper:** "Bridging The Gap between Low-rank and Orthogonal Adaptation via Householder Reflection Adaptation" (NeurIPS 2024)
4. **ELM:** Extreme Learning Machines - Random feature expansion + linear solving
5. **Flow Matching:** "Flow Matching for Generative Modeling" (Lipman et al., 2023)

---

*Document generated for Project Emergent-1 PIL Architecture*
