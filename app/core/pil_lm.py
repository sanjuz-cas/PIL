"""
PIL Language Model: Hybrid Transformer with PIL FFN Layers

A generative language model that replaces gradient-based FFN training
with pseudoinverse learning (PIL), while optionally using gradients
for attention and embeddings.

Architecture:
    Input Tokens → Embeddings → [Attention + Bi-PIL FFN] × N → PIL Output Head → Next Token

Key Innovation:
    - FFN layers solved via closed-form pseudoinverse (no backprop)
    - Output vocabulary projection solved via PIL
    - Only attention/embeddings use gradient descent (optional)

Reference:
    - "Bi-PIL: Bidirectional Gradient-Free Learning Scheme"
    - Project Emergent-1: Non-Gradient Learning Systems
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, List, Literal
from dataclasses import dataclass
import math


@dataclass
class PILLMConfig:
    """Configuration for PIL Language Model."""

    # Model dimensions
    vocab_size: int = 50257  # GPT-2 vocab size
    max_seq_len: int = 512
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 4

    # FFN dimensions
    ffn_expansion: int = 4  # FFN hidden = embed_dim * expansion

    # PIL parameters
    lambda_reg: float = 1e-5
    use_bipil: bool = True
    bipil_fusion: Literal["concat", "add"] = "concat"

    # Training mode
    train_embeddings: bool = True  # Use gradients for embeddings
    train_attention: bool = True  # Use gradients for attention

    # Generation
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9

    # Misc
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    seed: int = 42


def orthogonal_init(shape: Tuple[int, ...], seed: Optional[int] = None) -> torch.Tensor:
    """Initialize with orthogonal matrix for better conditioning."""
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    rows, cols = shape
    if rows < cols:
        M = torch.randn(cols, rows, generator=generator)
        Q, R = torch.linalg.qr(M)
        Q = Q.T[:rows, :]
    else:
        M = torch.randn(rows, cols, generator=generator)
        Q, R = torch.linalg.qr(M)
        Q = Q[:, :cols]

    return Q


def ridge_solve(
    H: torch.Tensor, Y: torch.Tensor, lambda_reg: float = 1e-5
) -> torch.Tensor:
    """
    Solve ridge regression: W = (H^T H + λI)^{-1} H^T Y

    Args:
        H: Hidden activations (N, hidden_dim)
        Y: Targets (N, output_dim)
        lambda_reg: Regularization parameter

    Returns:
        W: Solved weights (hidden_dim, output_dim)
    """
    N, hidden_dim = H.shape
    device = H.device
    dtype = H.dtype

    if N >= hidden_dim:
        # Standard form
        HtH = H.T @ H
        regularized = HtH + lambda_reg * torch.eye(
            hidden_dim, device=device, dtype=dtype
        )
        HtY = H.T @ Y

        try:
            L = torch.linalg.cholesky(regularized)
            W = torch.cholesky_solve(HtY, L)
        except RuntimeError:
            W = torch.linalg.solve(regularized, HtY)
    else:
        # Dual form for N < hidden_dim
        HHt = H @ H.T
        regularized = HHt + lambda_reg * torch.eye(N, device=device, dtype=dtype)

        try:
            L = torch.linalg.cholesky(regularized)
            temp = torch.cholesky_solve(Y, L)
        except RuntimeError:
            temp = torch.linalg.solve(regularized, Y)

        W = H.T @ temp

    return W


class BiPILFFN(nn.Module):
    """
    Bidirectional Pseudoinverse Learning Feed-Forward Network.

    Replaces standard FFN:
        Standard: h = GELU(x @ W1) @ W2  (both W1, W2 learned via backprop)

    With PIL:
        H_fwd = GELU(x @ W_fwd)   ← W_fwd: Fixed random
        H_bwd = GELU(x @ W_bwd)   ← W_bwd: Fixed random
        H = concat(H_fwd, H_bwd)
        out = H @ W_out           ← W_out: Solved via PIL
    """

    def __init__(
        self,
        embed_dim: int,
        expansion: int = 4,
        lambda_reg: float = 1e-5,
        use_bipil: bool = True,
        fusion: Literal["concat", "add"] = "concat",
        seed: Optional[int] = None,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = embed_dim * expansion
        self.lambda_reg = lambda_reg
        self.use_bipil = use_bipil
        self.fusion = fusion

        # Fused dimension
        if use_bipil and fusion == "concat":
            self.fused_dim = self.hidden_dim * 2
        else:
            self.fused_dim = self.hidden_dim

        # Forward random projection (FIXED)
        self.register_buffer(
            "W_fwd", orthogonal_init((embed_dim, self.hidden_dim), seed=seed)
        )

        # Backward random projection (FIXED) - for BiPIL
        if use_bipil:
            self.register_buffer(
                "W_bwd",
                orthogonal_init(
                    (embed_dim, self.hidden_dim), seed=seed + 1 if seed else None
                ),
            )

        # Output weights (SOLVED via PIL)
        self.register_buffer("W_out", torch.zeros(self.fused_dim, embed_dim))

        # Bias
        self.register_buffer("bias", torch.zeros(embed_dim))

        # Layer norm
        self.layer_norm = nn.LayerNorm(embed_dim)

        # State
        self._is_fitted = False

    def _expand(self, x: torch.Tensor) -> torch.Tensor:
        """Feature expansion with GELU activation."""
        H_fwd = F.gelu(x @ self.W_fwd)

        if self.use_bipil:
            H_bwd = F.gelu(x @ self.W_bwd)

            if self.fusion == "concat":
                return torch.cat([H_fwd, H_bwd], dim=-1)
            else:
                return H_fwd + H_bwd

        return H_fwd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input (batch, seq_len, embed_dim)

        Returns:
            Output (batch, seq_len, embed_dim)
        """
        residual = x

        # Feature expansion
        H = self._expand(x)

        # Output projection
        out = H @ self.W_out + self.bias

        # Residual + LayerNorm
        out = self.layer_norm(out + residual)

        return out

    @torch.no_grad()
    def fit(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict:
        """
        Solve for W_out using pseudoinverse.

        Args:
            x: Input tensor (N, embed_dim) - flattened
            target: Target tensor. If None, learns identity (residual)

        Returns:
            Fit statistics
        """
        if target is None:
            target = x  # Identity mapping for residual learning

        # Flatten if needed
        if x.dim() > 2:
            orig_shape = x.shape
            x = x.reshape(-1, self.embed_dim)
            target = target.reshape(-1, self.embed_dim)

        # Feature expansion
        H = self._expand(x)

        # Solve via PIL
        W_new = ridge_solve(H, target, self.lambda_reg)

        # Update weights
        self.W_out.copy_(W_new)

        # Compute bias
        residual = target - (H @ self.W_out)
        self.bias.copy_(residual.mean(dim=0))

        self._is_fitted = True

        # Statistics
        out = H @ self.W_out + self.bias
        mse = F.mse_loss(out, target).item()

        return {"success": True, "mse": mse}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class CausalSelfAttention(nn.Module):
    """
    Causal Self-Attention layer.

    Can be trained via backprop (standard) or kept frozen.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        trainable: bool = True,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        # QKV projection
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

        # Trainability
        if not trainable:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking.

        Args:
            x: Input (batch, seq_len, embed_dim)

        Returns:
            Output (batch, seq_len, embed_dim)
        """
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))

        # Softmax
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply to values
        out = attn @ v

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        out = self.out_proj(out)

        return out


class PILTransformerBlock(nn.Module):
    """
    Single Transformer block with PIL FFN.

    Structure:
        x → LayerNorm → Attention → + → LayerNorm → BiPIL FFN → +
            ↑__________________________|     ↑___________________|
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_expansion: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        lambda_reg: float = 1e-5,
        use_bipil: bool = True,
        train_attention: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Pre-norm for attention
        self.ln1 = nn.LayerNorm(embed_dim)

        # Causal self-attention
        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            trainable=train_attention,
        )

        # Pre-norm for FFN
        self.ln2 = nn.LayerNorm(embed_dim)

        # Bi-PIL FFN (gradient-free!)
        self.ffn = BiPILFFN(
            embed_dim=embed_dim,
            expansion=ffn_expansion,
            lambda_reg=lambda_reg,
            use_bipil=use_bipil,
            seed=seed,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Attention with residual
        x = x + self.dropout(self.attention(self.ln1(x)))

        # FFN with residual (handled inside FFN)
        x = self.ffn(self.ln2(x))

        return x

    @torch.no_grad()
    def fit_ffn(self, x: torch.Tensor, target: Optional[torch.Tensor] = None) -> Dict:
        """Fit the FFN layer via PIL."""
        # Apply attention first (frozen or trained)
        x_attn = x + self.attention(self.ln1(x))
        x_normed = self.ln2(x_attn)

        # Fit FFN
        return self.ffn.fit(x_normed, target if target is not None else x_normed)


class PILOutputHead(nn.Module):
    """
    Output head that maps hidden states to vocabulary logits.

    Solved via PIL: W_vocab = (H^T H + λI)^{-1} H^T Y_onehot
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        lambda_reg: float = 1e-5,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.lambda_reg = lambda_reg

        # Vocabulary projection (SOLVED via PIL)
        self.register_buffer("W_vocab", torch.zeros(embed_dim, vocab_size))

        self.register_buffer("bias", torch.zeros(vocab_size))

        self._is_fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits.

        Args:
            x: Hidden states (batch, seq_len, embed_dim)

        Returns:
            Logits (batch, seq_len, vocab_size)
        """
        return x @ self.W_vocab + self.bias

    @torch.no_grad()
    def fit(self, hidden: torch.Tensor, target_ids: torch.Tensor) -> Dict:
        """
        Fit vocabulary projection via PIL (memory-efficient).

        Uses chunked computation to avoid OOM with large vocabularies.
        Instead of creating full one-hot matrix, we solve column-by-column
        for active vocabulary tokens only.

        Args:
            hidden: Hidden states (N, embed_dim)
            target_ids: Target token IDs (N,)

        Returns:
            Fit statistics
        """
        N, D = hidden.shape
        device = hidden.device
        dtype = hidden.dtype

        # Get unique tokens in target (much smaller than full vocab)
        unique_tokens = torch.unique(target_ids)
        num_active = len(unique_tokens)

        print(
            f"    Fitting output head: {N} samples, {num_active}/{self.vocab_size} active tokens"
        )

        # Compute H^T H + λI (shared for all columns)
        HtH = hidden.T @ hidden
        regularized = HtH + self.lambda_reg * torch.eye(D, device=device, dtype=dtype)

        # Cholesky decomposition (solved once, reused)
        try:
            L = torch.linalg.cholesky(regularized)
            use_cholesky = True
        except RuntimeError:
            use_cholesky = False

        # Initialize W_vocab to zeros
        W_new = torch.zeros(D, self.vocab_size, device=device, dtype=dtype)

        # Solve for each active token's column (memory efficient)
        chunk_size = 1000  # Process tokens in chunks

        for i in range(0, num_active, chunk_size):
            chunk_tokens = unique_tokens[i : min(i + chunk_size, num_active)]

            # Create sparse target for this chunk
            # Y_chunk[j, k] = 1 if target_ids[j] == chunk_tokens[k]
            Y_chunk = (target_ids.unsqueeze(1) == chunk_tokens.unsqueeze(0)).float()

            # Solve: W_chunk = (H^T H + λI)^{-1} H^T Y_chunk
            HtY = hidden.T @ Y_chunk

            if use_cholesky:
                W_chunk = torch.cholesky_solve(HtY, L)
            else:
                W_chunk = torch.linalg.solve(regularized, HtY)

            # Store in correct columns
            W_new[:, chunk_tokens] = W_chunk

        # Update weights
        self.W_vocab.copy_(W_new)

        # Compute bias efficiently (in chunks to avoid OOM)
        self.bias.zero_()
        bias_chunk_size = 2000
        for i in range(0, num_active, bias_chunk_size):
            chunk_tokens = unique_tokens[i:min(i + bias_chunk_size, num_active)]
            logits_chunk = hidden @ W_new[:, chunk_tokens]
            Y_chunk = (target_ids.unsqueeze(1) == chunk_tokens.unsqueeze(0)).float()
            residual = Y_chunk - logits_chunk
            self.bias[chunk_tokens] = residual.mean(dim=0)
            del logits_chunk, Y_chunk, residual
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        self._is_fitted = True

        # Statistics (compute accuracy on small sample to avoid OOM)
        sample_size = min(5000, N)
        sample_idx = torch.randperm(N, device=device)[:sample_size]
        
        # Compute accuracy in chunks
        correct = 0
        acc_chunk_size = 1000
        for i in range(0, sample_size, acc_chunk_size):
            idx_chunk = sample_idx[i:min(i + acc_chunk_size, sample_size)]
            logits_chunk = hidden[idx_chunk] @ self.W_vocab + self.bias
            pred_chunk = logits_chunk.argmax(dim=-1)
            correct += (pred_chunk == target_ids[idx_chunk]).sum().item()
            del logits_chunk, pred_chunk
        
        accuracy = correct / sample_size

        return {"success": True, "accuracy": accuracy, "active_tokens": num_active}

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


class PILLanguageModel(nn.Module):
    """
    Hybrid PIL Language Model for Text Generation.

    Architecture:
        Tokens → Embeddings → [Attention + BiPIL FFN] × N → PIL Head → Next Token

    Training:
        - Embeddings: Gradients (optional)
        - Attention: Gradients (optional)
        - FFN layers: PIL (one-shot, no gradients)
        - Output head: PIL (one-shot, no gradients)
    """

    def __init__(self, config: PILLMConfig):
        super().__init__()

        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        # Position embeddings
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                PILTransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ffn_expansion=config.ffn_expansion,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                    lambda_reg=config.lambda_reg,
                    use_bipil=config.use_bipil,
                    train_attention=config.train_attention,
                    seed=config.seed + i if config.seed else None,
                )
                for i in range(config.num_layers)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Output head (PIL)
        self.output_head = PILOutputHead(
            embed_dim=config.embed_dim,
            vocab_size=config.vocab_size,
            lambda_reg=config.lambda_reg,
        )

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # Set trainability
        if not config.train_embeddings:
            self.token_embedding.weight.requires_grad = False
            self.position_embedding.weight.requires_grad = False

        # Initialize
        self._init_weights()

        # State
        self._pil_fitted = False

    def _init_weights(self):
        """Initialize weights."""
        # Standard initialization for embeddings
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            return_hidden: Return hidden states instead of logits

        Returns:
            Logits (batch, seq_len, vocab_size) or hidden states
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings
        tok_emb = self.token_embedding(input_ids)

        # Position embeddings
        positions = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)

        # Combined embeddings
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        if return_hidden:
            return x

        # Output logits
        logits = self.output_head(x)

        return logits

    @torch.no_grad()
    def fit_pil_layers(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        verbose: bool = True,
    ) -> Dict:
        """
        Fit all PIL layers (FFNs and output head) in one shot.

        This is the main training function for PIL components.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            target_ids: Target token IDs (batch, seq_len) - shifted by 1
            verbose: Print progress

        Returns:
            Training statistics
        """
        B, T = input_ids.shape
        device = input_ids.device

        if verbose:
            print(f"Fitting PIL layers: {B * T} tokens")

        stats = {"ffn_mse": [], "head_accuracy": 0}

        # Get embeddings
        tok_emb = self.token_embedding(input_ids)
        positions = torch.arange(T, device=device).unsqueeze(0)
        pos_emb = self.position_embedding(positions)
        x = tok_emb + pos_emb

        # Fit each block's FFN
        for i, block in enumerate(self.blocks):
            # Forward through attention
            x_attn = x + block.attention(block.ln1(x))
            x_normed = block.ln2(x_attn)

            # Flatten for PIL fitting
            x_flat = x_normed.reshape(-1, self.config.embed_dim)

            # Fit FFN (identity mapping for residual learning)
            ffn_stats = block.ffn.fit(x_flat)
            stats["ffn_mse"].append(ffn_stats.get("mse", 0))

            if verbose:
                print(
                    f"  Block {i + 1}/{self.config.num_layers}: FFN MSE = {ffn_stats.get('mse', 0):.6f}"
                )

            # Forward through fitted FFN
            x = block.ffn(x_normed)

        # Final layer norm
        x = self.ln_f(x)

        # Fit output head
        x_flat = x.reshape(-1, self.config.embed_dim)
        target_flat = target_ids.reshape(-1)

        head_stats = self.output_head.fit(x_flat, target_flat)
        stats["head_accuracy"] = head_stats.get("accuracy", 0)

        if verbose:
            print(f"  Output head accuracy: {stats['head_accuracy']:.4f}")

        self._pil_fitted = True

        return stats

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate new tokens autoregressively.

        Args:
            input_ids: Prompt token IDs (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (False = greedy)

        Returns:
            Generated token IDs (batch, seq_len + max_new_tokens)
        """
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p

        for _ in range(max_new_tokens):
            # Crop to max length
            idx_cond = input_ids[:, -self.config.max_seq_len :]

            # Forward pass
            logits = self(idx_cond)

            # Get last token logits
            logits = logits[:, -1, :] / temperature

            if do_sample:
                # Top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                        :, :-1
                    ].clone()
                    sorted_indices_to_remove[:, 0] = 0

                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                # Sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = logits.argmax(dim=-1, keepdim=True)

            # Append
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params

    @property
    def pil_fitted(self) -> bool:
        return self._pil_fitted
