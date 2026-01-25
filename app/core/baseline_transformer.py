"""
Baseline Transformer: Standard GPT-style model trained with AdamW.

This serves as the comparison baseline for PIL experiments.
Same architecture as PIL-LM but with standard gradient-based FFN training.

Usage:
    model = BaselineTransformer(config)
    model.train()
    for batch in dataloader:
        loss = model(input_ids, target_ids)
        loss.backward()
        optimizer.step()
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class BaselineConfig:
    """Configuration for Baseline Transformer (matches PILLMConfig)."""

    vocab_size: int = 50257
    max_seq_len: int = 512
    embed_dim: int = 384
    num_heads: int = 6
    num_layers: int = 4
    ffn_expansion: int = 4
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    weight_tying: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9


class CausalSelfAttention(nn.Module):
    """Standard causal self-attention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)

        return out


class StandardFFN(nn.Module):
    """Standard Feed-Forward Network with GELU activation."""

    def __init__(
        self,
        embed_dim: int,
        expansion: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = embed_dim * expansion

        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block with pre-norm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_expansion: int = 4,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )

        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = StandardFFN(
            embed_dim=embed_dim,
            expansion=ffn_expansion,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.ffn(self.ln2(x)))
        return x


class BaselineTransformer(nn.Module):
    """
    Standard GPT-style Transformer trained with backpropagation.

    This is the baseline for comparison with PIL-LM.
    """

    def __init__(self, config: BaselineConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=config.embed_dim,
                    num_heads=config.num_heads,
                    ffn_expansion=config.ffn_expansion,
                    max_seq_len=config.max_seq_len,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.embed_dim)

        # Output head (weight tied or separate)
        if config.weight_tying:
            self.lm_head = None  # Use embedding weights
        else:
            self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (batch, seq_len)
            targets: Target token IDs for loss computation (batch, seq_len)

        Returns:
            Dict with 'logits' and optionally 'loss'
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Embeddings
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = self.dropout(tok_emb + pos_emb)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Output projection
        if self.config.weight_tying:
            logits = F.linear(x, self.token_embedding.weight)
        else:
            logits = self.lm_head(x)

        result = {"logits": logits}

        # Compute loss if targets provided
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = None,
        top_k: int = None,
        top_p: float = None,
        do_sample: bool = True,
        eos_token_id: int = 50256,
        repetition_penalty: float = 1.2,
    ) -> torch.Tensor:
        """Generate new tokens autoregressively."""
        temperature = temperature or self.config.temperature
        top_k = top_k or self.config.top_k
        top_p = top_p or self.config.top_p

        batch_size = input_ids.shape[0]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            idx_cond = input_ids[:, -self.config.max_seq_len:]

            result = self(idx_cond)
            logits = result["logits"][:, -1, :] / temperature

            # Repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token in input_ids[i].unique():
                        if logits[i, token] > 0:
                            logits[i, token] /= repetition_penalty
                        else:
                            logits[i, token] *= repetition_penalty

            if do_sample:
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        F.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, eos_token_id),
                next_token,
            )

            input_ids = torch.cat([input_ids, next_token], dim=1)
            finished = finished | (next_token.squeeze(-1) == eos_token_id)

            if finished.all():
                break

        return input_ids

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()
        return n_params
