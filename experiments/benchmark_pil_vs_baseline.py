"""
Benchmark: PIL-LM vs Baseline Transformer

Compares:
1. Training time (wall-clock)
2. FLOPS estimation
3. Final perplexity
4. Generation quality

Outputs results in LaTeX table format for paper.

Usage:
    python experiments/benchmark_pil_vs_baseline.py --dataset wikitext --device cuda
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.pil_lm import PILLanguageModel, PILLMConfig
from app.core.baseline_transformer import BaselineTransformer, BaselineConfig


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    # Model
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 128

    # Training
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 5  # For baseline
    max_train_samples: int = 5000
    max_eval_samples: int = 1000

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    output_dir: str = "outputs/benchmark"


@dataclass
class BenchmarkResult:
    """Results from one model benchmark."""

    model_name: str
    num_params: int

    # Timing
    train_time_seconds: float
    tokens_per_second: float

    # Metrics
    final_loss: float
    final_perplexity: float
    top1_accuracy: float
    top5_accuracy: float

    # Details
    epochs_trained: int
    total_tokens: int


def load_tokenizer():
    """Load GPT-2 tokenizer."""
    try:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")


def load_dataset(dataset_name: str, max_train: int, max_eval: int):
    """Load dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        if dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            text_col = "text"
        elif dataset_name == "tinystories":
            ds = load_dataset("roneneldan/TinyStories")
            text_col = "text"
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        train_texts = [
            t
            for t in ds["train"][text_col][: max_train * 2]
            if t and len(t.strip()) > 20
        ][:max_train]
        eval_texts = [
            t
            for t in ds["validation"][text_col][: max_eval * 2]
            if t and len(t.strip()) > 20
        ][:max_eval]

        return train_texts, eval_texts
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")


def tokenize_data(
    texts: List[str],
    tokenizer,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize texts for language modeling."""
    all_input_ids = []
    all_target_ids = []

    for text in tqdm(texts, desc="Tokenizing", leave=False):
        tokens = tokenizer.encode(text, add_special_tokens=True)

        if len(tokens) < 10:
            continue

        # Truncate
        if len(tokens) > max_length + 1:
            tokens = tokens[: max_length + 1]

        # Pad
        if len(tokens) < max_length + 1:
            tokens = tokens + [tokenizer.pad_token_id] * (max_length + 1 - len(tokens))

        all_input_ids.append(tokens[:-1])
        all_target_ids.append(tokens[1:])

    return (
        torch.tensor(all_input_ids, dtype=torch.long),
        torch.tensor(all_target_ids, dtype=torch.long),
    )


def compute_metrics(
    model: nn.Module,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    batch_size: int,
    device: str,
    is_baseline: bool = False,
) -> Dict:
    """Compute evaluation metrics."""
    model.eval()

    total_loss = 0.0
    total_correct_top1 = 0
    total_correct_top5 = 0
    total_tokens = 0

    num_batches = (len(input_ids) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(input_ids))

            batch_input = input_ids[start:end].to(device)
            batch_target = target_ids[start:end].to(device)

            if is_baseline:
                result = model(batch_input, batch_target)
                logits = result["logits"]
            else:
                logits = model(batch_input)

            # Flatten
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = batch_target.reshape(-1)

            # Mask padding
            mask = target_flat != 50256  # GPT-2 EOS/PAD

            if mask.sum() > 0:
                loss = F.cross_entropy(logits_flat[mask], target_flat[mask])
                total_loss += loss.item() * mask.sum().item()

                # Top-1
                pred = logits_flat[mask].argmax(dim=-1)
                total_correct_top1 += (pred == target_flat[mask]).sum().item()

                # Top-5
                top5 = logits_flat[mask].topk(5, dim=-1).indices
                total_correct_top5 += (
                    (top5 == target_flat[mask].unsqueeze(-1)).any(dim=-1).sum().item()
                )

                total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 100))

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "top1_accuracy": total_correct_top1 / max(total_tokens, 1),
        "top5_accuracy": total_correct_top5 / max(total_tokens, 1),
        "num_tokens": total_tokens,
    }


def train_baseline(
    model: BaselineTransformer,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    eval_input: torch.Tensor,
    eval_target: torch.Tensor,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Train baseline transformer with AdamW."""
    device = config.device
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.01,
    )

    # Create dataloader
    dataset = TensorDataset(train_input, train_target)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    total_tokens = len(train_input) * train_input.shape[1]

    print("\n" + "=" * 60)
    print("Training Baseline Transformer (AdamW)")
    print("=" * 60)

    start_time = time.perf_counter()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch_input, batch_target in pbar:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()
            result = model(batch_input, batch_target)
            loss = result["loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Eval
        metrics = compute_metrics(
            model, eval_input, eval_target, config.batch_size, device, is_baseline=True
        )
        print(
            f"  Epoch {epoch + 1}: Loss={metrics['loss']:.4f}, PPL={metrics['perplexity']:.2f}, Acc={metrics['top1_accuracy']:.4f}"
        )

    train_time = time.perf_counter() - start_time

    # Final eval
    final_metrics = compute_metrics(
        model, eval_input, eval_target, config.batch_size, device, is_baseline=True
    )

    return BenchmarkResult(
        model_name="Baseline (AdamW)",
        num_params=model.get_num_params(),
        train_time_seconds=train_time,
        tokens_per_second=(total_tokens * config.num_epochs) / train_time,
        final_loss=final_metrics["loss"],
        final_perplexity=final_metrics["perplexity"],
        top1_accuracy=final_metrics["top1_accuracy"],
        top5_accuracy=final_metrics["top5_accuracy"],
        epochs_trained=config.num_epochs,
        total_tokens=total_tokens * config.num_epochs,
    )


def train_pil(
    model: PILLanguageModel,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    eval_input: torch.Tensor,
    eval_target: torch.Tensor,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Train PIL model with one-shot FFN + gradient attention."""
    device = config.device
    model = model.to(device)

    total_tokens = len(train_input) * train_input.shape[1]

    print("\n" + "=" * 60)
    print("Training PIL-LM (One-Shot FFN + Attention Fine-tune)")
    print("=" * 60)

    start_time = time.perf_counter()

    # Phase 1: PIL Training (One-Shot)
    print("\nPhase 1: PIL One-Shot Training")
    pil_input = train_input.to(device)
    pil_target = train_target.to(device)

    pil_stats = model.fit_pil_layers(pil_input, pil_target, verbose=True)

    pil_time = time.perf_counter() - start_time
    print(f"  PIL phase completed in {pil_time:.2f}s")

    # Phase 2: Attention Fine-tuning
    print("\nPhase 2: Attention Fine-tuning")

    # Only train attention parameters
    attn_params = []
    for block in model.blocks:
        attn_params.extend(block.attention.parameters())
        attn_params.extend(block.ln1.parameters())

    optimizer = torch.optim.AdamW(attn_params, lr=config.learning_rate)

    dataset = TensorDataset(train_input, train_target)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    for epoch in range(config.num_epochs):
        model.train()

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.num_epochs}")
        for batch_input, batch_target in pbar:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)

            optimizer.zero_grad()

            logits = model(batch_input)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_target.view(-1),
                ignore_index=50256,
            )
            loss.backward()

            torch.nn.utils.clip_grad_norm_(attn_params, 1.0)
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = compute_metrics(
            model, eval_input, eval_target, config.batch_size, device, is_baseline=False
        )
        print(
            f"  Epoch {epoch + 1}: Loss={metrics['loss']:.4f}, PPL={metrics['perplexity']:.2f}, Acc={metrics['top1_accuracy']:.4f}"
        )

    train_time = time.perf_counter() - start_time

    # Final eval
    final_metrics = compute_metrics(
        model, eval_input, eval_target, config.batch_size, device, is_baseline=False
    )

    return BenchmarkResult(
        model_name="PIL-LM (One-Shot + Attn)",
        num_params=model.get_num_params(),
        train_time_seconds=train_time,
        tokens_per_second=(total_tokens * (config.num_epochs + 1)) / train_time,
        final_loss=final_metrics["loss"],
        final_perplexity=final_metrics["perplexity"],
        top1_accuracy=final_metrics["top1_accuracy"],
        top5_accuracy=final_metrics["top5_accuracy"],
        epochs_trained=config.num_epochs + 1,  # +1 for PIL phase
        total_tokens=total_tokens * (config.num_epochs + 1),
    )


def train_pil_only(
    model: PILLanguageModel,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    eval_input: torch.Tensor,
    eval_target: torch.Tensor,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """Train PIL model with ONLY one-shot PIL (no attention fine-tuning)."""
    device = config.device
    model = model.to(device)

    total_tokens = len(train_input) * train_input.shape[1]

    print("\n" + "=" * 60)
    print("Training PIL-LM (PURE One-Shot, No Gradients)")
    print("=" * 60)

    start_time = time.perf_counter()

    # PIL Training ONLY
    pil_input = train_input.to(device)
    pil_target = train_target.to(device)

    pil_stats = model.fit_pil_layers(pil_input, pil_target, verbose=True)

    train_time = time.perf_counter() - start_time
    print(f"  PIL training completed in {train_time:.2f}s")

    # Final eval
    final_metrics = compute_metrics(
        model, eval_input, eval_target, config.batch_size, device, is_baseline=False
    )

    print(
        f"  Final: Loss={final_metrics['loss']:.4f}, PPL={final_metrics['perplexity']:.2f}"
    )

    return BenchmarkResult(
        model_name="PIL-LM (Pure One-Shot)",
        num_params=model.get_num_params(),
        train_time_seconds=train_time,
        tokens_per_second=total_tokens / train_time,
        final_loss=final_metrics["loss"],
        final_perplexity=final_metrics["perplexity"],
        top1_accuracy=final_metrics["top1_accuracy"],
        top5_accuracy=final_metrics["top5_accuracy"],
        epochs_trained=1,
        total_tokens=total_tokens,
    )


def format_latex_table(results: List[BenchmarkResult]) -> str:
    """Format results as LaTeX table."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Comparison of PIL-LM vs Baseline Transformer}",
        r"\label{tab:benchmark}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model & Params & Time (s) & PPL $\downarrow$ & Top-1 Acc $\uparrow$ & Tokens/s $\uparrow$ \\",
        r"\midrule",
    ]

    for r in results:
        lines.append(
            f"{r.model_name} & {r.num_params:,} & {r.train_time_seconds:.1f} & "
            f"{r.final_perplexity:.2f} & {r.top1_accuracy:.2%} & {r.tokens_per_second:.0f} \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def format_markdown_table(results: List[BenchmarkResult]) -> str:
    """Format results as Markdown table."""
    lines = [
        "| Model | Params | Time (s) | Perplexity | Top-1 Acc | Tokens/s |",
        "|-------|--------|----------|------------|-----------|----------|",
    ]

    for r in results:
        lines.append(
            f"| {r.model_name} | {r.num_params:,} | {r.train_time_seconds:.1f} | "
            f"{r.final_perplexity:.2f} | {r.top1_accuracy:.2%} | {r.tokens_per_second:.0f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Benchmark PIL vs Baseline")
    parser.add_argument(
        "--dataset", type=str, default="wikitext", choices=["wikitext", "tinystories"]
    )
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--max_train_samples", type=int, default=5000)
    parser.add_argument("--max_eval_samples", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmark")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    config = BenchmarkConfig(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        batch_size=args.batch_size,
        device=device,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("PIL vs Baseline Benchmark")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {config.num_layers}L, {config.embed_dim}D, {config.num_heads}H")
    print(f"Device: {device}")
    print(f"Epochs: {config.num_epochs}")
    print("=" * 60)

    # Load data
    print("\nLoading tokenizer...")
    tokenizer = load_tokenizer()

    print(f"Loading dataset: {args.dataset}")
    train_texts, eval_texts = load_dataset(
        args.dataset, config.max_train_samples, config.max_eval_samples
    )
    print(f"  Train: {len(train_texts)}, Eval: {len(eval_texts)}")

    print("\nTokenizing...")
    train_input, train_target = tokenize_data(
        train_texts, tokenizer, config.max_seq_len
    )
    eval_input, eval_target = tokenize_data(eval_texts, tokenizer, config.max_seq_len)
    print(f"  Train shape: {train_input.shape}")
    print(f"  Eval shape: {eval_input.shape}")

    results = []

    # === Baseline Transformer ===
    baseline_config = BaselineConfig(
        vocab_size=50257,
        max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    baseline_model = BaselineTransformer(baseline_config)
    print(f"\nBaseline params: {baseline_model.get_num_params():,}")

    baseline_result = train_baseline(
        baseline_model, train_input, train_target, eval_input, eval_target, config
    )
    results.append(baseline_result)

    # Clear GPU memory
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # === PIL-LM (One-Shot Only) ===
    pil_config_pure = PILLMConfig(
        vocab_size=50257,
        max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        use_bipil=True,
        train_attention=False,  # Don't train attention
    )
    pil_model_pure = PILLanguageModel(pil_config_pure)

    pil_pure_result = train_pil_only(
        pil_model_pure, train_input, train_target, eval_input, eval_target, config
    )
    results.append(pil_pure_result)

    del pil_model_pure
    if device == "cuda":
        torch.cuda.empty_cache()

    # === PIL-LM (One-Shot + Attention) ===
    pil_config = PILLMConfig(
        vocab_size=50257,
        max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        use_bipil=True,
        train_attention=True,
    )
    pil_model = PILLanguageModel(pil_config)

    pil_result = train_pil(
        pil_model, train_input, train_target, eval_input, eval_target, config
    )
    results.append(pil_result)

    # === Results Summary ===
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    print("\n" + format_markdown_table(results))

    print("\n\nLaTeX Table:")
    print(format_latex_table(results))

    # Compute speedup
    baseline_time = results[0].train_time_seconds
    pil_pure_time = results[1].train_time_seconds
    pil_full_time = results[2].train_time_seconds

    print("\n\nSpeedup Analysis:")
    print(f"  PIL Pure vs Baseline: {baseline_time / pil_pure_time:.1f}x faster")
    print(f"  PIL Full vs Baseline: {baseline_time / pil_full_time:.1f}x faster")

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)

    results_dict = {
        "config": asdict(config),
        "results": [asdict(r) for r in results],
    }

    with open(os.path.join(config.output_dir, "benchmark_results.json"), "w") as f:
        json.dump(results_dict, f, indent=2)

    with open(os.path.join(config.output_dir, "benchmark_table.md"), "w") as f:
        f.write("# PIL vs Baseline Benchmark Results\n\n")
        f.write(format_markdown_table(results))
        f.write("\n\n## LaTeX\n\n```latex\n")
        f.write(format_latex_table(results))
        f.write("\n```\n")

    print(f"\nResults saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
