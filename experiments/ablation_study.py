"""
Ablation Study: Understanding PIL Design Choices

Studies:
1. BiPIL vs Single-PIL (bidirectional vs unidirectional random projections)
2. Lambda (regularization) sensitivity
3. Expansion ratio (FFN hidden dim / embed_dim)
4. Number of subsampled tokens for PIL fitting

Outputs tables and plots for paper.

Usage:
    python experiments/ablation_study.py --study all --device cuda
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.pil_lm import PILLanguageModel, PILLMConfig


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""

    # Base model config
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 128

    # Training
    batch_size: int = 32
    max_train_samples: int = 3000
    max_eval_samples: int = 500
    num_epochs: int = 3  # For attention fine-tuning

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Output
    output_dir: str = "outputs/ablations"


def load_data(config: AblationConfig):
    """Load and prepare data."""
    from transformers import GPT2Tokenizer
    from datasets import load_dataset

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("wikitext", "wikitext-2-raw-v1")

    train_texts = [
        t
        for t in ds["train"]["text"][: config.max_train_samples * 2]
        if t and len(t.strip()) > 20
    ][: config.max_train_samples]
    eval_texts = [
        t
        for t in ds["validation"]["text"][: config.max_eval_samples * 2]
        if t and len(t.strip()) > 20
    ][: config.max_eval_samples]

    def tokenize(texts):
        all_input, all_target = [], []
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) < 10:
                continue
            if len(tokens) > config.max_seq_len + 1:
                tokens = tokens[: config.max_seq_len + 1]
            else:
                tokens = tokens + [tokenizer.pad_token_id] * (
                    config.max_seq_len + 1 - len(tokens)
                )
            all_input.append(tokens[:-1])
            all_target.append(tokens[1:])
        return torch.tensor(all_input), torch.tensor(all_target)

    train_input, train_target = tokenize(train_texts)
    eval_input, eval_target = tokenize(eval_texts)

    return train_input, train_target, eval_input, eval_target, tokenizer


def compute_metrics(model, eval_input, eval_target, config):
    """Compute evaluation metrics."""
    model.eval()
    device = config.device

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    batch_size = config.batch_size
    num_batches = (len(eval_input) + batch_size - 1) // batch_size

    with torch.no_grad():
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, len(eval_input))

            batch_input = eval_input[start:end].to(device)
            batch_target = eval_target[start:end].to(device)

            logits = model(batch_input)

            logits_flat = logits.view(-1, logits.size(-1))
            target_flat = batch_target.view(-1)

            mask = target_flat != 50256
            if mask.sum() > 0:
                loss = F.cross_entropy(logits_flat[mask], target_flat[mask])
                total_loss += loss.item() * mask.sum().item()

                pred = logits_flat[mask].argmax(dim=-1)
                total_correct += (pred == target_flat[mask]).sum().item()
                total_tokens += mask.sum().item()

    avg_loss = total_loss / max(total_tokens, 1)

    return {
        "loss": avg_loss,
        "perplexity": np.exp(min(avg_loss, 100)),
        "accuracy": total_correct / max(total_tokens, 1),
    }


def train_and_evaluate(
    model, train_input, train_target, eval_input, eval_target, config
):
    """Train model and return metrics."""
    device = config.device
    model = model.to(device)

    # PIL training
    start_time = time.perf_counter()

    pil_input = train_input.to(device)
    pil_target = train_target.to(device)

    try:
        model.fit_pil_layers(pil_input, pil_target, verbose=False)
    except Exception as e:
        print(f"    PIL fitting failed: {e}")
        return None

    pil_time = time.perf_counter() - start_time

    # Attention fine-tuning
    if config.num_epochs > 0:
        attn_params = []
        for block in model.blocks:
            attn_params.extend(block.attention.parameters())
            attn_params.extend(block.ln1.parameters())

        optimizer = torch.optim.AdamW(attn_params, lr=3e-4)

        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(train_input, train_target)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

        model.train()
        for epoch in range(config.num_epochs):
            for batch_input, batch_target in dataloader:
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
                optimizer.step()

    total_time = time.perf_counter() - start_time

    # Evaluate
    metrics = compute_metrics(model, eval_input, eval_target, config)
    metrics["pil_time"] = pil_time
    metrics["total_time"] = total_time

    return metrics


def ablation_bipil(
    config: AblationConfig, train_input, train_target, eval_input, eval_target
):
    """Ablation: BiPIL vs Single-PIL."""
    print("\n" + "=" * 60)
    print("ABLATION: BiPIL vs Single-PIL")
    print("=" * 60)

    results = []

    for use_bipil in [False, True]:
        name = "BiPIL" if use_bipil else "Single-PIL"
        print(f"\n  Testing {name}...")

        model_config = PILLMConfig(
            vocab_size=50257,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            use_bipil=use_bipil,
        )
        model = PILLanguageModel(model_config)

        metrics = train_and_evaluate(
            model, train_input, train_target, eval_input, eval_target, config
        )

        if metrics:
            results.append(
                {
                    "variant": name,
                    "use_bipil": use_bipil,
                    **metrics,
                }
            )
            print(
                f"    PPL: {metrics['perplexity']:.2f}, Acc: {metrics['accuracy']:.4f}"
            )

        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()

    return results


def ablation_lambda(
    config: AblationConfig, train_input, train_target, eval_input, eval_target
):
    """Ablation: Lambda (regularization) sensitivity."""
    print("\n" + "=" * 60)
    print("ABLATION: Lambda Sensitivity")
    print("=" * 60)

    lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    results = []

    for lam in lambdas:
        print(f"\n  Testing λ={lam:.0e}...")

        model_config = PILLMConfig(
            vocab_size=50257,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            use_bipil=True,
            lambda_reg=lam,
        )
        model = PILLanguageModel(model_config)

        metrics = train_and_evaluate(
            model, train_input, train_target, eval_input, eval_target, config
        )

        if metrics:
            results.append(
                {
                    "lambda": lam,
                    **metrics,
                }
            )
            print(
                f"    PPL: {metrics['perplexity']:.2f}, Acc: {metrics['accuracy']:.4f}"
            )

        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()

    return results


def ablation_expansion(
    config: AblationConfig, train_input, train_target, eval_input, eval_target
):
    """Ablation: FFN expansion ratio."""
    print("\n" + "=" * 60)
    print("ABLATION: FFN Expansion Ratio")
    print("=" * 60)

    expansions = [1, 2, 4, 8]
    results = []

    for exp in expansions:
        print(f"\n  Testing expansion={exp}x...")

        model_config = PILLMConfig(
            vocab_size=50257,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            use_bipil=True,
            ffn_expansion=exp,
        )
        model = PILLanguageModel(model_config)

        num_params = model.get_num_params()

        metrics = train_and_evaluate(
            model, train_input, train_target, eval_input, eval_target, config
        )

        if metrics:
            results.append(
                {
                    "expansion": exp,
                    "num_params": num_params,
                    **metrics,
                }
            )
            print(f"    Params: {num_params:,}, PPL: {metrics['perplexity']:.2f}")

        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()

    return results


def ablation_fusion(
    config: AblationConfig, train_input, train_target, eval_input, eval_target
):
    """Ablation: BiPIL fusion method (concat vs add)."""
    print("\n" + "=" * 60)
    print("ABLATION: BiPIL Fusion Method")
    print("=" * 60)

    fusions = ["concat", "add"]
    results = []

    for fusion in fusions:
        print(f"\n  Testing fusion={fusion}...")

        model_config = PILLMConfig(
            vocab_size=50257,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            max_seq_len=config.max_seq_len,
            use_bipil=True,
            bipil_fusion=fusion,
        )
        model = PILLanguageModel(model_config)

        metrics = train_and_evaluate(
            model, train_input, train_target, eval_input, eval_target, config
        )

        if metrics:
            results.append(
                {
                    "fusion": fusion,
                    **metrics,
                }
            )
            print(
                f"    PPL: {metrics['perplexity']:.2f}, Acc: {metrics['accuracy']:.4f}"
            )

        del model
        if config.device == "cuda":
            torch.cuda.empty_cache()

    return results


def format_results_table(results: List[Dict], key_column: str, title: str) -> str:
    """Format results as markdown table."""
    lines = [
        f"\n### {title}\n",
        f"| {key_column} | Perplexity | Accuracy | PIL Time (s) |",
        "|--------------|------------|----------|--------------|",
    ]

    for r in results:
        key_val = r.get(
            key_column.lower().replace(" ", "_"), r.get(key_column.lower(), "N/A")
        )
        lines.append(
            f"| {key_val} | {r['perplexity']:.2f} | {r['accuracy']:.4f} | {r['pil_time']:.2f} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="PIL Ablation Study")
    parser.add_argument(
        "--study",
        type=str,
        default="all",
        choices=["all", "bipil", "lambda", "expansion", "fusion"],
    )
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--max_train_samples", type=int, default=3000)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs/ablations")
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    config = AblationConfig(
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples,
        device=device,
        output_dir=args.output_dir,
    )

    print("=" * 60)
    print("PIL Ablation Study")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Study: {args.study}")

    # Load data
    print("\nLoading data...")
    train_input, train_target, eval_input, eval_target, tokenizer = load_data(config)
    print(f"  Train: {train_input.shape}, Eval: {eval_input.shape}")

    all_results = {}

    # Run ablations
    if args.study in ["all", "bipil"]:
        all_results["bipil"] = ablation_bipil(
            config, train_input, train_target, eval_input, eval_target
        )

    if args.study in ["all", "lambda"]:
        all_results["lambda"] = ablation_lambda(
            config, train_input, train_target, eval_input, eval_target
        )

    if args.study in ["all", "expansion"]:
        all_results["expansion"] = ablation_expansion(
            config, train_input, train_target, eval_input, eval_target
        )

    if args.study in ["all", "fusion"]:
        all_results["fusion"] = ablation_fusion(
            config, train_input, train_target, eval_input, eval_target
        )

    # Summary
    print("\n" + "=" * 60)
    print("ABLATION STUDY RESULTS")
    print("=" * 60)

    summary = ""

    if "bipil" in all_results:
        summary += format_results_table(
            all_results["bipil"], "Variant", "BiPIL vs Single-PIL"
        )

    if "lambda" in all_results:
        summary += format_results_table(
            all_results["lambda"], "Lambda", "Lambda Sensitivity"
        )

    if "expansion" in all_results:
        summary += format_results_table(
            all_results["expansion"], "Expansion", "FFN Expansion Ratio"
        )

    if "fusion" in all_results:
        summary += format_results_table(
            all_results["fusion"], "Fusion", "BiPIL Fusion Method"
        )

    print(summary)

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)

    with open(os.path.join(config.output_dir, "ablation_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    with open(os.path.join(config.output_dir, "ablation_tables.md"), "w") as f:
        f.write("# PIL Ablation Study Results\n")
        f.write(summary)

    print(f"\nResults saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
