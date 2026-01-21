"""
PIL Language Model Training & Evaluation on HuggingFace Datasets

This script:
1. Loads a standard HF dataset (WikiText-2, TinyStories, or similar)
2. Trains the hybrid PIL-Transformer
3. Evaluates with proper metrics (perplexity, loss, accuracy)
4. Compares PIL training speed vs gradient-based

Usage:
    python examples/train_pil_lm.py --dataset wikitext --epochs 3
"""

import torch
import torch.nn.functional as F
import numpy as np
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import os

# Add parent to path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.pil_lm import PILLanguageModel, PILLMConfig


@dataclass
class TrainingConfig:
    """Training configuration."""

    dataset: str = "wikitext"
    subset: str = "wikitext-2-raw-v1"
    batch_size: int = 32
    max_seq_len: int = 128
    num_epochs: int = 3
    eval_interval: int = 100
    max_train_samples: int = 10000
    max_eval_samples: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "outputs/pil_lm"


def load_tokenizer():
    """Load tokenizer (GPT-2 by default)."""
    try:
        from transformers import GPT2Tokenizer

        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except ImportError:
        print("Warning: transformers not installed, using simple tokenizer")
        return SimpleTokenizer()


class SimpleTokenizer:
    """Simple character-level tokenizer as fallback."""

    def __init__(self):
        self.vocab_size = 256
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text: str) -> List[int]:
        return [ord(c) % 256 for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(chr(i) for i in ids if i > 1)

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        max_len = kwargs.get("max_length", 128)
        truncation = kwargs.get("truncation", True)

        all_ids = []
        for text in texts:
            ids = self.encode(text)
            if truncation and len(ids) > max_len:
                ids = ids[:max_len]
            all_ids.append(ids)

        # Pad
        max_len_batch = max(len(ids) for ids in all_ids)
        padded = [
            ids + [self.pad_token_id] * (max_len_batch - len(ids)) for ids in all_ids
        ]

        return {"input_ids": padded}


def load_dataset_hf(config: TrainingConfig):
    """Load dataset from HuggingFace."""
    try:
        from datasets import load_dataset

        if config.dataset == "wikitext":
            dataset = load_dataset("wikitext", config.subset)
            text_column = "text"
        elif config.dataset == "tinystories":
            dataset = load_dataset("roneneldan/TinyStories")
            text_column = "text"
        elif config.dataset == "ptb":
            dataset = load_dataset("ptb_text_only")
            text_column = "sentence"
        else:
            # Try to load directly
            dataset = load_dataset(config.dataset)
            text_column = "text"

        return dataset, text_column

    except ImportError:
        print("Warning: datasets not installed, using synthetic data")
        return None, None


def create_synthetic_dataset(num_samples: int = 5000, vocab_size: int = 1000):
    """Create synthetic dataset for testing without HF dependencies."""
    print("Creating synthetic dataset...")

    # Simple patterns for language modeling
    patterns = [
        "The quick brown fox jumps over the lazy dog.",
        "A journey of a thousand miles begins with a single step.",
        "To be or not to be that is the question.",
        "All that glitters is not gold.",
        "The early bird catches the worm.",
        "Actions speak louder than words.",
        "Knowledge is power.",
        "Time flies when you are having fun.",
        "Practice makes perfect.",
        "Every cloud has a silver lining.",
    ]

    data = []
    for i in range(num_samples):
        # Create variations
        pattern = patterns[i % len(patterns)]
        data.append(pattern + " " + patterns[(i + 1) % len(patterns)])

    return data


def tokenize_dataset(
    texts: List[str],
    tokenizer,
    max_length: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize texts and create input/target pairs.

    For language modeling:
        input_ids:  [t0, t1, t2, ..., tn-1]
        target_ids: [t1, t2, t3, ..., tn]
    """
    all_input_ids = []
    all_target_ids = []

    for text in tqdm(texts, desc="Tokenizing"):
        if not text or len(text.strip()) < 10:
            continue

        # Tokenize
        encoded = tokenizer(
            text,
            max_length=max_length + 1,  # +1 for target shift
            truncation=True,
            padding=False,
        )

        ids = (
            encoded["input_ids"]
            if isinstance(encoded["input_ids"], list)
            else encoded["input_ids"].tolist()
        )

        if len(ids) < 3:
            continue

        # Create input/target pairs
        input_ids = ids[:-1]
        target_ids = ids[1:]

        # Pad to max_length
        pad_len = max_length - len(input_ids)
        if pad_len > 0:
            pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
            input_ids = input_ids + [pad_id] * pad_len
            target_ids = target_ids + [pad_id] * pad_len

        all_input_ids.append(input_ids[:max_length])
        all_target_ids.append(target_ids[:max_length])

    return (
        torch.tensor(all_input_ids, dtype=torch.long),
        torch.tensor(all_target_ids, dtype=torch.long),
    )


def compute_metrics(
    model: PILLanguageModel,
    input_ids: torch.Tensor,
    target_ids: torch.Tensor,
    batch_size: int = 32,
    device: str = "cpu",
) -> Dict:
    """
    Compute evaluation metrics.

    Metrics:
        - Loss (cross-entropy)
        - Perplexity (exp(loss))
        - Top-1 Accuracy
        - Top-5 Accuracy
    """
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

            # Forward
            logits = model(batch_input)

            # Compute loss (ignoring padding)
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = batch_target.reshape(-1)

            # Mask padding
            mask = target_flat != 0

            if mask.sum() > 0:
                loss = F.cross_entropy(
                    logits_flat[mask], target_flat[mask], reduction="sum"
                )
                total_loss += loss.item()

                # Top-1 accuracy
                pred = logits_flat[mask].argmax(dim=-1)
                total_correct_top1 += (pred == target_flat[mask]).sum().item()

                # Top-5 accuracy
                _, top5_pred = logits_flat[mask].topk(5, dim=-1)
                total_correct_top5 += (
                    (top5_pred == target_flat[mask].unsqueeze(-1))
                    .any(dim=-1)
                    .sum()
                    .item()
                )

                total_tokens += mask.sum().item()

    # Compute final metrics
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = np.exp(min(avg_loss, 100))  # Cap to avoid overflow
    top1_acc = total_correct_top1 / max(total_tokens, 1)
    top5_acc = total_correct_top5 / max(total_tokens, 1)

    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "num_tokens": total_tokens,
    }


def train_pil_model(
    model: PILLanguageModel,
    train_input: torch.Tensor,
    train_target: torch.Tensor,
    eval_input: torch.Tensor,
    eval_target: torch.Tensor,
    config: TrainingConfig,
) -> Dict:
    """
    Train PIL Language Model.

    PIL training is different from gradient descent:
    1. Collect all hidden activations
    2. Solve for optimal weights in ONE SHOT via pseudoinverse
    3. Optionally fine-tune attention via gradients

    Returns:
        Training results including metrics and timing
    """
    device = config.device
    model = model.to(device)

    results = {
        "pil_train_time": 0,
        "attention_train_time": 0,
        "epochs": [],
    }

    print(f"\n{'=' * 60}")
    print(f"PIL Language Model Training")
    print(f"{'=' * 60}")
    print(f"Model config: {model.config}")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_input)}")
    print(f"Eval samples: {len(eval_input)}")
    print(f"{'=' * 60}\n")

    # ===== PHASE 1: PIL Training (ONE-SHOT) =====
    print("Phase 1: PIL Training (One-Shot)")
    print("-" * 40)

    pil_start = time.perf_counter()

    # Sample subset for PIL fitting (memory efficient)
    pil_samples = min(config.max_train_samples, len(train_input))
    pil_indices = torch.randperm(len(train_input))[:pil_samples]

    pil_input = train_input[pil_indices].to(device)
    pil_target = train_target[pil_indices].to(device)

    # Fit PIL layers
    pil_stats = model.fit_pil_layers(pil_input, pil_target, verbose=True)

    pil_time = time.perf_counter() - pil_start
    results["pil_train_time"] = pil_time

    print(f"\nPIL training completed in {pil_time:.2f}s")
    print(f"FFN MSE values: {[f'{m:.6f}' for m in pil_stats['ffn_mse']]}")
    print(f"Output head accuracy: {pil_stats['head_accuracy']:.4f}")

    # Evaluate after PIL
    print("\nEvaluating after PIL training...")
    eval_metrics = compute_metrics(
        model, eval_input, eval_target, config.batch_size, device
    )

    results["epochs"].append({"epoch": 0, "phase": "PIL", **eval_metrics})

    print(f"  Loss: {eval_metrics['loss']:.4f}")
    print(f"  Perplexity: {eval_metrics['perplexity']:.2f}")
    print(f"  Top-1 Acc: {eval_metrics['top1_accuracy']:.4f}")
    print(f"  Top-5 Acc: {eval_metrics['top5_accuracy']:.4f}")

    # ===== PHASE 2: Optional Attention Fine-tuning (Gradient) =====
    if model.config.train_attention and config.num_epochs > 0:
        print(f"\nPhase 2: Attention Fine-tuning ({config.num_epochs} epochs)")
        print("-" * 40)

        # Only train attention parameters
        attn_params = []
        for block in model.blocks:
            attn_params.extend(block.attention.parameters())

        if model.config.train_embeddings:
            attn_params.append(model.token_embedding.weight)
            attn_params.append(model.position_embedding.weight)

        optimizer = torch.optim.AdamW(
            [p for p in attn_params if p.requires_grad],
            lr=1e-4,
            weight_decay=0.01,
        )

        attn_start = time.perf_counter()

        num_batches = (len(train_input) + config.batch_size - 1) // config.batch_size

        for epoch in range(config.num_epochs):
            model.train()
            epoch_loss = 0.0

            # Shuffle
            perm = torch.randperm(len(train_input))

            pbar = tqdm(
                range(num_batches), desc=f"Epoch {epoch + 1}/{config.num_epochs}"
            )

            for batch_idx in pbar:
                start = batch_idx * config.batch_size
                end = min(start + config.batch_size, len(train_input))

                indices = perm[start:end]
                batch_input = train_input[indices].to(device)
                batch_target = train_target[indices].to(device)

                # Forward
                logits = model(batch_input)

                # Loss
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    batch_target.reshape(-1),
                    ignore_index=0,  # Ignore padding
                )

                # Backward (only for attention!)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(attn_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Evaluate
            eval_metrics = compute_metrics(
                model, eval_input, eval_target, config.batch_size, device
            )

            results["epochs"].append(
                {
                    "epoch": epoch + 1,
                    "phase": "attention_finetune",
                    "train_loss": epoch_loss / num_batches,
                    **eval_metrics,
                }
            )

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"  Train Loss: {epoch_loss / num_batches:.4f}")
            print(f"  Eval Loss: {eval_metrics['loss']:.4f}")
            print(f"  Perplexity: {eval_metrics['perplexity']:.2f}")
            print(f"  Top-1 Acc: {eval_metrics['top1_accuracy']:.4f}")
            print(f"  Top-5 Acc: {eval_metrics['top5_accuracy']:.4f}")

        results["attention_train_time"] = time.perf_counter() - attn_start

    # ===== SUMMARY =====
    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    print(f"PIL training time: {results['pil_train_time']:.2f}s")
    print(f"Attention fine-tune time: {results['attention_train_time']:.2f}s")
    print(
        f"Total time: {results['pil_train_time'] + results['attention_train_time']:.2f}s"
    )
    print(f"Final Perplexity: {results['epochs'][-1]['perplexity']:.2f}")
    print(f"Final Top-1 Accuracy: {results['epochs'][-1]['top1_accuracy']:.4f}")
    print(f"{'=' * 60}\n")

    return results


def generate_samples(
    model: PILLanguageModel,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> List[str]:
    """Generate text samples from the model."""
    model.eval()
    model = model.to(device)

    generated = []

    for prompt in prompts:
        # Tokenize prompt
        encoded = tokenizer(
            prompt,
            return_tensors="pt" if hasattr(tokenizer, "return_tensors") else None,
        )

        if isinstance(encoded, dict):
            input_ids = torch.tensor(
                [encoded["input_ids"]]
                if isinstance(encoded["input_ids"][0], int)
                else encoded["input_ids"]
            )
        else:
            input_ids = torch.tensor([encoded])

        input_ids = input_ids.to(device)

        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
        )

        # Decode
        output_text = tokenizer.decode(output_ids[0].tolist())
        generated.append(output_text)

    return generated


def main():
    parser = argparse.ArgumentParser(description="Train PIL Language Model")
    parser.add_argument("--dataset", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--max_seq_len", type=int, default=128, help="Max sequence length"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Number of attention fine-tune epochs"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=256, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_layers", type=int, default=4, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--max_train_samples", type=int, default=5000, help="Max training samples"
    )
    parser.add_argument(
        "--max_eval_samples", type=int, default=500, help="Max eval samples"
    )
    parser.add_argument(
        "--no_bipil", action="store_true", help="Disable BiPIL (use single projection)"
    )
    parser.add_argument(
        "--no_train_attention", action="store_true", help="Don't fine-tune attention"
    )
    parser.add_argument(
        "--save_path", type=str, default="outputs/pil_lm", help="Save path"
    )
    args = parser.parse_args()

    # Training config
    train_config = TrainingConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_epochs=args.num_epochs,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        save_path=args.save_path,
    )

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = load_tokenizer()
    vocab_size = getattr(tokenizer, "vocab_size", 50257)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    dataset, text_column = load_dataset_hf(train_config)

    if dataset is not None:
        # Use HuggingFace dataset
        train_texts = [ex[text_column] for ex in dataset["train"]][
            : args.max_train_samples
        ]

        # Try to get validation split
        if "validation" in dataset:
            eval_texts = [ex[text_column] for ex in dataset["validation"]][
                : args.max_eval_samples
            ]
        elif "test" in dataset:
            eval_texts = [ex[text_column] for ex in dataset["test"]][
                : args.max_eval_samples
            ]
        else:
            # Split from train
            eval_texts = train_texts[-args.max_eval_samples :]
            train_texts = train_texts[: -args.max_eval_samples]
    else:
        # Use synthetic data
        all_texts = create_synthetic_dataset(
            args.max_train_samples + args.max_eval_samples
        )
        train_texts = all_texts[: args.max_train_samples]
        eval_texts = all_texts[-args.max_eval_samples :]

    # Filter empty texts
    train_texts = [t for t in train_texts if t and len(t.strip()) > 10]
    eval_texts = [t for t in eval_texts if t and len(t.strip()) > 10]

    print(f"Train texts: {len(train_texts)}")
    print(f"Eval texts: {len(eval_texts)}")

    # Tokenize
    print("\nTokenizing datasets...")
    train_input, train_target = tokenize_dataset(
        train_texts, tokenizer, args.max_seq_len
    )
    eval_input, eval_target = tokenize_dataset(eval_texts, tokenizer, args.max_seq_len)

    print(f"Train tensors: {train_input.shape}")
    print(f"Eval tensors: {eval_input.shape}")

    # Create model
    model_config = PILLMConfig(
        vocab_size=vocab_size,
        max_seq_len=args.max_seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_bipil=not args.no_bipil,
        train_attention=not args.no_train_attention,
    )

    print(f"\nCreating model...")
    model = PILLanguageModel(model_config)
    print(f"Model parameters: {model.get_num_params():,}")

    # Train
    results = train_pil_model(
        model,
        train_input,
        train_target,
        eval_input,
        eval_target,
        train_config,
    )

    # Generate samples
    print("\n" + "=" * 60)
    print("Generation Samples")
    print("=" * 60)

    prompts = [
        "The quick brown",
        "Knowledge is",
        "A journey of",
    ]

    for prompt in prompts:
        # Simple generation for demonstration
        encoded = tokenizer(prompt)
        input_ids = torch.tensor(
            [
                encoded["input_ids"]
                if isinstance(encoded["input_ids"][0], int)
                else encoded["input_ids"][0]
            ]
        )
        input_ids = input_ids.to(train_config.device)

        model = model.to(train_config.device)
        output_ids = model.generate(input_ids, max_new_tokens=30, temperature=0.8)

        output_text = tokenizer.decode(output_ids[0].tolist())
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{output_text}'")

    # Save results
    os.makedirs(args.save_path, exist_ok=True)
    results_path = os.path.join(args.save_path, "training_results.json")

    # Convert numpy types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_to_serializable(results), f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Save model
    model_path = os.path.join(args.save_path, "pil_lm_model.pt")
    torch.save(
        {
            "config": model_config,
            "state_dict": model.state_dict(),
        },
        model_path,
    )
    print(f"Model saved to: {model_path}")

    return results


if __name__ == "__main__":
    main()
