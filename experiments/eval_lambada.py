"""
LAMBADA Evaluation: Zero-shot Language Understanding

LAMBADA tests a model's ability to predict the final word of a passage
that requires understanding the full context.

Metric: Accuracy on predicting the last word correctly.

Reference: "The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context"
           https://arxiv.org/abs/1606.06031

Usage:
    python experiments/eval_lambada.py --model_path outputs/pil_lm/pil_lm_model.pt
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.pil_lm import PILLanguageModel, PILLMConfig
from app.core.baseline_transformer import BaselineTransformer, BaselineConfig


def load_lambada_dataset(max_samples: int = None) -> List[Dict]:
    """Load LAMBADA dataset from HuggingFace."""
    try:
        from datasets import load_dataset
        
        # LAMBADA dataset
        ds = load_dataset("lambada", split="test")
        
        samples = []
        for i, item in enumerate(ds):
            if max_samples and i >= max_samples:
                break
            
            text = item["text"]
            # Last word is the target
            words = text.strip().split()
            if len(words) < 2:
                continue
            
            target_word = words[-1]
            context = " ".join(words[:-1])
            
            samples.append({
                "context": context,
                "target": target_word,
                "full_text": text,
            })
        
        return samples
    
    except Exception as e:
        print(f"Error loading LAMBADA: {e}")
        print("Falling back to synthetic LAMBADA-like examples...")
        return create_synthetic_lambada()


def create_synthetic_lambada() -> List[Dict]:
    """Create synthetic LAMBADA-like examples for testing."""
    examples = [
        {
            "context": "The sun was setting over the horizon, painting the sky in brilliant shades of orange and",
            "target": "red",
        },
        {
            "context": "After years of hard work, she finally achieved her lifelong dream of becoming a",
            "target": "doctor",
        },
        {
            "context": "The ancient library contained thousands of rare books and",
            "target": "manuscripts",
        },
        {
            "context": "In the middle of the forest, they discovered a hidden",
            "target": "cave",
        },
        {
            "context": "The chef prepared a delicious meal using fresh vegetables and",
            "target": "herbs",
        },
    ]
    
    for ex in examples:
        ex["full_text"] = ex["context"] + " " + ex["target"]
    
    return examples * 20  # Repeat to have enough samples


def evaluate_lambada(
    model,
    tokenizer,
    samples: List[Dict],
    device: str = "cpu",
    is_baseline: bool = False,
) -> Dict:
    """
    Evaluate model on LAMBADA.
    
    Returns accuracy of predicting the final word.
    """
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    
    results = []
    
    for sample in tqdm(samples, desc="Evaluating LAMBADA"):
        context = sample["context"]
        target_word = sample["target"]
        
        # Tokenize context
        context_ids = tokenizer.encode(context)
        
        # Tokenize target (we want the FIRST token of the target word)
        # Add space before target to match how it appears in context
        target_ids = tokenizer.encode(" " + target_word)
        if len(target_ids) == 0:
            continue
        target_token = target_ids[0]
        
        # Forward pass
        input_ids = torch.tensor([context_ids], device=device)
        
        with torch.no_grad():
            if is_baseline:
                result = model(input_ids)
                logits = result["logits"]
            else:
                logits = model(input_ids)
        
        # Get prediction for last position
        last_logits = logits[0, -1, :]
        predicted_token = last_logits.argmax().item()
        
        is_correct = (predicted_token == target_token)
        correct += int(is_correct)
        total += 1
        
        # Store result
        predicted_word = tokenizer.decode([predicted_token]).strip()
        results.append({
            "context": context[:50] + "...",
            "target": target_word,
            "predicted": predicted_word,
            "correct": is_correct,
        })
    
    accuracy = correct / max(total, 1)
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "results": results[:10],  # First 10 for inspection
    }


def load_model(model_path: str, device: str):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    if "config" in checkpoint:
        config = checkpoint["config"]
        if hasattr(config, "use_bipil"):
            # PIL model
            model = PILLanguageModel(config)
            model.load_state_dict(checkpoint["state_dict"])
            return model, "pil"
        else:
            # Baseline model
            model = BaselineTransformer(config)
            model.load_state_dict(checkpoint["state_dict"])
            return model, "baseline"
    
    raise ValueError(f"Unknown checkpoint format in {model_path}")


def main():
    parser = argparse.ArgumentParser(description="LAMBADA Evaluation")
    parser.add_argument("--model_path", type=str, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max_samples", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="outputs/lambada")
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("LAMBADA Evaluation")
    print("=" * 60)
    print(f"Device: {device}")
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    # Load LAMBADA
    print("Loading LAMBADA dataset...")
    samples = load_lambada_dataset(args.max_samples)
    print(f"  Loaded {len(samples)} samples")
    
    # Load model
    if args.model_path:
        print(f"\nLoading model from: {args.model_path}")
        model, model_type = load_model(args.model_path, device)
        is_baseline = (model_type == "baseline")
        model_name = os.path.basename(args.model_path)
    else:
        # Create fresh model for testing
        print("\nNo model path provided, creating fresh PIL model...")
        config = PILLMConfig(
            vocab_size=50257,
            embed_dim=256,
            num_layers=4,
            num_heads=4,
        )
        model = PILLanguageModel(config)
        is_baseline = False
        model_name = "fresh_pil_model"
    
    # Evaluate
    print("\nEvaluating on LAMBADA...")
    results = evaluate_lambada(
        model, tokenizer, samples, device, is_baseline
    )
    
    # Results
    print("\n" + "=" * 60)
    print("LAMBADA RESULTS")
    print("=" * 60)
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Correct: {results['correct']} / {results['total']}")
    
    print("\nSample predictions:")
    for r in results["results"][:5]:
        status = "✓" if r["correct"] else "✗"
        print(f"  {status} Target: '{r['target']}' | Predicted: '{r['predicted']}'")
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, f"lambada_{model_name}.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": model_name,
            "accuracy": results["accuracy"],
            "correct": results["correct"],
            "total": results["total"],
            "samples": results["results"],
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
