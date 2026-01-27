"""
Ablation Study: Attention vs FFN Contribution

Measures how much attention vs FFN contributes to language modeling performance.

Experiments:
1. Attention-Only: FFN outputs zero
2. FFN-Only: Attention outputs identity (no mixing)  
3. Full Model: Both attention and FFN active
4. PIL FFN: Attention trained, FFN via PIL

Usage:
    python experiments/ablation_attention_vs_ffn.py --device cuda
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

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
class AblationConfig:
    """Ablation configuration."""
    embed_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    max_seq_len: int = 128
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 5
    max_train_samples: int = 5000
    max_eval_samples: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "outputs/ablation_attn_ffn"


def load_tokenizer():
    """Load GPT-2 tokenizer."""
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_dataset(max_train: int, max_eval: int):
    """Load WikiText-2."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [t for t in ds["train"]["text"][:max_train*2] if t and len(t.strip()) > 20][:max_train]
    eval_texts = [t for t in ds["validation"]["text"][:max_eval*2] if t and len(t.strip()) > 20][:max_eval]
    return train_texts, eval_texts


def tokenize_data(texts: List[str], tokenizer, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize texts."""
    all_input_ids = []
    all_target_ids = []
    
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if len(tokens) < 10:
            continue
        if len(tokens) > max_length + 1:
            tokens = tokens[:max_length + 1]
        if len(tokens) < max_length + 1:
            tokens = tokens + [tokenizer.pad_token_id] * (max_length + 1 - len(tokens))
        all_input_ids.append(tokens[:-1])
        all_target_ids.append(tokens[1:])
    
    return torch.tensor(all_input_ids, dtype=torch.long), torch.tensor(all_target_ids, dtype=torch.long)


def compute_metrics(model, input_ids, target_ids, batch_size, device, is_baseline=False):
    """Compute perplexity and accuracy."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
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
            
            logits_flat = logits.reshape(-1, logits.size(-1))
            target_flat = batch_target.reshape(-1)
            
            mask = target_flat != 50256
            if mask.sum() > 0:
                loss = F.cross_entropy(logits_flat[mask], target_flat[mask])
                total_loss += loss.item() * mask.sum().item()
                
                preds = logits_flat[mask].argmax(dim=-1)
                total_correct += (preds == target_flat[mask]).sum().item()
                total_tokens += mask.sum().item()
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = min(float('inf'), torch.exp(torch.tensor(avg_loss)).item())
    accuracy = total_correct / max(1, total_tokens)
    
    return {"loss": avg_loss, "perplexity": perplexity, "top1_accuracy": accuracy}


def train_baseline_full(model, train_input, train_target, eval_input, eval_target, config):
    """Train full baseline (attention + FFN)."""
    device = config.device
    model = model.to(device)
    
    print("\n" + "=" * 60)
    print("Training Full Baseline (Attention + FFN)")
    print("=" * 60)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    dataset = TensorDataset(train_input, train_target)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    start_time = time.perf_counter()
    
    for epoch in range(config.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_input, batch_target in pbar:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            result = model(batch_input, batch_target)
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        metrics = compute_metrics(model, eval_input, eval_target, config.batch_size, device, is_baseline=True)
        print(f"  Epoch {epoch+1}: PPL={metrics['perplexity']:.2f}, Acc={metrics['top1_accuracy']:.4f}")
    
    train_time = time.perf_counter() - start_time
    final_metrics = compute_metrics(model, eval_input, eval_target, config.batch_size, device, is_baseline=True)
    
    return {"name": "Full (Attn+FFN)", "ppl": final_metrics["perplexity"], 
            "acc": final_metrics["top1_accuracy"], "time": train_time}


def train_attention_only(model, train_input, train_target, eval_input, eval_target, config):
    """Train attention only, zero out FFN."""
    device = config.device
    model = model.to(device)
    
    print("\n" + "=" * 60)
    print("Training Attention-Only (FFN = 0)")
    print("=" * 60)
    
    # Zero out FFN weights
    with torch.no_grad():
        for block in model.blocks:
            block.ffn.fc1.weight.zero_()
            block.ffn.fc1.bias.zero_()
            block.ffn.fc2.weight.zero_()
            block.ffn.fc2.bias.zero_()
    
    # Only train attention params
    attn_params = []
    for block in model.blocks:
        attn_params.extend(block.attention.parameters())
        attn_params.extend(block.ln1.parameters())
        attn_params.extend(block.ln2.parameters())
    attn_params.extend(model.token_embedding.parameters())
    attn_params.extend(model.position_embedding.parameters())
    attn_params.extend(model.ln_f.parameters())
    attn_params.extend(model.output_head.parameters())
    
    optimizer = torch.optim.AdamW(attn_params, lr=config.learning_rate)
    dataset = TensorDataset(train_input, train_target)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    start_time = time.perf_counter()
    
    for epoch in range(config.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_input, batch_target in pbar:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            result = model(batch_input, batch_target)
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attn_params, 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        metrics = compute_metrics(model, eval_input, eval_target, config.batch_size, device, is_baseline=True)
        print(f"  Epoch {epoch+1}: PPL={metrics['perplexity']:.2f}, Acc={metrics['top1_accuracy']:.4f}")
    
    train_time = time.perf_counter() - start_time
    final_metrics = compute_metrics(model, eval_input, eval_target, config.batch_size, device, is_baseline=True)
    
    return {"name": "Attention-Only", "ppl": final_metrics["perplexity"], 
            "acc": final_metrics["top1_accuracy"], "time": train_time}


def train_ffn_only(model, train_input, train_target, eval_input, eval_target, config):
    """Train FFN only, attention = identity (no value projection)."""
    device = config.device
    model = model.to(device)
    
    print("\n" + "=" * 60)
    print("Training FFN-Only (Attention outputs input)")
    print("=" * 60)
    
    # Make attention output identity (zero out attention weights so residual = input)
    with torch.no_grad():
        for block in model.blocks:
            # Zero attention output projection
            block.attention.c_proj.weight.zero_()
            block.attention.c_proj.bias.zero_()
    
    # Only train FFN params
    ffn_params = []
    for block in model.blocks:
        ffn_params.extend(block.ffn.parameters())
    ffn_params.extend(model.token_embedding.parameters())
    ffn_params.extend(model.position_embedding.parameters())
    ffn_params.extend(model.ln_f.parameters())
    ffn_params.extend(model.output_head.parameters())
    
    optimizer = torch.optim.AdamW(ffn_params, lr=config.learning_rate)
    dataset = TensorDataset(train_input, train_target)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    start_time = time.perf_counter()
    
    for epoch in range(config.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_input, batch_target in pbar:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            result = model(batch_input, batch_target)
            loss = result["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ffn_params, 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        metrics = compute_metrics(model, eval_input, eval_target, config.batch_size, device, is_baseline=True)
        print(f"  Epoch {epoch+1}: PPL={metrics['perplexity']:.2f}, Acc={metrics['top1_accuracy']:.4f}")
    
    train_time = time.perf_counter() - start_time
    final_metrics = compute_metrics(model, eval_input, eval_target, config.batch_size, device, is_baseline=True)
    
    return {"name": "FFN-Only", "ppl": final_metrics["perplexity"], 
            "acc": final_metrics["top1_accuracy"], "time": train_time}


def train_pil_hybrid(model, train_input, train_target, eval_input, eval_target, config, use_target_prop=False):
    """Train PIL hybrid: attention via gradients, FFN via PIL."""
    device = config.device
    model = model.to(device)
    
    mode_name = "PIL+TargetProp" if use_target_prop else "PIL+Residual"
    print("\n" + "=" * 60)
    print(f"Training PIL Hybrid ({mode_name})")
    print("=" * 60)
    
    # Initialize FFN to zero
    with torch.no_grad():
        for block in model.blocks:
            block.ffn.W_out.zero_()
            block.ffn.bias.zero_()
    
    # Train attention
    attn_params = []
    for block in model.blocks:
        attn_params.extend(block.attention.parameters())
        attn_params.extend(block.ln1.parameters())
        attn_params.extend(block.ln2.parameters())
    attn_params.extend(model.token_embedding.parameters())
    attn_params.extend(model.position_embedding.parameters())
    attn_params.extend(model.ln_f.parameters())
    
    optimizer = torch.optim.AdamW(attn_params, lr=config.learning_rate)
    dataset = TensorDataset(train_input, train_target)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    start_time = time.perf_counter()
    
    for epoch in range(config.num_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        for batch_input, batch_target in pbar:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_input)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch_target.view(-1),
                ignore_index=50256
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attn_params, 1.0)
            optimizer.step()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        metrics = compute_metrics(model, eval_input, eval_target, config.batch_size, device)
        print(f"  Epoch {epoch+1}: PPL={metrics['perplexity']:.2f}, Acc={metrics['top1_accuracy']:.4f}")
    
    # Fit PIL
    print(f"\n  Fitting PIL FFN ({mode_name})...")
    pil_input = train_input.to(device)
    pil_target = train_target.to(device)
    model.fit_pil_layers(pil_input, pil_target, verbose=True, use_target_propagation=use_target_prop)
    
    train_time = time.perf_counter() - start_time
    final_metrics = compute_metrics(model, eval_input, eval_target, config.batch_size, device)
    
    return {"name": f"PIL ({mode_name})", "ppl": final_metrics["perplexity"], 
            "acc": final_metrics["top1_accuracy"], "time": train_time}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--output_dir", default="outputs/ablation_attn_ffn")
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    config = AblationConfig(device=device, num_epochs=args.num_epochs, output_dir=args.output_dir)
    
    print("=" * 60)
    print("Ablation: Attention vs FFN Contribution")
    print("=" * 60)
    print(f"Model: {config.num_layers}L, {config.embed_dim}D, {config.num_heads}H")
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    tokenizer = load_tokenizer()
    train_texts, eval_texts = load_dataset(config.max_train_samples, config.max_eval_samples)
    train_input, train_target = tokenize_data(train_texts, tokenizer, config.max_seq_len)
    eval_input, eval_target = tokenize_data(eval_texts, tokenizer, config.max_seq_len)
    print(f"  Train: {train_input.shape}, Eval: {eval_input.shape}")
    
    results = []
    
    # 1. Full Baseline
    baseline_config = BaselineConfig(
        vocab_size=50257, max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_layers
    )
    model_full = BaselineTransformer(baseline_config)
    results.append(train_baseline_full(model_full, train_input, train_target, eval_input, eval_target, config))
    del model_full
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 2. Attention-Only
    model_attn = BaselineTransformer(BaselineConfig(
        vocab_size=50257, max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_layers
    ))
    results.append(train_attention_only(model_attn, train_input, train_target, eval_input, eval_target, config))
    del model_attn
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 3. FFN-Only
    model_ffn = BaselineTransformer(BaselineConfig(
        vocab_size=50257, max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_layers
    ))
    results.append(train_ffn_only(model_ffn, train_input, train_target, eval_input, eval_target, config))
    del model_ffn
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 4. PIL Hybrid (Residual mode)
    pil_config = PILLMConfig(
        vocab_size=50257, max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_layers,
        use_bipil=True, train_attention=True
    )
    model_pil = PILLanguageModel(pil_config)
    results.append(train_pil_hybrid(model_pil, train_input, train_target, eval_input, eval_target, config, use_target_prop=False))
    del model_pil
    if device == "cuda":
        torch.cuda.empty_cache()
    
    # 5. PIL Hybrid (Target Propagation)
    model_pil_tp = PILLanguageModel(PILLMConfig(
        vocab_size=50257, max_seq_len=config.max_seq_len,
        embed_dim=config.embed_dim, num_heads=config.num_heads, num_layers=config.num_layers,
        use_bipil=True, train_attention=True
    ))
    results.append(train_pil_hybrid(model_pil_tp, train_input, train_target, eval_input, eval_target, config, use_target_prop=True))
    del model_pil_tp
    
    # Results
    print("\n" + "=" * 60)
    print("ABLATION RESULTS: Attention vs FFN")
    print("=" * 60)
    
    print("\n| Model | Perplexity | Accuracy | Time (s) |")
    print("|-------|------------|----------|----------|")
    for r in results:
        print(f"| {r['name']} | {r['ppl']:.2f} | {r['acc']:.2%} | {r['time']:.1f} |")
    
    # Analysis
    full_ppl = results[0]["ppl"]
    attn_ppl = results[1]["ppl"]
    ffn_ppl = results[2]["ppl"]
    
    print("\n" + "=" * 60)
    print("CONTRIBUTION ANALYSIS")
    print("=" * 60)
    print(f"Full Model PPL: {full_ppl:.2f}")
    print(f"Attention-Only PPL: {attn_ppl:.2f} ({full_ppl/attn_ppl:.2%} of full)")
    print(f"FFN-Only PPL: {ffn_ppl:.2f} ({full_ppl/ffn_ppl:.2%} of full)")
    
    if attn_ppl < ffn_ppl:
        attn_contrib = (ffn_ppl - full_ppl) / (ffn_ppl - attn_ppl) if ffn_ppl != attn_ppl else 0.5
        print(f"\nAttention contributes ~{attn_contrib:.0%} of the improvement")
        print(f"FFN contributes ~{1-attn_contrib:.0%} of the improvement")
    
    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "ablation_results.json"), "w") as f:
        json.dump({"config": asdict(config), "results": results}, f, indent=2)
    
    print(f"\nResults saved to: {config.output_dir}/")


if __name__ == "__main__":
    main()
