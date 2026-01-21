"""
Quick Test for PIL Language Model

Tests the implementation with synthetic data to verify everything works.
Run this first before trying with HuggingFace datasets.

Usage:
    python test_pil_lm.py
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.pil_lm import PILLanguageModel, PILLMConfig


def test_basic_functionality():
    """Test basic model functionality."""
    print("=" * 60)
    print("PIL Language Model - Basic Functionality Test")
    print("=" * 60)

    # Small model for testing
    config = PILLMConfig(
        vocab_size=1000,
        max_seq_len=64,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        use_bipil=True,
        train_attention=False,  # Pure PIL test
    )

    print(f"\nModel Config:")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  BiPIL: {config.use_bipil}")

    # Create model
    model = PILLanguageModel(config)
    print(f"\nModel parameters: {model.get_num_params():,}")

    # Test data
    batch_size = 50
    seq_len = 32

    # Create simple pattern data
    print(f"\nCreating test data: {batch_size} sequences, length {seq_len}")

    # Pattern: each token predicts next token in sequence
    input_ids = torch.randint(2, config.vocab_size, (batch_size, seq_len))
    target_ids = torch.cat(
        [input_ids[:, 1:], torch.randint(2, config.vocab_size, (batch_size, 1))], dim=1
    )

    # ===== Test 1: Forward Pass =====
    print("\n[Test 1] Forward Pass")
    print("-" * 40)

    with torch.no_grad():
        logits = model(input_ids)

    print(f"  Input shape: {input_ids.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: (batch={batch_size}, seq={seq_len}, vocab={config.vocab_size})")

    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Shape mismatch!"
    print("  ✓ Forward pass works!")

    # ===== Test 2: PIL Training =====
    print("\n[Test 2] PIL Training (One-Shot)")
    print("-" * 40)

    start_time = time.perf_counter()
    stats = model.fit_pil_layers(input_ids, target_ids, verbose=True)
    fit_time = time.perf_counter() - start_time

    print(f"\n  PIL training time: {fit_time:.3f}s")
    print(f"  Tokens/second: {(batch_size * seq_len) / fit_time:.0f}")
    print("  ✓ PIL training works!")

    # ===== Test 3: Loss Computation =====
    print("\n[Test 3] Loss Computation After PIL")
    print("-" * 40)

    with torch.no_grad():
        logits = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            target_ids.reshape(-1),
        )

    print(f"  Cross-entropy loss: {loss.item():.4f}")
    print(f"  Perplexity: {torch.exp(loss).item():.2f}")

    # Compute accuracy
    pred = logits.argmax(dim=-1)
    accuracy = (pred == target_ids).float().mean().item()
    print(f"  Accuracy: {accuracy:.4f}")
    print("  ✓ Loss computation works!")

    # ===== Test 4: Generation =====
    print("\n[Test 4] Text Generation")
    print("-" * 40)

    prompt = torch.randint(2, config.vocab_size, (1, 5))
    print(f"  Prompt tokens: {prompt[0].tolist()}")

    start_time = time.perf_counter()
    generated = model.generate(
        prompt, max_new_tokens=20, temperature=1.0, do_sample=True
    )
    gen_time = time.perf_counter() - start_time

    print(f"  Generated tokens: {generated[0].tolist()}")
    print(f"  Generation time: {gen_time:.3f}s")
    print(f"  Tokens/second: {20 / gen_time:.0f}")
    print("  ✓ Generation works!")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    return True


def test_bipil_vs_standard():
    """Compare BiPIL vs standard single projection."""
    print("\n" + "=" * 60)
    print("BiPIL vs Standard PIL Comparison")
    print("=" * 60)

    # Test data
    batch_size = 100
    seq_len = 32
    vocab_size = 500

    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    target_ids = torch.cat(
        [input_ids[:, 1:], torch.randint(2, vocab_size, (batch_size, 1))], dim=1
    )

    results = {}

    for use_bipil in [False, True]:
        name = "BiPIL" if use_bipil else "Standard"
        print(f"\n{name}:")

        config = PILLMConfig(
            vocab_size=vocab_size,
            max_seq_len=64,
            embed_dim=128,
            num_heads=4,
            num_layers=2,
            use_bipil=use_bipil,
            train_attention=False,
        )

        model = PILLanguageModel(config)

        # Train
        start = time.perf_counter()
        model.fit_pil_layers(input_ids, target_ids, verbose=False)
        train_time = time.perf_counter() - start

        # Evaluate
        with torch.no_grad():
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
            )

        pred = logits.argmax(dim=-1)
        accuracy = (pred == target_ids).float().mean().item()

        results[name] = {
            "train_time": train_time,
            "loss": loss.item(),
            "perplexity": torch.exp(loss).item(),
            "accuracy": accuracy,
        }

        print(f"  Train time: {train_time:.3f}s")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Perplexity: {torch.exp(loss).item():.2f}")
        print(f"  Accuracy: {accuracy:.4f}")

    # Comparison
    print("\n" + "-" * 40)
    print("Comparison:")
    improvement = (
        (results["Standard"]["loss"] - results["BiPIL"]["loss"])
        / results["Standard"]["loss"]
        * 100
    )
    print(f"  BiPIL loss improvement: {improvement:.2f}%")

    return results


def test_scalability():
    """Test scalability with different model sizes."""
    print("\n" + "=" * 60)
    print("Scalability Test")
    print("=" * 60)

    configs = [
        (64, 2, "Tiny"),
        (128, 2, "Small"),
        (256, 4, "Medium"),
    ]

    batch_size = 50
    seq_len = 32
    vocab_size = 500

    input_ids = torch.randint(2, vocab_size, (batch_size, seq_len))
    target_ids = torch.cat(
        [input_ids[:, 1:], torch.randint(2, vocab_size, (batch_size, 1))], dim=1
    )

    print(f"\nTest data: {batch_size} sequences × {seq_len} tokens")
    print("-" * 60)
    print(f"{'Model':<10} {'Params':>10} {'PIL Time':>12} {'Loss':>10} {'PPL':>10}")
    print("-" * 60)

    for embed_dim, num_layers, name in configs:
        config = PILLMConfig(
            vocab_size=vocab_size,
            max_seq_len=64,
            embed_dim=embed_dim,
            num_heads=4,
            num_layers=num_layers,
            use_bipil=True,
        )

        model = PILLanguageModel(config)
        num_params = model.get_num_params()

        # Train
        start = time.perf_counter()
        model.fit_pil_layers(input_ids, target_ids, verbose=False)
        train_time = time.perf_counter() - start

        # Evaluate
        with torch.no_grad():
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
            )

        ppl = torch.exp(loss).item()

        print(
            f"{name:<10} {num_params:>10,} {train_time:>10.3f}s {loss.item():>10.4f} {ppl:>10.2f}"
        )

    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("   PIL LANGUAGE MODEL - COMPREHENSIVE TEST SUITE")
    print("=" * 70)

    try:
        # Basic tests
        test_basic_functionality()

        # BiPIL comparison
        test_bipil_vs_standard()

        # Scalability
        test_scalability()

        print("\n" + "=" * 70)
        print("   ALL TESTS COMPLETED SUCCESSFULLY! ✓")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Install HuggingFace: pip install datasets transformers")
        print("2. Run with real data: python examples/train_pil_lm.py")
        print("=" * 70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
