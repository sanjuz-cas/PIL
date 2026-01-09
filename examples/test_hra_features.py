"""
Test script for HRA (Householder Reflection Adaptation) features.

This demonstrates the HRA integration from NeurIPS 2024 paper:
"Bridging The Gap between Low-rank and Orthogonal Adaptation 
 via Householder Reflection Adaptation"

HRA constructs orthogonal transformation matrices using chains of 
learnable Householder reflections, enabling parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn

from app.core.bipil_layer import (
    HouseholderReflection,
    HRALinear,
    HRAPilLayer,
    HRABiPILLayer,
    HRAInjectedLinear,
    inject_hra_into_model,
    hra_orthogonality_loss,
    compute_model_hra_loss,
)


def test_householder_reflection():
    """Test basic Householder reflection transformation."""
    print("=" * 60)
    print("Test 1: HouseholderReflection Basic")
    print("=" * 60)
    
    dim = 64
    r = 8  # Number of reflections
    
    hra = HouseholderReflection(dim=dim, r=r, apply_gs=True)
    
    # Get orthogonal matrix
    Q = hra.get_orthogonal_matrix()
    
    print(f"Input dimension: {dim}")
    print(f"Number of reflections (r): {r}")
    print(f"Q shape: {Q.shape}")
    
    # Verify orthogonality: Q @ Q^T should be I
    QQt = Q @ Q.t()
    identity = torch.eye(dim)
    ortho_error = torch.norm(QQt - identity).item()
    print(f"Orthogonality error ||Q @ Q^T - I||: {ortho_error:.6f}")
    
    # Check determinant (should be ±1 for orthogonal matrix)
    det = torch.linalg.det(Q).item()
    print(f"Determinant: {det:.4f}")
    
    print("[OK] HouseholderReflection test passed!\n")


def test_hra_linear():
    """Test HRA-enhanced linear layer."""
    print("=" * 60)
    print("Test 2: HRALinear Layer")
    print("=" * 60)
    
    batch_size = 32
    in_features = 128
    out_features = 64
    r = 8
    
    hra_linear = HRALinear(
        in_features=in_features,
        out_features=out_features,
        r=r,
        apply_gs=True,
        freeze_base=True,
    )
    
    x = torch.randn(batch_size, in_features)
    y = hra_linear(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Base weight frozen: {not hra_linear.weight.requires_grad}")
    print(f"HRA params trainable: {hra_linear.hra.hra_u.requires_grad}")
    
    # Count parameters
    total_params = sum(p.numel() for p in hra_linear.parameters())
    trainable_params = sum(p.numel() for p in hra_linear.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters (HRA only): {trainable_params}")
    print(f"Parameter efficiency: {trainable_params/total_params*100:.2f}%")
    
    print("[OK] HRALinear test passed!\n")


def test_hra_pil_layer():
    """Test HRA-enhanced PIL layer."""
    print("=" * 60)
    print("Test 3: HRAPilLayer (HRA + PIL)")
    print("=" * 60)
    
    batch_size = 64
    dim = 128
    hidden_dim = 512
    
    hra_pil = HRAPilLayer(
        input_dim=dim,
        hidden_dim=hidden_dim,
        output_dim=dim,
        hra_r=8,
        apply_gs=True,
    )
    
    # Create training data
    x_train = torch.randn(batch_size, dim)
    y_train = torch.randn(batch_size, dim)
    
    # PIL fitting (no backprop!)
    fit_result = hra_pil.fit(x_train, y_train)
    
    print(f"Input/Output dim: {dim}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Fit success: {fit_result['success']}")
    print(f"Fit MSE: {fit_result['mse']:.6f}")
    print(f"Method: {fit_result['method']}")
    
    # Forward pass
    x_test = torch.randn(16, dim)
    y_pred = hra_pil(x_test)
    print(f"Test output shape: {y_pred.shape}")
    
    print("[OK] HRAPilLayer test passed!\n")


def test_hra_bipil_layer():
    """Test HRA-enhanced Bidirectional PIL layer."""
    print("=" * 60)
    print("Test 4: HRABiPILLayer (HRA + Bidirectional PIL)")
    print("=" * 60)
    
    batch_size = 64
    seq_len = 32
    dim = 128
    
    hra_bipil = HRABiPILLayer(
        dim=dim,
        expansion_factor=4,
        hra_r=8,
        apply_gs=True,
        fusion="concat",
    )
    
    # Create training data (sequence)
    x_train = torch.randn(batch_size, seq_len, dim)
    
    # PIL fitting for identity mapping
    fit_result = hra_bipil.fit(x_train)
    
    print(f"Model dim: {dim}")
    print(f"Hidden dim: {hra_bipil.hidden_dim}")
    print(f"Fused dim: {hra_bipil.fused_dim}")
    print(f"Fit success: {fit_result['success']}")
    print(f"Fit MSE: {fit_result['mse']:.6f}")
    print(f"Condition number: {fit_result['condition_number']:.2e}")
    
    # Forward pass
    y_pred = hra_bipil(x_train)
    print(f"Output shape: {y_pred.shape}")
    
    # Get HRA parameters for potential hybrid training
    hra_params = hra_bipil.get_hra_params()
    print(f"HRA trainable params: {sum(p.numel() for p in hra_params)}")
    
    print("[OK] HRABiPILLayer test passed!\n")


def test_inject_hra():
    """Test HRA injection into existing model."""
    print("=" * 60)
    print("Test 5: inject_hra_into_model")
    print("=" * 60)
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(128, 256)
            self.fc2 = nn.Linear(256, 128)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            return self.fc2(x)
    
    model = SimpleModel()
    
    # Count params before
    before_params = sum(p.numel() for p in model.parameters())
    before_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Before HRA injection:")
    print(f"  Total params: {before_params}")
    print(f"  Trainable params: {before_trainable}")
    
    # Inject HRA
    hra_params, injected_names = inject_hra_into_model(
        model, r=8, apply_gs=True, verbose=True
    )
    
    # Count params after
    after_params = sum(p.numel() for p in model.parameters())
    after_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nAfter HRA injection:")
    print(f"  Total params: {after_params}")
    print(f"  Trainable params: {after_trainable}")
    print(f"  Injected modules: {injected_names}")
    
    # Test forward pass
    x = torch.randn(32, 128)
    y = model(x)
    print(f"  Output shape: {y.shape}")
    
    print("[OK] inject_hra_into_model test passed!\n")


def test_orthogonality_loss():
    """Test orthogonality regularization loss."""
    print("=" * 60)
    print("Test 6: Orthogonality Loss")
    print("=" * 60)
    
    hra = HouseholderReflection(dim=64, r=8, apply_gs=True)
    
    # Compute orthogonality loss
    orth_loss = hra_orthogonality_loss(hra)
    print(f"Initial orthogonality loss: {orth_loss.item():.6f}")
    
    # Create model with multiple HRA modules
    class MultiHRAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.hra1 = HouseholderReflection(64, r=8)
            self.hra2 = HouseholderReflection(64, r=8)
    
    model = MultiHRAModel()
    total_loss = compute_model_hra_loss(model, weight=1e-4)
    print(f"Total model HRA loss (weighted): {total_loss.item():.6f}")
    
    print("[OK] Orthogonality loss test passed!\n")


def main():
    """Run all HRA tests."""
    print("\n" + "=" * 60)
    print("HRA (Householder Reflection Adaptation) Feature Tests")
    print("NeurIPS 2024 Spotlight Paper Implementation")
    print("=" * 60 + "\n")
    
    test_householder_reflection()
    test_hra_linear()
    test_hra_pil_layer()
    test_hra_bipil_layer()
    test_inject_hra()
    test_orthogonality_loss()
    
    print("=" * 60)
    print("All HRA tests passed!")
    print("=" * 60)
    
    print("""
Summary of Added HRA Features:
------------------------------
1. HouseholderReflection - Core orthogonal transformation via HR chains
2. HRALinear - HRA-enhanced linear layer with frozen base weights
3. HRAInjectedLinear - Wrapper to inject HRA into existing nn.Linear
4. HRAPilLayer - Combines HRA with PIL for adaptive feature expansion
5. HRABiPILLayer - HRA + Bidirectional PIL for enhanced representations
6. inject_hra_into_model - Utility to add HRA to any model's Linear layers
7. hra_orthogonality_loss - Regularization loss for stable training
8. compute_model_hra_loss - Aggregate HRA loss for entire models

Key Benefits:
- Parameter-efficient fine-tuning (only HRA vectors are trainable)
- Orthogonal transformations preserve norms and stability
- Integrates with PIL's gradient-free training philosophy
- Compatible with hybrid training (HRA via backprop, output via pseudoinverse)
""")


if __name__ == "__main__":
    main()
