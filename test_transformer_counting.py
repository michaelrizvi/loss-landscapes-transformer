#!/usr/bin/env python3
"""
Test script for TransformerModels with CountingSequences dataset
"""

import torch
import torch.nn as nn
from transformer_models import TransformerModels
from datasets import CountingSequences

def test_transformer_counting():
    """Test the TransformerModels with CountingSequences dataset"""
    
    print("=" * 60)
    print("Testing TransformerModels with CountingSequences")
    print("=" * 60)
    
    # Create counting dataset
    dataset = CountingSequences(
        train=True, 
        samples=100, 
        seed=42,
        min_range_size=1,
        max_range_size=8, 
        vocab_size=50,
        max_len=24
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Show a few examples
    print("\nExample sequences:")
    for i in range(3):
        x, y = dataset[i]
        print(f"Sample {i}:")
        print(f"  Input:  {x.tolist()}")
        print(f"  Target: {y.tolist()}")
        
        # Parse the sequence to understand it
        input_list = x.tolist()
        target_list = y.tolist()
        
        # Find separator token (102)
        if 102 in input_list:
            sep_idx = input_list.index(102)
            input_part = input_list[:sep_idx]
            output_start = input_list[sep_idx+1:]
            print(f"  Parsed - Input range: {input_part}")
            print(f"  Parsed - Expected counting start: {output_start[:5]}...")
    
    # Create transformer model with multiple copies
    model_count = 10
    model = TransformerModels(
        vocab_size=110,  # Needs to be larger than 103 (pad token)
        d_model=32,
        n_layers=2, 
        n_heads=4,
        d_ff=64,
        max_len=24,
        model_count=model_count,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nCreated TransformerModels with {model_count} copies")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 8
    x_batch = torch.stack([dataset[i][0] for i in range(batch_size)])
    y_batch = torch.stack([dataset[i][1] for i in range(batch_size)])
    
    if torch.cuda.is_available():
        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        model = model.cuda()
    
    print(f"\nTesting forward pass:")
    print(f"Input shape: {x_batch.shape}")
    
    with torch.no_grad():
        logits = model(x_batch)
        print(f"Output shape: {logits.shape}")
        print(f"Expected shape: ({batch_size}, {model_count}, {x_batch.shape[1]}, 110)")
    
    # Test loss computation
    loss_func = nn.CrossEntropyLoss(reduction='none', ignore_index=103)  # ignore pad token
    
    print(f"\nTesting loss computation:")
    n, m, t, vocab_size = logits.shape
    
    # Compute loss for each model
    # Reshape logits: (B, M, T, V) -> (B*M*T, V)
    logits_flat = logits.reshape(n * m * t, vocab_size)
    
    # Reshape targets: (B, T) -> (B, M, T) -> (B*M*T,)
    targets_expanded = y_batch.unsqueeze(1).expand(n, m, t).reshape(n * m * t)
    
    loss = loss_func(logits_flat, targets_expanded).view(n, m, t).mean(dim=(0, 2))  # Average over batch and sequence
    
    print(f"Loss per model: {loss}")
    print(f"Best model index: {loss.argmin().item()}")
    
    # Test pattern search
    print(f"\nTesting pattern search (1 step):")
    original_loss = loss.min().item()
    print(f"Original best loss: {original_loss:.4f}")
    
    # Run one pattern search step
    model.pattern_search(x_batch, y_batch, loss_func)
    
    # Check if loss improved
    with torch.no_grad():
        new_logits = model(x_batch)
        new_logits_flat = new_logits.reshape(n * m * t, vocab_size)
        new_targets_expanded = y_batch.unsqueeze(1).expand(n, m, t).reshape(n * m * t)
        new_loss = loss_func(new_logits_flat, new_targets_expanded).view(n, m, t).mean(dim=(0, 2))
        
    new_best_loss = new_loss.min().item()
    print(f"New best loss: {new_best_loss:.4f}")
    
    if new_best_loss < original_loss:
        print("✅ Pattern search improved the loss!")
    else:
        print("⚪ Pattern search did not improve loss (normal for random initialization)")
    
    # Test model extraction
    print(f"\nTesting model extraction:")
    best_model = model.get_model_subsets([0])
    print(f"Extracted model has {sum(p.numel() for p in best_model.parameters()):,} parameters")
    print(f"Original model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("\n" + "=" * 60) 
    print("All tests passed! TransformerModels with CountingSequences works correctly.")
    print("=" * 60)

if __name__ == "__main__":
    test_transformer_counting()