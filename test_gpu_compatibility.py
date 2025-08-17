#!/usr/bin/env python3
"""
Test script to verify RTX A5000 compatibility with SEA model
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append('src')

def test_gpu_memory():
    """Test basic GPU memory allocation"""
    print("Testing GPU memory allocation...")
    device = torch.device('cuda:0')
    
    # Test with different tensor sizes to simulate model memory usage
    sizes = [1000, 2000, 5000, 10000]
    
    for size in sizes:
        try:
            # Allocate tensor similar to model parameters
            x = torch.randn(size, size, dtype=torch.float32).to(device)
            memory_used = torch.cuda.memory_allocated(0) / 1024**3
            print(f"Successfully allocated {size}x{size} tensor. Memory used: {memory_used:.2f} GB")
            del x
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Failed to allocate {size}x{size} tensor: {e}")
            break
    
    print("GPU memory test completed\n")

def test_model_architecture():
    """Test if we can create the model architecture"""
    print("Testing model architecture creation...")
    
    try:
        # Import model components
        from src.model.axial import AxialTransformer
        from src.model.aggregator import Aggregator
        
        # Create a simple args object
        class Args:
            def __init__(self):
                self.num_vars = 1000
                self.num_edge_types = 5
                self.embed_dim = 64
                self.transformer_num_layers = 4
                self.n_heads = 8
                self.ffn_embed_dim = 8
                self.dropout = 0.1
                self.max_length = 1000
        
        args = Args()
        
        # Test AxialTransformer
        print("Creating AxialTransformer...")
        transformer = AxialTransformer(args)
        print(f"AxialTransformer created successfully. Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
        
        # Test Aggregator
        print("Creating Aggregator...")
        aggregator = Aggregator(args)
        print(f"Aggregator created successfully. Parameters: {sum(p.numel() for p in aggregator.parameters()):,}")
        
        # Test moving to GPU
        print("Moving model to GPU...")
        device = torch.device('cuda:0')
        aggregator = aggregator.to(device)
        print("Model successfully moved to GPU")
        
        # Test forward pass with dummy data
        print("Testing forward pass...")
        batch_size = 1
        num_trials = 10
        k = 5
        
        # Create dummy batch data
        dummy_batch = {
            'input': torch.randint(0, 5, (batch_size, num_trials, k*k)).to(device),
            'index': torch.randint(0, 100, (batch_size, num_trials, k*k)).to(device),
            'feats': torch.randn(batch_size, args.num_vars, args.num_vars).to(device),
            'unique': torch.randint(0, args.num_vars, (batch_size, 100, 2)).to(device),
            'time': torch.zeros(batch_size),
            'label': torch.randint(0, 2, (batch_size, args.num_vars, args.num_vars)).long().to(device),
            'key': ['test_dataset'] * batch_size
        }
        
        # Test forward pass
        with torch.no_grad():
            output = aggregator(dummy_batch)
            print("Forward pass successful!")
            print(f"Output keys: {list(output.keys())}")
        
        print("Model architecture test completed successfully!\n")
        return True
        
    except Exception as e:
        print(f"Model architecture test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_checkpoint_loading():
    """Test if we can load the pretrained checkpoints"""
    print("Testing checkpoint loading...")
    
    try:
        import pytorch_lightning as pl
        from src.model.factory import load_model
        from src.args import parse_args
        
        # Test loading GIES checkpoint
        checkpoint_path = "checkpoints/gies_synthetic/model_best_epoch=535_auprc=0.849.ckpt"
        
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"Checkpoint loaded successfully. Keys: {list(checkpoint.keys())}")
            
            # Check model state dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print(f"State dict loaded. Number of parameters: {len(state_dict)}")
                
                # Check parameter sizes
                total_params = 0
                for key, param in state_dict.items():
                    if 'weight' in key or 'bias' in key:
                        total_params += param.numel()
                
                print(f"Total parameters in checkpoint: {total_params:,}")
                
                # Test moving to GPU
                device = torch.device('cuda:0')
                gpu_state_dict = {k: v.to(device) for k, v in state_dict.items()}
                print("Checkpoint successfully moved to GPU")
                
            print("Checkpoint loading test completed successfully!\n")
            return True
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return False
            
    except Exception as e:
        print(f"Checkpoint loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("RTX A5000 Compatibility Test for SEA Model")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    print()
    
    # Run tests
    test_gpu_memory()
    test_model_architecture()
    test_checkpoint_loading()
    
    print("=" * 60)
    print("Compatibility test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
