"""
Test the zero tokens functionality in HypercubeFunctionPredictor.
"""

import torch
import torch.nn.functional as F
from model import GPTConfig
from hypercube_function_framework import (
    HypercubeFunctionPredictor, 
    HypercubeFunctionDataset, 
    HypercubeFunctionTrainer,
    xor_function
)

def test_zero_tokens():
    """
    Test the zero tokens functionality.
    """
    print("Testing Zero Tokens Functionality")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    function_dim = 4
    num_zero_tokens = 2
    
    print(f"Function dimension: {function_dim}")
    print(f"Number of zero tokens: {num_zero_tokens}")
    print(f"Total sequence length: {function_dim + num_zero_tokens}")
    
    # Model configuration
    config = GPTConfig(block_size=function_dim + num_zero_tokens + 1, vocab_size=4)
    config.n_layers = 3
    config.n_heads = 2
    config.n_embd = 64
    
    # Create model with zero tokens
    model = HypercubeFunctionPredictor(config, function_dim, num_zero_tokens).to(device)
    
    print(f"\nModel configuration:")
    print(f"  Block size: {config.block_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Zero tokens: {num_zero_tokens}")
    
    # Test input processing
    print(f"\nTesting input processing:")
    test_inputs = torch.tensor([
        [1.0, -1.0, 1.0, -1.0],  # 4D hypercube input
        [-1.0, 1.0, -1.0, 1.0],
    ]).to(device)
    
    print(f"Original input shape: {test_inputs.shape}")
    print(f"Original inputs:")
    for i, input_bits in enumerate(test_inputs):
        print(f"  Input {i}: {input_bits.tolist()}")
    
    # Test forward pass
    print(f"\nTesting forward pass:")
    with torch.no_grad():
        predictions = model(test_inputs, injection_sequence=None)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions:")
        for i, pred in enumerate(predictions):
            print(f"  Input {i}: {test_inputs[i].tolist()} -> Prediction: {pred.item():.4f}")
    
    # Test with different numbers of zero tokens
    print(f"\nTesting different numbers of zero tokens:")
    for num_zeros in [0, 1, 2, 3]:
        print(f"\n  Testing with {num_zeros} zero tokens:")
        
        # Create model with different number of zero tokens
        config_test = GPTConfig(block_size=function_dim + num_zeros + 1, vocab_size=4)
        config_test.n_layers = 3
        config_test.n_heads = 2
        config_test.n_embd = 64
        
        model_test = HypercubeFunctionPredictor(config_test, function_dim, num_zeros).to(device)
        
        with torch.no_grad():
            predictions = model_test(test_inputs, injection_sequence=None)
            print(f"    Predictions: {[f'{p.item():.4f}' for p in predictions]}")
    
    # Test training with zero tokens
    print(f"\nTesting training with zero tokens:")
    
    # Create datasets
    train_dataset = HypercubeFunctionDataset(function_dim, 500, xor_function, seed=42)
    val_dataset = HypercubeFunctionDataset(function_dim, 100, xor_function, seed=123)
    
    # Create trainer
    trainer = HypercubeFunctionTrainer(model, config, learning_rate=1e-3)
    
    # Train for a few epochs
    print(f"Training for 20 epochs...")
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=20,
        batch_size=16,
        device=device,
        injection_sequence=None
    )
    
    print(f"Final training loss: {results['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {results['val_losses'][-1]:.6f}")
    
    # Test on validation set
    print(f"\nTesting on validation set:")
    model.eval()
    with torch.no_grad():
        val_inputs, val_targets = val_dataset.get_batch(5, device)
        predictions = model(val_inputs, injection_sequence=None)
        
        for i in range(5):
            input_bits = val_inputs[i].tolist()
            target = val_targets[i].item()
            prediction = predictions[i].item()
            error = abs(target - prediction)
            print(f"  Input: {input_bits} -> Target: {target:.1f}, Prediction: {prediction:.3f}, Error: {error:.3f}")
    
    # Test autoregressive generation
    print(f"\nTesting autoregressive generation:")
    try:
        generated = model.generate_autoregressive(function_dim, injection_sequence=None)
        generated_bits = generated[0].tolist()
        generated_value = xor_function(generated.unsqueeze(0))[0].item()
        print(f"  Generated: {generated_bits} -> Function value: {generated_value:.3f}")
    except Exception as e:
        print(f"  Autoregressive generation failed: {e}")
    
    print(f"\n" + "="*60)
    print(f"Zero tokens test completed!")
    print(f"="*60)
    
    return model, results

if __name__ == "__main__":
    test_zero_tokens()
