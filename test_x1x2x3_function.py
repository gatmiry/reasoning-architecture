"""
Test case for hypercube function x1*x2*x3 with dimension 8 and no injections.
This tests if the base model can learn this function without sequential injections.
"""

import torch
import torch.nn.functional as F
from model import GPTConfig
from hypercube_function_framework import (
    HypercubeFunctionPredictor, 
    HypercubeFunctionDataset, 
    HypercubeFunctionTrainer,
    create_injection_sequence_for_function
)

def x1x2x3_function(x: torch.Tensor) -> torch.Tensor:
    """
    Function: f(x) = x1 * x2 * x3
    For hypercube inputs where each xi is either +1 or -1.
    """
    # Extract first 3 dimensions and multiply them
    x1 = x[:, 0:1]  # First dimension
    x2 = x[:, 1:2]  # Second dimension  
    x3 = x[:, 2:3]  # Third dimension
    return x1 * x2 * x3

def test_x1x2x3_function():
    """
    Test the x1*x2*x3 function with dimension 8 and no injections.
    """
    print("Testing x1*x2*x3 Function on 8D Hypercube")
    print("="*60)
    print("Function: f(x) = x1 * x2 * x3")
    print("Dimension: 8")
    print("Sequential Injections: No")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    function_dim = 8
    
    # Model configuration
    config = GPTConfig(block_size=function_dim + 1, vocab_size=4)  # vocab_size=4 for -1, 0, +1, output
    config.n_layers = 4
    config.n_heads = 4
    config.n_embd = 128
    
    print(f"Model configuration:")
    print(f"  Block size: {config.block_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Embedding dim: {config.n_embd}")
    
    # Create model
    model = HypercubeFunctionPredictor(config, function_dim).to(device)
    
    # Create datasets
    train_dataset = HypercubeFunctionDataset(function_dim, 2000, x1x2x3_function, seed=42)
    val_dataset = HypercubeFunctionDataset(function_dim, 400, x1x2x3_function, seed=123)
    
    print(f"\nDataset information:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Function dimension: {function_dim}")
    
    # Test the function on some examples
    print(f"\nFunction examples:")
    test_inputs = torch.tensor([
        [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0],  # x1*x2*x3 = 1*1*1 = 1
        [1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0],   # x1*x2*x3 = 1*(-1)*1 = -1
        [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0],  # x1*x2*x3 = (-1)*1*(-1) = 1
        [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0],  # x1*x2*x3 = (-1)*(-1)*(-1) = -1
    ]).to(device)
    
    test_targets = x1x2x3_function(test_inputs)
    for i in range(len(test_inputs)):
        input_bits = test_inputs[i].tolist()
        target = test_targets[i].item()
        print(f"  Input: {input_bits[:3]}... -> Target: {target:.1f}")
    
    # Create trainer (no injection sequence)
    trainer = HypercubeFunctionTrainer(model, config, learning_rate=1e-3)
    
    # Train the model
    print(f"\nTraining the model...")
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=100,
        batch_size=32,
        device=device,
        injection_sequence=None  # No injections
    )
    
    # Test on validation set
    print(f"\nTesting on validation set:")
    model.eval()
    with torch.no_grad():
        val_inputs, val_targets = val_dataset.get_batch(10, device)
        predictions = model(val_inputs, injection_sequence=None)
        
        total_error = 0.0
        for i in range(10):
            input_bits = val_inputs[i].tolist()
            target = val_targets[i].item()
            prediction = predictions[i].item()
            error = abs(target - prediction)
            total_error += error
            
            print(f"  Input: {input_bits[:3]}... -> Target: {target:.1f}, Prediction: {prediction:.3f}, Error: {error:.3f}")
        
        avg_error = total_error / 10
        print(f"\nAverage error on validation set: {avg_error:.3f}")
    
    # Test on specific cases
    print(f"\nTesting on specific cases:")
    specific_cases = torch.tensor([
        [1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],  # Should be 1
        [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # Should be -1
        [1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # Should be -1
        [1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # Should be 1
        [-1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # Should be -1
        [-1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # Should be 1
        [-1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # Should be 1
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], # Should be -1
    ]).to(device)
    
    specific_targets = x1x2x3_function(specific_cases)
    specific_predictions = model(specific_cases, injection_sequence=None)
    
    correct_predictions = 0
    for i in range(len(specific_cases)):
        input_bits = specific_cases[i].tolist()
        target = specific_targets[i].item()
        prediction = specific_predictions[i].item()
        error = abs(target - prediction)
        
        # Check if prediction is correct (within 0.5 of target)
        is_correct = error < 0.5
        if is_correct:
            correct_predictions += 1
        
        print(f"  Input: {input_bits[:3]}... -> Target: {target:.1f}, Prediction: {prediction:.3f}, Error: {error:.3f} {'✓' if is_correct else '✗'}")
    
    accuracy = correct_predictions / len(specific_cases)
    print(f"\nAccuracy on specific cases: {accuracy:.1%} ({correct_predictions}/{len(specific_cases)})")
    
    # Generate some examples autoregressively
    print(f"\nAutoregressive generation:")
    try:
        generated = model.generate_autoregressive(function_dim, injection_sequence=None)
        generated_bits = generated[0].tolist()
        generated_value = x1x2x3_function(generated.unsqueeze(0))[0].item()
        print(f"  Generated: {generated_bits[:3]}... -> Function value: {generated_value:.3f}")
    except Exception as e:
        print(f"  Autoregressive generation failed: {e}")
    
    # Plot training curves
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(results['train_losses'], label='Training Loss', alpha=0.7)
        plt.plot(results['val_losses'], label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Curves for x1*x2*x3 Function (8D Hypercube)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig('/accounts/projects/peter/gatmiry/reasoning-architecture/x1x2x3_training_curves.png')
        print(f"\nTraining curves saved to x1x2x3_training_curves.png")
    except ImportError:
        print(f"\nMatplotlib not available, skipping plot generation")
    
    print(f"\n" + "="*60)
    print(f"Test completed!")
    print(f"Final training loss: {results['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {results['val_losses'][-1]:.6f}")
    print(f"="*60)
    
    return {
        'model': model,
        'trainer': trainer,
        'results': results,
        'accuracy': accuracy,
        'avg_error': avg_error
    }

if __name__ == "__main__":
    test_x1x2x3_function()



