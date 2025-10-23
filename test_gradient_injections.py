"""
Test the gradient-based injection specification functionality.
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

def test_gradient_injections():
    """
    Test the gradient-based injection specification functionality.
    """
    print("Testing Gradient-Based Injection Specifications")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    function_dim = 3
    num_zero_tokens = 2
    
    print(f"Function dimension: {function_dim}")
    print(f"Number of zero tokens: {num_zero_tokens}")
    
    # Model configuration
    config = GPTConfig(block_size=function_dim + num_zero_tokens + 1, vocab_size=4)
    config.n_layers = 3
    config.n_heads = 2
    config.n_embd = 64
    
    # Create model with zero tokens
    model = HypercubeFunctionPredictor(config, function_dim, num_zero_tokens).to(device)
    
    print(f"\nInitial injection specifications: {len(model.get_injection_specifications())}")
    
    # Create some test data
    test_inputs = torch.tensor([
        [1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0]
    ]).to(device)
    
    test_targets = xor_function(test_inputs)
    
    print(f"\nTest inputs:")
    for i, input_bits in enumerate(test_inputs):
        target = test_targets[i].item()
        print(f"  Input {i}: {input_bits.tolist()} -> Target: {target}")
    
    # Test gradient analysis with different thresholds
    thresholds = [0.01, 0.05, 0.1, 0.2]
    
    for threshold in thresholds:
        print(f"\n" + "="*40)
        print(f"Testing with threshold: {threshold}")
        print("="*40)
        
        # Clear previous specifications
        model.clear_injection_specifications()
        
        # Analyze gradients and add injections
        injection_specs = model.analyze_gradients_and_add_injections(
            test_inputs, test_targets, threshold=threshold
        )
        
        print(f"\nNumber of injection specifications added: {len(injection_specs)}")
        
        if injection_specs:
            print(f"\nInjection specifications:")
            for i, spec in enumerate(injection_specs):
                extraction = spec['extraction']
                injection = spec['injection']
                print(f"  {i+1}. Extract from token {extraction['position']} -> "
                      f"Inject to position {injection['position']} "
                      f"(weight: {injection['weight']:.3f})")
        
        # Test forward pass with injections
        if injection_specs:
            print(f"\nTesting forward pass with injections:")
            with torch.no_grad():
                predictions = model(test_inputs, injection_sequence=injection_specs)
                for i, (input_bits, target, pred) in enumerate(zip(test_inputs, test_targets, predictions)):
                    error = abs(target.item() - pred.item())
                    print(f"  Input {i}: {input_bits.tolist()} -> "
                          f"Target: {target.item():.1f}, "
                          f"Prediction: {pred.item():.3f}, "
                          f"Error: {error:.3f}")
    
    # Test iterative gradient analysis
    print(f"\n" + "="*60)
    print(f"Testing Iterative Gradient Analysis")
    print("="*60)
    
    # Clear specifications
    model.clear_injection_specifications()
    
    # Train the model a bit first
    print(f"\nTraining model for 10 epochs...")
    train_dataset = HypercubeFunctionDataset(function_dim, 200, xor_function, seed=42)
    val_dataset = HypercubeFunctionDataset(function_dim, 50, xor_function, seed=123)
    trainer = HypercubeFunctionTrainer(model, config, learning_rate=1e-3)
    
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=10,
        batch_size=16,
        device=device,
        injection_sequence=None
    )
    
    print(f"Final training loss: {results['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {results['val_losses'][-1]:.6f}")
    
    # Now test gradient analysis on trained model
    print(f"\nAnalyzing gradients on trained model...")
    
    # Get a batch of validation data
    val_inputs, val_targets = val_dataset.get_batch(8, device)
    
    # Analyze gradients multiple times to see how specifications accumulate
    for iteration in range(3):
        print(f"\nIteration {iteration + 1}:")
        injection_specs = model.analyze_gradients_and_add_injections(
            val_inputs, val_targets, threshold=0.05
        )
        print(f"Total injection specifications: {len(injection_specs)}")
    
    # Test final performance with all accumulated injections
    print(f"\nFinal performance with accumulated injections:")
    with torch.no_grad():
        predictions_no_inj = model(val_inputs, injection_sequence=None)
        predictions_with_inj = model(val_inputs, injection_sequence=injection_specs)
        
        mse_no_inj = F.mse_loss(predictions_no_inj, val_targets).item()
        mse_with_inj = F.mse_loss(predictions_with_inj, val_targets).item()
        
        print(f"MSE without injections: {mse_no_inj:.6f}")
        print(f"MSE with injections: {mse_with_inj:.6f}")
        print(f"Improvement: {((mse_no_inj - mse_with_inj) / mse_no_inj * 100):.2f}%")
    
    print(f"\n" + "="*60)
    print(f"Gradient injection test completed!")
    print(f"="*60)
    
    return model, injection_specs

if __name__ == "__main__":
    test_gradient_injections()
