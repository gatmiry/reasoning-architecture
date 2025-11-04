"""
Simple test to demonstrate gradient-based injection specifications.
"""

import torch
import torch.nn.functional as F
from model import GPTConfig
from hypercube_function_framework import HypercubeFunctionPredictor, xor_function

def test_simple_gradient_injections():
    """
    Simple test of gradient-based injection specifications.
    """
    print("Simple Gradient-Based Injection Test")
    print("="*50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    function_dim = 2
    num_zero_tokens = 1
    
    print(f"Function dimension: {function_dim}")
    print(f"Number of zero tokens: {num_zero_tokens}")
    
    # Model configuration
    config = GPTConfig(block_size=function_dim + num_zero_tokens + 1, vocab_size=4)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 32
    
    # Create model with zero tokens
    model = HypercubeFunctionPredictor(config, function_dim, num_zero_tokens).to(device)
    
    # Set a very low threshold to catch any gradients
    model.set_gradient_threshold(0.001)
    
    print(f"\nInitial injection specifications: {len(model.get_injection_specifications())}")
    
    # Create test data
    test_inputs = torch.tensor([
        [1.0, -1.0],
        [-1.0, 1.0],
        [1.0, 1.0],
        [-1.0, -1.0]
    ]).to(device)
    
    test_targets = xor_function(test_inputs)
    
    print(f"\nTest inputs:")
    for i, input_bits in enumerate(test_inputs):
        target = test_targets[i].item()
        print(f"  Input {i}: {input_bits.tolist()} -> Target: {target}")
    
    # Test gradient analysis with very low threshold
    print(f"\nAnalyzing gradients with threshold: 0.001")
    injection_specs = model.analyze_gradients_and_add_injections(
        test_inputs, test_targets, threshold=0.001
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
        print(f"\nTesting forward pass with injections:")
        with torch.no_grad():
            predictions = model(test_inputs, injection_sequence=injection_specs)
            for i, (input_bits, target, pred) in enumerate(zip(test_inputs, test_targets, predictions)):
                error = abs(target.item() - pred.item())
                print(f"  Input {i}: {input_bits.tolist()} -> "
                      f"Target: {target.item():.1f}, "
                      f"Prediction: {pred.item():.3f}, "
                      f"Error: {error:.3f}")
    else:
        print(f"\nNo injection specifications were added.")
        print(f"This might be because the gradients are too small or the model needs training.")
    
    # Test with a trained model
    print(f"\n" + "="*50)
    print(f"Testing with trained model")
    print("="*50)
    
    # Train the model a bit
    print(f"Training model for 5 epochs...")
    from hypercube_function_framework import HypercubeFunctionDataset, HypercubeFunctionTrainer
    
    train_dataset = HypercubeFunctionDataset(function_dim, 100, xor_function, seed=42)
    val_dataset = HypercubeFunctionDataset(function_dim, 20, xor_function, seed=123)
    trainer = HypercubeFunctionTrainer(model, config, learning_rate=1e-3)
    
    results = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=5,
        batch_size=8,
        device=device,
        injection_sequence=None
    )
    
    print(f"Final training loss: {results['train_losses'][-1]:.6f}")
    print(f"Final validation loss: {results['val_losses'][-1]:.6f}")
    
    # Clear previous specifications and analyze again
    model.clear_injection_specifications()
    
    # Analyze gradients on trained model
    print(f"\nAnalyzing gradients on trained model...")
    injection_specs = model.analyze_gradients_and_add_injections(
        test_inputs, test_targets, threshold=0.001
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
        
        # Test final performance
        print(f"\nFinal performance comparison:")
        with torch.no_grad():
            predictions_no_inj = model(test_inputs, injection_sequence=None)
            predictions_with_inj = model(test_inputs, injection_sequence=injection_specs)
            
            mse_no_inj = F.mse_loss(predictions_no_inj, test_targets).item()
            mse_with_inj = F.mse_loss(predictions_with_inj, test_targets).item()
            
            print(f"MSE without injections: {mse_no_inj:.6f}")
            print(f"MSE with injections: {mse_with_inj:.6f}")
            print(f"Improvement: {((mse_no_inj - mse_with_inj) / mse_no_inj * 100):.2f}%")
    
    print(f"\n" + "="*50)
    print(f"Simple gradient test completed!")
    print(f"="*50)
    
    return model, injection_specs

if __name__ == "__main__":
    test_simple_gradient_injections()
