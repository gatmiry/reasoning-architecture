#!/usr/bin/env python3
"""
Test script for HypercubeFunctionPredictor learning x₁x₂ function
with function_dim=8, showing training progress every 100 steps.
"""

import torch
import torch.nn.functional as F
from hypercube_function_framework import HypercubeFunctionPredictor, run_hypercube_function_experiment
from model import GPTConfig

def test_x1x2x3x4x5_function():
    """Test the HypercubeFunctionPredictor on x₁x₂ function with function_dim=8"""
    
    print("=" * 80)
    print("TESTING HYPERCUBE FUNCTION PREDICTOR")
    print("Function: x₁x₂")
    print("Function Dimension: 8")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(41)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    function_dim = 12
    num_zero_tokens = 10
    vocab_size = 4  # -1, 0, output, +1
    block_size = function_dim + num_zero_tokens + 1  # function_dim input tokens + zero tokens + 1 output token
    n_layers = 2
    n_heads = 1
    n_embd = 64
    dropout = 0.1
    batch_size = 1024
    learning_rate = 1e-4
    max_steps = 100000
    threshold = 0.03
    
    print(f"Configuration:")
    print(f"  Function Dimension: {function_dim}")
    print(f"  Zero Tokens: {num_zero_tokens}")
    print(f"  Vocabulary Size: {vocab_size}")
    print(f"  Block Size: {block_size}")
    print(f"  Layers: {n_layers}")
    print(f"  Heads: {n_heads}")
    print(f"  Embedding Dimension: {n_embd}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {learning_rate}")
    print(f"  Max Steps: {max_steps}")
    print()
    
    # Create GPT config
    config = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size
    )
    # Set additional parameters
    config.n_layers = n_layers
    config.n_heads = n_heads
    config.n_embd = n_embd
    
    # Create the predictor
    predictor = HypercubeFunctionPredictor(
        config=config,
        function_dim=function_dim,
        num_zero_tokens=num_zero_tokens
    )
    
    # Move model to device
    predictor = predictor.to(device)
    
    # Create optimizer
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=learning_rate)
    
    print("Starting training...")
    print("=" * 80)
    
    # Training loop
    for step in range(max_steps):
        # Generate random input data
        input_tokens = torch.randint(0, 2, (batch_size, function_dim), device=device)  # 0 or 1 for each dimension
        
        # Compute target function: x₁x₂ (only first two variables)
        # Convert 0,1 to -1,+1 for target computation
        input_values = input_tokens * 2 - 1  # Convert to -1 or +1
        target_values = (input_values[:, 0] * input_values[:, 1] * input_values[:, 2] * input_values[:, 3] * input_values[:, 4] * input_values[:, 5] * input_values[:, 6]).unsqueeze(1).float()
        
        # Forward pass
        predictions = predictor(input_tokens, injection_sequence=predictor.injection_specifications)
        
        # Compute loss
        loss = F.mse_loss(predictions, target_values)
        #def is_connected(loss, x):
        #   # loss can be scalar or tensor; if tensor, you may need grad_outputs (see below)
        #   (g,) = torch.autograd.grad(
        #       loss, x, retain_graph=True, allow_unused=True)
        #   return g is not None
        #print('is_connected(loss, predictor.transformer.transformer.wte.weight) is ', is_connected(loss, predictor.transformer.transformer.wte.weight))
        #print('is_connected(loss, predictor.transformer.hidden_states[layer_0]) is ', is_connected(loss, predictor.transformer.hidden_states['layer_0']))
        #print('grad of hidden states layer 0 is ', predictor.transformer.hidden_states['layer_0'].grad)
        # Backward pass
        optimizer.zero_grad()
        #print('grad of hidden states layer 0 is second time ', predictor.transformer.hidden_states['layer_0'].grad)
        loss.backward()  # Retain graph for gradient analysis
        optimizer.step()
        
        # Analyze gradients and potentially add injection specifications
        # Pass the predictions from the main training loop to avoid recomputation
        
        #input_tokens = torch.randint(0, 2, (batch_size, function_dim), device=device)
        #input_values = input_tokens * 2 - 1  # Convert to -1 or +1
        #target_values = (input_values[:, 0] * input_values[:, 1] * input_values[:, 2] * input_values[:, 3] * input_values[:, 4]).unsqueeze(1).float()
        
        
        
        
        # Print progress every 100 steps
        if step % 100 == 0 or step == max_steps - 1:
            predictor.analyze_gradients_and_add_injections(input_tokens, target_values, threshold)
            print(f"Step {step:4d}: Loss = {loss.item():.6f}, Injections: {len(predictor.injection_specifications)}")
            
            # Show some example predictions vs targets
            if step % 200 == 0:
                print(f"  Sample predictions vs targets:")
                for i in range(min(5, batch_size)):
                    pred_val = predictions[i].item()
                    target_val = target_values[i].item()
                    input_vals = input_tokens[i].tolist()
                    print(f"    Input {input_vals} -> Pred: {pred_val:.4f}, Target: {target_val:.4f}")
                print()
    
    print("=" * 80)
    print("Training completed!")
    print(f"Final loss: {loss.item():.6f}")
    
    # Test on some specific cases
    print("\nTesting on specific cases:")
    print("-" * 40)
    
    test_cases = [
        [0, 0, 0, 0, 0, 0, 0, 0],  # x₁=0, x₂=0 -> should be +1
        [1, 1, 1, 1, 1, 1, 1, 1],  # x₁=1, x₂=1 -> should be +1
        [0, 1, 0, 1, 0, 1, 0, 1],  # x₁=0, x₂=1 -> should be -1
        [1, 0, 1, 0, 1, 0, 1, 0],  # x₁=1, x₂=0 -> should be -1
        [0, 0, 1, 1, 0, 0, 1, 1],  # x₁=0, x₂=0 -> should be +1
    ]
    
    predictor.eval()
    with torch.no_grad():
        for i, test_input in enumerate(test_cases):
            input_tensor = torch.tensor(test_input, device=device).unsqueeze(0)
            prediction = predictor(input_tensor)
            # Convert to -1,+1 for target computation (only x₁ and x₂)
            input_values = input_tensor * 2 - 1
            target = (input_values[0, 0] * input_values[0, 1]).item()
            
            print(f"Test {i+1}: Input {test_input}")
            print(f"         Prediction: {prediction.item():.6f}")
            print(f"         Target:     {target:.6f}")
            print(f"         Error:      {abs(prediction.item() - target):.6f}")
            print()
    
    # Show injection specifications if any were added
    if predictor.injection_specifications:
        print("Injection Specifications:")
        print("-" * 40)
        for i, spec in enumerate(predictor.injection_specifications):
            print(f"Injection {i+1}: {spec}")
    else:
        print("No injection specifications were added during training.")
    
    print("=" * 80)

if __name__ == "__main__":
    test_x1x2x3x4x5_function()
