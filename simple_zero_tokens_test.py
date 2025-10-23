"""
Simple test to demonstrate zero tokens functionality.
"""

import torch
from model import GPTConfig
from hypercube_function_framework import HypercubeFunctionPredictor, xor_function

def test_zero_tokens_simple():
    """
    Simple test of zero tokens functionality.
    """
    print("Simple Zero Tokens Test")
    print("="*40)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    function_dim = 3
    num_zero_tokens = 2
    
    print(f"Function dimension: {function_dim}")
    print(f"Number of zero tokens: {num_zero_tokens}")
    print(f"Total sequence length: {function_dim + num_zero_tokens}")
    
    # Model configuration
    config = GPTConfig(block_size=function_dim + num_zero_tokens + 1, vocab_size=4)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 32
    
    # Create model with zero tokens
    model = HypercubeFunctionPredictor(config, function_dim, num_zero_tokens).to(device)
    
    # Test input
    test_input = torch.tensor([[1.0, -1.0, 1.0]]).to(device)  # 3D hypercube input
    
    print(f"\nTest input: {test_input[0].tolist()}")
    
    # Test forward pass
    with torch.no_grad():
        prediction = model(test_input, injection_sequence=None)
        print(f"Prediction: {prediction.item():.4f}")
        
        # Verify the actual function value
        actual_value = xor_function(test_input)[0].item()
        print(f"Actual XOR value: {actual_value}")
        print(f"Error: {abs(prediction.item() - actual_value):.4f}")
    
    print(f"\n" + "="*40)
    print(f"Test completed!")

if __name__ == "__main__":
    test_zero_tokens_simple()
