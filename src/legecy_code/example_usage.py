"""
Example usage of the Reasoning Framework for transformer hidden embeddings.

This script demonstrates how to:
1. Set up the enhanced GPT model
2. Extract hidden states from intermediate layers
3. Perform multiple reasoning passes
4. Analyze the reasoning progression
"""

import torch
import torch.nn as nn
from model import GPTConfig
from enhanced_gpt import EnhancedGPT
from reasoning_framework import ReasoningFramework

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration
    config = GPTConfig(block_size=512, vocab_size=50257)
    config.n_layers = 4  # Small model for demonstration
    config.n_heads = 4
    config.n_embd = 128
    
    # Create enhanced GPT model
    model = EnhancedGPT(config).to(device)
    model.eval()
    
    # Create reasoning framework
    reasoning_framework = ReasoningFramework(model, config)
    
    # Set up hidden state extraction from specific layers
    extraction_layers = [1, 2]  # Extract from layers 1 and 2 (0-indexed)
    reasoning_framework.set_extraction_layers(extraction_layers)
    
    # Create sample input
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample input tokens: {input_ids[0]}")
    
    print("\n" + "="*50)
    print("EXAMPLE 1: Basic Hidden State Extraction")
    print("="*50)
    
    # Perform initial pass and extract hidden states
    logits, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    print(f"Initial logits shape: {logits.shape}")
    print(f"Extracted hidden states: {list(hidden_states.keys())}")
    
    for layer_name, hidden in hidden_states.items():
        print(f"  {layer_name}: {hidden.shape}")
    
    print("\n" + "="*50)
    print("EXAMPLE 2: Single Reasoning Pass with Layer Injection")
    print("="*50)
    
    # Perform a reasoning pass using the extracted hidden states
    logits_reasoning, new_hidden_states = reasoning_framework.perform_reasoning_pass(
        input_ids=input_ids,
        hidden_states_to_use=hidden_states,
        injection_method='add',
        injection_layers=[1, 2],
        reasoning_strategy='layer_injection'
    )
    
    print(f"Reasoning pass logits shape: {logits_reasoning.shape}")
    print(f"New hidden states: {list(new_hidden_states.keys())}")
    
    # Compare logits between initial and reasoning passes
    logits_diff = torch.norm(logits_reasoning - logits).item()
    print(f"L2 difference between initial and reasoning logits: {logits_diff:.6f}")
    
    print("\n" + "="*50)
    print("EXAMPLE 3: Iterative Reasoning")
    print("="*50)
    
    # Perform multiple iterative reasoning passes
    results = reasoning_framework.perform_iterative_reasoning(
        input_ids=input_ids,
        num_iterations=3,
        injection_method='add',
        reasoning_strategy='layer_injection'
    )
    
    print(f"Performed {len(results)} iterations:")
    for i, result in enumerate(results):
        logits = result['logits']
        print(f"  Iteration {i}: logits shape {logits.shape}, "
              f"max logit: {torch.max(logits).item():.4f}")
    
    print("\n" + "="*50)
    print("EXAMPLE 4: Reasoning Analysis")
    print("="*50)
    
    # Analyze the reasoning progression
    analysis = reasoning_framework.analyze_reasoning_progression()
    
    print("Reasoning Analysis:")
    print(f"  Number of passes: {analysis['num_passes']}")
    print("  Logits evolution:")
    for logit_info in analysis['logits_evolution']:
        print(f"    Pass {logit_info['pass']}: "
              f"max={logit_info['max_logit']:.4f}, "
              f"mean={logit_info['mean_logit']:.4f}, "
              f"std={logit_info['std_logit']:.4f}")
    
    print("  Hidden state changes:")
    for layer_name, change_info in analysis['hidden_state_changes'].items():
        print(f"    {layer_name}: L2 change={change_info['l2_change']:.6f}, "
              f"relative change={change_info['relative_change']:.6f}")
    
    print("\n" + "="*50)
    print("EXAMPLE 5: Hidden State Summary")
    print("="*50)
    
    # Create summary for a specific layer
    layer_summary = reasoning_framework.create_hidden_state_summary('layer_1')
    print("Layer 1 Summary:")
    for key, value in layer_summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "="*50)
    print("EXAMPLE 6: Different Injection Methods")
    print("="*50)
    
    # Clear history for clean experiment
    reasoning_framework.clear_history()
    
    # Test different injection methods
    injection_methods = ['add', 'replace']
    
    for method in injection_methods:
        print(f"\nTesting injection method: {method}")
        
        # Initial pass
        logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
        
        # Reasoning pass with specific method
        logits_method, _ = reasoning_framework.perform_reasoning_pass(
            input_ids=input_ids,
            hidden_states_to_use=hidden_states,
            injection_method=method,
            reasoning_strategy='layer_injection'
        )
        
        # Calculate difference
        diff = torch.norm(logits_method - logits_initial).item()
        print(f"  L2 difference from initial: {diff:.6f}")
        
        # Clear for next method
        reasoning_framework.clear_history()
    
    print("\n" + "="*50)
    print("EXAMPLE 7: Custom Reasoning Strategy")
    print("="*50)
    
    # Test input enhancement strategy
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    logits_enhanced, _ = reasoning_framework.perform_reasoning_pass(
        input_ids=input_ids,
        hidden_states_to_use=hidden_states,
        reasoning_strategy='input_enhancement'
    )
    
    diff = torch.norm(logits_enhanced - logits_initial).item()
    print(f"Input enhancement L2 difference: {diff:.6f}")
    
    print("\n" + "="*50)
    print("Framework Usage Complete!")
    print("="*50)

def demonstrate_advanced_usage():
    """
    Demonstrate more advanced usage patterns.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a larger model for more interesting results
    config = GPTConfig(block_size=1024, vocab_size=50257)
    config.n_layers = 6
    config.n_heads = 8
    config.n_embd = 256
    
    model = EnhancedGPT(config).to(device)
    reasoning_framework = ReasoningFramework(model, config)
    
    # Extract from multiple layers
    extraction_layers = [0, 2, 4]  # Early, middle, and late layers
    reasoning_framework.set_extraction_layers(extraction_layers)
    
    # Longer sequence for more complex reasoning
    input_ids = torch.randint(0, config.vocab_size, (1, 50)).to(device)
    
    print("Advanced Usage - Multi-layer Extraction:")
    print(f"Extracting from layers: {extraction_layers}")
    print(f"Input sequence length: {input_ids.shape[1]}")
    
    # Perform reasoning with different strategies
    strategies = ['layer_injection', 'input_enhancement']
    
    for strategy in strategies:
        print(f"\nTesting strategy: {strategy}")
        
        # Clear history
        reasoning_framework.clear_history()
        
        # Initial pass
        logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
        
        # Multiple reasoning passes
        for i in range(3):
            logits, hidden_states = reasoning_framework.perform_reasoning_pass(
                input_ids=input_ids,
                hidden_states_to_use=hidden_states,
                injection_method='add',
                reasoning_strategy=strategy
            )
            
            diff = torch.norm(logits - logits_initial).item()
            print(f"  Pass {i+1}: L2 diff = {diff:.6f}")
        
        # Analyze progression
        analysis = reasoning_framework.analyze_reasoning_progression()
        print(f"  Final analysis: {analysis['num_passes']} passes completed")

if __name__ == "__main__":
    main()
    print("\n" + "="*60)
    print("ADVANCED USAGE DEMONSTRATION")
    print("="*60)
    demonstrate_advanced_usage()

