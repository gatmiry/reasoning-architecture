"""
Example demonstrating token-specific hidden embedding extraction and injection.

This example shows how to:
1. Extract hidden embeddings from specific token positions (e.g., third token)
2. Inject these embeddings into specific positions (e.g., first token)
3. Perform reasoning with token-specific manipulations
"""

import torch
import torch.nn as nn
from model import GPTConfig
from enhanced_gpt import EnhancedGPT
from reasoning_framework import ReasoningFramework

def main():
    print("="*60)
    print("TOKEN-SPECIFIC HIDDEN EMBEDDING EXTRACTION AND INJECTION")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create configuration
    config = GPTConfig(block_size=512, vocab_size=50257)
    config.n_layers = 4
    config.n_heads = 4
    config.n_embd = 128
    
    # Create enhanced GPT model
    model = EnhancedGPT(config).to(device)
    model.eval()
    
    # Create reasoning framework
    reasoning_framework = ReasoningFramework(model, config)
    
    # Set up hidden state extraction from specific layers
    extraction_layers = [1, 2]  # Extract from layers 1 and 2
    reasoning_framework.set_extraction_layers(extraction_layers)
    
    # Create sample input with longer sequence
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Sample input tokens (batch 0): {input_ids[0]}")
    print(f"Sample input tokens (batch 1): {input_ids[1]}")
    
    print("\n" + "="*50)
    print("EXAMPLE 1: Extract Third Token and Inject into First Token")
    print("="*50)
    
    # Define token extraction and injection positions
    # Extract token at position 2 (third token) from layer_1
    source_token_positions = {
        'layer_1': 2,  # Extract third token from layer 1
        'layer_2': 2   # Extract third token from layer 2
    }
    
    # Inject these tokens into position 0 (first token)
    target_token_positions = {
        'layer_1_token_2': 0,  # Inject third token from layer_1 into first position
        'layer_2_token_2': 0   # Inject third token from layer_2 into first position
    }
    
    print("Token extraction positions:")
    for layer, pos in source_token_positions.items():
        print(f"  {layer}: position {pos}")
    
    print("Token injection positions:")
    for embedding, pos in target_token_positions.items():
        print(f"  {embedding}: position {pos}")
    
    # Perform token-specific reasoning
    logits_reasoning, new_hidden_states = reasoning_framework.perform_token_specific_reasoning(
        input_ids=input_ids,
        source_token_positions=source_token_positions,
        target_token_positions=target_token_positions,
        injection_method='add'
    )
    
    print(f"\nReasoning pass logits shape: {logits_reasoning.shape}")
    
    # Compare with initial pass
    reasoning_framework.clear_history()
    logits_initial, _ = reasoning_framework.perform_initial_pass(input_ids)
    
    logits_diff = torch.norm(logits_reasoning - logits_initial).item()
    print(f"L2 difference between initial and reasoning logits: {logits_diff:.6f}")
    
    print("\n" + "="*50)
    print("EXAMPLE 2: Different Injection Methods")
    print("="*50)
    
    injection_methods = ['add', 'replace']
    
    for method in injection_methods:
        print(f"\nTesting injection method: {method}")
        
        # Clear history for clean experiment
        reasoning_framework.clear_history()
        
        # Perform token-specific reasoning
        logits_method, _ = reasoning_framework.perform_token_specific_reasoning(
            input_ids=input_ids,
            source_token_positions=source_token_positions,
            target_token_positions=target_token_positions,
            injection_method=method
        )
        
        # Compare with initial pass
        reasoning_framework.clear_history()
        logits_initial, _ = reasoning_framework.perform_initial_pass(input_ids)
        
        diff = torch.norm(logits_method - logits_initial).item()
        print(f"  L2 difference from initial: {diff:.6f}")
    
    print("\n" + "="*50)
    print("EXAMPLE 3: Multiple Token Extraction and Injection")
    print("="*50)
    
    # Extract multiple tokens from different positions
    source_positions = {
        'layer_1': 1,  # Second token from layer 1
        'layer_1': 3,  # Fourth token from layer 1 (will overwrite previous)
        'layer_2': 2   # Third token from layer 2
    }
    
    # Note: The above will only keep the last value for 'layer_1'
    # Let's use different layers for different tokens
    source_positions = {
        'layer_1': 1,  # Second token from layer 1
        'layer_2': 3,  # Fourth token from layer 2
        'layer_2': 5   # Sixth token from layer 2 (will overwrite)
    }
    
    # Actually, let's be more explicit with the extraction
    print("Extracting multiple tokens:")
    
    # Clear history
    reasoning_framework.clear_history()
    
    # Initial pass to get hidden states
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    # Extract tokens manually to show the process
    token_embeddings = model.extract_token_embeddings(
        hidden_states, 
        {'layer_1': 1, 'layer_2': 3}  # Second token from layer 1, fourth from layer 2
    )
    
    print("Extracted token embeddings:")
    for name, embedding in token_embeddings.items():
        print(f"  {name}: shape {embedding.shape}")
    
    # Inject into different positions
    injection_positions = {
        'layer_1_token_1': 0,  # Inject second token from layer_1 into first position
        'layer_2_token_3': 1   # Inject fourth token from layer_2 into second position
    }
    
    # Perform reasoning with token injection
    logits_reasoning, _ = reasoning_framework.perform_reasoning_pass(
        input_ids=input_ids,
        hidden_states_to_use=hidden_states,
        injection_method='add',
        reasoning_strategy='token_injection',
        token_extraction_positions={'layer_1': 1, 'layer_2': 3},
        token_injection_positions=injection_positions
    )
    
    diff = torch.norm(logits_reasoning - logits_initial).item()
    print(f"L2 difference with multiple token injection: {diff:.6f}")
    
    print("\n" + "="*50)
    print("EXAMPLE 4: Custom Token Selection")
    print("="*50)
    
    # Demonstrate extracting tokens from specific positions based on input analysis
    print("Custom token selection based on input analysis:")
    
    # Clear history
    reasoning_framework.clear_history()
    
    # Initial pass
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    # Analyze input to select interesting tokens (e.g., middle tokens)
    seq_len = input_ids.size(1)
    middle_pos = seq_len // 2
    interesting_positions = [middle_pos - 1, middle_pos, middle_pos + 1]
    
    print(f"Sequence length: {seq_len}")
    print(f"Interesting positions: {interesting_positions}")
    
    # Extract tokens from interesting positions
    custom_extractions = {}
    for pos in interesting_positions:
        if pos < seq_len:
            token_emb = model.extract_token_embeddings(
                hidden_states, {'layer_1': pos}
            )
            custom_extractions[f'position_{pos}'] = token_emb[f'layer_1_token_{pos}']
    
    print("Custom extracted embeddings:")
    for name, embedding in custom_extractions.items():
        print(f"  {name}: shape {embedding.shape}")
    
    # Inject custom tokens into first few positions
    custom_injection_positions = {}
    for i, (name, embedding) in enumerate(custom_extractions.items()):
        if i < seq_len:
            custom_injection_positions[name] = i
    
    print("Custom injection positions:")
    for name, pos in custom_injection_positions.items():
        print(f"  {name}: position {pos}")
    
    # Perform reasoning with custom token injection
    logits_custom, _ = reasoning_framework.perform_reasoning_pass(
        input_ids=input_ids,
        hidden_states_to_use=hidden_states,
        injection_method='add',
        reasoning_strategy='token_injection',
        token_extraction_positions={},
        token_injection_positions={}
    )
    
    # Note: The above won't work as expected because we need to properly format the parameters
    # Let's do it step by step
    
    # Create properly formatted token embeddings
    formatted_token_embeddings = {}
    for pos in interesting_positions:
        if pos < seq_len:
            token_emb = model.extract_token_embeddings(
                hidden_states, {'layer_1': pos}
            )
            formatted_token_embeddings[f'layer_1_token_{pos}'] = token_emb[f'layer_1_token_{pos}']
    
    # Create injection positions
    formatted_injection_positions = {}
    for i, pos in enumerate(interesting_positions):
        if i < seq_len and pos < seq_len:
            formatted_injection_positions[f'layer_1_token_{pos}'] = i
    
    # Perform the injection
    logits_custom, _ = model.forward_with_token_injection(
        input_ids, formatted_token_embeddings, formatted_injection_positions, 'add'
    )
    
    diff = torch.norm(logits_custom - logits_initial).item()
    print(f"L2 difference with custom token injection: {diff:.6f}")
    
    print("\n" + "="*50)
    print("EXAMPLE 5: Analysis of Token Injection Effects")
    print("="*50)
    
    # Analyze how token injection affects different positions
    reasoning_framework.clear_history()
    
    # Initial pass
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    # Extract third token and inject into first token
    logits_injected, _ = reasoning_framework.perform_token_specific_reasoning(
        input_ids=input_ids,
        source_token_positions={'layer_1': 2},
        target_token_positions={'layer_1_token_2': 0},
        injection_method='add'
    )
    
    # Analyze differences at different positions
    print("Analysis of token injection effects:")
    for pos in range(min(5, seq_len)):
        initial_logits = logits_initial[0, pos, :]  # First batch, position pos
        injected_logits = logits_injected[0, pos, :]
        
        diff = torch.norm(injected_logits - initial_logits).item()
        print(f"  Position {pos}: L2 difference = {diff:.6f}")
        
        # Show which tokens are most affected
        token_diff = torch.abs(injected_logits - initial_logits)
        top_affected = torch.topk(token_diff, k=3)
        print(f"    Top affected token indices: {top_affected.indices.tolist()}")
        print(f"    Top affected token differences: {top_affected.values.tolist()}")
    
    print("\n" + "="*60)
    print("TOKEN-SPECIFIC REASONING DEMONSTRATION COMPLETE!")
    print("="*60)

def demonstrate_advanced_token_manipulation():
    """
    Demonstrate more advanced token manipulation techniques.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a larger model for more interesting results
    config = GPTConfig(block_size=1024, vocab_size=50257)
    config.n_layers = 6
    config.n_heads = 8
    config.n_embd = 256
    
    model = EnhancedGPT(config).to(device)
    reasoning_framework = ReasoningFramework(model, config)
    
    # Longer sequence for more complex token manipulation
    input_ids = torch.randint(0, config.vocab_size, (1, 20)).to(device)
    
    print("Advanced Token Manipulation:")
    print(f"Input sequence length: {input_ids.shape[1]}")
    
    # Set up extraction from multiple layers
    reasoning_framework.set_extraction_layers([1, 3, 5])
    
    # Extract tokens from multiple layers and positions
    source_positions = {
        'layer_1': 5,   # Sixth token from layer 1
        'layer_3': 10,  # Eleventh token from layer 3
        'layer_5': 15   # Sixteenth token from layer 5
    }
    
    # Inject into strategic positions
    target_positions = {
        'layer_1_token_5': 0,   # Inject into first position
        'layer_3_token_10': 1,  # Inject into second position
        'layer_5_token_15': 2   # Inject into third position
    }
    
    print("Advanced extraction and injection:")
    for src_layer, src_pos in source_positions.items():
        print(f"  Extract from {src_layer} position {src_pos}")
    
    for tgt_emb, tgt_pos in target_positions.items():
        print(f"  Inject {tgt_emb} into position {tgt_pos}")
    
    # Perform advanced token reasoning
    logits_advanced, _ = reasoning_framework.perform_token_specific_reasoning(
        input_ids=input_ids,
        source_token_positions=source_positions,
        target_token_positions=target_positions,
        injection_method='add'
    )
    
    # Compare with initial pass
    reasoning_framework.clear_history()
    logits_initial, _ = reasoning_framework.perform_initial_pass(input_ids)
    
    diff = torch.norm(logits_advanced - logits_initial).item()
    print(f"Advanced token manipulation L2 difference: {diff:.6f}")

if __name__ == "__main__":
    main()
    print("\n" + "="*70)
    print("ADVANCED TOKEN MANIPULATION DEMONSTRATION")
    print("="*70)
    demonstrate_advanced_token_manipulation()

