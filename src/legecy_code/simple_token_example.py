"""
Simple example: Extract third token embedding and inject into first token position.
This directly demonstrates the specific functionality you requested.
"""

import torch
from model import GPTConfig
from enhanced_gpt import EnhancedGPT
from reasoning_framework import ReasoningFramework

def extract_third_token_inject_first():
    """
    Extract the third token embedding from every batch and inject it into 
    the first token position of the input.
    """
    print("Extracting third token and injecting into first token position")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=512, vocab_size=50257)
    config.n_layers = 4
    config.n_heads = 4
    config.n_embd = 128
    
    model = EnhancedGPT(config).to(device)
    reasoning_framework = ReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1, 2])
    
    # Create input with multiple batches
    batch_size = 3
    seq_length = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens:")
    for i in range(batch_size):
        print(f"  Batch {i}: {input_ids[i].tolist()}")
    
    # STEP 1: Extract third token (position 2) from layer_1
    print(f"\nStep 1: Extracting third token (position 2) from layer_1")
    
    # Initial pass to get hidden states
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    # Extract third token from layer_1
    source_positions = {'layer_1': 2}  # Third token (0-indexed)
    token_embeddings = model.extract_token_embeddings(hidden_states, source_positions)
    
    print(f"Extracted embedding shape: {token_embeddings['layer_1_token_2'].shape}")
    print(f"Extracted embedding for each batch:")
    for i in range(batch_size):
        embedding_norm = torch.norm(token_embeddings['layer_1_token_2'][i]).item()
        print(f"  Batch {i}: embedding norm = {embedding_norm:.4f}")
    
    # STEP 2: Inject third token into first token position (position 0)
    print(f"\nStep 2: Injecting third token into first token position")
    
    target_positions = {'layer_1_token_2': 0}  # Inject into first position
    
    # Perform reasoning pass with token injection
    logits_injected, _ = reasoning_framework.perform_reasoning_pass(
        input_ids=input_ids,
        hidden_states_to_use=hidden_states,
        injection_method='add',
        reasoning_strategy='token_injection',
        token_extraction_positions=source_positions,
        token_injection_positions=target_positions
    )
    
    # STEP 3: Compare results
    print(f"\nStep 3: Comparing results")
    
    logits_diff = torch.norm(logits_injected - logits_initial).item()
    print(f"Overall L2 difference: {logits_diff:.6f}")
    
    # Show differences at first token position (where we injected)
    first_token_diff = torch.norm(
        logits_injected[:, 0, :] - logits_initial[:, 0, :]
    ).item()
    print(f"First token L2 difference: {first_token_diff:.6f}")
    
    # Show differences at third token position (where we extracted from)
    third_token_diff = torch.norm(
        logits_injected[:, 2, :] - logits_initial[:, 2, :]
    ).item()
    print(f"Third token L2 difference: {third_token_diff:.6f}")
    
    print("\n" + "="*60)
    print("SUCCESS: Third token embedding extracted and injected into first token!")
    print("="*60)
    
    return logits_initial, logits_injected, token_embeddings

def demonstrate_different_injection_methods():
    """
    Show different ways to inject the third token into the first position.
    """
    print("\nDifferent injection methods:")
    print("="*40)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=256, vocab_size=50257)
    config.n_layers = 3
    config.n_heads = 2
    config.n_embd = 64
    
    model = EnhancedGPT(config).to(device)
    reasoning_framework = ReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1])
    
    input_ids = torch.randint(0, config.vocab_size, (2, 6)).to(device)
    
    # Initial pass
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    # Test different injection methods
    methods = ['add', 'replace']
    source_positions = {'layer_1': 2}  # Third token
    target_positions = {'layer_1_token_2': 0}  # First position
    
    for method in methods:
        print(f"\nMethod: {method}")
        
        reasoning_framework.clear_history()
        logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
        
        logits_method, _ = reasoning_framework.perform_reasoning_pass(
            input_ids=input_ids,
            hidden_states_to_use=hidden_states,
            injection_method=method,
            reasoning_strategy='token_injection',
            token_extraction_positions=source_positions,
            token_injection_positions=target_positions
        )
        
        diff = torch.norm(logits_method - logits_initial).item()
        print(f"  L2 difference: {diff:.6f}")

if __name__ == "__main__":
    # Main demonstration
    extract_third_token_inject_first()
    
    # Additional examples
    demonstrate_different_injection_methods()

