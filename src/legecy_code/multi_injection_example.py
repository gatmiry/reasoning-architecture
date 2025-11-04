"""
Comprehensive example demonstrating multiple hidden embedding injections
with full gradient backpropagation through all injection points.
"""

import torch
import torch.nn.functional as F
from model import GPTConfig
from multi_injection_gpt import MultiInjectionGPT, MultiInjectionReasoningFramework

def demonstrate_multiple_injections():
    """
    Demonstrate multiple hidden embedding injections with backpropagation.
    """
    print("Multiple Hidden Embedding Injections with Backpropagation")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=512, vocab_size=50257)
    config.n_layers = 4
    config.n_heads = 4
    config.n_embd = 128
    
    model = MultiInjectionGPT(config).to(device)
    reasoning_framework = MultiInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1, 2, 3])
    
    # Create input
    batch_size = 2
    seq_length = 8
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    target_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_tokens.shape}")
    print(f"Input tokens:")
    for i in range(batch_size):
        print(f"  Batch {i}: {input_ids[i].tolist()}")
    
    # STEP 1: Initial pass to extract hidden states
    print(f"\nStep 1: Initial pass to extract hidden states")
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    print(f"Extracted hidden states from layers: {list(hidden_states.keys())}")
    for layer_name, hidden in hidden_states.items():
        if hidden.dim() == 3:
            print(f"  {layer_name}: {hidden.shape}")
    
    # STEP 2: Extract multiple hidden embeddings
    print(f"\nStep 2: Extract multiple hidden embeddings")
    
    extraction_specs = [
        {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'},
        {'layer_name': 'layer_2', 'position': 4, 'key': 'token_4_layer_2'},
        {'layer_name': 'layer_3', 'position': 1, 'key': 'token_1_layer_3'},
        {'layer_name': 'layer_1', 'position': 6, 'key': 'token_6_layer_1'},
    ]
    
    hidden_embeddings = model.extract_hidden_embeddings(hidden_states, extraction_specs)
    
    print(f"Extracted embeddings:")
    for key, embedding in hidden_embeddings.items():
        print(f"  {key}: {embedding.shape}, norm = {torch.norm(embedding).item():.4f}")
    
    # STEP 3: Define multiple injection specifications
    print(f"\nStep 3: Define multiple injection specifications")
    
    injection_specs = [
        {
            'layer': 1,  # Inject into layer 1
            'position': 0,  # At position 0
            'hidden_embedding_key': 'token_2_layer_1',
            'method': 'add',
            'weight': 0.5
        },
        {
            'layer': 2,  # Inject into layer 2
            'position': 3,  # At position 3
            'hidden_embedding_key': 'token_4_layer_2',
            'method': 'replace'
        },
        {
            'layer': 1,  # Inject into layer 1 again
            'position': 5,  # At position 5
            'hidden_embedding_key': 'token_6_layer_1',
            'method': 'weighted_add',
            'weight': 0.3
        },
        {
            'layer': 3,  # Inject into layer 3
            'position': 2,  # At position 2
            'hidden_embedding_key': 'token_1_layer_3',
            'method': 'add',
            'weight': 0.8
        }
    ]
    
    print(f"Injection specifications:")
    for i, spec in enumerate(injection_specs):
        print(f"  {i+1}. Layer {spec['layer']}, Position {spec['position']}, "
              f"Method: {spec['method']}, Embedding: {spec['hidden_embedding_key']}")
    
    # STEP 4: Perform reasoning pass with multiple injections
    print(f"\nStep 4: Perform reasoning pass with multiple injections")
    
    logits_injected, new_hidden_states = reasoning_framework.perform_reasoning_pass_with_injections(
        input_ids, hidden_embeddings, injection_specs
    )
    
    # STEP 5: Compare results
    print(f"\nStep 5: Compare results")
    
    logits_diff = torch.norm(logits_injected - logits_initial).item()
    print(f"Overall L2 difference: {logits_diff:.6f}")
    
    # Show differences at specific positions
    for i, spec in enumerate(injection_specs):
        pos = spec['position']
        if pos < seq_length:
            pos_diff = torch.norm(
                logits_injected[:, pos, :] - logits_initial[:, pos, :]
            ).item()
            print(f"Position {pos} L2 difference: {pos_diff:.6f}")
    
    # STEP 6: Demonstrate gradient backpropagation
    print(f"\nStep 6: Demonstrate gradient backpropagation")
    
    # Compute loss
    loss_fn = F.cross_entropy
    loss = loss_fn(logits_injected.view(-1, logits_injected.size(-1)), target_tokens.view(-1))
    print(f"Loss: {loss.item():.6f}")
    
    # Backward pass
    loss.backward()
    
    # Analyze gradient flow
    gradient_analysis = reasoning_framework.compute_gradient_flow_analysis(loss_fn, target_tokens)
    
    print(f"Gradient analysis:")
    print(f"  Loss: {gradient_analysis['loss'].item():.6f}")
    print(f"  Gradient norms:")
    for name, norm in gradient_analysis['gradient_norms'].items():
        print(f"    {name}: {norm:.6f}")
    
    # Check if injection projection layers have gradients
    injection_grads = {k: v for k, v in gradient_analysis['gradient_norms'].items() 
                      if 'injection_projection' in k}
    if injection_grads:
        print(f"  Injection projection gradients:")
        for name, norm in injection_grads.items():
            print(f"    {name}: {norm:.6f}")
    
    print("\n" + "="*60)
    print("SUCCESS: Multiple injections with backpropagation completed!")
    print("="*60)
    
    return logits_initial, logits_injected, hidden_embeddings, gradient_analysis

def demonstrate_iterative_injections():
    """
    Demonstrate iterative reasoning with different injection sequences.
    """
    print("\nIterative Reasoning with Different Injection Sequences")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=256, vocab_size=50257)
    config.n_layers = 3
    config.n_heads = 2
    config.n_embd = 64
    
    model = MultiInjectionGPT(config).to(device)
    reasoning_framework = MultiInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1, 2])
    
    input_ids = torch.randint(0, config.vocab_size, (2, 6)).to(device)
    
    # Define injection sequences for multiple iterations
    injection_sequences = [
        # Iteration 1: Extract from layer 1, inject into layer 1
        [
            {'extraction': {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'}},
            {'layer': 1, 'position': 0, 'hidden_embedding_key': 'token_2_layer_1', 'method': 'add', 'weight': 0.5}
        ],
        # Iteration 2: Extract from layer 2, inject into layer 2
        [
            {'extraction': {'layer_name': 'layer_2', 'position': 4, 'key': 'token_4_layer_2'}},
            {'layer': 2, 'position': 1, 'hidden_embedding_key': 'token_4_layer_2', 'method': 'replace'}
        ],
        # Iteration 3: Multiple injections
        [
            {'extraction': {'layer_name': 'layer_1', 'position': 3, 'key': 'token_3_layer_1'}},
            {'extraction': {'layer_name': 'layer_2', 'position': 1, 'key': 'token_1_layer_2'}},
            {'layer': 1, 'position': 5, 'hidden_embedding_key': 'token_3_layer_1', 'method': 'add'},
            {'layer': 2, 'position': 2, 'hidden_embedding_key': 'token_1_layer_2', 'method': 'weighted_add', 'weight': 0.7}
        ]
    ]
    
    print(f"Injection sequences defined for {len(injection_sequences)} iterations")
    
    # Perform iterative reasoning
    results = reasoning_framework.perform_iterative_reasoning_with_injections(
        input_ids, injection_sequences
    )
    
    print(f"\nIterative reasoning results:")
    for i, result in enumerate(results):
        print(f"  Iteration {i}:")
        print(f"    Logits shape: {result['logits'].shape}")
        print(f"    Injection specs: {len(result['injection_specs'])}")
        
        if i > 0:
            # Compare with previous iteration
            prev_logits = results[i-1]['logits']
            curr_logits = result['logits']
            diff = torch.norm(curr_logits - prev_logits).item()
            print(f"    L2 difference from previous: {diff:.6f}")
    
    return results

def demonstrate_gradient_flow_analysis():
    """
    Demonstrate detailed gradient flow analysis through injection points.
    """
    print("\nDetailed Gradient Flow Analysis")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=128, vocab_size=50257)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 32
    
    model = MultiInjectionGPT(config).to(device)
    reasoning_framework = MultiInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1])
    
    input_ids = torch.randint(0, config.vocab_size, (1, 4)).to(device)
    target_tokens = torch.randint(0, config.vocab_size, (1, 4)).to(device)
    
    # Initial pass
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    # Extract and inject
    extraction_specs = [
        {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1'}
    ]
    
    hidden_embeddings = model.extract_hidden_embeddings(hidden_states, extraction_specs)
    
    injection_specs = [
        {
            'layer': 1,
            'position': 0,
            'hidden_embedding_key': 'token_1_layer_1',
            'method': 'add',
            'weight': 1.0
        }
    ]
    
    # Perform injection with gradient tracking
    logits_injected, _ = reasoning_framework.perform_reasoning_pass_with_injections(
        input_ids, hidden_embeddings, injection_specs
    )
    
    # Compute loss and gradients
    loss_fn = F.cross_entropy
    loss = loss_fn(logits_injected.view(-1, logits_injected.size(-1)), target_tokens.view(-1))
    loss.backward()
    
    # Analyze gradients
    gradient_analysis = reasoning_framework.compute_gradient_flow_analysis(loss_fn, target_tokens)
    
    print(f"Gradient flow analysis:")
    print(f"  Total loss: {gradient_analysis['loss'].item():.6f}")
    print(f"  Number of parameters with gradients: {len(gradient_analysis['gradients'])}")
    
    # Show gradient norms by category
    model_grads = {k: v for k, v in gradient_analysis['gradient_norms'].items() 
                  if k.startswith('model_')}
    injection_grads = {k: v for k, v in gradient_analysis['gradient_norms'].items() 
                      if k.startswith('injection_projection_')}
    
    print(f"  Model parameter gradients: {len(model_grads)}")
    for name, norm in sorted(model_grads.items()):
        print(f"    {name}: {norm:.6f}")
    
    print(f"  Injection projection gradients: {len(injection_grads)}")
    for name, norm in sorted(injection_grads.items()):
        print(f"    {name}: {norm:.6f}")
    
    return gradient_analysis

if __name__ == "__main__":
    # Main demonstrations
    print("Starting Multiple Injection Demonstrations")
    print("="*80)
    
    # 1. Basic multiple injections
    logits_initial, logits_injected, hidden_embeddings, gradient_analysis = demonstrate_multiple_injections()
    
    # 2. Iterative injections
    iterative_results = demonstrate_iterative_injections()
    
    # 3. Gradient flow analysis
    gradient_analysis_detailed = demonstrate_gradient_flow_analysis()
    
    print("\n" + "="*80)
    print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print("="*80)




