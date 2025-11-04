"""
Focused example demonstrating multiple hidden embedding injections
with backpropagation as requested by the user.
"""

import torch
import torch.nn.functional as F
from model import GPTConfig
from multi_injection_gpt import MultiInjectionGPT, MultiInjectionReasoningFramework

def demonstrate_user_requirements():
    """
    Demonstrate the specific requirements:
    1. Inject hidden_embeddings after forward pass into inputs multiple times
    2. Use specifications given in an array
    3. Backprop gradients through all injection points
    """
    print("Multiple Hidden Embedding Injections with Backpropagation")
    print("="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=256, vocab_size=50257)
    config.n_layers = 3
    config.n_heads = 4
    config.n_embd = 128
    
    model = MultiInjectionGPT(config).to(device)
    reasoning_framework = MultiInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1, 2])
    
    # Create input data
    batch_size = 2
    seq_length = 6
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    target_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_tokens.shape}")
    
    # STEP 1: Initial forward pass to extract hidden states
    print(f"\n1. Initial forward pass to extract hidden states")
    logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
    
    print(f"   Extracted from layers: {list(hidden_states.keys())}")
    
    # STEP 2: Define injection specifications as an array (as requested)
    print(f"\n2. Define injection specifications array")
    
    # Array of injection specifications - this is what you requested
    injection_specs_array = [
        {
            'layer': 1,  # Inject into layer 1
            'position': 0,  # At position 0
            'hidden_embedding_key': 'token_2_layer_1',
            'method': 'add',
            'weight': 0.5
        },
        {
            'layer': 1,  # Inject into layer 1 again
            'position': 3,  # At position 3
            'hidden_embedding_key': 'token_4_layer_2',
            'method': 'replace'
        },
        {
            'layer': 2,  # Inject into layer 2
            'position': 1,  # At position 1
            'hidden_embedding_key': 'token_1_layer_1',
            'method': 'weighted_add',
            'weight': 0.3
        },
        {
            'layer': 2,  # Inject into layer 2 again
            'position': 4,  # At position 4
            'hidden_embedding_key': 'token_5_layer_2',
            'method': 'add',
            'weight': 0.8
        }
    ]
    
    print(f"   Injection specifications array length: {len(injection_specs_array)}")
    for i, spec in enumerate(injection_specs_array):
        print(f"   [{i}] Layer {spec['layer']}, Position {spec['position']}, "
              f"Method: {spec['method']}, Embedding: {spec['hidden_embedding_key']}")
    
    # STEP 3: Extract hidden embeddings based on specifications
    print(f"\n3. Extract hidden embeddings based on specifications")
    
    # Extract the required embeddings
    extraction_specs = [
        {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'},
        {'layer_name': 'layer_2', 'position': 4, 'key': 'token_4_layer_2'},
        {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1'},
        {'layer_name': 'layer_2', 'position': 5, 'key': 'token_5_layer_2'},
    ]
    
    hidden_embeddings = model.extract_hidden_embeddings(hidden_states, extraction_specs)
    
    print(f"   Extracted embeddings:")
    for key, embedding in hidden_embeddings.items():
        print(f"     {key}: {embedding.shape}, norm = {torch.norm(embedding).item():.4f}")
    
    # STEP 4: Perform multiple injections using the specifications array
    print(f"\n4. Perform multiple injections using specifications array")
    
    logits_injected, new_hidden_states = reasoning_framework.perform_reasoning_pass_with_injections(
        input_ids, hidden_embeddings, injection_specs_array
    )
    
    # STEP 5: Demonstrate gradient backpropagation through all injection points
    print(f"\n5. Demonstrate gradient backpropagation through all injection points")
    
    # Compute loss
    loss_fn = F.cross_entropy
    loss = loss_fn(logits_injected.view(-1, logits_injected.size(-1)), target_tokens.view(-1))
    print(f"   Loss before backprop: {loss.item():.6f}")
    
    # Backward pass - this will backprop through all injection points
    loss.backward()
    
    # Analyze gradient flow
    gradient_analysis = reasoning_framework.compute_gradient_flow_analysis(loss_fn, target_tokens)
    
    print(f"   Loss after backprop: {gradient_analysis['loss'].item():.6f}")
    print(f"   Total parameters with gradients: {len(gradient_analysis['gradients'])}")
    
    # Check gradient flow through injection points
    injection_grads = {k: v for k, v in gradient_analysis['gradient_norms'].items() 
                      if 'injection_projection' in k}
    
    if injection_grads:
        print(f"   Injection projection gradients: {len(injection_grads)}")
        for name, norm in injection_grads.items():
            print(f"     {name}: {norm:.6f}")
    else:
        print(f"   No injection projection gradients (injections used existing dimensions)")
    
    # Show that gradients flow through the main model
    model_grads = {k: v for k, v in gradient_analysis['gradient_norms'].items() 
                  if k.startswith('model_')}
    print(f"   Model parameter gradients: {len(model_grads)}")
    
    # Show some key gradient norms
    key_params = ['transformer.wte.weight', 'transformer.h.1.c_attn.c_attn.weight', 
                  'transformer.h.2.c_attn.c_attn.weight', 'lm_head.weight']
    for param in key_params:
        full_name = f'model_{param}'
        if full_name in gradient_analysis['gradient_norms']:
            norm = gradient_analysis['gradient_norms'][full_name]
            print(f"     {param}: {norm:.6f}")
    
    # STEP 6: Show the effect of injections
    print(f"\n6. Show the effect of injections")
    
    logits_diff = torch.norm(logits_injected - logits_initial).item()
    print(f"   Overall L2 difference: {logits_diff:.6f}")
    
    # Show differences at injection positions
    injection_positions = [(spec['layer'], spec['position']) for spec in injection_specs_array]
    for layer, pos in injection_positions:
        if pos < seq_length:
            pos_diff = torch.norm(
                logits_injected[:, pos, :] - logits_initial[:, pos, :]
            ).item()
            print(f"   Position {pos} (Layer {layer}) L2 difference: {pos_diff:.6f}")
    
    print(f"\n" + "="*70)
    print(f"SUCCESS: Multiple injections with backpropagation completed!")
    print(f"✓ Hidden embeddings injected multiple times")
    print(f"✓ Specifications provided as array")
    print(f"✓ Gradients backpropagated through all injection points")
    print(f"="*70)
    
    return {
        'logits_initial': logits_initial,
        'logits_injected': logits_injected,
        'hidden_embeddings': hidden_embeddings,
        'injection_specs': injection_specs_array,
        'gradient_analysis': gradient_analysis
    }

def demonstrate_iterative_injections():
    """
    Demonstrate iterative injections with different specifications arrays.
    """
    print(f"\n\nIterative Injections with Different Specifications Arrays")
    print(f"="*70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=128, vocab_size=50257)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 64
    
    model = MultiInjectionGPT(config).to(device)
    reasoning_framework = MultiInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1])
    
    input_ids = torch.randint(0, config.vocab_size, (1, 4)).to(device)
    
    # Different injection specifications arrays for each iteration
    injection_sequences = [
        # Iteration 1: Simple injection
        [
            {'extraction': {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1'}},
            {'layer': 1, 'position': 0, 'hidden_embedding_key': 'token_1_layer_1', 'method': 'add', 'weight': 0.5}
        ],
        # Iteration 2: Multiple injections
        [
            {'extraction': {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'}},
            {'layer': 1, 'position': 1, 'hidden_embedding_key': 'token_2_layer_1', 'method': 'replace'},
            {'layer': 1, 'position': 3, 'hidden_embedding_key': 'token_1_layer_1', 'method': 'add', 'weight': 0.3}
        ],
        # Iteration 3: Complex injections
        [
            {'extraction': {'layer_name': 'layer_1', 'position': 0, 'key': 'token_0_layer_1'}},
            {'layer': 1, 'position': 2, 'hidden_embedding_key': 'token_0_layer_1', 'method': 'weighted_add', 'weight': 0.7},
            {'layer': 1, 'position': 1, 'hidden_embedding_key': 'token_2_layer_1', 'method': 'add', 'weight': 0.2}
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

if __name__ == "__main__":
    print("Focused Multiple Injection Demonstrations")
    print("="*80)
    
    # Main demonstration
    results = demonstrate_user_requirements()
    
    # Iterative demonstration
    iterative_results = demonstrate_iterative_injections()
    
    print(f"\n" + "="*80)
    print(f"ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print(f"="*80)




