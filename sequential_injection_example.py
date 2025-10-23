"""
Sequential Injection Example: Forward -> Inject -> Forward -> Inject -> ...

This demonstrates the sequential injection pattern where each injection
happens after a complete forward pass, with full gradient backpropagation.
"""

import torch
import torch.nn.functional as F
from model import GPTConfig
from sequential_injection_gpt import SequentialInjectionGPT, SequentialInjectionReasoningFramework

def demonstrate_sequential_injections():
    """
    Demonstrate sequential injections: forward -> inject -> forward -> inject -> ...
    """
    print("Sequential Hidden Embedding Injections")
    print("="*60)
    print("Pattern: Forward -> Inject -> Forward -> Inject -> ...")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=256, vocab_size=50257)
    config.n_layers = 3
    config.n_heads = 4
    config.n_embd = 128
    
    model = SequentialInjectionGPT(config).to(device)
    reasoning_framework = SequentialInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1, 2])
    
    # Create input data
    batch_size = 2
    seq_length = 6
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    target_tokens = torch.randint(0, config.vocab_size, (batch_size, seq_length)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_tokens.shape}")
    print(f"Input tokens:")
    for i in range(batch_size):
        print(f"  Batch {i}: {input_ids[i].tolist()}")
    
    # Define sequential injection sequence
    print(f"\nDefining sequential injection sequence:")
    
    injection_sequence = [
        # Step 1: Extract from layer 1, inject into layer 1
        {
            'extraction': {
                'layer_name': 'layer_1',
                'position': 2,
                'key': 'token_2_layer_1'
            },
            'injection': {
                'layer': 1,
                'position': 0,
                'method': 'add',
                'weight': 0.5
            }
        },
        # Step 2: Extract from layer 2, inject into layer 2
        {
            'extraction': {
                'layer_name': 'layer_2',
                'position': 4,
                'key': 'token_4_layer_2'
            },
            'injection': {
                'layer': 2,
                'position': 1,
                'method': 'replace'
            }
        },
        # Step 3: Extract from layer 1, inject into layer 2
        {
            'extraction': {
                'layer_name': 'layer_1',
                'position': 1,
                'key': 'token_1_layer_1'
            },
            'injection': {
                'layer': 2,
                'position': 3,
                'method': 'weighted_add',
                'weight': 0.3
            }
        },
        # Step 4: Extract from layer 2, inject into layer 1
        {
            'extraction': {
                'layer_name': 'layer_2',
                'position': 5,
                'key': 'token_5_layer_2'
            },
            'injection': {
                'layer': 1,
                'position': 4,
                'method': 'add',
                'weight': 0.8
            }
        }
    ]
    
    print(f"Sequential injection sequence with {len(injection_sequence)} steps:")
    for i, spec in enumerate(injection_sequence):
        extraction = spec['extraction']
        injection = spec['injection']
        print(f"  Step {i+1}: Extract {extraction['key']} from {extraction['layer_name']} pos {extraction['position']}")
        print(f"           -> Inject into layer {injection['layer']} pos {injection['position']} using {injection['method']}")
    
    # Perform sequential injections
    print(f"\nPerforming sequential injections:")
    print(f"Step 0: Initial forward pass")
    
    results = reasoning_framework.perform_sequential_injections(
        input_ids, injection_sequence
    )
    
    # Analyze results
    print(f"\nSequential injection results:")
    print(f"Total steps: {len(results)}")
    
    for i, result in enumerate(results):
        step_type = result['step_type']
        logits = result['logits']
        
        print(f"\nStep {i}: {step_type}")
        print(f"  Logits shape: {logits.shape}")
        print(f"  Logits norm: {torch.norm(logits).item():.4f}")
        
        if i > 0:
            # Compare with previous step
            prev_logits = results[i-1]['logits']
            diff = torch.norm(logits - prev_logits).item()
            print(f"  L2 difference from previous: {diff:.6f}")
            
            # Show injection details
            injection_spec = result['injection_spec']
            if injection_spec:
                extraction = injection_spec['extraction']
                injection = injection_spec['injection']
                print(f"  Injected: {extraction['key']} -> Layer {injection['layer']} Pos {injection['position']}")
    
    # Demonstrate gradient backpropagation
    print(f"\nDemonstrating gradient backpropagation:")
    
    # Get final logits
    final_logits = results[-1]['logits']
    
    # Compute loss
    loss_fn = F.cross_entropy
    loss = loss_fn(final_logits.view(-1, final_logits.size(-1)), target_tokens.view(-1))
    print(f"Final loss: {loss.item():.6f}")
    
    # Backward pass - gradients flow through all sequential injection points
    loss.backward()
    
    # Analyze gradient flow
    gradient_analysis = reasoning_framework.compute_gradient_flow_analysis(loss_fn, target_tokens)
    
    print(f"Gradient analysis:")
    print(f"  Total parameters with gradients: {len(gradient_analysis['gradients'])}")
    
    # Check gradient norms by category
    model_grads = {k: v for k, v in gradient_analysis['gradient_norms'].items() 
                  if k.startswith('model_')}
    injection_grads = {k: v for k, v in gradient_analysis['gradient_norms'].items() 
                      if k.startswith('injection_projection_')}
    
    print(f"  Model parameter gradients: {len(model_grads)}")
    print(f"  Injection projection gradients: {len(injection_grads)}")
    
    # Show some key gradient norms
    key_params = ['transformer.wte.weight', 'transformer.h.1.c_attn.c_attn.weight', 
                  'transformer.h.2.c_attn.c_attn.weight', 'lm_head.weight']
    for param in key_params:
        full_name = f'model_{param}'
        if full_name in gradient_analysis['gradient_norms']:
            norm = gradient_analysis['gradient_norms'][full_name]
            print(f"    {param}: {norm:.6f}")
    
    # Show overall progression
    print(f"\nOverall progression:")
    initial_logits = results[0]['logits']
    final_logits = results[-1]['logits']
    total_diff = torch.norm(final_logits - initial_logits).item()
    print(f"  Initial logits norm: {torch.norm(initial_logits).item():.4f}")
    print(f"  Final logits norm: {torch.norm(final_logits).item():.4f}")
    print(f"  Total L2 difference: {total_diff:.6f}")
    
    print(f"\n" + "="*60)
    print(f"SUCCESS: Sequential injections completed!")
    print(f"✓ Forward -> Inject -> Forward -> Inject pattern")
    print(f"✓ Full gradient backpropagation through all steps")
    print(f"✓ {len(injection_sequence)} sequential injection steps")
    print(f"="*60)
    
    return results, gradient_analysis

def demonstrate_sequential_vs_parallel():
    """
    Compare sequential injections vs parallel injections to show the difference.
    """
    print(f"\n\nSequential vs Parallel Injection Comparison")
    print(f"="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=128, vocab_size=50257)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 64
    
    model = SequentialInjectionGPT(config).to(device)
    reasoning_framework = SequentialInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1])
    
    input_ids = torch.randint(0, config.vocab_size, (1, 4)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Sequential approach
    print(f"\nSequential approach:")
    sequential_sequence = [
        {
            'extraction': {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1'},
            'injection': {'layer': 1, 'position': 0, 'method': 'add', 'weight': 0.5}
        },
        {
            'extraction': {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'},
            'injection': {'layer': 1, 'position': 3, 'method': 'replace'}
        }
    ]
    
    sequential_results = reasoning_framework.perform_sequential_injections(
        input_ids, sequential_sequence
    )
    
    print(f"  Sequential steps: {len(sequential_results)}")
    for i, result in enumerate(sequential_results):
        print(f"    Step {i}: {result['step_type']}, logits norm: {torch.norm(result['logits']).item():.4f}")
    
    # Show the difference between steps
    if len(sequential_results) > 1:
        step1_logits = sequential_results[1]['logits']
        step2_logits = sequential_results[2]['logits']
        diff = torch.norm(step2_logits - step1_logits).item()
        print(f"  Difference between injection steps: {diff:.6f}")
    
    print(f"\nSequential pattern:")
    print(f"  1. Forward pass -> Extract token_1")
    print(f"  2. Forward pass with token_1 injection -> Extract token_2")
    print(f"  3. Forward pass with token_2 injection")
    
    return sequential_results

if __name__ == "__main__":
    print("Sequential Injection Demonstrations")
    print("="*80)
    
    # Main demonstration
    results, gradient_analysis = demonstrate_sequential_injections()
    
    # Comparison demonstration
    sequential_results = demonstrate_sequential_vs_parallel()
    
    print(f"\n" + "="*80)
    print(f"ALL SEQUENTIAL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
    print(f"="*80)




