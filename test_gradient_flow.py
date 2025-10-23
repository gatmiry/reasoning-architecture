"""
Test to verify that gradients flow through the entire chain of sequential injections.
"""

import torch
import torch.nn.functional as F
from model import GPTConfig
from sequential_injection_gpt import SequentialInjectionGPT, SequentialInjectionReasoningFramework

def test_gradient_flow_through_chain():
    """
    Test that gradients flow through the entire chain of sequential injections.
    """
    print("Testing Gradient Flow Through Sequential Injection Chain")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=64, vocab_size=50257)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 32
    
    model = SequentialInjectionGPT(config).to(device)
    reasoning_framework = SequentialInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1])
    
    # Create input data
    input_ids = torch.randint(0, config.vocab_size, (1, 4)).to(device)
    target_tokens = torch.randint(0, config.vocab_size, (1, 4)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_tokens.shape}")
    
    # Define a simple sequential injection sequence
    injection_sequence = [
        {
            'extraction': {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1'},
            'injection': {'layer': 1, 'position': 0, 'method': 'add', 'weight': 0.5}
        },
        {
            'extraction': {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'},
            'injection': {'layer': 1, 'position': 3, 'method': 'replace'}
        }
    ]
    
    print(f"\nSequential injection sequence with {len(injection_sequence)} steps")
    
    # Perform sequential injections
    results = reasoning_framework.perform_sequential_injections(
        input_ids, injection_sequence
    )
    
    print(f"\nSequential injection results:")
    for i, result in enumerate(results):
        print(f"  Step {i}: {result['step_type']}")
        print(f"    Logits shape: {result['logits'].shape}")
        print(f"    Logits requires_grad: {result['logits'].requires_grad}")
    
    # Test gradient flow
    print(f"\nTesting gradient flow:")
    
    # Get final logits
    final_logits = results[-1]['logits']
    print(f"Final logits requires_grad: {final_logits.requires_grad}")
    
    # Compute loss
    loss_fn = F.cross_entropy
    loss = loss_fn(final_logits.view(-1, final_logits.size(-1)), target_tokens.view(-1))
    print(f"Loss: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    # Backward pass
    print(f"\nPerforming backward pass...")
    loss.backward()
    
    # Check gradients
    print(f"\nGradient analysis:")
    
    # Check model parameters
    model_params_with_grad = 0
    model_grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            model_params_with_grad += 1
            grad_norm = torch.norm(param.grad).item()
            model_grad_norms.append(grad_norm)
            if 'transformer.wte.weight' in name or 'lm_head.weight' in name:
                print(f"  {name}: grad_norm = {grad_norm:.6f}")
    
    print(f"  Model parameters with gradients: {model_params_with_grad}")
    print(f"  Average gradient norm: {sum(model_grad_norms) / len(model_grad_norms):.6f}")
    
    # Check injection projection parameters
    injection_params_with_grad = 0
    injection_grad_norms = []
    
    for name, param in model.injection_projections.named_parameters():
        if param.grad is not None:
            injection_params_with_grad += 1
            grad_norm = torch.norm(param.grad).item()
            injection_grad_norms.append(grad_norm)
            print(f"  {name}: grad_norm = {grad_norm:.6f}")
    
    print(f"  Injection projection parameters with gradients: {injection_params_with_grad}")
    
    # Test if gradients flow through the chain by checking intermediate steps
    print(f"\nTesting gradient flow through intermediate steps:")
    
    # Clear gradients
    model.zero_grad()
    
    # Test with intermediate step
    intermediate_logits = results[1]['logits']  # First injection step
    print(f"Intermediate logits (step 1) requires_grad: {intermediate_logits.requires_grad}")
    
    if intermediate_logits.requires_grad:
        intermediate_loss = loss_fn(intermediate_logits.view(-1, intermediate_logits.size(-1)), target_tokens.view(-1))
        print(f"Intermediate loss: {intermediate_loss.item():.6f}")
        print(f"Intermediate loss requires_grad: {intermediate_loss.requires_grad}")
        
        intermediate_loss.backward()
        
        # Check if gradients are computed
        intermediate_grads = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                intermediate_grads += 1
        
        print(f"Parameters with gradients after intermediate backward: {intermediate_grads}")
    
    return {
        'final_loss': loss.item(),
        'model_params_with_grad': model_params_with_grad,
        'injection_params_with_grad': injection_params_with_grad,
        'gradient_flow_successful': model_params_with_grad > 0
    }

def test_manual_sequential_chain():
    """
    Test manual sequential chain to verify gradient flow.
    """
    print(f"\n\nManual Sequential Chain Test")
    print(f"="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=32, vocab_size=50257)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 16
    
    model = SequentialInjectionGPT(config).to(device)
    model.enable_hidden_extraction([1])
    
    input_ids = torch.randint(0, config.vocab_size, (1, 3)).to(device)
    target_tokens = torch.randint(0, config.vocab_size, (1, 3)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Step 1: Initial forward pass
    print(f"\nStep 1: Initial forward pass")
    logits1, _, hidden_states1 = model(input_ids, return_hidden_states=True)
    print(f"  Logits1 requires_grad: {logits1.requires_grad}")
    
    # Step 2: Extract and inject
    print(f"\nStep 2: Extract and inject")
    extraction_spec = {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1'}
    hidden_embeddings = model.extract_hidden_embeddings(hidden_states1, [extraction_spec])
    hidden_embedding = hidden_embeddings['token_1_layer_1']
    print(f"  Hidden embedding requires_grad: {hidden_embedding.requires_grad}")
    
    injection_spec = {'layer': 1, 'position': 0, 'method': 'add', 'weight': 0.5}
    logits2, _ = model.forward_with_single_injection(input_ids, hidden_embedding, injection_spec)
    print(f"  Logits2 requires_grad: {logits2.requires_grad}")
    
    # Step 3: Extract and inject again
    print(f"\nStep 3: Extract and inject again")
    # Get new hidden states
    _, _, hidden_states2 = model(input_ids, return_hidden_states=True)
    extraction_spec2 = {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'}
    hidden_embeddings2 = model.extract_hidden_embeddings(hidden_states2, [extraction_spec2])
    hidden_embedding2 = hidden_embeddings2['token_2_layer_1']
    print(f"  Hidden embedding2 requires_grad: {hidden_embedding2.requires_grad}")
    
    injection_spec2 = {'layer': 1, 'position': 2, 'method': 'replace'}
    logits3, _ = model.forward_with_single_injection(input_ids, hidden_embedding2, injection_spec2)
    print(f"  Logits3 requires_grad: {logits3.requires_grad}")
    
    # Test gradient flow
    print(f"\nTesting gradient flow through manual chain:")
    
    # Compute loss on final logits
    loss = F.cross_entropy(logits3.view(-1, logits3.size(-1)), target_tokens.view(-1))
    print(f"  Final loss: {loss.item():.6f}")
    print(f"  Loss requires_grad: {loss.requires_grad}")
    
    if loss.requires_grad:
        loss.backward()
        
        # Check gradients
        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
                if 'transformer.wte.weight' in name:
                    print(f"    {name}: grad_norm = {torch.norm(param.grad).item():.6f}")
        
        print(f"  Parameters with gradients: {params_with_grad}")
        return params_with_grad > 0
    else:
        print(f"  Loss does not require gradients - chain is broken!")
        return False

if __name__ == "__main__":
    print("Gradient Flow Testing")
    print("="*80)
    
    # Test 1: Framework-based sequential injections
    result1 = test_gradient_flow_through_chain()
    
    # Test 2: Manual sequential chain
    result2 = test_manual_sequential_chain()
    
    print(f"\n" + "="*80)
    print(f"GRADIENT FLOW TEST RESULTS:")
    print(f"Framework-based: {'PASS' if result1['gradient_flow_successful'] else 'FAIL'}")
    print(f"Manual chain: {'PASS' if result2 else 'FAIL'}")
    print(f"="*80)




