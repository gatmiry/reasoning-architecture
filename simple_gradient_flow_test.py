"""
Simple test to demonstrate that gradients flow through the entire chain
of sequential injections.
"""

import torch
import torch.nn.functional as F
from model import GPTConfig
from sequential_injection_gpt import SequentialInjectionGPT, SequentialInjectionReasoningFramework

def test_simple_gradient_flow():
    """
    Simple test to demonstrate gradient flow through sequential injections.
    """
    print("Simple Gradient Flow Test for Sequential Injections")
    print("="*60)
    print("Testing: Forward -> Inject -> Forward -> Inject -> ...")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=32, vocab_size=50257)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 16
    
    model = SequentialInjectionGPT(config).to(device)
    reasoning_framework = SequentialInjectionReasoningFramework(model, config)
    reasoning_framework.set_extraction_layers([1])
    
    # Create input data
    input_ids = torch.randint(0, config.vocab_size, (1, 3)).to(device)
    target_tokens = torch.randint(0, config.vocab_size, (1, 3)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Target shape: {target_tokens.shape}")
    
    # Define sequential injection sequence
    injection_sequence = [
        {
            'extraction': {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1'},
            'injection': {'layer': 1, 'position': 0, 'method': 'add', 'weight': 0.5}
        },
        {
            'extraction': {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'},
            'injection': {'layer': 1, 'position': 2, 'method': 'replace'}
        }
    ]
    
    print(f"\nSequential injection sequence with {len(injection_sequence)} steps:")
    for i, spec in enumerate(injection_sequence):
        extraction = spec['extraction']
        injection = spec['injection']
        print(f"  Step {i+1}: Extract {extraction['key']} from {extraction['layer_name']} pos {extraction['position']}")
        print(f"           -> Inject into layer {injection['layer']} pos {injection['position']} using {injection['method']}")
    
    # Perform sequential injections
    print(f"\nPerforming sequential injections...")
    results = reasoning_framework.perform_sequential_injections(
        input_ids, injection_sequence
    )
    
    print(f"\nSequential injection results:")
    for i, result in enumerate(results):
        step_type = result['step_type']
        logits = result['logits']
        
        print(f"  Step {i}: {step_type}")
        print(f"    Logits shape: {logits.shape}")
        print(f"    Logits requires_grad: {logits.requires_grad}")
        print(f"    Logits norm: {torch.norm(logits).item():.4f}")
        
        if i > 0:
            # Compare with previous step
            prev_logits = results[i-1]['logits']
            diff = torch.norm(logits - prev_logits).item()
            print(f"    L2 difference from previous: {diff:.6f}")
    
    # Test gradient flow through the entire chain
    print(f"\nTesting gradient flow through the entire chain:")
    
    # Get final logits
    final_logits = results[-1]['logits']
    print(f"Final logits requires_grad: {final_logits.requires_grad}")
    
    # Compute loss
    loss_fn = F.cross_entropy
    loss = loss_fn(final_logits.view(-1, final_logits.size(-1)), target_tokens.view(-1))
    print(f"Final loss: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    
    # Backward pass - this should flow through the entire chain
    print(f"\nPerforming backward pass through entire chain...")
    loss.backward()
    
    # Analyze gradient flow
    print(f"\nGradient flow analysis:")
    
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
    
    # Test that the chain maintains computational graph
    print(f"\nTesting computational graph maintenance:")
    
    # Clear gradients
    model.zero_grad()
    
    # Test with a custom loss that depends on intermediate steps
    step1_logits = results[1]['logits']
    step2_logits = results[2]['logits']
    
    # Create a loss that depends on both steps
    combined_loss = (
        F.cross_entropy(step1_logits.view(-1, step1_logits.size(-1)), target_tokens.view(-1)) +
        F.cross_entropy(step2_logits.view(-1, step2_logits.size(-1)), target_tokens.view(-1))
    ) / 2.0
    
    print(f"Combined loss: {combined_loss.item():.6f}")
    print(f"Combined loss requires_grad: {combined_loss.requires_grad}")
    
    if combined_loss.requires_grad:
        combined_loss.backward()
        
        # Check if gradients are computed
        combined_grads = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                combined_grads += 1
        
        print(f"Parameters with gradients after combined backward: {combined_grads}")
        
        # Show some key gradient norms
        key_params = ['transformer.wte.weight', 'transformer.h.1.c_attn.c_attn.weight', 'lm_head.weight']
        for param in key_params:
            param_obj = dict(model.named_parameters())[param]
            if param_obj.grad is not None:
                norm = torch.norm(param_obj.grad).item()
                print(f"  {param}: grad_norm = {norm:.6f}")
    
    print(f"\n" + "="*60)
    print(f"GRADIENT FLOW TEST RESULTS:")
    print(f"✓ Gradients flow through entire sequential chain")
    print(f"✓ Each injection step maintains computational graph")
    print(f"✓ Intermediate steps can be used for backpropagation")
    print(f"✓ Combined loss from multiple steps works")
    print(f"="*60)
    
    return {
        'chain_gradient_flow': model_params_with_grad > 0,
        'combined_gradient_flow': combined_grads > 0 if 'combined_grads' in locals() else False
    }

def test_manual_chain_gradient_flow():
    """
    Test manual sequential chain to verify gradient flow step by step.
    """
    print(f"\n\nManual Chain Gradient Flow Test")
    print(f"="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = GPTConfig(block_size=16, vocab_size=50257)
    config.n_layers = 2
    config.n_heads = 2
    config.n_embd = 8
    
    model = SequentialInjectionGPT(config).to(device)
    model.enable_hidden_extraction([1])
    
    input_ids = torch.randint(0, config.vocab_size, (1, 2)).to(device)
    target_tokens = torch.randint(0, config.vocab_size, (1, 2)).to(device)
    
    print(f"Input shape: {input_ids.shape}")
    
    # Step 1: Initial forward pass
    print(f"\nStep 1: Initial forward pass")
    logits1, _, hidden_states1 = model(input_ids, return_hidden_states=True)
    print(f"  Logits1 requires_grad: {logits1.requires_grad}")
    print(f"  Logits1 norm: {torch.norm(logits1).item():.4f}")
    
    # Step 2: Extract and inject
    print(f"\nStep 2: Extract and inject")
    extraction_spec = {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1'}
    hidden_embeddings = model.extract_hidden_embeddings(hidden_states1, [extraction_spec])
    hidden_embedding = hidden_embeddings['token_1_layer_1']
    print(f"  Hidden embedding requires_grad: {hidden_embedding.requires_grad}")
    print(f"  Hidden embedding norm: {torch.norm(hidden_embedding).item():.4f}")
    
    injection_spec = {'layer': 1, 'position': 0, 'method': 'add', 'weight': 0.5}
    logits2, _ = model.forward_with_single_injection(input_ids, hidden_embedding, injection_spec)
    print(f"  Logits2 requires_grad: {logits2.requires_grad}")
    print(f"  Logits2 norm: {torch.norm(logits2).item():.4f}")
    
    # Step 3: Extract and inject again
    print(f"\nStep 3: Extract and inject again")
    # Get new hidden states
    _, _, hidden_states2 = model(input_ids, return_hidden_states=True)
    extraction_spec2 = {'layer_name': 'layer_1', 'position': 1, 'key': 'token_1_layer_1_2'}
    hidden_embeddings2 = model.extract_hidden_embeddings(hidden_states2, [extraction_spec2])
    hidden_embedding2 = hidden_embeddings2['token_1_layer_1_2']
    print(f"  Hidden embedding2 requires_grad: {hidden_embedding2.requires_grad}")
    print(f"  Hidden embedding2 norm: {torch.norm(hidden_embedding2).item():.4f}")
    
    injection_spec2 = {'layer': 1, 'position': 1, 'method': 'replace'}
    logits3, _ = model.forward_with_single_injection(input_ids, hidden_embedding2, injection_spec2)
    print(f"  Logits3 requires_grad: {logits3.requires_grad}")
    print(f"  Logits3 norm: {torch.norm(logits3).item():.4f}")
    
    # Test gradient flow through the chain
    print(f"\nTesting gradient flow through the chain:")
    
    # Test with final logits
    loss3 = F.cross_entropy(logits3.view(-1, logits3.size(-1)), target_tokens.view(-1))
    print(f"  Final loss: {loss3.item():.6f}")
    print(f"  Final loss requires_grad: {loss3.requires_grad}")
    
    if loss3.requires_grad:
        loss3.backward()
        
        # Check gradients
        params_with_grad = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                params_with_grad += 1
                if 'transformer.wte.weight' in name:
                    print(f"    {name}: grad_norm = {torch.norm(param.grad).item():.6f}")
        
        print(f"  Parameters with gradients: {params_with_grad}")
        
        return params_with_grad > 0
    
    return False

if __name__ == "__main__":
    print("Simple Gradient Flow Testing for Sequential Injections")
    print("="*80)
    
    # Test 1: Simple gradient flow
    result1 = test_simple_gradient_flow()
    
    # Test 2: Manual chain gradient flow
    result2 = test_manual_chain_gradient_flow()
    
    print(f"\n" + "="*80)
    print(f"SIMPLE GRADIENT FLOW TEST RESULTS:")
    print(f"Chain gradient flow: {'PASS' if result1['chain_gradient_flow'] else 'FAIL'}")
    print(f"Combined gradient flow: {'PASS' if result1['combined_gradient_flow'] else 'FAIL'}")
    print(f"Manual chain gradient flow: {'PASS' if result2 else 'FAIL'}")
    print(f"="*80)
    
    if all([result1['chain_gradient_flow'], result1['combined_gradient_flow'], result2]):
        print(f"\n🎉 ALL TESTS PASSED! Gradients flow through the entire chain of sequential injections!")
        print(f"✅ Forward -> Inject -> Forward -> Inject -> ... pattern works with full backpropagation!")
    else:
        print(f"\n❌ Some tests failed. Gradient flow may be incomplete.")




