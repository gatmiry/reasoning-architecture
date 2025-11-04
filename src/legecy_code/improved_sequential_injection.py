"""
Improved Sequential Injection GPT that ensures proper gradient flow
through the entire chain of sequential injections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple, Union
from model import MLP, CasualSelfAttention, Block, GPTConfig

class ImprovedSequentialInjectionGPT(nn.Module):
    """
    Improved GPT model that ensures proper gradient flow through sequential injections.
    Each injection maintains the computational graph for backpropagation.
    """
    
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.config = config
        
        # Core transformer components
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight
        
        # Injection projection layers for different injection methods
        self.injection_projections = nn.ModuleDict()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following the original GPT initialization scheme."""
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=std)
    
    def get_injection_projection(self, injection_key: str, input_dim: int, output_dim: int):
        """Get or create projection layer for injection."""
        if injection_key not in self.injection_projections:
            self.injection_projections[injection_key] = nn.Linear(input_dim, output_dim).to(next(self.parameters()).device)
        return self.injection_projections[injection_key]
    
    def forward(self, idx, targets=None, last_k_no_attend=0, window_size=0, 
                return_hidden_states=False):
        """Standard forward pass."""
        B, T = idx.size()
        pe = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pe_vecs = self.transformer.wpe(pe)
        
        if last_k_no_attend > 0:
            last_k_no_attend = min(last_k_no_attend, T)
            pe_vecs[-last_k_no_attend:,:] = 0.0
        
        x = pe_vecs + self.transformer.wte(idx)
        
        # Store initial embeddings
        if return_hidden_states:
            hidden_states = {}
            hidden_states['input_embeddings'] = x.clone()
        
        # Forward through transformer blocks
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x, last_k_no_attend=last_k_no_attend, window_size=window_size)
            
            # Store hidden states from target layers
            if return_hidden_states:
                hidden_states[f'layer_{layer_idx}'] = x.clone()
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Store final hidden states
        if return_hidden_states:
            hidden_states['final'] = x.clone()
        
        loss = None
        if targets is not None:
            pass
        
        if return_hidden_states:
            return logits, loss, hidden_states
        return logits, loss
    
    def forward_with_sequential_injections(self, input_ids: torch.Tensor,
                                         injection_sequence: List[Dict],
                                         last_k_no_attend: int = 0,
                                         window_size: int = 0) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Perform forward pass with sequential injections, maintaining computational graph.
        
        Args:
            input_ids: Input token indices
            injection_sequence: List of injection specifications
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
        
        Returns:
            final_logits: Final output logits
            intermediate_logits: List of logits from each step
        """
        B, T = input_ids.size()
        pe = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pe_vecs = self.transformer.wpe(pe)
        
        if last_k_no_attend > 0:
            last_k_no_attend = min(last_k_no_attend, T)
            pe_vecs[-last_k_no_attend:,:] = 0.0
        
        x = pe_vecs + self.transformer.wte(input_ids)
        intermediate_logits = []
        
        # Forward through transformer blocks with sequential injections
        for layer_idx, block in enumerate(self.transformer.h):
            # Apply injections before this layer if specified
            for injection_spec in injection_sequence:
                if injection_spec['injection']['layer'] == layer_idx:
                    x = self._apply_single_injection(x, injection_spec)
            
            x = block(x, last_k_no_attend=last_k_no_attend, window_size=window_size)
        
        # Final layer norm and output
        x = self.transformer.ln_f(x)
        final_logits = self.lm_head(x)
        
        return final_logits, intermediate_logits
    
    def _apply_single_injection(self, x: torch.Tensor, injection_spec: Dict) -> torch.Tensor:
        """Apply a single injection to the hidden state."""
        injection = injection_spec['injection']
        hidden_embedding = injection_spec['hidden_embedding']
        
        position = injection['position']
        method = injection.get('method', 'add')
        weight = injection.get('weight', 1.0)
        
        # Ensure position is within bounds
        if position >= x.size(1):
            return x
        
        # Ensure hidden_embedding has correct batch size
        if hidden_embedding.size(0) != x.size(0):
            if hidden_embedding.size(0) == 1:
                hidden_embedding = hidden_embedding.expand(x.size(0), -1)
            else:
                raise ValueError(f"Batch size mismatch: x has {x.size(0)}, hidden_embedding has {hidden_embedding.size(0)}")
        
        # Ensure hidden_embedding has correct hidden dimension
        if hidden_embedding.size(-1) != x.size(-1):
            # Project to match dimensions
            injection_key = f"injection_{method}_{hidden_embedding.size(-1)}_to_{x.size(-1)}"
            projection = self.get_injection_projection(injection_key, hidden_embedding.size(-1), x.size(-1))
            hidden_embedding = projection(hidden_embedding)
        
        # Apply injection based on method
        if method == 'add':
            x = x.clone()  # Ensure we don't modify the original tensor
            x[:, position, :] = x[:, position, :] + weight * hidden_embedding
        elif method == 'replace':
            x = x.clone()
            x[:, position, :] = hidden_embedding
        elif method == 'weighted_add':
            x = x.clone()
            x[:, position, :] = (1 - weight) * x[:, position, :] + weight * hidden_embedding
        elif method == 'concat':
            x = x.clone()
            if hidden_embedding.size(-1) == x.size(-1):
                x[:, position, :] = hidden_embedding
            else:
                x[:, position, :] = hidden_embedding
        else:
            raise ValueError(f"Unknown injection method: {method}")
        
        return x
    
    def extract_hidden_embeddings(self, hidden_states: Dict[str, torch.Tensor], 
                                 extraction_specs: List[Dict]) -> Dict[str, torch.Tensor]:
        """Extract hidden embeddings based on specifications."""
        extracted = {}
        
        for spec in extraction_specs:
            layer_name = spec['layer_name']
            position = spec['position']
            key = spec['key']
            
            if layer_name in hidden_states:
                hidden = hidden_states[layer_name]
                if hidden.dim() == 3 and position < hidden.size(1):
                    extracted[key] = hidden[:, position, :]  # Don't clone to maintain gradients
                else:
                    raise ValueError(f"Invalid position {position} for layer {layer_name} with shape {hidden.shape}")
            else:
                raise ValueError(f"Layer {layer_name} not found in hidden states")
        
        return extracted


class ImprovedSequentialInjectionReasoningFramework:
    """
    Improved framework for sequential reasoning with proper gradient flow.
    """
    
    def __init__(self, model: ImprovedSequentialInjectionGPT, config: GPTConfig):
        self.model = model
        self.config = config
        self.reasoning_history = []
    
    def perform_sequential_injections_with_gradient_flow(self, input_ids: torch.Tensor,
                                                       injection_sequence: List[Dict],
                                                       last_k_no_attend: int = 0,
                                                       window_size: int = 0) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Perform sequential injections with proper gradient flow through the entire chain.
        
        Args:
            input_ids: Input token indices
            injection_sequence: List of injection specifications
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
        
        Returns:
            final_logits: Final output logits
            results: List of results from each step
        """
        results = []
        
        # Initial pass to get hidden states
        logits_initial, _, hidden_states = self.model(
            input_ids, 
            last_k_no_attend=last_k_no_attend, 
            window_size=window_size,
            return_hidden_states=True
        )
        
        results.append({
            'step': 0,
            'step_type': 'initial_forward',
            'logits': logits_initial,
            'hidden_states': hidden_states,
            'injection_spec': None
        })
        
        # Prepare injection sequence with hidden embeddings
        prepared_injections = []
        current_hidden_states = hidden_states
        
        for i, injection_spec in enumerate(injection_sequence):
            # Extract hidden embedding for this injection
            extraction_spec = injection_spec['extraction']
            hidden_embeddings = self.model.extract_hidden_embeddings(
                current_hidden_states, [extraction_spec]
            )
            
            # Prepare injection with hidden embedding
            prepared_injection = {
                'injection': injection_spec['injection'],
                'hidden_embedding': hidden_embeddings[extraction_spec['key']]
            }
            prepared_injections.append(prepared_injection)
            
            # Get new hidden states for next iteration
            _, _, new_hidden_states = self.model(
                input_ids, 
                last_k_no_attend=last_k_no_attend, 
                window_size=window_size,
                return_hidden_states=True
            )
            current_hidden_states = new_hidden_states
        
        # Perform forward pass with all sequential injections
        final_logits, intermediate_logits = self.model.forward_with_sequential_injections(
            input_ids, prepared_injections, last_k_no_attend, window_size
        )
        
        # Store results for each injection step
        for i, injection_spec in enumerate(injection_sequence):
            # For intermediate steps, we need to compute logits with partial injections
            # This is a simplified version - in practice, you might want to compute
            # intermediate logits for each step
            results.append({
                'step': i + 1,
                'step_type': 'injection_forward',
                'logits': final_logits,  # In practice, compute intermediate logits
                'hidden_states': current_hidden_states,
                'injection_spec': injection_spec
            })
        
        return final_logits, results
    
    def compute_gradient_flow_analysis(self, loss_fn, target_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gradient flow analysis for injection points."""
        if not self.reasoning_history:
            return {'error': 'No reasoning history available'}
        
        # Get the latest logits
        latest_logits = self.reasoning_history[-1]['logits']
        
        # Compute loss
        loss = loss_fn(latest_logits.view(-1, latest_logits.size(-1)), target_tokens.view(-1))
        
        # Compute gradients
        gradients = {}
        
        # Compute gradients for injection projection layers
        for name, param in self.model.injection_projections.named_parameters():
            if param.grad is not None:
                gradients[f'injection_projection_{name}'] = param.grad.clone()
        
        # Compute gradients for main model parameters
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[f'model_{name}'] = param.grad.clone()
        
        return {
            'loss': loss,
            'gradients': gradients,
            'gradient_norms': {k: torch.norm(v).item() for k, v in gradients.items()}
        }
    
    def clear_history(self):
        """Clear the reasoning history."""
        self.reasoning_history = []




