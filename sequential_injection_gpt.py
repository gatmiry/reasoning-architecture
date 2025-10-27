"""
Sequential Injection GPT model that performs:
1. Forward pass
2. Inject first hidden embedding
3. Forward pass again
4. Inject second hidden embedding
5. Forward pass again
... and so on

Each injection happens after a complete forward pass, with full gradient backpropagation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple, Union
from model import MLP, CasualSelfAttention, Block, GPTConfig

class SequentialInjectionGPT(nn.Module):
    """
    GPT model that supports sequential hidden embedding injections.
    Each injection happens after a complete forward pass.
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
        
        # Store hidden states from intermediate layers
        self.hidden_states = {}
        self.extract_hidden = True
        self.target_layers = []
        
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
    
    def forward(self, idx, targets=None, return_hidden_states=False):
        """
        Standard forward pass with optional hidden state extraction.
        """
        #print("DEBUG: Entered SequentialInjectionGPT.forward()")
        #print(f"DEBUG: idx shape: {idx.shape}")
        #print(f"DEBUG: return_hidden_states: {return_hidden_states}")
        
        B, T = idx.size()
        pe = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pe_vecs = self.transformer.wpe(pe)
        x = pe_vecs + self.transformer.wte(idx)
        
        # Store initial embeddings as layer_0
        if return_hidden_states:
            hidden_states = {}
            hidden_states['layer_0'] = x
        
        # Always store layer_0 in hidden_states
        self.hidden_states['layer_0'] = x
        if self.training:
            #print('im retaining gradients for x layer_0')
            x.retain_grad()  # Retain gradients for this tensor
        
        # Forward through transformer blocks
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x)
            
            # Always store hidden states from all layers
            self.hidden_states[f'layer_{layer_idx + 1}'] = x
            if self.training:
                x.retain_grad()  # Retain gradients for this tensor
            
            if return_hidden_states:
                hidden_states[f'layer_{layer_idx + 1}'] = x
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        
        # Store final hidden states BEFORE computing logits
        # Always store final hidden states
        self.hidden_states['final'] = x
        if self.training:
            x.retain_grad()  # Retain gradients for this tensor
        
        if return_hidden_states:
            hidden_states['final'] = x
        
        # Compute logits
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Loss computation can be added here if needed
            pass
        
        if return_hidden_states:
            return logits, loss, hidden_states
        return logits, loss
    
    def forward_with_single_injection(self, input_ids: torch.Tensor,
                                    hidden_embedding: torch.Tensor,
                                    injection_spec: Dict, 
                                    return_hidden_states: bool = False) -> Tuple[torch.Tensor, None, Optional[Dict]]:
        """
        Perform a single forward pass with one hidden embedding injection.
        
        Args:
            input_ids: Input token indices
            hidden_embedding: Hidden embedding to inject [batch, hidden_dim]
            injection_spec: Injection specification dict
            return_hidden_states: Whether to return hidden states
        
        Returns:
            logits: Output logits
            loss: Loss (None in this case)
            hidden_states: Hidden states if return_hidden_states=True, else None
        """
        B, T = input_ids.size()
        #pe = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        #pe_vecs = self.transformer.wpe(pe)
        #x = pe_vecs + self.transformer.wte(input_ids)
        x = self.hidden_states['layer_0']
        # Store hidden states if requested
        hidden_states = {} if return_hidden_states else None
        if return_hidden_states:
            hidden_states['layer_0'] = x
        
        # Apply injection at layer_0 (initial embeddings) if specified
        if injection_spec['layer'] == 0:
            x = self._apply_single_injection(x, hidden_embedding, injection_spec)
            if return_hidden_states:
                hidden_states['layer_0'] = x
        
        # Forward through transformer blocks with single injection
        for layer_idx, block in enumerate(self.transformer.h):
            # Apply injection before this layer if specified (layer_idx+1 because layer_0 is initial embeddings)
            if injection_spec['layer'] == layer_idx + 1:
                x = self._apply_single_injection(x, hidden_embedding, injection_spec)
            
            x = block(x)
            
            # Store hidden states if requested
            if return_hidden_states:
                hidden_states[f'layer_{layer_idx + 1}'] = x
        
        # Final layer norm and output
        x = self.transformer.ln_f(x)
        if return_hidden_states:
            hidden_states['final'] = x
        
        logits = self.lm_head(x)
        
        return logits, None, hidden_states
    
    def _apply_single_injection(self, x: torch.Tensor, hidden_embedding: torch.Tensor, 
                               spec: Dict) -> torch.Tensor:
        """
        Apply a single injection to the hidden state.
        
        Args:
            x: Current hidden state tensor [batch, seq_len, hidden_dim]
            hidden_embedding: Hidden embedding to inject [batch, hidden_dim]
            spec: Injection specification dict
        
        Returns:
            Modified hidden state tensor
        """

        x = x.clone()

        position = spec['position']
        method = spec.get('method', 'add')
        weight = spec.get('weight', 1.0)
        
        
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
            x[:, position, :] = x[:, position, :] + weight * hidden_embedding
        elif method == 'replace':
            x[:, position, :] = hidden_embedding
        elif method == 'weighted_add':
            x[:, position, :] = (1 - weight) * x[:, position, :] + weight * hidden_embedding
        elif method == 'concat':
            # For concat, we need to handle the dimension increase
            if hidden_embedding.size(-1) == x.size(-1):
                x[:, position, :] = hidden_embedding
            else:
                # Project and replace
                x[:, position, :] = hidden_embedding
        else:
            raise ValueError(f"Unknown injection method: {method}")
        
        return x
    
    def extract_hidden_embeddings(self, hidden_states: Dict[str, torch.Tensor], 
                                 extraction_specs: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Extract hidden embeddings based on specifications.
        """
        extracted = {}
        
        for spec in extraction_specs:
            layer_name = spec['layer_name']
            position = spec['position']
            key = spec['key']
            
            if layer_name in hidden_states:
                hidden = hidden_states[layer_name]
                if hidden.dim() == 3 and position < hidden.size(1):
                    extracted[key] = hidden[:, position, :]
                else:
                    raise ValueError(f"Invalid position {position} for layer {layer_name} with shape {hidden.shape}")
            else:
                raise ValueError(f"Layer {layer_name} not found in hidden states")
        
        return extracted
    
    def get_hidden_states(self, layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Get stored hidden states."""
        if layer_name is None:
            return self.hidden_states.copy()
        return self.hidden_states.get(layer_name, None)
    
    def clear_hidden_states(self):
        """Clear stored hidden states."""
        self.hidden_states = {}


class SequentialInjectionReasoningFramework:
    """
    Framework for performing sequential reasoning with hidden embedding injections.
    Each injection happens after a complete forward pass.
    """
    
    def __init__(self, model: SequentialInjectionGPT, config: GPTConfig):
        self.model = model
        self.config = config
        self.reasoning_history = []
    
    
    def perform_initial_pass(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform the initial forward pass and extract hidden states."""
        logits, loss, hidden_states = self.model(
            input_ids, 
            return_hidden_states=True
        )
        
        # Store in reasoning history
        self.reasoning_history.append({
            'pass_type': 'initial',
            'input_ids': input_ids,
            'logits': logits,
            'hidden_states': {k: v for k, v in hidden_states.items()},
            'timestamp': len(self.reasoning_history)
        })
        
        return logits, hidden_states
    
    def perform_sequential_injections(self, input_ids: torch.Tensor,
                                    injection_sequence: List[Dict]) -> List[Dict]:
        """
        Perform sequential injections: forward -> inject -> forward -> inject -> ...
        
        Args:
            input_ids: Input token indices
            injection_sequence: List of injection specifications
                Each spec should have:
                - 'extraction': dict with 'layer_name', 'position', 'key' for extraction
                - 'injection': dict with 'layer', 'position', 'method', 'weight' for injection
        
        Returns:
            List of results from each step
        """
        results = []
        
        # Initial pass
        #print('performing initial pass')
        logits, hidden_states = self.perform_initial_pass(input_ids)
        results.append({
            'step': 0,
            'step_type': 'initial_forward',
            'logits': logits,
            'hidden_states': hidden_states,
            'injection_spec': None
        })
        
        # Sequential injections
        
        for i, injection_spec in enumerate(injection_sequence):
            extraction_spec = injection_spec['extraction']
            layer_name = extraction_spec['layer_name']
            position = extraction_spec['position']
            #print(f'performing injection {i+1} of {len(injection_sequence)}, from layer {layer_name} at position {position}')
            # Extract hidden embedding for this injection
            extraction_spec = injection_spec['extraction']
            ## there is no dependecy of extract_hidden_embeddings function on the model
            hidden_embeddings = self.model.extract_hidden_embeddings(
                hidden_states, [extraction_spec]
            )
            
            # Get the injection specification
            injection_config = injection_spec['injection']
            hidden_embedding = hidden_embeddings[extraction_spec['key']]
            
            # Perform forward pass with single injection
            logits, _, new_hidden_states = self.model.forward_with_single_injection(
                input_ids, hidden_embedding, injection_config, return_hidden_states=True
            )
            
            # Use hidden states from the injection result
            if new_hidden_states is not None:
                hidden_states = new_hidden_states
            
            # Store in reasoning history
            self.reasoning_history.append({
                'pass_type': 'sequential_injection',
                'input_ids': input_ids,
                'logits': logits,
                'hidden_states': {k: v for k, v in hidden_states.items()},
                'injection_spec': injection_spec,
                'timestamp': len(self.reasoning_history)
            })
            
            results.append({
                'step': i + 1,
                'step_type': 'injection_forward',
                'logits': logits,
                'hidden_states': hidden_states,
                'injection_spec': injection_spec
            })

        #print('length of results inside perform_sequential_injections is ', len(results))
        
        return results
    
    def compute_gradient_flow_analysis(self, loss_fn, target_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute gradient flow analysis for injection points.
        """
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
        self.model.clear_hidden_states()




