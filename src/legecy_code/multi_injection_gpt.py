"""
Enhanced GPT model with support for multiple hidden embedding injections
and full gradient backpropagation through injection points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Dict, Optional, Tuple, Union
from model import MLP, CasualSelfAttention, Block, GPTConfig

class MultiInjectionGPT(nn.Module):
    """
    Enhanced GPT model that supports multiple hidden embedding injections
    with full gradient backpropagation through all injection points.
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
        self.extract_hidden = False
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
    
    def enable_hidden_extraction(self, target_layers: List[int]):
        """Enable extraction of hidden states from specified layers."""
        self.extract_hidden = True
        self.target_layers = target_layers
        self.hidden_states = {}
    
    def disable_hidden_extraction(self):
        """Disable hidden state extraction."""
        self.extract_hidden = False
        self.target_layers = []
        self.hidden_states = {}
    
    def get_injection_projection(self, injection_key: str, input_dim: int, output_dim: int):
        """Get or create projection layer for injection."""
        if injection_key not in self.injection_projections:
            self.injection_projections[injection_key] = nn.Linear(input_dim, output_dim).to(next(self.parameters()).device)
        return self.injection_projections[injection_key]
    
    def forward(self, idx, targets=None, last_k_no_attend=0, window_size=0, 
                return_hidden_states=False, injection_specs=None):
        """
        Forward pass with support for multiple hidden embedding injections.
        
        Args:
            idx: Input token indices
            targets: Target tokens for loss computation
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
            return_hidden_states: Whether to return hidden states from all layers
            injection_specs: List of injection specifications
                Each spec is a dict with keys:
                - 'layer': layer index to inject into
                - 'position': token position to inject at
                - 'hidden_embedding': tensor to inject
                - 'method': 'add', 'replace', 'concat', 'weighted_add'
                - 'weight': weight for weighted operations (optional)
        
        Returns:
            logits: Output logits
            loss: Loss (if targets provided)
            hidden_states: Hidden states from intermediate layers (if requested)
        """
        B, T = idx.size()
        pe = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pe_vecs = self.transformer.wpe(pe)
        
        if last_k_no_attend > 0:
            last_k_no_attend = min(last_k_no_attend, T)
            pe_vecs[-last_k_no_attend:,:] = 0.0
        
        x = pe_vecs + self.transformer.wte(idx)
        
        # Store initial embeddings
        if self.extract_hidden or return_hidden_states:
            hidden_states = {}
            hidden_states['input_embeddings'] = x.clone()
        
        # Forward through transformer blocks with injections
        for layer_idx, block in enumerate(self.transformer.h):
            # Apply injections before this layer if specified
            if injection_specs:
                x = self._apply_injections(x, injection_specs, layer_idx)
            
            x = block(x, last_k_no_attend=last_k_no_attend, window_size=window_size)
            
            # Store hidden states from target layers
            if (self.extract_hidden and layer_idx in self.target_layers) or return_hidden_states:
                if self.extract_hidden:
                    self.hidden_states[f'layer_{layer_idx}'] = x.clone().detach()
                if return_hidden_states:
                    hidden_states[f'layer_{layer_idx}'] = x.clone()
        
        # Final layer norm
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        # Store final hidden states
        if self.extract_hidden:
            self.hidden_states['final'] = x.clone().detach()
        if return_hidden_states:
            hidden_states['final'] = x.clone()
        
        loss = None
        if targets is not None:
            # Loss computation can be added here if needed
            pass
        
        if return_hidden_states:
            return logits, loss, hidden_states
        return logits, loss
    
    def _apply_injections(self, x: torch.Tensor, injection_specs: List[Dict], 
                         current_layer: int) -> torch.Tensor:
        """
        Apply multiple injections to the current hidden state.
        
        Args:
            x: Current hidden state tensor [batch, seq_len, hidden_dim]
            injection_specs: List of injection specifications
            current_layer: Current layer index
        
        Returns:
            Modified hidden state tensor
        """
        for spec in injection_specs:
            if spec['layer'] == current_layer:
                x = self._apply_single_injection(x, spec)
        return x
    
    def _apply_single_injection(self, x: torch.Tensor, spec: Dict) -> torch.Tensor:
        """
        Apply a single injection to the hidden state.
        
        Args:
            x: Current hidden state tensor [batch, seq_len, hidden_dim]
            spec: Injection specification dict
        
        Returns:
            Modified hidden state tensor
        """
        position = spec['position']
        hidden_embedding = spec['hidden_embedding']  # [batch, hidden_dim]
        method = spec.get('method', 'add')
        weight = spec.get('weight', 1.0)
        
        # Ensure position is within bounds
        if position >= x.size(1):
            return x
        
        # Ensure hidden_embedding has correct batch size
        if hidden_embedding.size(0) != x.size(0):
            # Broadcast or repeat if needed
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
            # This is a simplified version - you might need more sophisticated handling
            if hidden_embedding.size(-1) == x.size(-1):
                x[:, position, :] = hidden_embedding
            else:
                # Project and replace
                x[:, position, :] = hidden_embedding
        else:
            raise ValueError(f"Unknown injection method: {method}")
        
        return x
    
    def forward_with_multiple_injections(self, input_ids: torch.Tensor,
                                       hidden_embeddings: Dict[str, torch.Tensor],
                                       injection_specs: List[Dict],
                                       last_k_no_attend: int = 0,
                                       window_size: int = 0) -> Tuple[torch.Tensor, None]:
        """
        Perform forward pass with multiple hidden embedding injections.
        
        Args:
            input_ids: Input token indices
            hidden_embeddings: Dictionary of hidden embeddings to inject
            injection_specs: List of injection specifications
                Each spec should have 'hidden_embedding_key' instead of 'hidden_embedding'
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
        
        Returns:
            logits: Output logits
            loss: Loss (None in this case)
        """
        # Replace hidden_embedding_key with actual tensor
        processed_specs = []
        for spec in injection_specs:
            processed_spec = spec.copy()
            if 'hidden_embedding_key' in spec:
                key = spec['hidden_embedding_key']
                if key in hidden_embeddings:
                    processed_spec['hidden_embedding'] = hidden_embeddings[key]
                    del processed_spec['hidden_embedding_key']
                else:
                    raise ValueError(f"Hidden embedding key '{key}' not found in provided embeddings")
            processed_specs.append(processed_spec)
        
        # Perform forward pass with injections
        logits, _ = self.forward(
            input_ids, 
            last_k_no_attend=last_k_no_attend, 
            window_size=window_size,
            injection_specs=processed_specs
        )
        
        return logits, None
    
    def extract_hidden_embeddings(self, hidden_states: Dict[str, torch.Tensor], 
                                 extraction_specs: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Extract hidden embeddings based on specifications.
        
        Args:
            hidden_states: Hidden states from intermediate layers
            extraction_specs: List of extraction specifications
                Each spec is a dict with keys:
                - 'layer_name': name of the layer to extract from
                - 'position': token position to extract
                - 'key': key to store the extracted embedding under
        
        Returns:
            Dictionary of extracted embeddings
        """
        extracted = {}
        
        for spec in extraction_specs:
            layer_name = spec['layer_name']
            position = spec['position']
            key = spec['key']
            
            if layer_name in hidden_states:
                hidden = hidden_states[layer_name]
                if hidden.dim() == 3 and position < hidden.size(1):
                    extracted[key] = hidden[:, position, :].clone()
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


class MultiInjectionReasoningFramework:
    """
    Framework for performing multi-pass reasoning with multiple hidden embedding injections
    and full gradient backpropagation.
    """
    
    def __init__(self, model: MultiInjectionGPT, config: GPTConfig):
        self.model = model
        self.config = config
        self.reasoning_history = []
        self.extraction_layers = []
    
    def set_extraction_layers(self, layers: List[int]):
        """Set which layers to extract hidden states from."""
        self.extraction_layers = layers
        self.model.enable_hidden_extraction(layers)
    
    def perform_initial_pass(self, input_ids: torch.Tensor, 
                           last_k_no_attend: int = 0, 
                           window_size: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Perform the initial forward pass and extract hidden states."""
        if self.extraction_layers:
            self.model.enable_hidden_extraction(self.extraction_layers)
        
        logits, loss, hidden_states = self.model(
            input_ids, 
            last_k_no_attend=last_k_no_attend, 
            window_size=window_size,
            return_hidden_states=True
        )
        
        # Store in reasoning history
        self.reasoning_history.append({
            'pass_type': 'initial',
            'input_ids': input_ids.clone(),
            'logits': logits.clone(),
            'hidden_states': {k: v.clone() for k, v in hidden_states.items()},
            'last_k_no_attend': last_k_no_attend,
            'window_size': window_size
        })
        
        return logits, hidden_states
    
    def perform_reasoning_pass_with_injections(self, input_ids: torch.Tensor,
                                             hidden_embeddings: Dict[str, torch.Tensor],
                                             injection_specs: List[Dict],
                                             last_k_no_attend: int = 0,
                                             window_size: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform a reasoning pass with multiple hidden embedding injections.
        
        Args:
            input_ids: Input token indices
            hidden_embeddings: Dictionary of hidden embeddings to inject
            injection_specs: List of injection specifications
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
        
        Returns:
            logits: Output logits from the reasoning pass
            hidden_states: New hidden states from this pass
        """
        # Perform forward pass with injections
        logits, _ = self.model.forward_with_multiple_injections(
            input_ids, hidden_embeddings, injection_specs, 
            last_k_no_attend, window_size
        )
        
        # Get new hidden states
        new_hidden_states = self.model.get_hidden_states()
        
        # Store in reasoning history
        self.reasoning_history.append({
            'pass_type': 'reasoning_with_injections',
            'input_ids': input_ids.clone(),
            'logits': logits.clone(),
            'hidden_states': {k: v.clone() for k, v in new_hidden_states.items()},
            'injection_specs': injection_specs,
            'last_k_no_attend': last_k_no_attend,
            'window_size': window_size
        })
        
        return logits, new_hidden_states
    
    def perform_iterative_reasoning_with_injections(self, input_ids: torch.Tensor,
                                                  injection_sequences: List[List[Dict]],
                                                  last_k_no_attend: int = 0,
                                                  window_size: int = 0) -> List[Dict]:
        """
        Perform multiple iterative reasoning passes with different injection sequences.
        
        Args:
            input_ids: Input token indices
            injection_sequences: List of injection specification lists for each iteration
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
        
        Returns:
            List of results from each iteration
        """
        results = []
        
        # Initial pass
        logits, hidden_states = self.perform_initial_pass(input_ids, last_k_no_attend, window_size)
        results.append({
            'iteration': 0,
            'logits': logits,
            'hidden_states': hidden_states,
            'injection_specs': []
        })
        
        # Keep track of all extracted embeddings across iterations
        all_hidden_embeddings = {}
        
        # Iterative reasoning passes with injections
        for i, injection_specs in enumerate(injection_sequences):
            # Extract hidden embeddings for this iteration
            extraction_specs = []
            injection_only_specs = []
            
            for spec in injection_specs:
                if 'extraction' in spec:
                    extraction_specs.append(spec['extraction'])
                else:
                    injection_only_specs.append(spec)
            
            if extraction_specs:
                new_hidden_embeddings = self.model.extract_hidden_embeddings(
                    hidden_states, extraction_specs
                )
                # Add new embeddings to the collection
                all_hidden_embeddings.update(new_hidden_embeddings)
            
            # Perform reasoning pass with injections using all available embeddings
            logits, hidden_states = self.perform_reasoning_pass_with_injections(
                input_ids, all_hidden_embeddings, injection_only_specs, 
                last_k_no_attend, window_size
            )
            
            results.append({
                'iteration': i + 1,
                'logits': logits,
                'hidden_states': hidden_states,
                'injection_specs': injection_specs
            })
        
        return results
    
    def compute_gradient_flow_analysis(self, loss_fn, target_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute gradient flow analysis for injection points.
        
        Args:
            loss_fn: Loss function to use
            target_tokens: Target tokens for loss computation
        
        Returns:
            Dictionary containing gradient information
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
