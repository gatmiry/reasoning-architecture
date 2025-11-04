import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from typing import List, Dict, Optional, Tuple
from model import MLP, CasualSelfAttention, Block, GPTConfig

class EnhancedGPT(nn.Module):
    """
    Enhanced GPT model that can extract hidden embeddings from intermediate layers
    and perform additional forward passes with modified inputs.
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
        """
        Enable extraction of hidden states from specified layers.
        
        Args:
            target_layers: List of layer indices to extract hidden states from (0-indexed)
        """
        self.extract_hidden = True
        self.target_layers = target_layers
        self.hidden_states = {}
    
    def disable_hidden_extraction(self):
        """Disable hidden state extraction."""
        self.extract_hidden = False
        self.target_layers = []
        self.hidden_states = {}
    
    def forward(self, idx, targets=None, last_k_no_attend=0, window_size=0, 
                return_hidden_states=False):
        """
        Forward pass with optional hidden state extraction.
        
        Args:
            idx: Input token indices
            targets: Target tokens for loss computation
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
            return_hidden_states: Whether to return hidden states from all layers
            
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
        
        # Forward through transformer blocks
        for layer_idx, block in enumerate(self.transformer.h):
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
    
    def get_hidden_states(self, layer_name: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Get stored hidden states.
        
        Args:
            layer_name: Specific layer to retrieve, or None for all layers
            
        Returns:
            Dictionary of hidden states or specific layer state
        """
        if layer_name is None:
            return self.hidden_states.copy()
        return self.hidden_states.get(layer_name, None)
    
    def clear_hidden_states(self):
        """Clear stored hidden states."""
        self.hidden_states = {}
    
    def forward_with_hidden_injection(self, idx, hidden_states_to_inject: Dict[str, torch.Tensor],
                                    injection_layers: List[int], injection_method: str = 'add',
                                    last_k_no_attend=0, window_size=0):
        """
        Perform forward pass with injected hidden states from previous passes.
        
        Args:
            idx: Input token indices
            hidden_states_to_inject: Hidden states to inject from previous passes
            injection_layers: List of layer indices where to inject hidden states
            injection_method: Method for injection ('add', 'concat', 'replace')
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
            
        Returns:
            logits: Output logits
            loss: Loss (if targets provided)
        """
        B, T = idx.size()
        pe = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pe_vecs = self.transformer.wpe(pe)
        
        if last_k_no_attend > 0:
            last_k_no_attend = min(last_k_no_attend, T)
            pe_vecs[-last_k_no_attend:,:] = 0.0
        
        x = pe_vecs + self.transformer.wte(idx)
        
        # Forward through transformer blocks with injection
        for layer_idx, block in enumerate(self.transformer.h):
            # Inject hidden states if this is a target layer
            if layer_idx in injection_layers and f'layer_{layer_idx}' in hidden_states_to_inject:
                hidden_to_inject = hidden_states_to_inject[f'layer_{layer_idx}']
                
                if injection_method == 'add':
                    x = x + hidden_to_inject
                elif injection_method == 'concat':
                    # For concat, we need to adjust the linear layers
                    # This is a simplified version - you might need to modify the Block class
                    x = torch.cat([x, hidden_to_inject], dim=-1)
                elif injection_method == 'replace':
                    x = hidden_to_inject
            
            x = block(x, last_k_no_attend=last_k_no_attend, window_size=window_size)
        
        # Final layer norm and output
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, None
    
    def create_reasoning_input(self, original_idx: torch.Tensor, 
                             hidden_states: Dict[str, torch.Tensor],
                             reasoning_method: str = 'layer_wise') -> torch.Tensor:
        """
        Create enhanced input for reasoning by combining original input with hidden states.
        
        Args:
            original_idx: Original input token indices
            hidden_states: Hidden states from intermediate layers
            reasoning_method: Method for creating reasoning input
            
        Returns:
            Enhanced input tensor
        """
        if reasoning_method == 'layer_wise':
            # Use hidden states from a specific layer as additional context
            # This is a simplified approach - you can customize this further
            if 'layer_0' in hidden_states:
                # Project hidden states to vocabulary space for concatenation
                hidden_projected = torch.matmul(hidden_states['layer_0'], 
                                              self.transformer.wte.weight.T)
                # Take the most relevant tokens
                _, top_indices = torch.topk(hidden_projected, k=min(10, original_idx.size(1)), dim=-1)
                return torch.cat([original_idx, top_indices], dim=1)
        
        return original_idx
    
    def extract_token_embeddings(self, hidden_states: Dict[str, torch.Tensor], 
                                token_positions: Dict[int, str] = None) -> Dict[str, torch.Tensor]:
        """
        Extract specific token embeddings from hidden states.
        
        Args:
            hidden_states: Hidden states from intermediate layers
            token_positions: Dict mapping layer names to token positions to extract
                           e.g., {'layer_1': 2} means extract token at position 2 from layer_1
                           If None, extracts from all layers at position 0
        
        Returns:
            Dictionary of extracted token embeddings
        """
        extracted = {}
        
        if token_positions is None:
            # Default: extract first token from all layers
            for layer_name, hidden in hidden_states.items():
                if hidden.dim() == 3:  # [batch, seq_len, hidden_dim]
                    extracted[f"{layer_name}_token_0"] = hidden[:, 0, :].clone()
        else:
            for layer_name, token_pos in token_positions.items():
                if layer_name in hidden_states:
                    hidden = hidden_states[layer_name]
                    if hidden.dim() == 3 and token_pos < hidden.size(1):
                        extracted[f"{layer_name}_token_{token_pos}"] = hidden[:, token_pos, :].clone()
        
        return extracted
    
    def inject_token_embeddings(self, input_ids: torch.Tensor,
                               token_embeddings: Dict[str, torch.Tensor],
                               injection_positions: Dict[str, int] = None,
                               injection_method: str = 'add') -> torch.Tensor:
        """
        Inject specific token embeddings into the input at specified positions.
        
        Args:
            input_ids: Original input token indices
            token_embeddings: Token embeddings to inject
            injection_positions: Dict mapping embedding names to positions where to inject
                               e.g., {'layer_1_token_2': 0} means inject token_2 from layer_1 at position 0
            injection_method: Method for injection ('add', 'replace', 'concat')
        
        Returns:
            Modified input tensor
        """
        if injection_positions is None:
            # Default: inject all embeddings at position 0
            injection_positions = {name: 0 for name in token_embeddings.keys()}
        
        # Get input embeddings
        B, T = input_ids.size()
        pe = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
        pe_vecs = self.transformer.wpe(pe)
        input_embeddings = pe_vecs + self.transformer.wte(input_ids)
        
        # Apply injections
        for embedding_name, pos in injection_positions.items():
            if embedding_name in token_embeddings and pos < T:
                token_embedding = token_embeddings[embedding_name]  # [batch, hidden_dim]
                
                if injection_method == 'add':
                    input_embeddings[:, pos, :] += token_embedding
                elif injection_method == 'replace':
                    input_embeddings[:, pos, :] = token_embedding
                elif injection_method == 'concat':
                    # For concat, we need to handle dimension mismatch
                    # This is a simplified version - you might need more sophisticated handling
                    if token_embedding.size(-1) == input_embeddings.size(-1):
                        input_embeddings[:, pos, :] = token_embedding
                    else:
                        # Project to match dimensions
                        if not hasattr(self, 'token_projection'):
                            self.token_projection = nn.Linear(token_embedding.size(-1), 
                                                            input_embeddings.size(-1)).to(input_embeddings.device)
                        projected = self.token_projection(token_embedding)
                        input_embeddings[:, pos, :] = projected
        
        return input_embeddings
    
    def forward_with_token_injection(self, input_ids: torch.Tensor,
                                   token_embeddings: Dict[str, torch.Tensor],
                                   injection_positions: Dict[str, int] = None,
                                   injection_method: str = 'add',
                                   last_k_no_attend: int = 0,
                                   window_size: int = 0) -> Tuple[torch.Tensor, None]:
        """
        Perform forward pass with specific token embeddings injected at specified positions.
        
        Args:
            input_ids: Input token indices
            token_embeddings: Token embeddings to inject
            injection_positions: Positions where to inject embeddings
            injection_method: Method for injection
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
        
        Returns:
            logits: Output logits
            loss: Loss (None in this case)
        """
        # Inject token embeddings into input
        modified_embeddings = self.inject_token_embeddings(
            input_ids, token_embeddings, injection_positions, injection_method
        )
        
        # Forward through transformer blocks
        x = modified_embeddings
        for block in self.transformer.h:
            x = block(x, last_k_no_attend=last_k_no_attend, window_size=window_size)
        
        # Final layer norm and output
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        
        return logits, None
