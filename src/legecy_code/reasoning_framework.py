import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from enhanced_gpt import EnhancedGPT
from model import GPTConfig

class ReasoningFramework:
    """
    Framework for performing multi-pass reasoning with transformer hidden embeddings.
    This class manages the extraction of intermediate representations and orchestrates
    multiple forward passes with enhanced inputs.
    """
    
    def __init__(self, model: EnhancedGPT, config: GPTConfig):
        self.model = model
        self.config = config
        self.reasoning_history = []
        self.extraction_layers = []
        
    def set_extraction_layers(self, layers: List[int]):
        """
        Set which layers to extract hidden states from.
        
        Args:
            layers: List of layer indices (0-indexed) to extract from
        """
        self.extraction_layers = layers
        self.model.enable_hidden_extraction(layers)
    
    def perform_initial_pass(self, input_ids: torch.Tensor, 
                           last_k_no_attend: int = 0, 
                           window_size: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform the initial forward pass and extract hidden states.
        
        Args:
            input_ids: Input token indices
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
            
        Returns:
            logits: Output logits from the first pass
            hidden_states: Extracted hidden states from intermediate layers
        """
        # Enable hidden extraction for specified layers
        if self.extraction_layers:
            self.model.enable_hidden_extraction(self.extraction_layers)
        
        # Perform forward pass
        logits, loss = self.model(input_ids, last_k_no_attend=last_k_no_attend, 
                                window_size=window_size)
        
        # Get extracted hidden states
        hidden_states = self.model.get_hidden_states()
        
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
    
    def perform_reasoning_pass(self, input_ids: torch.Tensor,
                             hidden_states_to_use: Dict[str, torch.Tensor],
                             injection_method: str = 'add',
                             injection_layers: Optional[List[int]] = None,
                             reasoning_strategy: str = 'layer_injection',
                             last_k_no_attend: int = 0,
                             window_size: int = 0,
                             # Token-specific parameters
                             token_extraction_positions: Optional[Dict[str, int]] = None,
                             token_injection_positions: Optional[Dict[str, int]] = None,
                             token_reasoning_strategy: str = 'token_injection') -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform a reasoning pass using hidden states from previous passes.
        
        Args:
            input_ids: Input token indices for this pass
            hidden_states_to_use: Hidden states from previous passes to use
            injection_method: Method for injecting hidden states ('add', 'concat', 'replace')
            injection_layers: Layers to inject hidden states into (defaults to extraction layers)
            reasoning_strategy: Strategy for reasoning ('layer_injection', 'input_enhancement', 'token_injection')
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
            token_extraction_positions: Dict mapping layer names to token positions to extract
            token_injection_positions: Dict mapping embedding names to positions where to inject
            token_reasoning_strategy: Strategy for token-specific reasoning
            
        Returns:
            logits: Output logits from the reasoning pass
            hidden_states: New hidden states from this pass
        """
        if injection_layers is None:
            injection_layers = self.extraction_layers
        
        if reasoning_strategy == 'layer_injection':
            # Inject hidden states into specific layers
            logits, _ = self.model.forward_with_hidden_injection(
                input_ids, hidden_states_to_use, injection_layers, 
                injection_method, last_k_no_attend, window_size
            )
            
            # Get new hidden states from this pass
            new_hidden_states = self.model.get_hidden_states()
            
        elif reasoning_strategy == 'input_enhancement':
            # Enhance input with hidden states before forward pass
            enhanced_input = self.model.create_reasoning_input(
                input_ids, hidden_states_to_use, 'layer_wise'
            )
            
            # Perform forward pass with enhanced input
            logits, _ = self.model(enhanced_input, last_k_no_attend=last_k_no_attend,
                                 window_size=window_size)
            
            # Get new hidden states
            new_hidden_states = self.model.get_hidden_states()
            
        elif reasoning_strategy == 'token_injection':
            # Extract specific token embeddings and inject them at specified positions
            token_embeddings = self.model.extract_token_embeddings(
                hidden_states_to_use, token_extraction_positions
            )
            
            # Perform forward pass with token injection
            logits, _ = self.model.forward_with_token_injection(
                input_ids, token_embeddings, token_injection_positions,
                injection_method, last_k_no_attend, window_size
            )
            
            # Get new hidden states
            new_hidden_states = self.model.get_hidden_states()
        
        # Store in reasoning history
        self.reasoning_history.append({
            'pass_type': 'reasoning',
            'input_ids': input_ids.clone(),
            'logits': logits.clone(),
            'hidden_states': {k: v.clone() for k, v in new_hidden_states.items()},
            'injection_method': injection_method,
            'reasoning_strategy': reasoning_strategy,
            'last_k_no_attend': last_k_no_attend,
            'window_size': window_size
        })
        
        return logits, new_hidden_states
    
    def perform_token_specific_reasoning(self, input_ids: torch.Tensor,
                                       source_token_positions: Dict[str, int],
                                       target_token_positions: Dict[str, int],
                                       injection_method: str = 'add',
                                       last_k_no_attend: int = 0,
                                       window_size: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Perform reasoning by extracting specific tokens and injecting them at specific positions.
        
        Args:
            input_ids: Input token indices
            source_token_positions: Dict mapping layer names to token positions to extract
                                  e.g., {'layer_1': 2} means extract token at position 2 from layer_1
            target_token_positions: Dict mapping embedding names to positions where to inject
                                  e.g., {'layer_1_token_2': 0} means inject token_2 from layer_1 at position 0
            injection_method: Method for injection ('add', 'replace', 'concat')
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
            
        Returns:
            logits: Output logits from the reasoning pass
            hidden_states: New hidden states from this pass
        """
        # First, perform initial pass to get hidden states
        logits_initial, hidden_states = self.perform_initial_pass(
            input_ids, last_k_no_attend, window_size
        )
        
        # Extract specific token embeddings
        token_embeddings = self.model.extract_token_embeddings(
            hidden_states, source_token_positions
        )
        
        print(f"Extracted token embeddings: {list(token_embeddings.keys())}")
        for name, embedding in token_embeddings.items():
            print(f"  {name}: shape {embedding.shape}")
        
        # Perform reasoning pass with token injection
        logits_reasoning, new_hidden_states = self.perform_reasoning_pass(
            input_ids=input_ids,
            hidden_states_to_use=hidden_states,
            injection_method=injection_method,
            reasoning_strategy='token_injection',
            token_extraction_positions=source_token_positions,
            token_injection_positions=target_token_positions,
            last_k_no_attend=last_k_no_attend,
            window_size=window_size
        )
        
        return logits_reasoning, new_hidden_states
    
    def perform_iterative_reasoning(self, input_ids: torch.Tensor,
                                  num_iterations: int = 3,
                                  injection_method: str = 'add',
                                  reasoning_strategy: str = 'layer_injection',
                                  last_k_no_attend: int = 0,
                                  window_size: int = 0) -> List[Dict[str, Any]]:
        """
        Perform multiple iterative reasoning passes.
        
        Args:
            input_ids: Initial input token indices
            num_iterations: Number of reasoning iterations to perform
            injection_method: Method for injecting hidden states
            reasoning_strategy: Strategy for reasoning
            last_k_no_attend: Number of last tokens to not attend to
            window_size: Window size for attention
            
        Returns:
            List of results from each iteration
        """
        results = []
        
        # Initial pass
        logits, hidden_states = self.perform_initial_pass(
            input_ids, last_k_no_attend, window_size
        )
        results.append({
            'iteration': 0,
            'logits': logits,
            'hidden_states': hidden_states
        })
        
        # Iterative reasoning passes
        for i in range(1, num_iterations):
            logits, hidden_states = self.perform_reasoning_pass(
                input_ids, hidden_states, injection_method, 
                reasoning_strategy=reasoning_strategy,
                last_k_no_attend=last_k_no_attend, window_size=window_size
            )
            results.append({
                'iteration': i,
                'logits': logits,
                'hidden_states': hidden_states
            })
        
        return results
    
    def analyze_reasoning_progression(self) -> Dict[str, Any]:
        """
        Analyze the progression of reasoning across multiple passes.
        
        Returns:
            Dictionary containing analysis metrics
        """
        if len(self.reasoning_history) < 2:
            return {'error': 'Need at least 2 passes for analysis'}
        
        analysis = {
            'num_passes': len(self.reasoning_history),
            'logits_evolution': [],
            'hidden_state_changes': {},
            'attention_patterns': []
        }
        
        # Analyze logits evolution
        for i, pass_data in enumerate(self.reasoning_history):
            logits = pass_data['logits']
            analysis['logits_evolution'].append({
                'pass': i,
                'max_logit': torch.max(logits).item(),
                'mean_logit': torch.mean(logits).item(),
                'std_logit': torch.std(logits).item()
            })
        
        # Analyze hidden state changes
        for layer_name in self.reasoning_history[0]['hidden_states'].keys():
            if layer_name in self.reasoning_history[-1]['hidden_states']:
                initial = self.reasoning_history[0]['hidden_states'][layer_name]
                final = self.reasoning_history[-1]['hidden_states'][layer_name]
                
                if initial.shape == final.shape:
                    diff = torch.norm(final - initial).item()
                    analysis['hidden_state_changes'][layer_name] = {
                        'l2_change': diff,
                        'relative_change': diff / torch.norm(initial).item()
                    }
        
        return analysis
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get the complete reasoning history."""
        return self.reasoning_history.copy()
    
    def clear_history(self):
        """Clear the reasoning history."""
        self.reasoning_history = []
        self.model.clear_hidden_states()
    
    def save_reasoning_state(self, filepath: str):
        """
        Save the current reasoning state to a file.
        
        Args:
            filepath: Path to save the state
        """
        state = {
            'reasoning_history': self.reasoning_history,
            'extraction_layers': self.extraction_layers,
            'config': self.config.__dict__
        }
        torch.save(state, filepath)
    
    def load_reasoning_state(self, filepath: str):
        """
        Load a reasoning state from a file.
        
        Args:
            filepath: Path to load the state from
        """
        state = torch.load(filepath, map_location='cpu')
        self.reasoning_history = state['reasoning_history']
        self.extraction_layers = state['extraction_layers']
        
        # Restore model state if needed
        if self.extraction_layers:
            self.model.enable_hidden_extraction(self.extraction_layers)
    
    def create_hidden_state_summary(self, layer_name: str) -> Dict[str, float]:
        """
        Create a summary of hidden states for a specific layer across all passes.
        
        Args:
            layer_name: Name of the layer to summarize
            
        Returns:
            Dictionary containing summary statistics
        """
        if not self.reasoning_history:
            return {'error': 'No reasoning history available'}
        
        layer_data = []
        for pass_data in self.reasoning_history:
            if layer_name in pass_data['hidden_states']:
                hidden = pass_data['hidden_states'][layer_name]
                layer_data.append(hidden)
        
        if not layer_data:
            return {'error': f'Layer {layer_name} not found in any pass'}
        
        # Compute statistics across all passes
        all_hidden = torch.cat(layer_data, dim=0)
        
        summary = {
            'mean_activation': torch.mean(all_hidden).item(),
            'std_activation': torch.std(all_hidden).item(),
            'max_activation': torch.max(all_hidden).item(),
            'min_activation': torch.min(all_hidden).item(),
            'num_passes': len(layer_data),
            'shape': list(all_hidden.shape)
        }
        
        return summary
