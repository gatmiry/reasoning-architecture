"""
Enhanced configuration class for the reasoning framework.
Extends the base GPTConfig with reasoning-specific parameters.
"""

from model import GPTConfig
from typing import List, Optional, Dict, Any

class ReasoningConfig(GPTConfig):
    """
    Enhanced configuration class that extends GPTConfig with reasoning-specific parameters.
    """
    
    def __init__(self, block_size: int, vocab_size: int, 
                 # Reasoning-specific parameters
                 extraction_layers: Optional[List[int]] = None,
                 max_reasoning_passes: int = 5,
                 injection_method: str = 'add',
                 reasoning_strategy: str = 'layer_injection',
                 hidden_state_memory: bool = True,
                 attention_modification: bool = False,
                 # Advanced reasoning parameters
                 reasoning_temperature: float = 1.0,
                 reasoning_dropout: float = 0.1,
                 layer_wise_attention: bool = False,
                 cross_layer_attention: bool = False,
                 # Analysis parameters
                 enable_analysis: bool = True,
                 save_intermediate_states: bool = False,
                 analysis_metrics: Optional[List[str]] = None):
        """
        Initialize the enhanced configuration.
        
        Args:
            block_size: Maximum sequence length
            vocab_size: Vocabulary size
            extraction_layers: List of layer indices to extract hidden states from
            max_reasoning_passes: Maximum number of reasoning passes allowed
            injection_method: Method for injecting hidden states ('add', 'concat', 'replace')
            reasoning_strategy: Strategy for reasoning ('layer_injection', 'input_enhancement')
            hidden_state_memory: Whether to keep hidden states in memory across passes
            attention_modification: Whether to modify attention patterns during reasoning
            reasoning_temperature: Temperature for reasoning pass sampling
            reasoning_dropout: Dropout rate for reasoning passes
            layer_wise_attention: Whether to use layer-wise attention mechanisms
            cross_layer_attention: Whether to enable cross-layer attention
            enable_analysis: Whether to enable automatic analysis
            save_intermediate_states: Whether to save intermediate states to disk
            analysis_metrics: List of metrics to compute during analysis
        """
        super().__init__(block_size, vocab_size)
        
        # Reasoning-specific parameters
        self.extraction_layers = extraction_layers or [1, 2]
        self.max_reasoning_passes = max_reasoning_passes
        self.injection_method = injection_method
        self.reasoning_strategy = reasoning_strategy
        self.hidden_state_memory = hidden_state_memory
        self.attention_modification = attention_modification
        
        # Advanced reasoning parameters
        self.reasoning_temperature = reasoning_temperature
        self.reasoning_dropout = reasoning_dropout
        self.layer_wise_attention = layer_wise_attention
        self.cross_layer_attention = cross_layer_attention
        
        # Analysis parameters
        self.enable_analysis = enable_analysis
        self.save_intermediate_states = save_intermediate_states
        self.analysis_metrics = analysis_metrics or [
            'logits_evolution', 'hidden_state_changes', 'attention_patterns'
        ]
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate the configuration parameters."""
        if self.max_reasoning_passes < 1:
            raise ValueError("max_reasoning_passes must be at least 1")
        
        if self.injection_method not in ['add', 'concat', 'replace']:
            raise ValueError("injection_method must be one of: 'add', 'concat', 'replace'")
        
        if self.reasoning_strategy not in ['layer_injection', 'input_enhancement']:
            raise ValueError("reasoning_strategy must be one of: 'layer_injection', 'input_enhancement'")
        
        if self.reasoning_temperature <= 0:
            raise ValueError("reasoning_temperature must be positive")
        
        if not 0 <= self.reasoning_dropout <= 1:
            raise ValueError("reasoning_dropout must be between 0 and 1")
        
        if self.extraction_layers:
            if any(layer < 0 or layer >= self.n_layers for layer in self.extraction_layers):
                raise ValueError(f"extraction_layers must be between 0 and {self.n_layers - 1}")
    
    def update_extraction_layers(self, layers: List[int]):
        """Update the extraction layers and validate."""
        self.extraction_layers = layers
        self._validate_config()
    
    def set_reasoning_strategy(self, strategy: str):
        """Update the reasoning strategy and validate."""
        self.reasoning_strategy = strategy
        self._validate_config()
    
    def set_injection_method(self, method: str):
        """Update the injection method and validate."""
        self.injection_method = method
        self._validate_config()
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get the configuration as a dictionary."""
        config_dict = {
            # Base GPT parameters
            'block_size': self.block_size,
            'vocab_size': self.vocab_size,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'n_embd': self.n_embd,
            
            # Reasoning parameters
            'extraction_layers': self.extraction_layers,
            'max_reasoning_passes': self.max_reasoning_passes,
            'injection_method': self.injection_method,
            'reasoning_strategy': self.reasoning_strategy,
            'hidden_state_memory': self.hidden_state_memory,
            'attention_modification': self.attention_modification,
            'reasoning_temperature': self.reasoning_temperature,
            'reasoning_dropout': self.reasoning_dropout,
            'layer_wise_attention': self.layer_wise_attention,
            'cross_layer_attention': self.cross_layer_attention,
            
            # Analysis parameters
            'enable_analysis': self.enable_analysis,
            'save_intermediate_states': self.save_intermediate_states,
            'analysis_metrics': self.analysis_metrics
        }
        return config_dict
    
    def save_config(self, filepath: str):
        """Save the configuration to a file."""
        import json
        config_dict = self.get_config_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_config(cls, filepath: str) -> 'ReasoningConfig':
        """Load configuration from a file."""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Extract base parameters
        block_size = config_dict.pop('block_size')
        vocab_size = config_dict.pop('vocab_size')
        
        # Create instance with base parameters
        config = cls(block_size, vocab_size)
        
        # Update with loaded parameters
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def create_preset_configs(self) -> Dict[str, 'ReasoningConfig']:
        """Create preset configurations for different use cases."""
        presets = {}
        
        # Lightweight preset for quick experimentation
        presets['lightweight'] = ReasoningConfig(
            block_size=256, vocab_size=50257,
            extraction_layers=[1],
            max_reasoning_passes=2,
            injection_method='add',
            reasoning_strategy='layer_injection',
            enable_analysis=False
        )
        
        # Standard preset for general use
        presets['standard'] = ReasoningConfig(
            block_size=512, vocab_size=50257,
            extraction_layers=[1, 2],
            max_reasoning_passes=3,
            injection_method='add',
            reasoning_strategy='layer_injection',
            enable_analysis=True
        )
        
        # Advanced preset for complex reasoning
        presets['advanced'] = ReasoningConfig(
            block_size=1024, vocab_size=50257,
            extraction_layers=[0, 2, 4],
            max_reasoning_passes=5,
            injection_method='add',
            reasoning_strategy='layer_injection',
            hidden_state_memory=True,
            attention_modification=True,
            layer_wise_attention=True,
            enable_analysis=True,
            save_intermediate_states=True
        )
        
        # Research preset for experimentation
        presets['research'] = ReasoningConfig(
            block_size=1024, vocab_size=50257,
            extraction_layers=list(range(6)),  # Extract from all layers
            max_reasoning_passes=10,
            injection_method='concat',  # More complex injection
            reasoning_strategy='input_enhancement',
            hidden_state_memory=True,
            attention_modification=True,
            cross_layer_attention=True,
            reasoning_temperature=0.8,
            enable_analysis=True,
            save_intermediate_states=True,
            analysis_metrics=['logits_evolution', 'hidden_state_changes', 
                            'attention_patterns', 'gradient_flow']
        )
        
        return presets
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        return f"""ReasoningConfig(
    block_size={self.block_size},
    vocab_size={self.vocab_size},
    n_layers={self.n_layers},
    n_heads={self.n_heads},
    n_embd={self.n_embd},
    extraction_layers={self.extraction_layers},
    max_reasoning_passes={self.max_reasoning_passes},
    injection_method='{self.injection_method}',
    reasoning_strategy='{self.reasoning_strategy}',
    enable_analysis={self.enable_analysis}
)"""
    
    def __repr__(self) -> str:
        """Detailed representation of the configuration."""
        return self.__str__()

# Convenience functions for creating common configurations
def create_lightweight_config() -> ReasoningConfig:
    """Create a lightweight configuration for quick experiments."""
    return ReasoningConfig(
        block_size=256, vocab_size=50257,
        extraction_layers=[1],
        max_reasoning_passes=2,
        enable_analysis=False
    )

def create_standard_config() -> ReasoningConfig:
    """Create a standard configuration for general use."""
    return ReasoningConfig(
        block_size=512, vocab_size=50257,
        extraction_layers=[1, 2],
        max_reasoning_passes=3,
        enable_analysis=True
    )

def create_advanced_config() -> ReasoningConfig:
    """Create an advanced configuration for complex reasoning tasks."""
    return ReasoningConfig(
        block_size=1024, vocab_size=50257,
        extraction_layers=[0, 2, 4],
        max_reasoning_passes=5,
        hidden_state_memory=True,
        attention_modification=True,
        enable_analysis=True
    )

