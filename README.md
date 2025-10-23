# Transformer Reasoning Framework

A comprehensive framework for performing multi-pass reasoning with transformer hidden embeddings. This framework allows you to extract intermediate layer representations and perform additional forward passes with enhanced inputs.

## Features

- **Hidden State Extraction**: Extract hidden states from any intermediate transformer layers
- **Multiple Reasoning Strategies**: Support for layer injection and input enhancement strategies
- **Iterative Reasoning**: Perform multiple reasoning passes with progressive refinement
- **Analysis Tools**: Built-in analysis of reasoning progression and hidden state evolution
- **Flexible Configuration**: Extensive configuration options for different use cases
- **Memory Management**: Efficient handling of hidden states across multiple passes

## Files Overview

- `model.py` - Original GPT model implementation
- `enhanced_gpt.py` - Enhanced GPT model with hidden state extraction capabilities
- `reasoning_framework.py` - Main framework for orchestrating reasoning passes
- `enhanced_config.py` - Enhanced configuration class with reasoning parameters
- `example_usage.py` - Comprehensive examples demonstrating framework usage

## Quick Start

```python
import torch
from model import GPTConfig
from enhanced_gpt import EnhancedGPT
from reasoning_framework import ReasoningFramework

# Create configuration
config = GPTConfig(block_size=512, vocab_size=50257)
config.n_layers = 4
config.n_heads = 4
config.n_embd = 128

# Create model and framework
model = EnhancedGPT(config)
reasoning_framework = ReasoningFramework(model, config)

# Set extraction layers
reasoning_framework.set_extraction_layers([1, 2])

# Sample input
input_ids = torch.randint(0, config.vocab_size, (1, 10))

# Perform reasoning
logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)
logits_reasoning, new_hidden = reasoning_framework.perform_reasoning_pass(
    input_ids, hidden_states, injection_method='add'
)
```

## Core Components

### EnhancedGPT

The `EnhancedGPT` class extends the original GPT model with:

- **Hidden State Extraction**: Extract representations from intermediate layers
- **Multiple Forward Passes**: Support for different injection strategies
- **Memory Management**: Store and retrieve hidden states efficiently

### ReasoningFramework

The `ReasoningFramework` class provides:

- **Pass Orchestration**: Manage multiple reasoning passes
- **Strategy Support**: Layer injection and input enhancement strategies
- **Analysis Tools**: Analyze reasoning progression and hidden state evolution
- **History Management**: Keep track of reasoning history across passes

### ReasoningConfig

The `ReasoningConfig` class extends the base configuration with:

- **Reasoning Parameters**: Configure extraction layers, injection methods, etc.
- **Analysis Settings**: Control analysis and logging behavior
- **Preset Configurations**: Pre-defined configurations for common use cases

## Usage Examples

### Basic Hidden State Extraction

```python
# Enable hidden extraction from specific layers
model.enable_hidden_extraction([1, 2])

# Perform forward pass
logits, loss = model(input_ids, return_hidden_states=True)

# Access extracted hidden states
hidden_states = model.get_hidden_states()
print(f"Extracted from layers: {list(hidden_states.keys())}")
```

### Multi-Pass Reasoning

```python
# Perform iterative reasoning
results = reasoning_framework.perform_iterative_reasoning(
    input_ids=input_ids,
    num_iterations=3,
    injection_method='add',
    reasoning_strategy='layer_injection'
)

# Analyze progression
analysis = reasoning_framework.analyze_reasoning_progression()
print(f"Completed {analysis['num_passes']} reasoning passes")
```

### Custom Injection Methods

```python
# Test different injection methods
for method in ['add', 'replace', 'concat']:
    logits, hidden = reasoning_framework.perform_reasoning_pass(
        input_ids, previous_hidden_states, 
        injection_method=method
    )
    # Analyze results...
```

## Configuration Options

### Extraction Settings
- `extraction_layers`: List of layer indices to extract from
- `hidden_state_memory`: Whether to keep states in memory
- `save_intermediate_states`: Save states to disk

### Reasoning Settings
- `injection_method`: How to inject hidden states ('add', 'concat', 'replace')
- `reasoning_strategy`: Overall reasoning approach
- `max_reasoning_passes`: Maximum number of passes allowed

### Analysis Settings
- `enable_analysis`: Enable automatic analysis
- `analysis_metrics`: Which metrics to compute
- `reasoning_temperature`: Temperature for sampling

## Advanced Features

### Layer-wise Attention
Enable attention mechanisms that consider hidden states from multiple layers:

```python
config = ReasoningConfig(
    block_size=1024, vocab_size=50257,
    layer_wise_attention=True,
    cross_layer_attention=True
)
```

### Memory Optimization
For large models or long sequences:

```python
# Disable memory storage to save GPU memory
config.hidden_state_memory = False

# Use selective extraction
config.extraction_layers = [2, 4]  # Only extract from specific layers
```

### Custom Analysis
Add custom analysis metrics:

```python
config.analysis_metrics = [
    'logits_evolution',
    'hidden_state_changes', 
    'attention_patterns',
    'custom_metric'
]
```

## Performance Considerations

- **Memory Usage**: Hidden state extraction increases memory usage significantly
- **Computational Cost**: Multiple forward passes increase inference time
- **Layer Selection**: Choose extraction layers carefully to balance performance and capability
- **Batch Size**: Consider reducing batch size when using hidden state extraction

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or disable hidden state memory
2. **Layer Index Errors**: Ensure extraction layer indices are within valid range
3. **Shape Mismatches**: Check that hidden state shapes match across passes

### Debug Mode

Enable detailed logging:

```python
config.enable_analysis = True
config.save_intermediate_states = True
```

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic hidden state extraction
- Multi-pass reasoning
- Different injection strategies
- Analysis and visualization
- Advanced usage patterns

## Contributing

This framework is designed to be extensible. Key areas for extension:
- New injection methods
- Additional reasoning strategies
- Custom analysis metrics
- Memory optimization techniques

## License

This framework extends the original GPT implementation and follows the same licensing terms.

