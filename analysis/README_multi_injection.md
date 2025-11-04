# Multiple Hidden Embedding Injections with Backpropagation

This implementation provides a comprehensive solution for injecting hidden embeddings multiple times into transformer inputs with full gradient backpropagation through all injection points.

## Key Features

✅ **Multiple Injections**: Inject hidden embeddings multiple times at different layers and positions  
✅ **Array-based Specifications**: Define injection specifications as arrays for easy management  
✅ **Full Gradient Flow**: Backpropagate gradients through all injection points to model weights  
✅ **Flexible Injection Methods**: Support for add, replace, weighted_add, and concat methods  
✅ **Iterative Reasoning**: Support for iterative reasoning with different injection sequences  

## Files

- `multi_injection_gpt.py` - Core implementation with MultiInjectionGPT and MultiInjectionReasoningFramework
- `focused_injection_example.py` - Focused example demonstrating the key requirements
- `multi_injection_example.py` - Comprehensive example with multiple demonstrations

## Usage

### Basic Multiple Injections

```python
from multi_injection_gpt import MultiInjectionGPT, MultiInjectionReasoningFramework
from model import GPTConfig

# Setup
config = GPTConfig(block_size=256, vocab_size=50257)
config.n_layers = 3
config.n_heads = 4
config.n_embd = 128

model = MultiInjectionGPT(config)
reasoning_framework = MultiInjectionReasoningFramework(model, config)
reasoning_framework.set_extraction_layers([1, 2])

# Initial pass
logits_initial, hidden_states = reasoning_framework.perform_initial_pass(input_ids)

# Define injection specifications array
injection_specs_array = [
    {
        'layer': 1,  # Inject into layer 1
        'position': 0,  # At position 0
        'hidden_embedding_key': 'token_2_layer_1',
        'method': 'add',
        'weight': 0.5
    },
    {
        'layer': 2,  # Inject into layer 2
        'position': 3,  # At position 3
        'hidden_embedding_key': 'token_4_layer_2',
        'method': 'replace'
    }
]

# Extract hidden embeddings
extraction_specs = [
    {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'},
    {'layer_name': 'layer_2', 'position': 4, 'key': 'token_4_layer_2'},
]
hidden_embeddings = model.extract_hidden_embeddings(hidden_states, extraction_specs)

# Perform multiple injections
logits_injected, new_hidden_states = reasoning_framework.perform_reasoning_pass_with_injections(
    input_ids, hidden_embeddings, injection_specs_array
)

# Backpropagate gradients
loss = F.cross_entropy(logits_injected.view(-1, logits_injected.size(-1)), targets.view(-1))
loss.backward()  # Gradients flow through all injection points
```

### Injection Methods

1. **'add'**: Add the hidden embedding to the existing value
2. **'replace'**: Replace the existing value with the hidden embedding
3. **'weighted_add'**: Weighted combination: `(1-weight) * existing + weight * hidden`
4. **'concat'**: Concatenate (requires dimension matching or projection)

### Injection Specification Format

Each injection specification is a dictionary with:
- `'layer'`: Layer index to inject into (0-indexed)
- `'position'`: Token position to inject at
- `'hidden_embedding_key'`: Key of the hidden embedding to inject
- `'method'`: Injection method ('add', 'replace', 'weighted_add', 'concat')
- `'weight'`: Weight for weighted operations (optional)

### Gradient Flow

The implementation ensures that gradients flow through all injection points:

1. **Direct Injections**: When hidden embeddings have the same dimension as the target, gradients flow directly
2. **Projection Injections**: When dimensions don't match, learnable projection layers are created and gradients flow through them
3. **Model Parameters**: All model parameters receive gradients from the injection points

### Iterative Reasoning

```python
# Define injection sequences for multiple iterations
injection_sequences = [
    # Iteration 1
    [
        {'extraction': {'layer_name': 'layer_1', 'position': 2, 'key': 'token_2_layer_1'}},
        {'layer': 1, 'position': 0, 'hidden_embedding_key': 'token_2_layer_1', 'method': 'add'}
    ],
    # Iteration 2
    [
        {'extraction': {'layer_name': 'layer_2', 'position': 4, 'key': 'token_4_layer_2'}},
        {'layer': 2, 'position': 1, 'hidden_embedding_key': 'token_4_layer_2', 'method': 'replace'}
    ]
]

# Perform iterative reasoning
results = reasoning_framework.perform_iterative_reasoning_with_injections(
    input_ids, injection_sequences
)
```

## Key Implementation Details

### MultiInjectionGPT Class

- **`forward()`**: Main forward pass with support for injection specifications
- **`_apply_injections()`**: Apply multiple injections to hidden states
- **`_apply_single_injection()`**: Apply a single injection with specified method
- **`extract_hidden_embeddings()`**: Extract embeddings based on specifications
- **`forward_with_multiple_injections()`**: Forward pass with multiple injections

### MultiInjectionReasoningFramework Class

- **`perform_initial_pass()`**: Extract hidden states from initial forward pass
- **`perform_reasoning_pass_with_injections()`**: Perform reasoning with injections
- **`perform_iterative_reasoning_with_injections()`**: Iterative reasoning with different sequences
- **`compute_gradient_flow_analysis()`**: Analyze gradient flow through injection points

## Testing

Run the examples to see the implementation in action:

```bash
python focused_injection_example.py
python multi_injection_example.py
```

The examples demonstrate:
- Multiple injections with different methods
- Gradient backpropagation through all injection points
- Iterative reasoning with different injection sequences
- Comprehensive gradient flow analysis

## Requirements Met

✅ **Inject hidden_embeddings after forward pass into inputs multiple times**  
✅ **Use specifications given in an array**  
✅ **Backprop gradients through all injection points to model weights**  

The implementation provides a robust, flexible, and efficient solution for multiple hidden embedding injections with full gradient support.




