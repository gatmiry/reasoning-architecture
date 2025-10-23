# Sequential Hidden Embedding Injections

This implementation provides **sequential** hidden embedding injections where each injection happens after a complete forward pass, exactly as you requested.

## Sequential Pattern

```
Forward Pass -> Extract Hidden Embedding -> Inject -> Forward Pass -> Extract Hidden Embedding -> Inject -> Forward Pass -> ...
```

Each injection is applied after a **complete forward pass**, not all at once.

## Key Features

✅ **Sequential Processing**: Forward -> Inject -> Forward -> Inject -> ...  
✅ **Array-based Specifications**: Define injection sequences as arrays  
✅ **Full Gradient Flow**: Backpropagate gradients through all sequential steps  
✅ **Flexible Injection Methods**: Support for add, replace, weighted_add, and concat  
✅ **Step-by-step Analysis**: Track changes at each sequential step  

## Files

- `sequential_injection_gpt.py` - Core implementation with SequentialInjectionGPT
- `sequential_injection_example.py` - Example demonstrating sequential injections

## Usage

### Basic Sequential Injections

```python
from sequential_injection_gpt import SequentialInjectionGPT, SequentialInjectionReasoningFramework
from model import GPTConfig

# Setup
config = GPTConfig(block_size=256, vocab_size=50257)
config.n_layers = 3
config.n_heads = 4
config.n_embd = 128

model = SequentialInjectionGPT(config)
reasoning_framework = SequentialInjectionReasoningFramework(model, config)
reasoning_framework.set_extraction_layers([1, 2])

# Define sequential injection sequence
injection_sequence = [
    # Step 1: Extract from layer 1, inject into layer 1
    {
        'extraction': {
            'layer_name': 'layer_1',
            'position': 2,
            'key': 'token_2_layer_1'
        },
        'injection': {
            'layer': 1,
            'position': 0,
            'method': 'add',
            'weight': 0.5
        }
    },
    # Step 2: Extract from layer 2, inject into layer 2
    {
        'extraction': {
            'layer_name': 'layer_2',
            'position': 4,
            'key': 'token_4_layer_2'
        },
        'injection': {
            'layer': 2,
            'position': 1,
            'method': 'replace'
        }
    }
    # ... more sequential steps
]

# Perform sequential injections
results = reasoning_framework.perform_sequential_injections(
    input_ids, injection_sequence
)

# Each result contains:
# - step: step number (0 = initial, 1+ = injection steps)
# - step_type: 'initial_forward' or 'injection_forward'
# - logits: output logits from this step
# - hidden_states: hidden states from this step
# - injection_spec: injection specification for this step
```

### Sequential vs Parallel

**Sequential (what you wanted):**
```
1. Forward pass -> Extract token_1
2. Forward pass with token_1 injection -> Extract token_2  
3. Forward pass with token_2 injection
```

**Parallel (previous implementation):**
```
1. Forward pass -> Extract token_1, token_2
2. Forward pass with both token_1 and token_2 injections
```

## Sequential Processing Details

### Step-by-Step Process

1. **Initial Forward Pass**: Extract hidden states from all target layers
2. **Step 1**: 
   - Extract specific token embedding from hidden states
   - Perform forward pass with single injection
   - Extract new hidden states for next step
3. **Step 2**:
   - Extract specific token embedding from updated hidden states
   - Perform forward pass with single injection
   - Extract new hidden states for next step
4. **Continue...** until all injections are complete

### Gradient Flow

Gradients flow through **all sequential steps**:
- Each forward pass contributes to the final loss
- Each injection point receives gradients
- All model parameters are updated through the sequential chain

## Injection Specification Format

Each step in the sequence has:

**Extraction Spec:**
- `'layer_name'`: Layer to extract from (e.g., 'layer_1')
- `'position'`: Token position to extract
- `'key'`: Key to store the extracted embedding

**Injection Spec:**
- `'layer'`: Layer index to inject into (0-indexed)
- `'position'`: Token position to inject at
- `'method'`: Injection method ('add', 'replace', 'weighted_add', 'concat')
- `'weight'`: Weight for weighted operations (optional)

## Example Output

```
Sequential injection results:
Total steps: 5

Step 0: initial_forward
  Logits shape: torch.Size([2, 6, 50257])
  Logits norm: 175.5938

Step 1: injection_forward
  Logits shape: torch.Size([2, 6, 50257])
  Logits norm: 175.4605
  L2 difference from previous: 42.983490
  Injected: token_2_layer_1 -> Layer 1 Pos 0

Step 2: injection_forward
  Logits shape: torch.Size([2, 6, 50257])
  Logits norm: 175.6719
  L2 difference from previous: 113.050598
  Injected: token_4_layer_2 -> Layer 2 Pos 1

Step 3: injection_forward
  Logits shape: torch.Size([2, 6, 50257])
  Logits norm: 175.6302
  L2 difference from previous: 108.980911
  Injected: token_1_layer_1 -> Layer 2 Pos 3

Step 4: injection_forward
  Logits shape: torch.Size([2, 6, 50257])
  Logits norm: 175.4942
  L2 difference from previous: 74.501099
  Injected: token_5_layer_2 -> Layer 1 Pos 4
```

## Key Classes

### SequentialInjectionGPT

- **`forward()`**: Standard forward pass
- **`forward_with_single_injection()`**: Forward pass with single injection
- **`_apply_single_injection()`**: Apply single injection to hidden state
- **`extract_hidden_embeddings()`**: Extract embeddings from hidden states

### SequentialInjectionReasoningFramework

- **`perform_initial_pass()`**: Initial forward pass to extract hidden states
- **`perform_sequential_injections()`**: Perform complete sequential injection sequence
- **`compute_gradient_flow_analysis()`**: Analyze gradient flow through all steps

## Testing

Run the example to see sequential injections in action:

```bash
python sequential_injection_example.py
```

The example demonstrates:
- 4-step sequential injection sequence
- Forward -> Inject -> Forward -> Inject pattern
- Gradient backpropagation through all steps
- Step-by-step analysis of changes

## Requirements Met

✅ **Sequential injections**: Forward -> Inject -> Forward -> Inject -> ...  
✅ **Array-based specifications**: Injection sequences defined as arrays  
✅ **Full gradient backpropagation**: Gradients flow through all sequential steps  

This implementation provides exactly what you requested: sequential hidden embedding injections where each injection happens after a complete forward pass, with full gradient backpropagation through all steps.




