"""
Hypercube Function Prediction Framework using Transformers with Sequential Injections

This framework tests the ability of transformers with sequential injections to predict
real-valued functions over hypercubes. Each input token represents a bit (+1 or -1),
and the output is obtained using a linear layer on the last token's hidden embedding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable, Optional
from model import GPTConfig
from sequential_injection_gpt import SequentialInjectionGPT, SequentialInjectionReasoningFramework

class HypercubeFunctionPredictor(nn.Module):
    """
    Transformer-based predictor for real-valued functions over hypercubes.
    Uses sequential injections to enhance function prediction.
    """
    
    def __init__(self, config: GPTConfig, function_dim: int, num_zero_tokens: int = 0):
        super().__init__()
        self.config = config
        self.function_dim = function_dim
        self.num_zero_tokens = num_zero_tokens
        
        # Core transformer with sequential injection capability
        self.transformer = SequentialInjectionGPT(config)
        
        # Function prediction head
        self.function_head = nn.Linear(config.n_embd, 1)
        
        # Gradient-based injection specification tracking
        self.injection_specifications = []
        self.gradient_threshold = 0.8  # Threshold for adding injection specifications
        self.zero_position_counter = function_dim  # Counter for zero token positions
        
        # Learnable base weight for injection specifications, initialized close to zero
        self.base_weight = torch.tensor(0.1)  # Small initialization
        self.learnable_weights = nn.ParameterList([nn.Parameter(self.base_weight.clone()) for _ in range(self.num_zero_tokens)])  # Store learnable weights for each injection
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_tokens: torch.Tensor, 
                injection_sequence: Optional[List[Dict]] = None) -> torch.Tensor:
        """
        Forward pass for function prediction.
        
        Args:
            input_tokens: Input tokens representing bits (+1/-1) [batch, seq_len]
            injection_sequence: Optional sequential injection sequence
        
        Returns:
            function_values: Predicted function values [batch, 1]
        """
        batch_size, seq_len = input_tokens.shape
        
        # Convert bit tokens to token indices (-1 for -1, +1 for +1)
        token_indices = input_tokens.long()
        
        # Add fixed number of zero tokens after the hypercube bit tokens
        if self.num_zero_tokens > 0:
            zero_tokens = torch.zeros(batch_size, self.num_zero_tokens, dtype=torch.long, device=token_indices.device)
            token_indices = torch.cat([token_indices, zero_tokens], dim=1)
        
        # Add output token (index 2) at the end of all sequences
        output_tokens = torch.full((batch_size, 1), 2, dtype=torch.long, device=token_indices.device)
        token_indices = torch.cat([token_indices, output_tokens], dim=1)
        
        if injection_sequence is None:
            print("no injection sequence, doing standard forward pass")
            # Standard forward pass
            _, _, hidden_states = self.transformer(token_indices, return_hidden_states=True)
            last_layer_hidden = hidden_states[f'layer_{self.config.n_layers}']
        else:
            # Forward pass with sequential injections
            reasoning_framework = SequentialInjectionReasoningFramework(self.transformer, self.config)
            
            results = reasoning_framework.perform_sequential_injections(
                token_indices, injection_sequence
            )
            last_layer_hidden = results[-1]['hidden_states'][f'layer_{self.config.n_layers}']
            #print('length of results is ', len(results))
        
        # Get hidden states from the last layer for the last token
        
        
        # Extract output token's hidden state (at position function_dim + num_zero_tokens)
        output_token_position = self.function_dim + self.num_zero_tokens
        output_token_hidden = last_layer_hidden[:, output_token_position, :]  # [batch, hidden_dim]
        
        # Predict function value
        function_values = self.function_head(output_token_hidden)  # [batch, 1]
        
        return function_values
    
    def analyze_gradients_and_add_injections(self, input_tokens: torch.Tensor, 
                                           target_values: torch.Tensor,
                                           threshold: Optional[float] = None) -> List[Dict]:
        """
        Analyze gradients of zero input tokens and add injection specifications if above threshold.
        Only runs during training mode.
        
        Args:
            input_tokens: Input tokens representing bits (+1/-1) [batch, seq_len]
            target_values: Target function values [batch, 1]
            threshold: Optional threshold override (uses self.gradient_threshold if None)
        
        Returns:
            Updated injection specifications list
        """
        # Only run gradient analysis during training
        if not self.training:
            return self.injection_specifications
            
        if threshold is None:
            threshold = self.gradient_threshold
            
        if self.num_zero_tokens == 0:
            return self.injection_specifications
            
        batch_size, seq_len = input_tokens.shape
        
        # Convert bit tokens to token indices and add zero tokens
        token_indices = input_tokens.long()
        if self.num_zero_tokens > 0:
            zero_tokens = torch.zeros(batch_size, self.num_zero_tokens, dtype=torch.long, device=token_indices.device)
            token_indices = torch.cat([token_indices, zero_tokens], dim=1)
        
        # Add output token (index 2) at the end of all sequences
        output_tokens = torch.full((batch_size, 1), 2, dtype=torch.long, device=token_indices.device)
        token_indices = torch.cat([token_indices, output_tokens], dim=1)
        
        # Check the current zero token position using counter
        zero_position = self.zero_position_counter
        
        # Hidden states are always extracted automatically
        
        # Forward pass to get predictions and enable gradient tracking
        predictions = self.forward(input_tokens, injection_sequence=self.injection_specifications)
        
        # Compute loss (averaged across batch)
        loss = F.mse_loss(predictions, target_values.clone().detach())
        
        # Backward pass to get gradients
        loss.backward()
        #print('predictor.transformer.transformer.wte.weight.grad is ', self.transformer.transformer.wte.weight.grad)
        if len(self.injection_specifications) > 0:
            position =self.injection_specifications[-1]['extraction']['position']
            layer_name = self.injection_specifications[-1]['extraction']['layer_name']
            #print('added layer name is ', layer_name, 'and position is ', position)
            #print('predictor.transformer.hiddden_states[layer_0].grad in the added layer and position is ', self.transformer.hidden_states[layer_name].grad[0,position,:])
            # Get gradients directly from the transformer's internal hidden states
        # We need to access the original tensors, not the cloned ones
        if hasattr(self.transformer, 'hidden_states') and 'layer_0' in self.transformer.hidden_states:
            # Use the final hidden state which is definitely part of the computation graph
            zero_hidden = self.transformer.hidden_states['layer_0']
            if zero_hidden.grad is not None:
                zero_token_grads = zero_hidden.grad[:, zero_position, :]  # [batch, n_embd]
                #print('zero_token_grads major is ', zero_token_grads[:3, :])
            else:
                print("DEBUG: No gradients found in layer_0 hidden state")
                return self.injection_specifications
        else:
            print("DEBUG: No gradients found in predictions.grad")
            return self.injection_specifications
        
        # Compute dot products with all intermediate hidden embeddings using tensor operations
        best_token_idx = 0
        best_layer = 0
        best_dot_product_sign = 1
        
        # Check all layers - access from transformer's internal hidden states
        max_cosine_similarity = 0
        for layer_idx in range(1, self.config.n_layers):
            if f'layer_{layer_idx}' in self.transformer.hidden_states:
                layer_hidden = self.transformer.hidden_states[f'layer_{layer_idx}'].clone().detach()  # [batch, seq_len, n_embd]
                
                # Check all hypercube bit tokens (intermediate tokens)
                for token_idx in range(self.function_dim + self.num_zero_tokens + 1):
                    # Get hidden embedding for this token at this layer
                    token_hidden = layer_hidden[:, token_idx, :]  # [batch, n_embd]
                    
                    # Compute dot products per sample using tensor operations
                    # zero_token_grads: [batch, n_embd], token_hidden: [batch, n_embd]
                    #print('zero_token_grads is ', zero_token_grads[:3, :], 'token_hidden is ', token_hidden[:3, :])
                    dot_products = torch.sum(zero_token_grads * token_hidden, dim=1)  # [batch]
                    average_norm_zero_token_grads = torch.mean(torch.norm(zero_token_grads, dim=1))
                    average_norm_token_hidden = torch.mean(torch.norm(token_hidden, dim=1))
                    scaled_dot_product_sign = torch.sign(torch.mean(dot_products)).item()
                    if average_norm_zero_token_grads > 0 and average_norm_token_hidden > 0:
                        cosine_similarity = torch.mean(dot_products) / (average_norm_zero_token_grads * average_norm_token_hidden)
                        #print('cosine_similarity is ', cosine_similarity)
                        if cosine_similarity * scaled_dot_product_sign > max_cosine_similarity:
                            max_cosine_similarity = cosine_similarity * scaled_dot_product_sign
                            best_token_idx = token_idx
                            best_layer = layer_idx
                            best_dot_product_sign = scaled_dot_product_sign
                    
                    
                   
        
        # If dot product is above threshold, add injection specification
        print('max_cosine_similarity is ', max_cosine_similarity, 'threshold is ', threshold)
        if max_cosine_similarity > threshold and len(self.injection_specifications) < self.num_zero_tokens:
            #assert len(self.injection_specifications) == 1, 'injection specifications should be empty'
            # Determine the injection weight with correct sign
            # If gradient dot product is positive, inject negative of hidden embedding
            # If gradient dot product is negative, inject positive of hidden embedding
            # Use learnable base_weight parameter instead of computed value
            base_weight = min(1.0, max_cosine_similarity / threshold)
            injection_weight = -base_weight if best_dot_product_sign > 0 else base_weight
            with torch.no_grad():
                self.learnable_weights[zero_position - self.function_dim].mul_(injection_weight)
            
            
            injection_spec = {
                'extraction': {
                    'key': f'token_{best_token_idx}_layer_{best_layer}',
                    'layer_name': f'layer_{best_layer}',
                    'layer': best_layer,
                    'position': best_token_idx
                },
                'injection': {
                    'layer': 0,  # This corresponds to layer_0 (token_embd + position_embd)
                    'position': zero_position,
                    'method': 'add',
                    'weight': self.learnable_weights[zero_position - self.function_dim]  # Weight with correct sign based on gradient direction
                }
            }
            
            # Add to specifications if not already present
            if injection_spec not in self.injection_specifications:
                print('adding injection specification inside iffff')
                self.injection_specifications.append(injection_spec)
                print(f"Added injection specification for zero token at position {zero_position}: "
                      f"dot_product={max_cosine_similarity:.4f} (sign={best_dot_product_sign:+.0f}), "
                      f"from_token={best_token_idx}_layer={best_layer} to_position={zero_position}, "
                      f"weight={injection_weight:.4f}")
                
                # Increment counter for next zero token position
                self.zero_position_counter += 1
        
        # Clear gradients
        self.zero_grad()
        
        return self.injection_specifications
    
    def get_injection_specifications(self) -> List[Dict]:
        """Get current injection specifications."""
        return self.injection_specifications.copy()
    
    def clear_injection_specifications(self):
        """Clear all injection specifications."""
        self.injection_specifications = []
    
    def set_gradient_threshold(self, threshold: float):
        """Set the gradient threshold for adding injection specifications."""
        self.gradient_threshold = threshold
    
    def reset_zero_position_counter(self):
        """Reset the zero position counter to start from function_dim."""
        self.zero_position_counter = self.function_dim
    


class HypercubeFunctionDataset:
    """
    Dataset for hypercube function prediction tasks.
    """
    
    def __init__(self, function_dim: int, num_samples: int, 
                 function: Callable[[torch.Tensor], torch.Tensor],
                 seed: int = 42):
        self.function_dim = function_dim
        self.num_samples = num_samples
        self.function = function
        
        # Generate random hypercube points
        torch.manual_seed(seed)
        self.inputs = torch.randint(0, 2, (num_samples, function_dim)).float() * 2 - 1  # Convert to -1,1
        self.targets = self.function(self.inputs)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
    def get_batch(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a random batch from the dataset."""
        indices = torch.randint(0, len(self), (batch_size,))
        inputs = self.inputs[indices].to(device)
        targets = self.targets[indices].to(device)
        return inputs, targets


class HypercubeFunctionTrainer:
    """
    Trainer for hypercube function prediction with sequential injections.
    """
    
    def __init__(self, model: HypercubeFunctionPredictor, 
                 config: GPTConfig,
                 learning_rate: float = 1e-3):
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_dataset: HypercubeFunctionDataset, 
                   batch_size: int, device: torch.device,
                   injection_sequence: Optional[List[Dict]] = None) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_dataset), batch_size):
            # Get batch
            inputs, targets = train_dataset.get_batch(batch_size, device)
            
            # Forward pass
            predictions = self.model(inputs, injection_sequence)
            
            # Compute loss
            loss = self.criterion(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def evaluate(self, val_dataset: HypercubeFunctionDataset, 
                batch_size: int, device: torch.device,
                injection_sequence: Optional[List[Dict]] = None) -> float:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_dataset), batch_size):
                # Get batch
                inputs, targets = val_dataset.get_batch(batch_size, device)
                
                # Forward pass
                predictions = self.model(inputs, injection_sequence)
                
                # Compute loss
                loss = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_dataset: HypercubeFunctionDataset, 
             val_dataset: HypercubeFunctionDataset,
             num_epochs: int, batch_size: int, device: torch.device,
             injection_sequence: Optional[List[Dict]] = None) -> Dict[str, List[float]]:
        """Train the model."""
        print(f"Training hypercube function predictor...")
        print(f"Function dimension: {train_dataset.function_dim}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Sequential injections: {'Yes' if injection_sequence else 'No'}")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(train_dataset, batch_size, device, injection_sequence)
            
            # Validate
            val_loss = self.evaluate(val_dataset, batch_size, device, injection_sequence)
            
            # Store losses
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }


# Test functions for hypercubes
def linear_function(x: torch.Tensor) -> torch.Tensor:
    """Linear function: f(x) = sum(x)"""
    return torch.sum(x, dim=1, keepdim=True)

def quadratic_function(x: torch.Tensor) -> torch.Tensor:
    """Quadratic function: f(x) = sum(x^2)"""
    return torch.sum(x**2, dim=1, keepdim=True)

def parity_function(x: torch.Tensor) -> torch.Tensor:
    """Parity function: f(x) = (-1)^(sum of positive bits)"""
    positive_bits = (x > 0).sum(dim=1, keepdim=True)
    return (-1) ** positive_bits.float()

def xor_function(x: torch.Tensor) -> torch.Tensor:
    """XOR function: f(x) = x[0] XOR x[1] XOR ... XOR x[n-1]"""
    return torch.prod(x, dim=1, keepdim=True)

def polynomial_function(x: torch.Tensor) -> torch.Tensor:
    """Polynomial function: f(x) = sum(x) + sum(x^2) + sum(x^3)"""
    return torch.sum(x, dim=1, keepdim=True) + torch.sum(x**2, dim=1, keepdim=True) + torch.sum(x**3, dim=1, keepdim=True)


def create_injection_sequence_for_function(function_dim: int, 
                                         function_name: str) -> List[Dict]:
    """
    Create a sequential injection sequence tailored for specific functions.
    Note: layer 0 = initial embeddings (token_embd + position_embd)
          layer 1 = output of first transformer block
          layer 2 = output of second transformer block, etc.
    """
    if function_name == "linear":
        # For linear functions, inject information from early layers to later layers
        return [
            {
                'extraction': {'layer_name': 'layer_0', 'position': 0, 'key': 'first_token'},
                'injection': {'layer': 1, 'position': function_dim - 1, 'method': 'add', 'weight': 0.3}
            }
        ]
    elif function_name == "quadratic":
        # For quadratic functions, inject squared information
        return [
            {
                'extraction': {'layer_name': 'layer_0', 'position': 0, 'key': 'first_token'},
                'injection': {'layer': 1, 'position': function_dim - 1, 'method': 'add', 'weight': 0.5}
            },
            {
                'extraction': {'layer_name': 'layer_1', 'position': function_dim - 1, 'key': 'last_token'},
                'injection': {'layer': 2, 'position': function_dim - 1, 'method': 'weighted_add', 'weight': 0.7}
            }
        ]
    elif function_name == "parity":
        # For parity functions, inject information about bit counts
        return [
            {
                'extraction': {'layer_name': 'layer_0', 'position': 0, 'key': 'first_token'},
                'injection': {'layer': 1, 'position': function_dim - 1, 'method': 'replace'}
            }
        ]
    else:
        # Default injection sequence
        return [
            {
                'extraction': {'layer_name': 'layer_0', 'position': 0, 'key': 'first_token'},
                'injection': {'layer': 1, 'position': function_dim - 1, 'method': 'add', 'weight': 0.5}
            }
        ]


def run_hypercube_function_experiment(function_dim: int = 4,
                                    function_name: str = "linear",
                                    num_epochs: int = 100,
                                    batch_size: int = 32,
                                    use_injections: bool = True,
                                    num_zero_tokens: int = 0):
    """
    Run a complete experiment for hypercube function prediction.
    """
    print(f"Hypercube Function Prediction Experiment")
    print(f"="*60)
    print(f"Function: {function_name}")
    print(f"Dimension: {function_dim}")
    print(f"Sequential Injections: {'Yes' if use_injections else 'No'}")
    print(f"="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model configuration
    config = GPTConfig(block_size=function_dim + num_zero_tokens + 1, vocab_size=4)  # vocab_size=4 for -1, 0, output, +1
    config.n_layers = 3
    config.n_heads = 2
    config.n_embd = 64
    
    # Create model
    model = HypercubeFunctionPredictor(config, function_dim, num_zero_tokens).to(device)
    
    # Select function
    functions = {
        "linear": linear_function,
        "quadratic": quadratic_function,
        "parity": parity_function,
        "xor": xor_function,
        "polynomial": polynomial_function
    }
    
    if function_name not in functions:
        raise ValueError(f"Unknown function: {function_name}")
    
    function = functions[function_name]
    
    # Create datasets
    train_dataset = HypercubeFunctionDataset(function_dim, 1000, function, seed=42)
    val_dataset = HypercubeFunctionDataset(function_dim, 200, function, seed=123)
    
    # Create injection sequence
    injection_sequence = None
    if use_injections:
        injection_sequence = create_injection_sequence_for_function(function_dim, function_name)
        print(f"Injection sequence: {len(injection_sequence)} steps")
        for i, spec in enumerate(injection_sequence):
            extraction = spec['extraction']
            injection = spec['injection']
            print(f"  Step {i+1}: Extract {extraction['key']} from {extraction['layer_name']} pos {extraction['position']}")
            print(f"           -> Inject into layer {injection['layer']} pos {injection['position']} using {injection['method']}")
    
    # Create trainer
    trainer = HypercubeFunctionTrainer(model, config)
    
    # Train
    results = trainer.train(train_dataset, val_dataset, num_epochs, batch_size, device, injection_sequence)
    
    # Test on some examples
    print(f"\nTesting on examples:")
    model.eval()
    with torch.no_grad():
        test_inputs, test_targets = val_dataset.get_batch(5, device)
        predictions = model(test_inputs, injection_sequence)
        
        for i in range(5):
            input_bits = test_inputs[i].tolist()
            target = test_targets[i].item()
            prediction = predictions[i].item()
            error = abs(target - prediction)
            print(f"  Input: {input_bits} -> Target: {target:.4f}, Prediction: {prediction:.4f}, Error: {error:.4f}")
    
    
    return {
        'model': model,
        'trainer': trainer,
        'results': results,
        'injection_sequence': injection_sequence
    }


if __name__ == "__main__":
    print("Hypercube Function Prediction Framework")
    print("="*80)
    
    # Run experiments
    experiments = [
        ("linear", 4, False),      # Linear function without injections
        ("linear", 4, True),      # Linear function with injections
        ("quadratic", 4, False),  # Quadratic function without injections
        ("quadratic", 4, True),   # Quadratic function with injections
        ("parity", 4, False),     # Parity function without injections
        ("parity", 4, True),      # Parity function with injections
    ]
    
    results = {}
    
    for function_name, function_dim, use_injections in experiments:
        experiment_name = f"{function_name}_{function_dim}d_{'with' if use_injections else 'without'}_injections"
        print(f"\n\nRunning experiment: {experiment_name}")
        print("="*80)
        
        try:
            result = run_hypercube_function_experiment(
                function_dim=function_dim,
                function_name=function_name,
                num_epochs=50,
                batch_size=16,
                use_injections=use_injections
            )
            results[experiment_name] = result
        except Exception as e:
            print(f"Error in experiment {experiment_name}: {e}")
    
    print(f"\n" + "="*80)
    print(f"All experiments completed!")
    print(f"="*80)



