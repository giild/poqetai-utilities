import os
import torch
import torch.nn as nn
import json
from collections import OrderedDict

def chunk_model(model, output_dir='model_chunks'):
    """
    Load a PyTorch model and save each layer in a separate folder with its weights
    and a summary file.
    
    Args:
        model: A PyTorch model or path to a saved model file
        output_dir: Directory where the chunked model will be saved
    """
    # If model is a file path, load it
    if isinstance(model, str):
        try:
            model = torch.load(model, map_location=torch.device('cpu'))
        except Exception as e:
            raise ValueError(f"Failed to load model from {model}: {e}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each named module in the model
    for name, module in model.named_modules():
        # Skip the model itself and container modules
        if name == '' or len(list(module.children())) > 0:
            continue
        
        # Create a folder for this layer
        layer_dir = os.path.join(output_dir, name.replace('.', '_'))
        os.makedirs(layer_dir, exist_ok=True)
        
        # Extract the state dict for this module
        layer_state_dict = OrderedDict()
        for param_name, param in module.state_dict().items():
            full_param_name = f"{name}.{param_name}" if name else param_name
            if full_param_name in model.state_dict():
                layer_state_dict[param_name] = param
        
        # Save the weights
        if layer_state_dict:
            torch.save(layer_state_dict, os.path.join(layer_dir, 'weights.pth'))
        
        # Create a summary of the layer
        summary = {
            'name': name,
            'type': module.__class__.__name__,
            'parameters': sum(p.numel() for p in module.parameters()),
            'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad),
            'shape': [list(p.shape) for p in module.parameters()],
            'attributes': {k: str(v) for k, v in module.__dict__.items() 
                          if not k.startswith('_') and not callable(v)}
        }
        
        # Save the summary
        with open(os.path.join(layer_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    
    print(f"Model chunked successfully. Chunks saved to {output_dir}")

def main():
    """Example usage"""
    # Example 1: Load a model from a file
    # chunk_model('path/to/model.pth')
    
    # Example 2: Create and chunk a simple model
    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    chunk_model(model, './example_model_chunks')

if __name__ == '__main__':
    main()
