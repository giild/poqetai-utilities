import torch
import json
import numpy as np
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from safetensors import safe_open

def extract_layer_number(key: str) -> tuple:
    """
    Extract layer number from key for sorting purposes.
    
    Args:
        key: Layer key name
        
    Returns:
        Tuple of (prefix, layer_number, suffix) for sorting
    """
    # Look for patterns like "layer.12", "layers.5", "transformer.h.8", etc.
    patterns = [
        r'(.*?\.(?:layer?|blocks?|h)\.?)(\d+)(\..*|$)',
        r'(.*?\.(?:layers?|blocks?|h)\.?)(\d+)(\..*|$)',
        r'(.*)transformer\.h\.(\d+)(.*)',
        r'(.*)block\.(\d+)(.*)',
        r'(.*)encoder\.layer\.(\d+)(.*)',
        r'(.*)decoder\.layer\.(\d+)(.*)'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, key, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            layer_num = int(match.group(2))
            suffix = match.group(3)
            return (prefix, layer_num, suffix)
    
    # If no layer number found, return the key as is for alphabetical sorting
    return (key, -1, "")

def sort_keys(keys: list) -> list:
    """
    Sort keys with special handling for numbered layers.
    
    Args:
        keys: List of layer keys
        
    Returns:
        Sorted list of keys
    """
    def sort_key(key):
        prefix, layer_num, suffix = extract_layer_number(key)
        # Sort first by prefix, then by layer number, then by suffix
        return (prefix, layer_num, suffix)
    
    return sorted(keys, key=sort_key)

def load_safetensors(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a checkpoint file in safetensors format.
    
    Args:
        checkpoint_path: Path to the safetensors checkpoint file
        
    Returns:
        Dictionary mapping layer names to tensors
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
   
    # First, get all keys
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())
    
    # Sort the keys
    sorted_keys = sort_keys(all_keys)
    #print(f"{sorted_keys} keys in checkpoint: {checkpoint_path}")
    
    # Load tensors in sorted order
    tensors = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
        for key in sorted_keys:
            tensors[key] = f.get_tensor(key)
    
    return tensors

def load_checkpoint(checkpoint_path: str) -> Tuple[torch.nn.Module, str]:
    """
    Load a specific checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        
    Returns:
        Tuple of (model_state_dict, checkpoint_path)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = None
    if checkpoint_path.endswith('.pt') or checkpoint_path.endswith('.pth'):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    elif checkpoint_path.endswith(".safetensors"):
        print(f"{checkpoint_path} is a safetensors file, loading with safetensors library")
        checkpoint = load_safetensors(checkpoint_path)
    
    # Extract model state dict (handle different checkpoint formats)
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        model_state_dict = checkpoint['state_dict']
    elif 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    else:
        # Assume the entire checkpoint is the state dict
        model_state_dict = checkpoint
    
    return model_state_dict, checkpoint_path

def check_tensor_overflow(tensor: torch.Tensor, threshold: float = 1e30) -> Dict[str, Any]:
    """
    Check if a tensor has values that could cause overflow.
    
    Args:
        tensor: PyTorch tensor to check
        threshold: Threshold for considering values as potential overflow
        
    Returns:
        Dictionary with overflow information
    """
    # Convert to float32 if needed for analysis
    if tensor.dtype in [torch.float16, torch.bfloat16]:
        tensor_float = tensor.float()
    else:
        tensor_float = tensor
    
    # Check for various overflow conditions
    has_inf = torch.isinf(tensor_float).any().item()
    has_nan = torch.isnan(tensor_float).any().item()
    
    # Check for very large values that might cause overflow in operations
    abs_tensor = torch.abs(tensor_float)
    max_val = abs_tensor.max().item()
    has_large_values = (abs_tensor > threshold).any().item()
    
    # Count problematic values
    inf_count = torch.isinf(tensor_float).sum().item()
    nan_count = torch.isnan(tensor_float).sum().item()
    large_count = (abs_tensor > threshold).sum().item()
    
    return {
        'has_overflow_risk': has_inf or has_nan or has_large_values,
        'has_inf': has_inf,
        'has_nan': has_nan,
        'has_large_values': has_large_values,
        'max_abs_value': max_val,
        'inf_count': inf_count,
        'nan_count': nan_count,
        'large_values_count': large_count,
        'tensor_shape': list(tensor.shape),
        'tensor_dtype': str(tensor.dtype)
    }

def extract_attention_weights(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract attention-related weights from the model state dict.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        Dictionary of attention weights
    """
    attention_weights = {}
    
    for name, tensor in state_dict.items():
        # Look for common attention layer patterns
        if any(pattern in name.lower() for pattern in [
            'attention', 'attn', 'query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj',
            'self_attn', 'cross_attn', 'multihead', 'mha'
        ]):
            attention_weights[name] = tensor
    
    return attention_weights

def validate_attention_layers(checkpoint_file: str, output_dir: str, output_filename: str, 
                            threshold: float = 1e30) -> None:
    """
    Main function to validate attention layers for overflow issues.
    
    Args:
        checkpoint_file: Path to the checkpoint file
        output_dir: Directory to save the output file
        output_filename: Name of the output JSON file
        threshold: Threshold for considering values as potential overflow
    """
    resultarray = []
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Full path for output file
        output_path = os.path.join(output_dir, output_filename)
        
        # Load the checkpoint
        state_dict, checkpoint_path = load_checkpoint(checkpoint_file)
        
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Total parameters in checkpoint: {len(state_dict)}")
        
        # Extract attention weights
        attention_weights = extract_attention_weights(state_dict)
        
        if not attention_weights:
            print("No attention layers found in the checkpoint!")
            print("Available layer names:")
            for name in list(state_dict.keys())[:20]:  # Show first 20 layer names
                print(f"  {name}")
            return
        
        print(f"Found {len(attention_weights)} attention-related parameters")
        
        # Analyze each attention weight
        overflow_issues = {}
        total_issues = 0
        
        for param_name, tensor in attention_weights.items():
            print(f"Checking: {param_name} (shape: {tensor.shape})")
            
            overflow_info = check_tensor_overflow(tensor, threshold)
            
            if overflow_info['has_overflow_risk']:
                total_issues += 1
                overflow_issues[param_name] = overflow_info
                
                # Add the actual problematic values for detailed analysis
                if overflow_info['has_inf'] or overflow_info['has_nan']:
                    # For inf/nan, store locations
                    inf_mask = torch.isinf(tensor.float())
                    nan_mask = torch.isnan(tensor.float())
                    
                    if inf_mask.any():
                        inf_indices = torch.where(inf_mask)
                        overflow_issues[param_name]['inf_locations'] = [
                            tuple(idx.item() for idx in indices) 
                            for indices in zip(*inf_indices)
                        ][:100]  # Limit to first 100 locations
                    
                    if nan_mask.any():
                        nan_indices = torch.where(nan_mask)
                        overflow_issues[param_name]['nan_locations'] = [
                            tuple(idx.item() for idx in indices) 
                            for indices in zip(*nan_indices)
                        ][:100]  # Limit to first 100 locations
                
                if overflow_info['has_large_values']:
                    # Store some of the large values
                    abs_tensor = torch.abs(tensor.float())
                    large_mask = abs_tensor > threshold
                    large_indices = torch.where(large_mask)
                    large_values = tensor.float()[large_mask]
                    
                    overflow_issues[param_name]['large_value_samples'] = [
                        {
                            'location': tuple(idx.item() for idx in indices),
                            'value': val.item()
                        }
                        for indices, val in zip(zip(*large_indices), large_values)
                    ][:50]  # Limit to first 50 samples
                
                print(f"  ⚠️  OVERFLOW RISK DETECTED!")
                print(f"      Max absolute value: {overflow_info['max_abs_value']:.2e}")
                print(f"      Has inf: {overflow_info['has_inf']} (count: {overflow_info['inf_count']})")
                print(f"      Has nan: {overflow_info['has_nan']} (count: {overflow_info['nan_count']})")
                print(f"      Has large values: {overflow_info['has_large_values']} (count: {overflow_info['large_values_count']})")
                rsjson = {"layer": param_name,
                          "shape": tensor.shape,
                          "overflow":True,"max abs value":overflow_info['max_abs_value'],
                          "inf locations": overflow_info.get('inf_locations', []),
                          "nan locations": overflow_info.get('nan_locations', []),
                          "large value samples": overflow_info.get('large_value_samples', [])}
                resultarray.append(rsjson)
            else:
                rsjson = {"layer":param_name,"shape":tensor.shape,"overflow":False,"max abs value":overflow_info['max_abs_value']}
                print(f"  ✅ No overflow risk detected (max abs value: {overflow_info['max_abs_value']})")
                resultarray.append(rsjson)
        
        # Prepare summary
        summary = {
            'checkpoint_path': checkpoint_path,
            'total_attention_parameters': len(attention_weights),
            'parameters_with_overflow_risk': total_issues,
            'analysis_threshold': threshold,
            'overflow_details': overflow_issues,
            'results': resultarray
        }
        
        # Save results to JSON
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total attention parameters analyzed: {len(attention_weights)}")
        print(f"Parameters with overflow risk: {total_issues}")
        print(f"Results saved to: {output_path}")
        
        if total_issues > 0:
            print(f"\n⚠️  WARNING: Found {total_issues} attention parameters with potential overflow issues!")
            print("This could be the cause of your NaN errors during training.")
            print("Consider:")
            print("  - Using gradient clipping")
            print("  - Reducing learning rate")
            print("  - Using mixed precision training")
            print("  - Initializing weights with smaller variance")
        else:
            print("\n✅ No overflow issues detected in attention layers.")
            print("The NaN error might be coming from other parts of the model or training process.")
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise

def main():
    """
    Main function that handles command line arguments and runs the validation.
    """
    import argparse
    
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Validate attention layers for overflow issues in PyTorch checkpoints')
    parser.add_argument('checkpoint_file', type=str, 
                       help='Path to the checkpoint file (e.g., checkpoint_epoch_70.pt)')
    parser.add_argument('output_folder', type=str,
                       help='Folder to save the output JSON file')
    parser.add_argument('output_filename', type=str,
                       help='Name of the output JSON file (e.g., overflow_analysis.json)')
    parser.add_argument('--threshold', type=float, default=1e30,
                       help='Threshold for considering values as potential overflow (default: 1e30)')
    
    args = parser.parse_args()
    
    print("Starting attention layer overflow validation...")
    print(f"Checkpoint file: {args.checkpoint_file}")
    print(f"Output folder: {args.output_folder}")
    print(f"Output filename: {args.output_filename}")
    print(f"Overflow threshold: {args.threshold}")
    
    validate_attention_layers(args.checkpoint_file, args.output_folder, args.output_filename, args.threshold)

if __name__ == "__main__":
    main()
