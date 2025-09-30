#!/usr/bin/env python3
"""
Script to compare two SigLIP checkpoint files in safetensors format.
Calculates weight changes between checkpoints and saves results to JSON.
"""
import time
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
from safetensors import safe_open
import torch

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
        r'(.*)layer\.(\d+)(.*)',
        r'(.*)layers\.(\d+)(.*)',
        r'(.*)transformer\.h\.(\d+)(.*)',
        r'(.*)block\.(\d+)(.*)',
        r'(.*)encoder\.layer\.(\d+)(.*)',
        r'(.*)decoder\.layer\.(\d+)(.*)'
    ]
    
    for pattern in patterns:
        key = key.replace("_", ".")  # Normalize keys with underscores
        match = re.match(pattern, key, re.IGNORECASE)
        if match:
            prefix, layer_num, suffix = match.groups()
            return (prefix, int(layer_num), suffix)
    
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

def loadTorchToDict(filename):
    modeldata = torch.load(filename, weights_only=False, map_location='cpu')
    return modeldata

def is_attention_layer(layer_name: str) -> bool:
    """
    Determine if a layer is an attention layer based on its name.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        True if it's an attention layer, False otherwise
    """
    attention_keywords = [
        'attention', 'attn', 'self_attn', 'cross_attn',
        'q_proj', 'k_proj', 'v_proj', 'out_proj',
        'query', 'key', 'value', 'qkv'
    ]
    layer_name_lower = layer_name.lower()
    return any(keyword in layer_name_lower for keyword in attention_keywords)

def is_fused_attention(layer_name: str) -> bool:
    """
    Check if the layer is using fused attention weights.
    
    Args:
        layer_name: Name of the layer
        
    Returns:
        True if it uses fused attention weights, False otherwise
    """
    return 'qkv' in layer_name.lower()

def is_bias_layer(layer_name: str) -> bool:
    """
    Check if the layer name has bias

    Args:
        layer_name (str): _description_

    Returns:
        bool: _description_
    """
    return 'bias' in layer_name.lower()

def unfuse_qkv_weights(fused_weight: torch.Tensor, fused_bias: Optional[torch.Tensor] = None,
    embed_dim: Optional[int] = None, num_heads: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Unfuse the fused QKV weights from a transformer attention layer.
    
    Args:
        fused_weight: Fused weight tensor of shape (3 * embed_dim, embed_dim)
                     where the first dimension contains Q, K, V weights concatenated
        fused_bias: Optional fused bias tensor of shape (3 * embed_dim,)
        embed_dim: Embedding dimension. If None, inferred from weight shape
        num_heads: Number of attention heads (for validation)
        
    Returns:
        Dictionary containing:
        - 'query_weight': Query projection weights
        - 'key_weight': Key projection weights  
        - 'value_weight': Value projection weights
        - 'query_bias': Query bias (if fused_bias provided)
        - 'key_bias': Key bias (if fused_bias provided)
        - 'value_bias': Value bias (if fused_bias provided)
    """
    
    # Validate input shapes
    if len(fused_weight.shape) != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape {fused_weight.shape}")
    
    # Infer embed_dim if not provided
    if embed_dim is None:
        embed_dim = fused_weight.shape[1]
    
    # Validate that the first dimension is 3 * embed_dim
    expected_fused_dim = 3 * embed_dim
    if fused_weight.shape[0] != expected_fused_dim:
        raise ValueError(
            f"Expected fused weight first dimension to be {expected_fused_dim} "
            f"(3 * embed_dim={embed_dim}), got {fused_weight.shape[0]}"
        )
    
    # Validate bias if provided
    if fused_bias is not None:
        if len(fused_bias.shape) != 1:
            raise ValueError(f"Expected 1D bias tensor, got shape {fused_bias.shape}")
        if fused_bias.shape[0] != expected_fused_dim:
            raise ValueError(
                f"Expected fused bias dimension to be {expected_fused_dim}, "
                f"got {fused_bias.shape[0]}"
            )
    
    # Validate num_heads if provided
    if num_heads is not None and embed_dim % num_heads != 0:
        raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
    
    # Split the fused weights
    query_weight = fused_weight[0:embed_dim, :]
    key_weight = fused_weight[embed_dim:2*embed_dim, :]
    value_weight = fused_weight[2*embed_dim:3*embed_dim, :]
    
    result = {
        'query_weight': query_weight,
        'key_weight': key_weight,
        'value_weight': value_weight
    }
    
    # Split the fused bias if provided
    if fused_bias is not None:
        query_bias = fused_bias[0:embed_dim]
        key_bias = fused_bias[embed_dim:2*embed_dim]
        value_bias = fused_bias[2*embed_dim:3*embed_dim]
        
        result.update({
            'query_bias': query_bias,
            'key_bias': key_bias,
            'value_bias': value_bias
        })
    
    return result
    
def extract_weight(checkpoint1: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Calculate weight changes between two checkpoints.
    
    Args:
        checkpoint1: First checkpoint tensors
        checkpoint2: Second checkpoint tensors
        
    Returns:
        Dictionary containing individual weight values and layer change counts
    """
    attenweights = {}
    
    for key in checkpoint1.keys():
        tensor1 = checkpoint1[key]
        
        # Check if this is an attention layer
        is_attention = is_attention_layer(key)
        is_fused = is_fused_attention(key)
        is_bias = is_bias_layer(key)
        
        # Create structured weight data
        if is_attention and is_fused and not is_bias:
            #weights_data = create_weight_structure(tensor1, is_attention, is_fused, is_bias)
            weights_data = unfuse_qkv_weights(tensor1)
            layer_data = {
                "shape": list(tensor1.shape),
                "dtype": str(tensor1.dtype),
                "layer_type": "attention",
                "weights": {
                    "query": weights_data['query_weight'].detach().cpu().numpy().tolist(),
                    "key": weights_data['key_weight'].detach().cpu().numpy().tolist(),
                    "value": weights_data['value_weight'].detach().cpu().numpy().tolist()
                }
            }
            attenweights[key] = layer_data
    
    return attenweights


def run(checkpoint1_path: str, outputdir:str, outputfile: str) -> None:
    """
    Main function to compare two checkpoints and save results.
    
    Args:
        checkpoint1_path: Path to first checkpoint file
        outputdir: Directory to save the results
        outputfile: Name of the model for output file naming
    """
    print(f"Loading checkpoint: {checkpoint1_path}")
    checkpoint1 = loadTorchToDict(checkpoint1_path)
    print(f"Loaded {len(checkpoint1)} tensors from checkpoint 1")
    
    print("Extracting weights...")
    calstart = time.time()
    layers = extract_weight(checkpoint1)
    calend = time.time()
    print(f"Extracted in {calend - calstart:.2f} seconds, {(calend - calstart)/60:.2f} min")
    
    # Prepare output data
    output_data = {
        "model_name": outputfile,
        "checkpoint1_path": checkpoint1_path,
        "layers": layers
    }
    
    # Save to JSON file
    output_filename = f"{outputdir}/{outputfile}_attenx.json"
    print(f"Saving results to: {output_filename}")
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Extract complete! Results saved to {output_filename}")    

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Compare two SigLIP checkpoint files and analyze weight changes"
    )
    parser.add_argument(
        "checkpoint1", 
        type=str, 
        help="Path to the first checkpoint file (.pt)"
    )
    parser.add_argument(
        "outputdir", 
        type=str, 
        help="directory to save the results)"
    )
    parser.add_argument(
        "outputfile", 
        type=str, 
        help="Output file name to save the results)"
    )
    
    args = parser.parse_args()
    
    try:
        start = time.time()
        run(args.checkpoint1, args.outputdir, args.outputfile)
        end = time.time()
        print(f"Total time taken: {end - start:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
