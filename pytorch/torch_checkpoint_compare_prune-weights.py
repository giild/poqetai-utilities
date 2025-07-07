#!/usr/bin/env python3
"""
Script to compare two SigLIP checkpoint files and identify unchanged weights.
Creates a list of all weights that didn't change for each layer.
"""
import time
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, List

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
    modeldata = torch.load(filename, weights_only=False)
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


def get_unchanged_weights(tensor1: torch.Tensor, tensor2: torch.Tensor, is_attention: bool) -> List[Dict[str, Any]]:
    """
    Get list of unchanged weights with their indices and values.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        is_attention: Whether this is an attention layer
        
    Returns:
        List of unchanged weight data maintaining tensor structure
    """
    diff = tensor2 - tensor1
    unchanged_mask = (diff == 0)
    
    if len(tensor1.shape) == 1:  # 1D tensor (bias)
        unchanged_indices = torch.nonzero(unchanged_mask, as_tuple=False).squeeze(-1)
        return [
            {
                "index": [int(i.item())],
                "value": float(tensor1[i].item())
            }
            for i in unchanged_indices
        ]
    
    elif len(tensor1.shape) == 2:  # 2D tensor (linear layer weights)
        unchanged_indices = torch.nonzero(unchanged_mask, as_tuple=False)
        if is_attention:
            # For attention layers, add position information
            return [
                {
                    "index": [int(idx[0].item()), int(idx[1].item())],
                    "position": f"head_{idx[0].item()//64}_dim_{idx[1].item()}" if tensor1.shape[0] % 64 == 0 else f"row_{idx[0].item()}_col_{idx[1].item()}",
                    "value": float(tensor1[idx[0], idx[1]].item())
                }
                for idx in unchanged_indices
            ]
        else:
            return [
                {
                    "index": [int(idx[0].item()), int(idx[1].item())],
                    "value": float(tensor1[idx[0], idx[1]].item())
                }
                for idx in unchanged_indices
            ]
    
    elif len(tensor1.shape) == 3:  # 3D tensor
        unchanged_indices = torch.nonzero(unchanged_mask, as_tuple=False)
        return [
            {
                "index": [int(idx[0].item()), int(idx[1].item()), int(idx[2].item())],
                "value": float(tensor1[idx[0], idx[1], idx[2]].item())
            }
            for idx in unchanged_indices
        ]
    
    elif len(tensor1.shape) == 4:  # 4D tensor (conv layers)
        unchanged_indices = torch.nonzero(unchanged_mask, as_tuple=False)
        return [
            {
                "index": [int(idx[0].item()), int(idx[1].item()), int(idx[2].item()), int(idx[3].item())],
                "value": float(tensor1[idx[0], idx[1], idx[2], idx[3]].item())
            }
            for idx in unchanged_indices
        ]
    
    else:  # Higher dimensional tensors
        unchanged_indices = torch.nonzero(unchanged_mask, as_tuple=False)
        return [
            {
                "index": idx.tolist(),
                "value": float(tensor1[tuple(idx)].item())
            }
            for idx in unchanged_indices
        ]


def find_unchanged_weights(checkpoint1: Dict[str, torch.Tensor], 
                          checkpoint2: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Find unchanged weights between two checkpoints.
    
    Args:
        checkpoint1: First checkpoint tensors
        checkpoint2: Second checkpoint tensors
        
    Returns:
        Dictionary containing unchanged weights for each layer
    """
    unchanged_data = {}
    
    # Get common keys between both checkpoints
    common_keys = set(checkpoint1.keys()) & set(checkpoint2.keys())
    keys_only_in_cp1 = set(checkpoint1.keys()) - set(checkpoint2.keys())
    keys_only_in_cp2 = set(checkpoint2.keys()) - set(checkpoint1.keys())
    
    # Log keys that are not in both checkpoints
    if keys_only_in_cp1:
        print(f"Keys only in checkpoint 1: {list(keys_only_in_cp1)}")
    if keys_only_in_cp2:
        print(f"Keys only in checkpoint 2: {list(keys_only_in_cp2)}")
    
    for key in sorted(common_keys):  # Process in sorted order
        tensor1 = checkpoint1[key]
        tensor2 = checkpoint2[key]
        
        # Ensure tensors have the same shape
        if tensor1.shape != tensor2.shape:
            print(f"Warning: Shape mismatch for {key}: {tensor1.shape} vs {tensor2.shape}")
            continue
        
        # Check if this is an attention layer
        is_attention = is_attention_layer(key)
        
        # Calculate difference for counting
        diff = tensor2 - tensor1
        
        # Get unchanged weights
        unchanged_weights = get_unchanged_weights(tensor1, tensor2, is_attention)
        
        # Count changes
        num_changed = int(torch.sum(diff != 0).item())
        num_unchanged = int(torch.sum(diff == 0).item())
        
        layer_data = {
            "shape": list(tensor1.shape),
            "dtype": str(tensor1.dtype),
            "layer_type": "attention" if is_attention else "standard",
            "weights_changed": num_changed,
            "weights_unchanged": num_unchanged,
            "total_weights": int(tensor1.numel()),
            "unchanged_weights": unchanged_weights
        }
        
        unchanged_data[key] = layer_data
    
    return unchanged_data


def calculate_summary_stats(unchanged_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary statistics for unchanged weights.
    
    Args:
        unchanged_data: Dictionary of per-layer unchanged weight data
        
    Returns:
        Dictionary containing summary statistics
    """
    if not unchanged_data:
        return {}
    
    total_weights = sum(layer["total_weights"] for layer in unchanged_data.values())
    total_unchanged = sum(layer["weights_unchanged"] for layer in unchanged_data.values())
    total_changed = sum(layer["weights_changed"] for layer in unchanged_data.values())
    
    layers_fully_unchanged = sum(1 for layer in unchanged_data.values() 
                                if layer["weights_unchanged"] == layer["total_weights"])
    layers_partially_unchanged = sum(1 for layer in unchanged_data.values() 
                                   if 0 < layer["weights_unchanged"] < layer["total_weights"])
    layers_fully_changed = sum(1 for layer in unchanged_data.values() 
                             if layer["weights_unchanged"] == 0)
    
    summary = {
        "total_layers": len(unchanged_data),
        "total_weights": total_weights,
        "total_weights_unchanged": total_unchanged,
        "total_weights_changed": total_changed,
        "percent_weights_unchanged": float((total_unchanged / total_weights) * 100) if total_weights > 0 else 0.0,
        "layers_fully_unchanged": layers_fully_unchanged,
        "layers_partially_unchanged": layers_partially_unchanged,
        "layers_fully_changed": layers_fully_changed
    }
    
    return summary


def run(checkpoint1_path: str, checkpoint2_path: str, model_name: str, outputdir:str) -> None:
    """
    Main function to compare two checkpoints and identify unchanged weights.
    
    Args:
        checkpoint1_path: Path to first checkpoint file
        checkpoint2_path: Path to second checkpoint file
        model_name: Name of the model for output file naming
    """
    print(f"Loading checkpoint 1: {checkpoint1_path}")
    checkpoint1 = loadTorchToDict(checkpoint1_path)
    print(f"Loaded {len(checkpoint1)} tensors from checkpoint 1")
    
    print(f"Loading checkpoint 2: {checkpoint2_path}")
    checkpoint2 = loadTorchToDict(checkpoint2_path)
    print(f"Loaded {len(checkpoint2)} tensors from checkpoint 2")
    
    print("Finding unchanged weights...")
    calstart = time.time()
    unchanged_data = find_unchanged_weights(checkpoint1, checkpoint2)
    calend = time.time()
    print(f"Weight changes calculated in {calend - calstart:.2f} seconds, {(calend - calstart)/60:.2f} min")
    
    print("Calculating summary statistics...")
    summary = calculate_summary_stats(unchanged_data)
    
    # Prepare output data
    output_data = {
        "model_name": model_name,
        "checkpoint1_path": checkpoint1_path,
        "checkpoint2_path": checkpoint2_path,
        "summary": summary,
        "layer_unchanged_weights": unchanged_data
    }
    
    # Save to JSON file
    output_filename = f"{outputdir}/{model_name}_unchanged_weights_analysis.json"
    print(f"Saving results to: {output_filename}")
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Analysis complete! Results saved to {output_filename}")
    
    # Print summary
    if summary:
        print("\n--- Unchanged Weights Summary ---")
        print(f"Total layers: {summary['total_layers']}")
        print(f"Total weights: {summary['total_weights']:,}")
        print(f"Weights unchanged: {summary['total_weights_unchanged']:,} ({summary['percent_weights_unchanged']:.2f}%)")
        print(f"Weights changed: {summary['total_weights_changed']:,}")
        print(f"Layers fully unchanged: {summary['layers_fully_unchanged']}")
        print(f"Layers partially unchanged: {summary['layers_partially_unchanged']}")
        print(f"Layers fully changed: {summary['layers_fully_changed']}")


def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Analyze unchanged weights between two SigLIP checkpoint files"
    )
    parser.add_argument(
        "checkpoint1", 
        type=str, 
        help="Path to the first checkpoint file (.safetensors)"
    )
    parser.add_argument(
        "checkpoint2", 
        type=str, 
        help="Path to the second checkpoint file (.safetensors)"
    )
    parser.add_argument(
        "outputdir", 
        type=str, 
        help="Directory to save the output JSON file"
    )
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Name of the model (used for output file naming)"
    )
    
    args = parser.parse_args()
    
    try:
        start = time.time()
        run(args.checkpoint1, args.checkpoint2, args.model_name, args.outputdir)
        end = time.time()
        print(f"Total time taken: {end - start:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
