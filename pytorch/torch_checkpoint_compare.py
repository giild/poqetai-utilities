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
from typing import Dict, Any, Tuple

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


def create_weight_structure(tensor1: torch.Tensor, tensor2: torch.Tensor, is_attention: bool) -> list:
    """
    Create weight structure based on tensor dimensions and layer type.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        is_attention: Whether this is an attention layer
        
    Returns:
        List of weight comparison data maintaining tensor structure
    """
    diff = tensor2 - tensor1
    
    if len(tensor1.shape) == 1:  # 1D tensor (bias)
        return [
            {
                "index": [i],
                "left_value": float(tensor1[i].item()),
                "right_value": float(tensor2[i].item()),
                "delta": float(diff[i].item())
            }
            for i in range(tensor1.shape[0])
        ]
    
    elif len(tensor1.shape) == 2:  # 2D tensor (linear layer weights)
        if is_attention:
            # For attention layers, preserve the 2D structure more explicitly
            return [
                {
                    "index": [i, j],
                    "position": f"head_{i//64}_dim_{j}" if tensor1.shape[0] % 64 == 0 else f"row_{i}_col_{j}",
                    "left_value": float(tensor1[i, j].item()),
                    "right_value": float(tensor2[i, j].item()),
                    "delta": float(diff[i, j].item())
                }
                for i in range(tensor1.shape[0])
                for j in range(tensor1.shape[1])
            ]
        else:
            # Regular 2D tensor
            return [
                {
                    "index": [i, j],
                    "left_value": float(tensor1[i, j].item()),
                    "right_value": float(tensor2[i, j].item()),
                    "delta": float(diff[i, j].item())
                }
                for i in range(tensor1.shape[0])
                for j in range(tensor1.shape[1])
            ]
    
    elif len(tensor1.shape) == 3:  # 3D tensor
        return [
            {
                "index": [i, j, k],
                "left_value": float(tensor1[i, j, k].item()),
                "right_value": float(tensor2[i, j, k].item()),
                "delta": float(diff[i, j, k].item())
            }
            for i in range(tensor1.shape[0])
            for j in range(tensor1.shape[1])
            for k in range(tensor1.shape[2])
        ]
    
    elif len(tensor1.shape) == 4:  # 4D tensor (conv layers)
        return [
            {
                "index": [i, j, k, l],
                "left_value": float(tensor1[i, j, k, l].item()),
                "right_value": float(tensor2[i, j, k, l].item()),
                "delta": float(diff[i, j, k, l].item())
            }
            for i in range(tensor1.shape[0])
            for j in range(tensor1.shape[1])
            for k in range(tensor1.shape[2])
            for l in range(tensor1.shape[3])
        ]
    
    else:  # Higher dimensional tensors - flatten with index tracking
        indices = torch.nonzero(torch.ones_like(tensor1), as_tuple=False)
        return [
            {
                "index": idx.tolist(),
                "left_value": float(tensor1[tuple(idx)].item()),
                "right_value": float(tensor2[tuple(idx)].item()),
                "delta": float(diff[tuple(idx)].item())
            }
            for idx in indices
        ]


def calculate_weight_changes(checkpoint1: Dict[str, torch.Tensor], 
                           checkpoint2: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Calculate weight changes between two checkpoints.
    
    Args:
        checkpoint1: First checkpoint tensors
        checkpoint2: Second checkpoint tensors
        
    Returns:
        Dictionary containing individual weight values and layer change counts
    """
    changes = {}
    
    # Get common keys between both checkpoints
    common_keys = set(checkpoint1.keys()) & set(checkpoint2.keys())
    keys_only_in_cp1 = set(checkpoint1.keys()) - set(checkpoint2.keys())
    keys_only_in_cp2 = set(checkpoint2.keys()) - set(checkpoint1.keys())
    
    # Log keys that are not in both checkpoints
    if keys_only_in_cp1:
        print(f"Keys only in checkpoint 1: {list(keys_only_in_cp1)}")
    if keys_only_in_cp2:
        print(f"Keys only in checkpoint 2: {list(keys_only_in_cp2)}")
    
    for key in common_keys:
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
        
        # Create structured weight data
        weights_data = create_weight_structure(tensor1, tensor2, is_attention)
        
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
            "weights": weights_data
        }
        
        changes[key] = layer_data
    
    return changes


def calculate_summary_stats(changes: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate summary statistics across all layers.
    
    Args:
        changes: Dictionary of per-layer changes
        
    Returns:
        Dictionary containing summary statistics
    """
    if not changes:
        return {}
    
    total_weights = sum(layer["total_weights"] for layer in changes.values())
    total_changed = sum(layer["weights_changed"] for layer in changes.values())
    total_unchanged = sum(layer["weights_unchanged"] for layer in changes.values())
    
    summary = {
        "total_layers": len(changes),
        "total_weights": total_weights,
        "total_weights_changed": total_changed,
        "total_weights_unchanged": total_unchanged,
        "percent_weights_changed": float((total_changed / total_weights) * 100) if total_weights > 0 else 0.0,
        "layers_with_changes": sum(1 for layer in changes.values() if layer["weights_changed"] > 0)
    }
    
    return summary


def run(checkpoint1_path: str, checkpoint2_path: str, model_name: str) -> None:
    """
    Main function to compare two checkpoints and save results.
    
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
    
    print("Calculating weight changes...")
    calstart = time.time()
    changes = calculate_weight_changes(checkpoint1, checkpoint2)
    calend = time.time()
    print(f"Weight changes calculated in {calend - calstart:.2f} seconds, {(calend - calstart)/60:.2f} min")
    
    print("Calculating summary statistics...")
    summary = calculate_summary_stats(changes)
    
    # Prepare output data
    output_data = {
        "model_name": model_name,
        "checkpoint1_path": checkpoint1_path,
        "checkpoint2_path": checkpoint2_path,
        "summary": summary,
        "layer_changes": changes
    }
    
    # Save to JSON file
    output_filename = f"{model_name}_checkpoint_comparison.json"
    print(f"Saving results to: {output_filename}")
    
    with open(output_filename, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Comparison complete! Results saved to {output_filename}")
    
    # Print summary
    if summary:
        print("\n--- Summary ---")
        print(f"Total layers: {summary['total_layers']}")
        print(f"Total weights: {summary['total_weights']:,}")
        print(f"Weights changed: {summary['total_weights_changed']:,} ({summary['percent_weights_changed']:.2f}%)")
        print(f"Weights unchanged: {summary['total_weights_unchanged']:,}")
        print(f"Layers with changes: {summary['layers_with_changes']}")


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
        help="Path to the first checkpoint file (.safetensors)"
    )
    parser.add_argument(
        "checkpoint2", 
        type=str, 
        help="Path to the second checkpoint file (.safetensors)"
    )
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Name of the model (used for output file naming)"
    )
    
    args = parser.parse_args()
    
    try:
        start = time.time()
        run(args.checkpoint1, args.checkpoint2, args.model_name)
        end = time.time()
        print(f"Total time taken: {end - start:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
