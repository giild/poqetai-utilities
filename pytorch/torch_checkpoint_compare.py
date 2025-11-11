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
import torch_utils

def sort_keys(keys: list) -> list:
    """
    Sort keys with special handling for numbered layers.
    
    Args:
        keys: List of layer keys
        
    Returns:
        Sorted list of keys
    """
    def sort_key(key):
        prefix, layer_num, suffix = torch_utils.extract_layer_number(key)
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
    
def create_weight_structure(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                            is_attention: bool, is_fused:bool, is_bias:bool) -> list:
    """
    Create weight structure based on tensor dimensions and layer type.
    
    Args:
        tensor1: First tensor
        tensor2: Second tensor
        is_attention: Whether this is an attention layer
        
    Returns:
        List of weight comparison data maintaining tensor structure
    """
    if is_attention and is_fused and not is_bias:
        print(f" the attention is fused, we need to unfuse them.")
        unfuseddict1 = unfuse_qkv_weights(tensor1)
        unfuseddict2 = unfuse_qkv_weights(tensor2)
        query1 = unfuseddict1['query_weight']
        key1 = unfuseddict1['key_weight']
        value1 = unfuseddict1['value_weight']
        query2 = unfuseddict2['query_weight']
        key2 = unfuseddict2['key_weight']
        value2 = unfuseddict2['value_weight']
        querydiff = query2 - query1
        keydiff = key2 - key1
        valuediff = value2 - value1
        print(f" unfused the qkv weights.")
        return [
            {
                "index": [i,j],
                "faw":"q",
                "left_value": float(query1[i,j].item()),
                "right_value": float(query2[i,j].item()),
                "delta": float(querydiff[i,j].item())
            }
            for i in range(query1.shape[0])
            for j in range(query1.shape[1])
        ], [
            {
                "index": [i,j],
                "faw":"k",
                "left_value": float(key1[i,j].item()),
                "right_value": float(key2[i,j].item()),
                "delta": float(keydiff[i,j].item())
            }
            for i in range(key1.shape[0])
            for j in range(key1.shape[1])
        ], [
            {
                "index": [i,j],
                "faw":"v",
                "left_value": float(value1[i,j].item()),
                "right_value": float(value2[i,j].item()),
                "delta": float(valuediff[i,j].item())
            }
            for i in range(value1.shape[0])
            for j in range(value1.shape[1])
        ]
    else:
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
    
    for key in checkpoint1.keys():
        tensor1 = checkpoint1[key]
        tensor2 = checkpoint2[key]
        
        # Ensure tensors have the same shape
        if tensor1.shape != tensor2.shape:
            print(f"Warning: Shape mismatch for {key}: {tensor1.shape} vs {tensor2.shape}")
            continue
        
        # Check if this is an attention layer
        is_attention = is_attention_layer(key)
        is_fused = is_fused_attention(key)
        is_bias = is_bias_layer(key)
        
        # Calculate difference for counting
        diff = tensor2 - tensor1
        
        # Create structured weight data
        weights_data = create_weight_structure(tensor1, tensor2, is_attention, is_fused, is_bias)
        
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
        print(f"Layer {key}: {num_changed} weights changed, {num_unchanged} unchanged")
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


def run(checkpoint1_path: str, checkpoint2_path: str, outputdir:str, outputfile: str) -> None:
    """
    Main function to compare two checkpoints and save results.
    
    Args:
        checkpoint1_path: Path to first checkpoint file
        checkpoint2_path: Path to second checkpoint file
        outputfile: Name of the model for output file naming
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
        "model_name": outputfile,
        "checkpoint1_path": checkpoint1_path,
        "checkpoint2_path": checkpoint2_path,
        "summary": summary,
        "layer_changes": changes
    }
    
    # Save to JSON file
    output_filename = f"{outputdir}/{outputfile}_checkpoint_comparison.json"
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
        run(args.checkpoint1, args.checkpoint2, args.outputdir, args.outputfile)
        end = time.time()
        print(f"Total time taken: {end - start:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
