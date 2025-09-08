#!/usr/bin/env python3
"""
Script to modify PyTorch model attention weights based on JSON data.
Handles fused QKV attention layers where query, key, and value weights are concatenated.
"""

import json
import torch
import argparse
import sys
from pathlib import Path

prune = False

def load_json_data(json_file):
    """Load and parse the JSON file containing weight modifications."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in '{json_file}': {e}")
        sys.exit(1)

def load_checkpoint(checkpoint_file):
    """Load the PyTorch checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        return checkpoint
    except FileNotFoundError:
        print(f"Error: Checkpoint file '{checkpoint_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

def get_model_state_dict(checkpoint):
    """Extract the state dict from the checkpoint, handling different checkpoint formats."""
    if isinstance(checkpoint, dict):
        # Common checkpoint formats
        if 'state_dict' in checkpoint:
            return checkpoint['state_dict']
        elif 'model' in checkpoint:
            return checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']
        else:
            # Assume the checkpoint is already a state dict
            return checkpoint
    else:
        # Direct model state dict
        return checkpoint

def modify_attention_weights(state_dict, json_data):
    """
    Modify attention weights based on the JSON data.
    Assumes fused QKV format where weights are concatenated as [query, key, value].
    """
    layer_name = json_data['layer_name']
    
    if layer_name not in state_dict:
        print(f"Warning: Layer '{layer_name}' not found in model state dict.")
        print(f"Available layers: {list(state_dict.keys())}")
        return False
    
    # Get the original weight tensor
    original_weight = state_dict[layer_name]
    print(f"Original weight shape: {original_weight.shape}")
    
    # Clone the weight to avoid modifying the original
    modified_weight = original_weight.clone()
    
    # For fused QKV, we need to determine the split points
    # Assuming equal splits for Q, K, V
    total_dim = original_weight.shape[0]
    if total_dim % 3 != 0:
        print(f"Warning: Weight dimension {total_dim} is not divisible by 3. Assuming custom split.")
        # You might need to adjust this based on your specific model architecture
        qkv_dim = total_dim // 3
    else:
        qkv_dim = total_dim // 3
    
    query_start = 0
    key_start = qkv_dim
    value_start = 2 * qkv_dim
    
    modifications_made = 0
    
    # Modify query values
    if 'query' in json_data['common_values']:
        for query_mod in json_data['common_values']['query']:
            row_idx = query_mod['row_index']
            col_idx = query_mod['col_index']
            new_value = query_mod['value']
            
            # Adjust row index for query section
            actual_row_idx = query_start + row_idx
            
            if actual_row_idx < original_weight.shape[0] and col_idx < original_weight.shape[1]:
                if modified_weight[actual_row_idx, col_idx] != new_value:
                    # Only modify if the value is different
                    if prune:
                        modified_weight[actual_row_idx, col_idx] = 0.0
                    else:
                        modified_weight[actual_row_idx, col_idx] = new_value
                        modifications_made += 1
                        print(f"Modified query weight at ({actual_row_idx}, {col_idx}) = {new_value}")
                else:
                    print(f"-- No change for query weight at ({actual_row_idx}, {col_idx}), already {new_value}")
            else:
                print(f"Warning: Query index ({actual_row_idx}, {col_idx}) out of bounds")
    
    # Modify key values
    if 'key' in json_data['common_values']:
        for key_mod in json_data['common_values']['key']:
            row_idx = key_mod['row_index']
            col_idx = key_mod['col_index']
            new_value = key_mod['value']
            
            # Adjust row index for key section
            actual_row_idx = key_start + row_idx
            
            if actual_row_idx < original_weight.shape[0] and col_idx < original_weight.shape[1]:
                if modified_weight[actual_row_idx, col_idx] != new_value:
                    # Only modify if the value is different
                    if prune:
                        modified_weight[actual_row_idx, col_idx] = 0.0
                    else:
                        modified_weight[actual_row_idx, col_idx] = new_value
                        modifications_made += 1
                        print(f"Modified key weight at ({actual_row_idx}, {col_idx}) = {new_value}")
                else:
                    print(f"-- No change for key weight at ({actual_row_idx}, {col_idx}), already {new_value}")
            else:
                print(f"Warning: Key index ({actual_row_idx}, {col_idx}) out of bounds")
    
    # Update the state dict
    state_dict[layer_name] = modified_weight
    
    print(f"Total modifications made: {modifications_made}")
    return True

def save_checkpoint(checkpoint, output_file, modified_state_dict):
    """Save the modified checkpoint."""
    try:
        # Update the checkpoint with modified state dict
        if isinstance(checkpoint, dict):
            # Preserve original checkpoint structure
            if 'state_dict' in checkpoint:
                checkpoint['state_dict'] = modified_state_dict
            elif 'model' in checkpoint:
                checkpoint['model'] = modified_state_dict
            elif 'model_state_dict' in checkpoint:
                checkpoint['model_state_dict'] = modified_state_dict
            else:
                # Replace entire checkpoint with state dict
                checkpoint = modified_state_dict
        else:
            checkpoint = modified_state_dict
        
        torch.save(checkpoint, output_file)
        print(f"Modified model saved to: {output_file}")
        
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Modify PyTorch model attention weights based on JSON data"
    )
    parser.add_argument(
        "checkpoint", 
        help="Path to the input PyTorch checkpoint file (.pth)"
    )
    parser.add_argument(
        "json_file", 
        help="Path to the JSON file containing weight modifications"
    )
    parser.add_argument(
        "output", 
        help="Path for the output modified model file"
    )
    parser.add_argument(
        "p",
        help="Prune the query and keys",
        default="False"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    print(f"- prune setting {args.p} -")
    
    if args.p.lower() in ['true']:
        prune = True

    # Validate input files exist
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file '{args.checkpoint}' does not exist.")
        sys.exit(1)
    
    if not Path(args.json_file).exists():
        print(f"Error: JSON file '{args.json_file}' does not exist.")
        sys.exit(1)
    
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)
    
    print(f"Loading JSON data: {args.json_file}")
    json_data = load_json_data(args.json_file)
    
    # Extract state dict
    state_dict = get_model_state_dict(checkpoint)
    
    if args.verbose:
        print(f"JSON data layer: {json_data['layer_name']}")
        print(f"Total query modifications: {len(json_data['common_values'].get('query', []))}")
        print(f"Total key modifications: {len(json_data['common_values'].get('key', []))}")
    
    print("Modifying attention weights...")
    success = modify_attention_weights(state_dict, json_data)
    
    if success:
        print(f"Saving modified model: {args.output}")
        save_checkpoint(checkpoint, args.output, state_dict)
        print("Process completed successfully!")
    else:
        print("Failed to modify weights. Please check the layer name and JSON format.")
        sys.exit(1)

if __name__ == "__main__":
    main()