import json
import os
import sys
import re
from pathlib import Path
import numpy as np
import argparse
import torch
from collections import defaultdict, Counter
from collections import OrderedDict

def parse_block_number(layer_name):
    """
    Parse block number from layer name.
    
    Examples:
        'blocks.0.attn.proj.weight' -> 0
        'blocks.5.attn.qkv.weight' -> 5
        'vision_model.encoder.layers.0.self_attn.q_proj.weight' -> 0
        'encoder.layer.12.attention.self.query.weight' -> 12
    
    Args:
        layer_name: String name of the layer
    
    Returns:
        Block number (int) or None if no block number found
    """
    # Common patterns for block numbers in layer names
    patterns = [
        r'^blocks\.(\d+)\.',       # blocks.X. at start
        r'\.blocks\.(\d+)\.',      # .blocks.X.
        r'^layers\.(\d+)\.',       # layers.X. at start
        r'\.layers\.(\d+)\.',      # .layers.X.
        r'^layer\.(\d+)\.',        # layer.X. at start
        r'\.layer\.(\d+)\.',       # .layer.X.
        r'^block\.(\d+)\.',        # block.X. at start
        r'\.block\.(\d+)\.',       # .block.X.
        r'\.h\.(\d+)\.',           # .h.X. (GPT-style)
    ]
    
    for pattern in patterns:
        match = re.search(pattern, layer_name)
        if match:
            return int(match.group(1))
    
    return None


def get_block_info(state_dict):
    """
    Analyze state dict to find all blocks and their layer names.
    
    Args:
        state_dict: Model state dictionary
    
    Returns:
        Dictionary mapping block numbers to lists of layer names
    """
    blocks = {}
    other_layers = []
    
    for name in state_dict.keys():
        block_num = parse_block_number(name)
        
        if block_num is not None:
            if block_num not in blocks:
                blocks[block_num] = []
            blocks[block_num].append(name)
        else:
            other_layers.append(name)
    
    return blocks, other_layers


def prune_blocks(state_dict, blocks_to_remove):
    """
    Remove specified blocks from the state dictionary.
    Keeps original block numbers in layer names (no renumbering).
    
    Args:
        state_dict: Model state dictionary
        blocks_to_remove: List of block indices to remove
    
    Returns:
        New state dict with specified blocks removed
    """
    blocks, other_layers = get_block_info(state_dict)
    
    print(f"Found {len(blocks)} blocks in the model")
    print(f"Block indices: {sorted(blocks.keys())}")
    print(f"Blocks to remove: {sorted(blocks_to_remove)}")
    
    # Validate block indices
    for block_num in blocks_to_remove:
        if block_num not in blocks:
            raise ValueError(f"Block {block_num} not found in model. Available blocks: {sorted(blocks.keys())}")
    
    # Determine which blocks to keep
    all_blocks = sorted(blocks.keys())
    blocks_to_keep = [b for b in all_blocks if b not in blocks_to_remove]
    
    print(f"Keeping blocks: {blocks_to_keep}")
    print(f"Removing {len(blocks_to_remove)} blocks, keeping {len(blocks_to_keep)} blocks")
    
    # Create new state dict without renumbering
    new_state_dict = OrderedDict()
    
    # First, add all non-block layers
    for name in other_layers:
        new_state_dict[name] = state_dict[name]
    
    # Add kept blocks with original indices preserved
    for block_idx in blocks_to_keep:
        for layer_name in blocks[block_idx]:
            # Keep the original layer name unchanged
            new_state_dict[layer_name] = state_dict[layer_name]
    
    return new_state_dict


def load_model(filename:str):
    """Load SigLIP model from path or HuggingFace hub."""
    modeldata = torch.load(filename, weights_only=False, map_location='cpu')
    return modeldata


def save_model(state_dict, original_checkpoint, output_path):
    """Save the pruned model."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare checkpoint to save
    if isinstance(original_checkpoint, dict) and 'state_dict' in original_checkpoint:
        # Preserve original checkpoint structure
        new_checkpoint = original_checkpoint.copy()
        new_checkpoint['state_dict'] = state_dict
    elif isinstance(original_checkpoint, dict) and 'model' in original_checkpoint:
        new_checkpoint = original_checkpoint.copy()
        new_checkpoint['model'] = state_dict
    else:
        # Just save state dict
        new_checkpoint = state_dict
    
    save_path = output_path / "pruned_model.pt"
    print(f"Saving pruned model to: {save_path}")
    torch.save(new_checkpoint, save_path)
    
    print("Model saved successfully!")


def main():
    parser = argparse.ArgumentParser(description="Prune blocks from SigLIP-2 Vision Transformer")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or HuggingFace model name (e.g., 'google/siglip-so400m-patch14-384')"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the pruned model"
    )
    parser.add_argument(
        "--bp",
        type=str,
        required=True,
        help="Comma-separated list of block indices to keep (e.g., '0,2,4,6,8,10')"
    )
    
    args = parser.parse_args()
    
    # Parse blocks to keep
    block_to_prune = [int(x.strip()) for x in args.bp.split(",")]
    block_to_prune = sorted(block_to_prune)  # Ensure sorted order
    
    # Load model
    model = load_model(args.model_path)
    
    # Prune blocks
    pruned_model = prune_blocks(model, block_to_prune)
    
    # Save the pruned model
    save_model(pruned_model, model, args.output_path)
    
    print("\nPruning completed successfully!")


if __name__ == "__main__":
    
    main()
