import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary

def load_torch_cpu(filename:str):
    """Load a PyTorch model file to CPU."""
    model = torch.load(filename, map_location=torch.device('cpu'), weights_only=False)
    return model

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