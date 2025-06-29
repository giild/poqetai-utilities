#!/usr/bin/env python3
"""
Script to convert PyTorch .pth files to Hugging Face SafeTensors format.
"""

import os
import sys
import torch
from safetensors.torch import save_file
from pathlib import Path


def convert_pth_to_safetensors(pth_filename):
    """
    Convert a PyTorch .pth file to SafeTensors format.
    
    Args:
        pth_filename (str): Path to the .pth file to convert
        
    Returns:
        str: Path to the generated .safetensors file
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        Exception: If conversion fails
    """
    # Validate input file
    if not os.path.exists(pth_filename):
        raise FileNotFoundError(f"Input file not found: {pth_filename}")
    
    if not pth_filename.endswith('.pth'):
        raise ValueError(f"Input file must have .pth extension: {pth_filename}")
    
    # Generate output filename
    output_filename = pth_filename.replace('.pth', '.safetensors')
    
    try:
        print(f"Loading PyTorch model from: {pth_filename}")
        
        # Load the PyTorch model
        # Use map_location='cpu' to ensure compatibility across devices
        state_dict = torch.load(pth_filename, map_location='cpu')
        
        # Handle different PyTorch save formats
        if isinstance(state_dict, dict):
            # Check if it's a full checkpoint with 'state_dict' key
            if 'state_dict' in state_dict:
                tensors = state_dict['state_dict']
                print("Found 'state_dict' key in checkpoint")
            elif 'model' in state_dict:
                tensors = state_dict['model']
                print("Found 'model' key in checkpoint")
            else:
                # Assume the dict itself contains the tensors
                tensors = state_dict
                print("Using entire dict as tensor collection")
        else:
            raise ValueError("Unsupported PyTorch file format")
        
        # Validate that all values are tensors
        invalid_keys = []
        for key, value in tensors.items():
            if not torch.is_tensor(value):
                invalid_keys.append(key)
        
        if invalid_keys:
            print(f"Warning: Removing non-tensor keys: {invalid_keys}")
            tensors = {k: v for k, v in tensors.items() if torch.is_tensor(v)}
        
        if not tensors:
            raise ValueError("No valid tensors found in the input file")
        
        print(f"Converting {len(tensors)} tensors to SafeTensors format")
        
        # Save as SafeTensors
        save_file(tensors, output_filename)
        
        print(f"Successfully converted to: {output_filename}")
        
        # Print file size comparison
        input_size = os.path.getsize(pth_filename) / (1024 * 1024)  # MB
        output_size = os.path.getsize(output_filename) / (1024 * 1024)  # MB
        print(f"File sizes - Input: {input_size:.2f} MB, Output: {output_size:.2f} MB")
        
        return output_filename
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        # Clean up partial output file if it exists
        if os.path.exists(output_filename):
            os.remove(output_filename)
        raise


def main():
    """Main function to handle command line arguments and run conversion."""
    if len(sys.argv) != 2:
        print("Usage: python convert_pth_to_safetensors.py <input_file.pth>")
        print("Example: python convert_pth_to_safetensors.py model.pth")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    try:
        output_file = convert_pth_to_safetensors(input_file)
        print(f"\n✅ Conversion completed successfully!")
        print(f"Output file: {output_file}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()