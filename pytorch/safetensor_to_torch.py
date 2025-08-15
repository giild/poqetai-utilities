#!/usr/bin/env python3
"""
Script to convert Hugging Face SafeTensor checkpoint models to PyTorch .pth format.
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
    from safetensors import safe_open
except ImportError as e:
    print(f"Error: Required package not found - {e}")
    print("Please install required packages:")
    print("pip install torch safetensors")
    sys.exit(1)


def convert_safetensor_to_pth(input_file, output_file=None):
    """
    Convert a SafeTensor file to PyTorch .pth format.
    
    Args:
        input_file (str): Path to the input .safetensors file
        output_file (str, optional): Path to the output .pth file. 
                                   If None, uses input filename with .pth extension
    
    Returns:
        str: Path to the created .pth file
    """
    input_path = Path(input_file)
    
    # Validate input file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not input_path.suffix.lower() == '.safetensors':
        print(f"Warning: Input file doesn't have .safetensors extension: {input_file}")
    
    # Generate output filename if not provided
    if output_file is None:
        output_file = input_path.with_suffix('.pth')
    else:
        output_file = Path(output_file)
    
    print(f"Converting {input_file} to {output_file}")
    
    # Load the safetensor file
    state_dict = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        
        print(f"Loaded {len(state_dict)} tensors from safetensor file")
        
        # Save as PyTorch .pth file
        torch.save(state_dict, output_file)
        print(f"Successfully saved to {output_file}")
        
        # Print some info about the converted model
        total_params = sum(tensor.numel() for tensor in state_dict.values())
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"Model info: {total_params:,} total parameters, {file_size_mb:.1f} MB")
        
        return str(output_file)
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        # Clean up partial file if it exists
        if output_file.exists():
            output_file.unlink()
        raise


def main():
    """Main function to handle command line arguments and perform conversion."""
    parser = argparse.ArgumentParser(
        description="Convert Hugging Face SafeTensor checkpoint models to PyTorch .pth format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_safetensor.py model.safetensors
  python convert_safetensor.py model.safetensors -o converted_model.pth
  python convert_safetensor.py /path/to/model.safetensors -o /path/to/output.pth
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Path to the input .safetensors file"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Path to the output .pth file (default: same as input with .pth extension)"
    )
    
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists"
    )
    
    args = parser.parse_args()
    
    # Check if output file exists and handle overwrite
    output_path = Path(args.output) if args.output else Path(args.input_file).with_suffix('.pth')
    
    if output_path.exists() and not args.overwrite:
        response = input(f"Output file {output_path} already exists. Overwrite? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("Conversion cancelled.")
            return
    
    try:
        output_file = convert_safetensor_to_pth(args.input_file, args.output)
        print(f"\nConversion completed successfully!")
        print(f"Output saved to: {output_file}")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
