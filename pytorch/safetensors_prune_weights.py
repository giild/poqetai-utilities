import json
import torch
from safetensors.torch import load_file, save_file
import numpy as np
import argparse
from pathlib import Path

def load_json_data(json_path):
    """Load the JSON file containing unchanged weight information."""
    with open(json_path, 'r') as f:
        return json.load(f)

def convert_flat_indices_to_coordinates(flat_indices, shape):
    """Convert flat indices to multi-dimensional coordinates."""
    coordinates = []
    for flat_idx in flat_indices:
        # Convert flat index to multi-dimensional coordinates
        coords = np.unravel_index(flat_idx, shape)
        coordinates.append(coords)
    return coordinates

def zero_unchanged_weights(checkpoint_path, json_data, output_path=None):
    """
    Load checkpoint and set unchanged weights to zero based on JSON data.
    
    Args:
        checkpoint_path: Path to the safetensor checkpoint file
        json_data: Dictionary containing unchanged weight information
        output_path: Path to save modified checkpoint (optional)
    
    Returns:
        Modified state dict
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the checkpoint
    state_dict = load_file(checkpoint_path)
    
    # Get unchanged weights information
    layer_unchanged_weights = json_data.get('layer_unchanged_weights', {})
    
    total_zeroed = 0
    processed_layers = 0
    
    print(f"Processing {len(layer_unchanged_weights)} layers...")
    
    for layer_name, layer_info in layer_unchanged_weights.items():
        if layer_name not in state_dict:
            print(f"Warning: Layer '{layer_name}' not found in checkpoint")
            continue
            
        # Get the weight tensor
        weight_tensor = state_dict[layer_name]
        
        # Verify shape matches
        expected_shape = tuple(layer_info['shape'])
        if weight_tensor.shape != expected_shape:
            print(f"Warning: Shape mismatch for layer '{layer_name}'. "
                  f"Expected {expected_shape}, got {weight_tensor.shape}")
            continue
            
        # Get unchanged weight indices
        unchanged_indices = layer_info.get('unchanged_weights', [])
        
        if not unchanged_indices:
            print(f"No unchanged weights for layer '{layer_name}'")
            continue
            
        # Convert flat indices to coordinates
        coordinates = convert_flat_indices_to_coordinates(unchanged_indices, weight_tensor.shape)
        
        # Set unchanged weights to zero
        weights_zeroed_in_layer = 0
        for coords in coordinates:
            try:
                # Set the weight to zero at the specified coordinates
                weight_tensor[coords] = 0.0
                weights_zeroed_in_layer += 1
            except IndexError as e:
                print(f"Warning: Invalid coordinates {coords} for layer '{layer_name}': {e}")
                continue
        
        total_zeroed += weights_zeroed_in_layer
        processed_layers += 1
        
        print(f"Layer '{layer_name}': Zeroed {weights_zeroed_in_layer} weights")
    
    print(f"\nSummary:")
    print(f"Processed layers: {processed_layers}")
    print(f"Total weights zeroed: {total_zeroed}")
    
    # Save modified checkpoint if output path is provided
    if output_path:
        print(f"Saving modified checkpoint to: {output_path}")
        save_file(state_dict, output_path)
    
    return state_dict

def main():
    parser = argparse.ArgumentParser(description='Zero unchanged weights in safetensor checkpoint')
    parser.add_argument('json_file', type=str, help='Path to JSON file with unchanged weight info')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file (overrides JSON)')
    parser.add_argument('--output', type=str, help='Output path for modified checkpoint')
    parser.add_argument('--use-checkpoint1', action='store_true', 
                       help='Use checkpoint1_path from JSON (default)')
    parser.add_argument('--use-checkpoint2', action='store_true', 
                       help='Use checkpoint2_path from JSON')
    
    args = parser.parse_args()
    
    # Load JSON data
    try:
        json_data = load_json_data(args.json_file)
        print(f"Loaded JSON data for model: {json_data.get('model_name', 'Unknown')}")
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.use_checkpoint2:
        checkpoint_path = json_data.get('checkpoint2_path')
    else:
        checkpoint_path = json_data.get('checkpoint1_path')
    
    if not checkpoint_path:
        print("Error: No checkpoint path specified")
        return
    
    # Check if checkpoint file exists
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    # Generate output path if not provided
    if not args.output:
        checkpoint_path_obj = Path(checkpoint_path)
        output_path = checkpoint_path_obj.parent / f"{checkpoint_path_obj.stem}_zeroed{checkpoint_path_obj.suffix}"
    else:
        output_path = args.output
    
    try:
        # Process the checkpoint
        modified_state_dict = zero_unchanged_weights(checkpoint_path, json_data, output_path)
        print(f"\nSuccessfully processed checkpoint and saved to: {output_path}")
        
    except Exception as e:
        print(f"Error processing checkpoint: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# Example usage as a module:
# from zero_weights_script import zero_unchanged_weights, load_json_data
# 
# json_data = load_json_data('unchanged_weights.json')
# modified_state_dict = zero_unchanged_weights('model.safetensors', json_data, 'model_zeroed.safetensors')