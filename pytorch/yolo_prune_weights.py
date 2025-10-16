import torch
import json
import argparse
from pathlib import Path


def load_json_data(json_path):
    """Load the JSON file containing weight change information."""
    with open(json_path, 'r') as f:
        return json.load(f)

def loadTorchToDict(filename):
    modeldata = torch.load(filename, weights_only=False, map_location='cpu')
    modeldata = modeldata['ema']
    modeldata = modeldata.state_dict()
    return modeldata

def modify_weights(checkpoint_path, json_data, output_folder, output_filename):
    """
    Load a YOLOv12 checkpoint and modify weights where delta is zero.
    
    Args:
        checkpoint_path: Path to the PyTorch checkpoint file
        json_data: Dictionary containing weight change information
        output_folder: Folder to save the modified checkpoint
        output_filename: Name of the output checkpoint file
    """
    # Load the checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = loadTorchToDict(checkpoint_path)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_state = checkpoint['model']
        if hasattr(model_state, 'state_dict'):
            state_dict = model_state.state_dict()
        else:
            state_dict = model_state
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Track statistics
    total_weights_modified = 0
    layers_modified = 0
    
    # Process each layer in the JSON
    layer_changes = json_data.get('layer_changes', {})
    
    for layer_name, layer_info in layer_changes.items():
        if layer_name not in state_dict:
            print(f"Warning: Layer '{layer_name}' not found in checkpoint")
            continue
        
        weights_list = layer_info.get('weights', [])
        if not weights_list:
            continue
        
        # Get the tensor for this layer
        tensor = state_dict[layer_name]
        modified_count = 0
        
        # Process each weight change
        for weight_info in weights_list:
            delta = weight_info.get('delta', None)
            
            # Only modify if delta is zero
            if delta == 0:
                index = weight_info['index']
                
                # Convert index list to tuple for indexing
                idx_tuple = tuple(index)
                
                # Modify the weight
                tensor[idx_tuple] = 0.0
                modified_count += 1
        
        if modified_count > 0:
            layers_modified += 1
            total_weights_modified += modified_count
            print(f"Modified {modified_count} weights in layer: {layer_name}")
    
    # Save the modified checkpoint
    output_path = Path(output_folder) / output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in the same format as loaded
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        checkpoint['model'] = state_dict if isinstance(checkpoint['model'], dict) else checkpoint['model']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint['state_dict'] = state_dict
    else:
        checkpoint = state_dict
    
    torch.save(checkpoint, output_path)
    
    print(f"\n=== Modification Summary ===")
    print(f"Total weights modified: {total_weights_modified}")
    print(f"Layers modified: {layers_modified}")
    print(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Modify YOLOv12 model weights where delta is zero based on JSON data'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the input checkpoint file (.pt)'
    )
    parser.add_argument(
        '--json',
        type=str,
        required=True,
        help='Path to the JSON file containing weight change data'
    )
    parser.add_argument(
        '--output-folder',
        type=str,
        required=True,
        help='Folder to save the modified checkpoint'
    )
    parser.add_argument(
        '--output-filename',
        type=str,
        required=True,
        help='Name of the output checkpoint file'
    )
    
    args = parser.parse_args()
    
    # Load JSON data
    print("Loading JSON data...")
    json_data = load_json_data(args.json)
    
    # Modify weights
    modify_weights(
        args.checkpoint,
        json_data,
        args.output_folder,
        args.output_filename
    )


if __name__ == '__main__':
    main()
