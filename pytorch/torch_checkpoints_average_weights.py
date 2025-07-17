import torch
import argparse
import os
from collections import OrderedDict

def load_checkpoint(checkpoint_path):
    """Load a PyTorch checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load checkpoint (handles both GPU and CPU)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract state_dict (handle different checkpoint formats)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        # Assume the checkpoint is already a state_dict
        state_dict = checkpoint
    
    return state_dict, checkpoint

def average_weights(state_dict1, state_dict2):
    """Average the weights of two state dictionaries."""
    averaged_state_dict = OrderedDict()
    
    # Get all keys from both state dicts
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    # Check if keys match
    if keys1 != keys2:
        print("Warning: State dict keys don't match exactly!")
        print(f"Keys only in model 1: {keys1 - keys2}")
        print(f"Keys only in model 2: {keys2 - keys1}")
        # Use intersection of keys
        common_keys = keys1 & keys2
    else:
        common_keys = keys1
    
    print(f"Averaging {len(common_keys)} layers...")
    
    # Average weights for each layer
    for key in common_keys:
        tensor1 = state_dict1[key]
        tensor2 = state_dict2[key]
        
        # Check if tensors have the same shape
        if tensor1.shape != tensor2.shape:
            print(f"Warning: Shape mismatch for {key}: {tensor1.shape} vs {tensor2.shape}")
            continue
        
        # Average the tensors
        averaged_tensor = (tensor1 + tensor2) / 2.0
        averaged_state_dict[key] = averaged_tensor
        
        print(f"Averaged layer: {key} (shape: {averaged_tensor.shape})")
    
    return averaged_state_dict

def save_checkpoint(averaged_state_dict, reference_checkpoint, out_dir, save_model):
    """Save the averaged weights as a new checkpoint."""
    # Create new checkpoint based on reference structure
    new_checkpoint = {}
    
    # Copy metadata from reference checkpoint if it exists
    if isinstance(reference_checkpoint, dict):
        for key, value in reference_checkpoint.items():
            if key not in ['state_dict', 'model']:
                new_checkpoint[key] = value
    
    # Add the averaged state dict
    if 'state_dict' in reference_checkpoint:
        new_checkpoint['state_dict'] = averaged_state_dict
    elif 'model' in reference_checkpoint:
        new_checkpoint['model'] = averaged_state_dict
    else:
        # If reference was just a state_dict, save as state_dict
        new_checkpoint = averaged_state_dict
    
    # Save the checkpoint
    output_path = os.path.join(out_dir, save_model)
    os.makedirs(out_dir, exist_ok=True)
    torch.save(new_checkpoint, output_path)
    print(f"Averaged model saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Average weights of two PyTorch model checkpoints')
    parser.add_argument('model1', help='Path to first model checkpoint')
    parser.add_argument('model2', help='Path to second model checkpoint')
    parser.add_argument('outdir', help='Path for output directory')
    parser.add_argument('savemodel', help='Save the model using this filename')
    parser.add_argument('--device', default='cpu', help='Device to use (cpu/cuda)')
    
    args = parser.parse_args()
    
    print(f"Loading first model: {args.model1}")
    state_dict1, checkpoint1 = load_checkpoint(args.model1)
    
    print(f"Loading second model: {args.model2}")
    state_dict2, checkpoint2 = load_checkpoint(args.model2)
    
    print("Averaging weights...")
    averaged_state_dict = average_weights(state_dict1, state_dict2)
    
    print("Saving averaged model...")
    save_checkpoint(averaged_state_dict, checkpoint1, args.outdir, args.savemodel)
    
    print("Done!")

if __name__ == "__main__":
    # Example usage without command line arguments
    if len(os.sys.argv) == 1:
        print("Example usage:")
        print("python script.py model1.pth model2.pth averaged_model.pth")
        print("\nOr modify the paths below for direct execution:")
        
        # Uncomment and modify these lines for direct execution
        # model1_path = "path/to/your/first/model.pth"
        # model2_path = "path/to/your/second/model.pth"
        # output_path = "path/to/your/averaged_model.pth"
        # 
        # state_dict1, checkpoint1 = load_checkpoint(model1_path)
        # state_dict2, checkpoint2 = load_checkpoint(model2_path)
        # averaged_state_dict = average_weights(state_dict1, state_dict2)
        # save_checkpoint(averaged_state_dict, checkpoint1, output_path)
    else:
        main()
