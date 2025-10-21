import argparse
import json
import torch
import os
from pathlib import Path

def load_checkpoint(checkpoint_path):
    """Load PyTorch checkpoint file."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def load_json_analysis(json_path):
    """Load JSON analysis file."""
    print(f"Loading JSON analysis: {json_path}")
    with open(json_path, 'r') as f:
        analysis = json.load(f)
    return analysis

def modify_parameters(checkpoint, analysis, threshold):
    """
    Modify parameters in checkpoint based on delta analysis.
    Sets parameters to 0 where 0 < |delta| < threshold.
    """
    # Get the model state dict
    if 'ema' in checkpoint:
        ema = checkpoint['ema']
        state_dict = ema.state_dict()
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    modifications_count = 0
    layers_modified = 0
    
    layer_changes = analysis.get('layer_changes', {})
    
    for layer_name, layer_info in layer_changes.items():
        if layer_name not in state_dict:
            print(f"Warning: Layer '{layer_name}' not found in checkpoint")
            continue
        
        # Get the parameter tensor
        param_tensor = state_dict[layer_name]
        weights_data = layer_info.get('weights', [])
        
        layer_had_modifications = False
        
        for weight_info in weights_data:
            delta = abs(weight_info['delta'])
            
            # Check if delta is non-zero and below threshold
            if 0 < delta < threshold:
                index = tuple(weight_info['index'])
                
                # Set this weight to 0
                param_tensor[index] = 0.0
                modifications_count += 1
                layer_had_modifications = True

        if layer_had_modifications:
            layers_modified += 1
    
    print(f"\nModifications applied:")
    print(f"  Total weights set to 0: {modifications_count}")
    print(f"  Layers modified: {layers_modified}")

    return checkpoint, modifications_count, layers_modified

def save_checkpoint(checkpoint, output_folder, output_filename):
    """Save modified checkpoint to file."""
    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Full output file path
    output_file = output_path / output_filename
    
    print(f"\nSaving modified checkpoint to: {output_file}")
    torch.save(checkpoint, output_file)
    print("Checkpoint saved successfully!")

    return output_file

def main():
    parser = argparse.ArgumentParser(
        description='Modify checkpoint parameters based on delta threshold analysis'
    )
    parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to checkpoint model file (.pt)'
    )
    parser.add_argument(
        'json_file',
        type=str,
        help='Path to JSON analysis file'
    )
    parser.add_argument(
        'output_folder',
        type=str,
        help='Output folder path'
    )
    parser.add_argument(
        'output_filename',
        type=str,
        help='Output filename for modified checkpoint'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0000001,
        help='Delta threshold (default: 0.0000001). Weights with 0 < |delta| < threshold will be set to 0'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Checkpoint Parameter Modifier")
    print("="*60)
    print(f"Threshold: {args.threshold}")
    print(f"Weights with 0 < |delta| < {args.threshold} will be set to 0")
    print("="*60)
    
    # Load checkpoint and analysis
    checkpoint = load_checkpoint(args.checkpoint)
    analysis = load_json_analysis(args.json_file)
    
    # Display analysis summary
    if 'summary' in analysis:
        summary = analysis['summary']
        print("\nAnalysis Summary:")
        print(f"  Total layers: {summary.get('total_layers', 'N/A')}")
        print(f"  Total weights: {summary.get('total_weights', 'N/A')}")
        print(f"  Weights changed: {summary.get('total_weights_changed', 'N/A')}")
        print(f"  Percent changed: {summary.get('percent_weights_changed', 'N/A'):.2f}%")
    
    # Modify parameters
    modified_checkpoint, mods_count, layers_count = modify_parameters(
        checkpoint, analysis, args.threshold
    )
    
    # Save modified checkpoint
    output_file = save_checkpoint(
        modified_checkpoint, 
        args.output_folder, 
        args.output_filename
    )
    
    print("\n" + "="*60)
    print("Process completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
