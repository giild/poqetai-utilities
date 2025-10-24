import argparse
import json
import torch
import numpy as np
from pathlib import Path

def count_weights_in_ranges(weights):
    """
    Count weights in different magnitude ranges.
    
    Args:
        weights: numpy array or torch tensor of weights
    
    Returns:
        dict: counts for each range
    """
    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()
    
    # Take absolute values for magnitude analysis
    all_weights = weights.flatten()
    
    # Define ranges (upper bounds)
    ranges = [
        (0.1, 0.01, "0.1_to_0.01"),
        (0.01, 0.001, "0.01_to_0.001"),
        (0.001, 0.0001, "0.001_to_0.0001"),
        (0.0001, 0.00001, "0.0001_to_0.00001"),
        (0.00001, 0.000001, "0.00001_to_0.000001"),
        (0.000001, 0.0000001, "0.000001_to_0.0000001"),
        (0.0000001, 0.00000001, "0.0000001_to_0.00000001"),
        (0.00000001, 0.000000001, "0.00000001_to_0.000000001")
    ]
    
    counts = {}
    
    # Count weights in each range
    for upper, lower, name in ranges:
        count = np.sum((all_weights < upper) & (all_weights >= lower))
        counts[name] = int(count)
    
    # Also count weights outside these ranges
    counts["greater_than_0.1"] = int(np.sum(all_weights >= 0.1))
    counts["less_than_0.00000001"] = int(np.sum(all_weights < 0.000000001))
    counts["total_weights"] = len(all_weights)
    
    return counts

def analyze_checkpoint(checkpoint_path):
    """
    Load checkpoint and analyze weight statistics.
    
    Args:
        checkpoint_path: path to the model checkpoint
    
    Returns:
        dict: statistics for all layers and overall
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model weights (handle different checkpoint formats)
    if 'ema' in checkpoint:
        ema = checkpoint['ema']
        state_dict = ema.state_dict()
    else:
        state_dict = checkpoint
    
    results = {
        "checkpoint_path": str(checkpoint_path),
        "layer_statistics": {},
        "overall_statistics": {}
    }
    
    # Collect all weights for overall statistics
    all_weights = []
    
    # Analyze each layer
    print("\nAnalyzing weights per layer...")
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor) and param.numel() > 0:
            layer_stats = count_weights_in_ranges(param)
            results["layer_statistics"][name] = layer_stats
            all_weights.append(param.cpu().numpy().flatten())
            print(f"  Processed: {name} ({param.numel()} weights)")
    
    # Calculate overall statistics
    if all_weights:
        print("\nCalculating overall statistics...")
        all_weights_array = np.concatenate(all_weights)
        results["overall_statistics"] = count_weights_in_ranges(all_weights_array)
        
        # Add percentage information
        total = results["overall_statistics"]["total_weights"]
        results["overall_statistics"]["percentages"] = {}
        for key, count in results["overall_statistics"].items():
            if key not in ["total_weights", "percentages"]:
                percentage = (count / total * 100) if total > 0 else 0
                results["overall_statistics"]["percentages"][key] = round(percentage, 4)
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="Analyze YOLOv12 model weights and generate statistics"
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the model checkpoint file"
    )
    parser.add_argument(
        "output",
        type=str,
        default=None,
        help="Output JSON file path (default: <checkpoint_name>_weight_stats.json)"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Analyze checkpoint
    results = analyze_checkpoint(checkpoint_path)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_weight_stats.json"
    
    # Save results
    print(f"\nSaving results to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("OVERALL STATISTICS SUMMARY")
    print("="*60)
    overall = results["overall_statistics"]
    print(f"Total weights: {overall['total_weights']:,}")
    print(f"\nWeight distribution:")
    print(f"  > 0.1:              {overall['greater_than_0.1']:>12,} ({overall['percentages']['greater_than_0.1']:>6.2f}%)")
    print(f"  0.1 to 0.01:        {overall['0.1_to_0.01']:>12,} ({overall['percentages']['0.1_to_0.01']:>6.2f}%)")
    print(f"  0.01 to 0.001:      {overall['0.01_to_0.001']:>12,} ({overall['percentages']['0.01_to_0.001']:>6.2f}%)")
    print(f"  0.001 to 0.0001:    {overall['0.001_to_0.0001']:>12,} ({overall['percentages']['0.001_to_0.0001']:>6.2f}%)")
    print(f"  0.0001 to 0.00001:  {overall['0.0001_to_0.00001']:>12,} ({overall['percentages']['0.0001_to_0.00001']:>6.2f}%)")
    print(f"  0.00001 to 0.000001: {overall['0.00001_to_0.000001']:>11,} ({overall['percentages']['0.00001_to_0.000001']:>6.2f}%)")
    print(f"  < 0.00000001:       {overall['less_than_0.00000001']:>12,} ({overall['percentages']['less_than_0.00000001']:>6.2f}%)")
    print("="*60)

if __name__ == "__main__":
    main()
