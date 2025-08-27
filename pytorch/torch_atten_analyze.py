#!/usr/bin/env python3
"""
Attention Weights Common Values Analyzer

This script analyzes attention weights from Vision Transformer models across multiple training epochs
to find common query and key values in a specified layer.

Usage:
    python attention_analyzer.py <json_folder> <layer_name> <output_file>

Example:
    python attention_analyzer.py ./attention_weights blocks.0.attn.qkv.weight summary.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from typing import Dict, List, Set, Tuple, Any

counthreshold = 2 # Minimum number of epochs a value must appear in to be considered "common"

def load_json_files(folder_path: str) -> List[Dict]:
    """
    Load all JSON files from the specified folder.
    
    Args:
        folder_path: Path to folder containing JSON files
        
    Returns:
        List of loaded JSON data dictionaries
    """
    json_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    json_file_paths = list(folder.glob("*.json"))
    
    if not json_file_paths:
        raise ValueError(f"No JSON files found in folder: {folder_path}")
    
    # Sort files naturally (handles numeric sorting properly)
    json_file_paths.sort(key=lambda x: x.name)
    
    print(f"Found {len(json_file_paths)} JSON files")
    print("Loading files in order:")
    for path in json_file_paths:
        print(f"  - {path.name}")
    print()
    
    for json_path in json_file_paths:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                json_files.append({
                    'filename': json_path.name,
                    'data': data
                })
                print(f"Loaded: {json_path.name}")
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    return json_files


def find_common_values(all_values: Dict[str, List[List[float]]], threshold: float = 1e-6, countlimit: int=2) -> List[Tuple[Tuple[int, int], float, int]]:
    """
    Find common values across all epochs with their 2D indices and occurrence counts.
    
    Args:
        all_values: Dictionary mapping epoch filenames to 2D arrays of values
        threshold: Tolerance for considering values as equal
        
    Returns:
        List of tuples ((row_idx, col_idx), common_value, occurrence_count)
    """
    if not all_values:
        return []
    
    # Filter out epochs with empty arrays
    valid_epochs = {epoch: vals for epoch, vals in all_values.items() if vals and len(vals) > 0}
    if not valid_epochs:
        return []
    
    # Get dimensions from first valid epoch
    first_epoch = next(iter(valid_epochs.values()))
    if not first_epoch or len(first_epoch) == 0:
        return []
    
    num_rows = len(first_epoch)
    num_cols = len(first_epoch[0]) if first_epoch[0] else 0
    
    common_indices = []
    
    # Iterate over each position in the 2D array
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            # Get values at this 2D position from all epochs
            values_at_position = []
            for epoch, epoch_values in valid_epochs.items():
                if (row_idx < len(epoch_values) and 
                    col_idx < len(epoch_values[row_idx])):
                    try:
                        val = float(epoch_values[row_idx][col_idx])
                        values_at_position.append(val)
                    except (ValueError, TypeError):
                        # Skip if can't convert to float
                        continue
            
            if len(values_at_position) < 2:
                continue
                
            # Check if multiple values at this position are similar (within threshold)
            first_value = values_at_position[0]
            try:
                similar_count = sum(1 for val in values_at_position if abs(val - first_value) <= threshold)
                is_common = similar_count >= countlimit  # At least 2 epochs have similar values
            except (ValueError, TypeError):
                continue
            
            if is_common:
                common_indices.append(((row_idx, col_idx), first_value, len(values_at_position)))
    
    return common_indices


def analyze_attention_weights(json_files: List[Dict], layer_name: str, count:int) -> Dict[str, Any]:
    """
    Analyze attention weights to find common values in query and key.
    
    Args:
        json_files: List of loaded JSON data
        layer_name: Name of the layer to analyze
        
    Returns:
        Dictionary containing analysis results
    """
    query_values_by_epoch = {}
    key_values_by_epoch = {}
    
    valid_epochs = []
    
    for file_info in json_files:
        filename = file_info['filename']
        data = file_info['data']
        
        # Check if layer exists in this file
        if 'layers' not in data or layer_name not in data['layers']:
            print(f"Warning: Layer '{layer_name}' not found in {filename}")
            continue
            
        layer_data = data['layers'][layer_name]
        
        if 'weights' not in layer_data:
            print(f"Warning: No weights found in layer '{layer_name}' in {filename}")
            continue
            
        weights = layer_data['weights']
        
        # Extract query and key values
        query_vals = weights.get('query', [])
        key_vals = weights.get('key', [])
        
        if not query_vals and not key_vals:
            print(f"Warning: No query or key values found in {filename}")
            continue
            
        query_values_by_epoch[filename] = query_vals
        key_values_by_epoch[filename] = key_vals
        valid_epochs.append(filename)
    
    if not valid_epochs:
        raise ValueError(f"No valid data found for layer '{layer_name}'")
    
    print(f"Analyzing {len(valid_epochs)} valid epochs for layer '{layer_name}'")
    
    # Find common values
    common_query = find_common_values(query_values_by_epoch, countlimit=count)
    common_key = find_common_values(key_values_by_epoch, countlimit=count)
    
    # Calculate statistics
    first_valid_query = next(iter(query_values_by_epoch.values())) if query_values_by_epoch else []
    first_valid_key = next(iter(key_values_by_epoch.values())) if key_values_by_epoch else []
    
    total_query_values = len(first_valid_query) * (len(first_valid_query[0]) if first_valid_query else 0)
    total_key_values = len(first_valid_key) * (len(first_valid_key[0]) if first_valid_key else 0)
    
    results = {
        'layer_name': layer_name,
        'total_epochs_analyzed': len(valid_epochs),
        'valid_epochs': valid_epochs,
        'statistics': {
            'total_query_values': total_query_values,
            'total_key_values': total_key_values,
            'common_query_count': len(common_query),
            'common_key_count': len(common_key),
            'query_commonality_ratio': len(common_query) / total_query_values if total_query_values > 0 else 0,
            'key_commonality_ratio': len(common_key) / total_key_values if total_key_values > 0 else 0
        },
        'common_values': {
            'query': [
                {
                    'row_index': pos[0],
                    'col_index': pos[1],
                    'value': float(val),
                    'occurrences': count
                }
                for pos, val, count in common_query
            ],
            'key': [
                {
                    'row_index': pos[0],
                    'col_index': pos[1],
                    'value': float(val),
                    'occurrences': count
                }
                for pos, val, count in common_key
            ]
        }
    }
    
    return results


def save_summary(results: Dict[str, Any], output_file: str):
    """
    Save the analysis results to a JSON file.
    
    Args:
        results: Analysis results dictionary
        output_file: Path to output file
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Summary saved to: {output_file}")
    except Exception as e:
        raise IOError(f"Error saving summary to {output_file}: {e}")


def print_summary_stats(results: Dict[str, Any]):
    """
    Print summary statistics to console.
    
    Args:
        results: Analysis results dictionary
    """
    stats = results['statistics']
    
    print("\n" + "="*60)
    print("ATTENTION WEIGHTS ANALYSIS SUMMARY")
    print("="*60)
    print(f"Layer analyzed: {results['layer_name']}")
    print(f"Total epochs: {results['total_epochs_analyzed']}")
    print(f"Total query values: {stats['total_query_values']}")
    print(f"Total key values: {stats['total_key_values']}")
    print(f"Common query values: {stats['common_query_count']} ({stats['query_commonality_ratio']:.2%})")
    print(f"Common key values: {stats['common_key_count']} ({stats['key_commonality_ratio']:.2%})")
    
    # Show first few common values as examples
    common_query = results['common_values']['query']
    common_key = results['common_values']['key']
    
    if common_query:
        print(f"\nFirst 5 common query values:")
        for item in common_query[:5]:
            print(f"  Position ({item['row_index']}, {item['col_index']}): {item['value']:.6f} (in {item['occurrences']} epochs)")
    
    if common_key:
        print(f"\nFirst 5 common key values:")
        for item in common_key[:5]:
            print(f"  Position ({item['row_index']}, {item['col_index']}): {item['value']:.6f} (in {item['occurrences']} epochs)")


def main():
    """Main function to run the attention weights analyzer."""
    parser = argparse.ArgumentParser(
        description="Analyze attention weights to find common query and key values across epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python attention_analyzer.py ./attention_weights blocks.0.attn.qkv.weight summary.json
  python attention_analyzer.py /path/to/weights blocks.1.attn.qkv.weight results.json
        """
    )
    
    parser.add_argument('json_folder', 
                       help='Path to folder containing JSON files with attention weights')
    parser.add_argument('layer_name', 
                       help='Name of the layer to analyze (e.g., blocks.0.attn.qkv.weight)')
    parser.add_argument('output_file', 
                       help='Path to output summary file (JSON format)')
    parser.add_argument('--count_threshold', type=int, default=2,
                       help='Count threshold for considering values as common (default: 2 )')
    
    args = parser.parse_args()
    
    try:
        if args.count_threshold > 2:
            counthreshold = args.count_threshold
            print(f"Using count threshold: {counthreshold}")
        else:
            counthreshold = 2
        
        print("Loading JSON files...")
        json_files = load_json_files(args.json_folder)
        
        print(f"Analyzing layer: {args.layer_name}")
        results = analyze_attention_weights(json_files, args.layer_name, counthreshold)
        
        print("Saving summary...")
        save_summary(results, args.output_file)
        
        print_summary_stats(results)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()