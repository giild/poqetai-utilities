import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def load_and_visualize_json(json_file_path, output_png_path='heatmap_output.png'):
    """
    Load JSON data and create a 768x768 heatmap visualization.
    
    Args:
        json_file_path (str): Path to the JSON file
        output_png_path (str): Path for the output PNG file
    """
    
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Initialize a 768x768 matrix with zeros
    heatmap_matrix = np.zeros((768, 768))
    
    # Extract query data and populate the matrix
    query_data = data['common_values']['query']
    
    for entry in query_data:
        row_idx = entry['row_index']
        col_idx = entry['col_index']
        value = entry['value']
        
        # Ensure indices are within bounds
        if 0 <= row_idx < 768 and 0 <= col_idx < 768:
            heatmap_matrix[row_idx, col_idx] = value
    
    # Create custom colormap: red for negative, blue for positive
    colors = ['red', 'white', 'blue']
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list('red_blue', colors, N=n_bins)
    
    # Create the plot
    plt.figure(figsize=(10, 10))
    
    # Create heatmap
    im = plt.imshow(heatmap_matrix, cmap=cmap, aspect='equal', 
                    vmin=-np.abs(heatmap_matrix).max() if heatmap_matrix.min() < 0 else 0,
                    vmax=np.abs(heatmap_matrix).max() if heatmap_matrix.max() > 0 else 0)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    # Set title and labels
    layer_name = data.get('layer_name', 'Unknown Layer')
    plt.title(f'Query Values Heatmap - {layer_name}', fontsize=14, pad=20)
    plt.xlabel('Column Index', fontsize=12)
    plt.ylabel('Row Index', fontsize=12)
    
    # Add statistics as text
    stats = data['statistics']
    stats_text = f"Total Query Values: {stats['total_query_values']:,}\n"
    stats_text += f"Common Query Count: {stats['common_query_count']:,}\n"
    stats_text += f"Query Commonality Ratio: {stats['query_commonality_ratio']:.2f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Heatmap saved as: {output_png_path}")
    print(f"Matrix shape: {heatmap_matrix.shape}")
    print(f"Non-zero values: {np.count_nonzero(heatmap_matrix)}")
    print(f"Value range: [{heatmap_matrix.min():.6f}, {heatmap_matrix.max():.6f}]")

# Example usage:
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_json_file> <output_png_file>")
        print("Example: python script.py data.json heatmap.png")
        sys.exit(1)
    
    json_file_path = sys.argv[1]
    output_path = sys.argv[2]
    
    try:
        load_and_visualize_json(json_file_path, output_path)
    except FileNotFoundError:
        print(f"Error: Could not find file '{json_file_path}'")
        print("Please make sure the JSON file exists and the path is correct.")
    except KeyError as e:
        print(f"Error: Missing key in JSON data: {e}")
        print("Please check that your JSON file has the expected structure.")
    except Exception as e:
        print(f"An error occurred: {e}")

