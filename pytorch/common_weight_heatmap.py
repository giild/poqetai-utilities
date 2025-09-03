import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def create_heatmap(data_entries, data_type, layer_name, stats, output_path):
    """
    Create a single heatmap for either query or key data.
    
    Args:
        data_entries (list): List of data entries with row_index, col_index, value
        data_type (str): 'query' or 'key'
        layer_name (str): Name of the layer
        stats (dict): Statistics dictionary
        output_path (str): Output file path
    """
    # Initialize a 768x768 matrix with zeros
    heatmap_matrix = np.zeros((768, 768))
    
    # Populate the matrix
    for entry in data_entries:
        row_idx = entry['row_index']
        col_idx = entry['col_index']
        value = entry['value']
        
        # Ensure indices are within bounds
        if 0 <= row_idx < 768 and 0 <= col_idx < 768:
            heatmap_matrix[row_idx, col_idx] = value
    
    # Create custom colormap: red for negative, blue for positive
    colors = ['red', 'white', 'blue']
    n_bins = 512
    cmap = LinearSegmentedColormap.from_list('red_blue', colors, N=n_bins)
    
    # Create the plot with larger figure size
    plt.figure(figsize=(15, 15))
    
    # Create heatmap with interpolation to make blocks more visible
    im = plt.imshow(heatmap_matrix, cmap=cmap, aspect='equal', 
                    interpolation='nearest',  # Makes individual pixels/blocks more distinct
                    vmin=-np.abs(heatmap_matrix).max() if heatmap_matrix.min() < 0 else 0,
                    vmax=np.abs(heatmap_matrix).max() if heatmap_matrix.max() > 0 else 0)
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    # Set title and labels
    plt.title(f'{data_type.capitalize()} Values Heatmap - {layer_name}', fontsize=14, pad=20)
    plt.xlabel('Column Index', fontsize=12)
    plt.ylabel('Row Index', fontsize=12)
    
    # Add statistics as text
    if data_type == 'query':
        stats_text = f"Total Query Values: {stats.get('total_query_values', 0):,}\n"
        stats_text += f"Common Query Count: {stats.get('common_query_count', 0):,}\n"
        stats_text += f"Query Commonality Ratio: {stats.get('query_commonality_ratio', 0):.2f}"
    else:  # key
        stats_text = f"Total Key Values: {stats.get('total_key_values', 0):,}\n"
        stats_text += f"Common Key Count: {stats.get('common_key_count', 0):,}\n"
        stats_text += f"Key Commonality Ratio: {stats.get('key_commonality_ratio', 0):.2f}"
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    
    print(f"{data_type.capitalize()} heatmap saved as: {output_path}")
    print(f"Matrix shape: {heatmap_matrix.shape}")
    print(f"Non-zero values: {np.count_nonzero(heatmap_matrix)}")
    print(f"Value range: [{heatmap_matrix.min():.6f}, {heatmap_matrix.max():.6f}]")
    print()

def load_and_visualize_json(json_file_path, output_png_path='heatmap_output.png'):
    """
    Load JSON data and create 768x768 heatmap visualizations for both query and key data.
    
    Args:
        json_file_path (str): Path to the JSON file
        output_png_path (str): Base path for the output PNG files (without extension)
    """
    
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract base filename and extension
    if output_png_path.endswith('.png'):
        base_name = output_png_path[:-4]  # Remove .png extension
    else:
        base_name = output_png_path
    
    # Get layer name and statistics
    layer_name = data.get('layer_name', 'Unknown Layer')
    stats = data['statistics']
    
    # Create query heatmap
    query_data = data['common_values']['query']
    if query_data:
        query_output_path = f"{base_name}_query.png"
        create_heatmap(query_data, 'query', layer_name, stats, query_output_path)
    else:
        print("No query data found in the JSON file.")
    
    # Create key heatmap
    key_data = data['common_values']['key']
    if key_data:
        key_output_path = f"{base_name}_key.png"
        create_heatmap(key_data, 'key', layer_name, stats, key_output_path)
    else:
        print("No key data found in the JSON file.")
    
    print(f"Processing complete for layer: {layer_name}")

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
