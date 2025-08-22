import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Any
import argparse

class CommonAttentionFinder:
    def __init__(self, input_folder: str, output_folder: str, layer_name: str, 
                 summary_file: str, similarity_threshold: float = 1e-6):
        """
        Initialize the common attention weight finder.
        
        Args:
            input_folder: Path to folder containing JSON files
            output_folder: Path to output folder for results
            layer_name: Name of the specific layer to analyze
            summary_file: Name of the summary JSON file
            similarity_threshold: Threshold for considering vectors as identical
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.layer_name = layer_name
        self.summary_file = summary_file
        self.similarity_threshold = similarity_threshold
        self.epoch_data = {}
        
        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
    def load_all_epochs(self) -> Dict[str, Any]:
        """Load all JSON files from the input folder."""
        json_files = list(self.input_folder.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {self.input_folder}")

        # Sort files in ascending order by filename
        json_files = sorted(json_files, key=lambda x: x.name)        
        print(f"Loading {len(json_files)} JSON files...")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    epoch_name = json_file.stem
                    self.epoch_data[epoch_name] = data
                    print(f"  ✓ Loaded: {json_file.name}")
            except Exception as e:
                print(f"  ✗ Error loading {json_file}: {e}")
        
        print(f"Successfully loaded {len(self.epoch_data)} epochs")
        return self.epoch_data
    
    def extract_layer_weights(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract query and key weights for the specified layer from all epochs.
        
        Returns:
            Dictionary with epoch names as keys and query/key weights as values
        """
        layer_weights = {}
        
        for epoch_name, data in self.epoch_data.items():
            layers = data.get("layers", {})
            
            if self.layer_name not in layers:
                print(f"  ⚠ Layer '{self.layer_name}' not found in epoch '{epoch_name}'")
                continue
            
            layer_data = layers[self.layer_name]
            
            if layer_data.get("layer_type") != "attention":
                print(f"  ⚠ Layer '{self.layer_name}' is not an attention layer in epoch '{epoch_name}'")
                continue
            
            weights = layer_data.get("weights", {})
            query_weights = weights.get("query", [])
            key_weights = weights.get("key", [])
            
            if not query_weights or not key_weights:
                print(f"  ⚠ Empty query or key weights in epoch '{epoch_name}'")
                continue
            
            layer_weights[epoch_name] = {
                "query": np.array(query_weights),
                "key": np.array(key_weights)
            }
            
            print(f"  ✓ Extracted weights from epoch '{epoch_name}' - "
                  f"Query shape: {np.array(query_weights).shape}, "
                  f"Key shape: {np.array(key_weights).shape}")
        
        return layer_weights
    
    def vectors_are_similar(self, vec1: np.ndarray, vec2: np.ndarray, tolerance: float = 1e-6) -> List[float]:
        """Compare two vectors and return a list of values that are similar at the same indices."""
        if vec1.shape != vec2.shape:
            return []
        
        # Flatten vectors to handle multi-dimensional arrays
        v1_flat = vec1.flatten()
        v2_flat = vec2.flatten()
        
        similar_values = []
        
        # Compare values element-wise and collect similar ones
        for i in range(len(v1_flat)):
            val1 = v1_flat[i]
            val2 = v2_flat[i]
            
            # Check if values are within tolerance
            if abs(val1 - val2) <= tolerance:
                similar_values.append(float(val1))  # Use val1 as reference
        
        return similar_values

    def vector_intersection(arr1, arr2, threshold):
        """
        Find the intersection of two numpy arrays with a given threshold for float comparison.
        
        Args:
            arr1 (numpy.ndarray): First input array (will be flattened)
            arr2 (numpy.ndarray): Second input array (will be flattened)
            threshold (float): Threshold for considering two float values as equal
            
        Returns:
            list: List of float values that are present in both arrays within the threshold
        """
        # Flatten the arrays
        flat1 = arr1.flatten()
        flat2 = arr2.flatten()
        
        intersection = []
        
        # For each value in the first array
        for val1 in flat1:
            # Check if there's a matching value in the second array within threshold
            for val2 in flat2:
                if abs(val1 - val2) <= threshold:
                    # Add to intersection if not already present (within threshold)
                    is_duplicate = False
                    for existing in intersection:
                        if abs(val1 - existing) <= threshold:
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        intersection.append(float(val1))
                    break  # Found a match, move to next val1
        
        return intersection
    
    def find_common_vectors(self, all_vectors: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Find vectors that repeat (appear in multiple epochs).
        
        Args:
            all_vectors: Dict with epoch names as keys and vector arrays as values
            
        Returns:
            List of dictionaries containing information about vectors that repeat across epochs
        """
        if len(all_vectors) < 2:
            return []
        
        epoch_names = list(all_vectors.keys())
        common_vectors = []
        processed_vectors = set()  # Track vectors we've already processed
        
        # Compare vectors across all epochs
        for i, epoch1 in enumerate(epoch_names):
            vectors1 = all_vectors[epoch1]
            
            for vec_idx1, vector1 in enumerate(vectors1):
                # Create a unique identifier for this vector to avoid duplicates
                vector_id = (epoch1, vec_idx1)
                if vector_id in processed_vectors:
                    continue
                
                # Find all occurrences of this vector across epochs
                occurrences = [(epoch1, vec_idx1)]  # Include the original occurrence
                similar_values_list = []
                
                # Check remaining epochs
                for j, epoch2 in enumerate(epoch_names[i:], i):
                    if epoch2 == epoch1:
                        continue  # Skip the same epoch
                    
                    vectors2 = all_vectors[epoch2]
                    for vec_idx2, vector2 in enumerate(vectors2):
                        #similar_values = self.vectors_are_similar(vector1, vector2, self.similarity_threshold)
                        similar_values = self.vector_intersection(vector1, vector2, self.similarity_threshold)
                        
                        # Check if there are similar values (length > 0)
                        if len(similar_values) > 0:
                            occurrences.append((epoch2, vec_idx2))
                            similar_values_list.append({
                                "epoch": epoch2,
                                "vector_index": vec_idx2,
                                "similar_values": similar_values,
                                "similarity_count": len(similar_values)
                            })
                
                # If this vector appears in multiple epochs, it's a repeating vector
                if len(occurrences) > 1:
                    # Mark all occurrences as processed to avoid duplicates
                    for occurrence in occurrences:
                        processed_vectors.add(occurrence)
                    
                    # Create epochs and indices dictionaries
                    epochs_with_vector = [occ[0] for occ in occurrences]
                    vector_indices = {occ[0]: occ[1] for occ in occurrences}
                    
                    common_vectors.append({
                        "vector_id": len(common_vectors),
                        "reference_vector": vector1.tolist(),
                        "epochs": epochs_with_vector,
                        "indices": vector_indices,
                        "shape": list(vector1.shape),
                        "repeat_count": len(occurrences),
                        "similar_values_details": similar_values_list
                    })
        
        return common_vectors
    
    def analyze_layer(self) -> Dict[str, Any]:
        """Analyze the specified layer and find common queries and keys."""
        print(f"\nAnalyzing layer: {self.layer_name}")
        print("-" * 50)
        
        # Extract weights for the specified layer
        layer_weights = self.extract_layer_weights()
        
        if not layer_weights:
            raise ValueError(f"No valid weight data found for layer '{self.layer_name}'")
        
        # Separate queries and keys across all epochs
        all_queries = {}
        all_keys = {}
        
        for epoch_name, weights in layer_weights.items():
            # Split query and key matrices into individual vectors
            query_matrix = weights["query"]
            key_matrix = weights["key"]
            
            # Assuming the vectors are along one of the dimensions
            # You may need to adjust this based on your data structure
            if len(query_matrix.shape) == 2:
                # If 2D, treat each row as a vector
                all_queries[epoch_name] = [query_matrix[i] for i in range(query_matrix.shape[0])]
                all_keys[epoch_name] = [key_matrix[i] for i in range(key_matrix.shape[0])]
            else:
                # If 1D or other shape, treat as single vector
                all_queries[epoch_name] = [query_matrix]
                all_keys[epoch_name] = [key_matrix]
        
        print(f"Extracted vectors from {len(layer_weights)} epochs")
        
        # Find common queries and keys
        print("Finding common queries...")
        common_queries = self.find_common_vectors(all_queries)
        print(f"Found {len(common_queries)} common queries")
        
        print("Finding common keys...")
        common_keys = self.find_common_vectors(all_keys)
        print(f"Found {len(common_keys)} common keys")
        
        return {
            "common_queries": common_queries,
            "common_keys": common_keys,
            "analysis_metadata": {
                "layer_name": self.layer_name,
                "epochs_analyzed": list(layer_weights.keys()),
                "num_epochs": len(layer_weights),
                "similarity_threshold": self.similarity_threshold
            }
        }
    
    def save_summary(self, results: Dict[str, Any]):
        """Save the analysis results to a JSON file."""
        summary_path = self.output_folder / self.summary_file
        
        # Add timestamp and additional metadata
        results["summary_metadata"] = {
            "input_folder": str(self.input_folder),
            "layer_analyzed": self.layer_name,
            "total_common_queries": len(results["common_queries"]),
            "total_common_keys": len(results["common_keys"]),
            "similarity_threshold": self.similarity_threshold
        }
        
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Summary saved to: {summary_path}")
    
    def print_summary(self, results: Dict[str, Any]):
        """Print a readable summary of the results."""
        metadata = results["analysis_metadata"]
        
        print("\n" + "="*80)
        print("COMMON ATTENTION WEIGHTS ANALYSIS RESULTS")
        print("="*80)
        print(f"Layer analyzed: {metadata['layer_name']}")
        print(f"Number of epochs: {metadata['num_epochs']}")
        print(f"Epochs: {', '.join(metadata['epochs_analyzed'])}")
        print(f"Similarity threshold: {metadata['similarity_threshold']}")
        print("\n📊 RESULTS:")
        print(f"🔍 Common queries found: {len(results['common_queries'])}")
        print(f"🗝️  Common keys found: {len(results['common_keys'])}")
        
        if results['common_queries']:
            print(f"\nFirst common query shape: {results['common_queries'][0]['shape']}")
        if results['common_keys']:
            print(f"First common key shape: {results['common_keys'][0]['shape']}")

def main():
    parser = argparse.ArgumentParser(description="Find common queries and keys across all epochs")
    parser.add_argument("input_folder", help="Path to folder containing JSON files")
    parser.add_argument("output_folder", help="Path to output folder")
    parser.add_argument("layer_name", help="Name of the layer to analyze")
    parser.add_argument("summary_file", help="Name of the summary JSON file")
    parser.add_argument("--threshold", type=float, default=0.000001 
                       help="Similarity threshold (default: 0.000001)")
    
    args = parser.parse_args()
    
    try:
        # Initialize finder
        finder = CommonAttentionFinder(
            args.input_folder, 
            args.output_folder, 
            args.layer_name, 
            args.summary_file,
            args.threshold
        )
        
        # Load all epoch data
        finder.load_all_epochs()
        
        # Analyze the specified layer
        results = finder.analyze_layer()
        
        # Print and save results
        finder.print_summary(results)
        finder.save_summary(results)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
