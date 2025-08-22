import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import argparse

class AttentionWeightComparer:
    def __init__(self, folder_path: str, similarity_threshold: float = 0.95):
        """
        Initialize the attention weight comparer.
        
        Args:
            folder_path: Path to folder containing JSON files
            similarity_threshold: Cosine similarity threshold for considering weights as "common"
        """
        self.folder_path = Path(folder_path)
        self.similarity_threshold = similarity_threshold
        self.attention_data = {}
        
    def load_json_files(self) -> Dict[str, Any]:
        """Load all JSON files from the specified folder."""
        json_files = list(self.folder_path.glob("*.json"))
        
        if not json_files:
            raise ValueError(f"No JSON files found in {self.folder_path}")
        
        print(f"Found {len(json_files)} JSON files to process")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    # Use filename as key (without extension)
                    file_key = json_file.stem
                    self.attention_data[file_key] = data
                    print(f"Loaded: {json_file.name}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        return self.attention_data
    
    def extract_qk_weights(self, data: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Extract query and key weights from the loaded data.
        
        Args:
            data: Loaded JSON data for one epoch
            
        Returns:
            Dictionary with layer names as keys and query/key weights as values
        """
        qk_weights = {}
        
        for layer_name, layer_data in data.get("layers", {}).items():
            if layer_data.get("layer_type") == "attention":
                weights = layer_data.get("weights", {})
                
                # Convert lists to numpy arrays if they're not empty
                query_weights = weights.get("query", [])
                key_weights = weights.get("key", [])
                
                if query_weights and key_weights:
                    qk_weights[layer_name] = {
                        "query": np.array(query_weights),
                        "key": np.array(key_weights)
                    }
        
        return qk_weights
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors/matrices."""
        # Flatten arrays to 1D for similarity calculation
        a_flat = a.flatten()
        b_flat = b.flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(a_flat, b_flat)
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def find_common_weights(self) -> Dict[str, Dict[str, List[Tuple[str, str, float]]]]:
        """
        Find queries and keys that are common (similar) across different epochs.
        
        Returns:
            Dictionary with layer names and lists of common weight pairs
        """
        if len(self.attention_data) < 2:
            raise ValueError("Need at least 2 files to compare")
        
        # Extract QK weights for all epochs
        all_qk_weights = {}
        for epoch_name, data in self.attention_data.items():
            all_qk_weights[epoch_name] = self.extract_qk_weights(data)
        
        # Get all layer names (assuming all epochs have same layers)
        epoch_names = list(all_qk_weights.keys())
        first_epoch = epoch_names[0]
        layer_names = list(all_qk_weights[first_epoch].keys())
        
        common_weights = {}
        
        for layer_name in layer_names:
            print(f"\nAnalyzing layer: {layer_name}")
            common_weights[layer_name] = {"queries": [], "keys": []}
            
            # Compare all pairs of epochs for this layer
            for i, epoch1 in enumerate(epoch_names):
                for j, epoch2 in enumerate(epoch_names[i+1:], i+1):
                    
                    # Check if both epochs have this layer
                    if (layer_name not in all_qk_weights[epoch1] or 
                        layer_name not in all_qk_weights[epoch2]):
                        continue
                    
                    epoch1_weights = all_qk_weights[epoch1][layer_name]
                    epoch2_weights = all_qk_weights[epoch2][layer_name]
                    
                    # Compare queries
                    if ("query" in epoch1_weights and "query" in epoch2_weights):
                        query_sim = self.cosine_similarity(
                            epoch1_weights["query"], 
                            epoch2_weights["query"]
                        )
                        
                        if query_sim >= self.similarity_threshold:
                            common_weights[layer_name]["queries"].append(
                                (epoch1, epoch2, query_sim)
                            )
                    
                    # Compare keys
                    if ("key" in epoch1_weights and "key" in epoch2_weights):
                        key_sim = self.cosine_similarity(
                            epoch1_weights["key"], 
                            epoch2_weights["key"]
                        )
                        
                        if key_sim >= self.similarity_threshold:
                            common_weights[layer_name]["keys"].append(
                                (epoch1, epoch2, key_sim)
                            )
        
        return common_weights
    
    def print_results(self, common_weights: Dict[str, Dict[str, List[Tuple[str, str, float]]]]):
        """Print the results in a readable format."""
        print("\n" + "="*80)
        print("ATTENTION WEIGHT COMPARISON RESULTS")
        print("="*80)
        print(f"Similarity threshold: {self.similarity_threshold}")
        print(f"Number of files compared: {len(self.attention_data)}")
        
        for layer_name, results in common_weights.items():
            print(f"\n📊 Layer: {layer_name}")
            print("-" * 60)
            
            # Query results
            queries = results["queries"]
            print(f"🔍 Common Queries: {len(queries)} pairs found")
            for epoch1, epoch2, similarity in queries:
                print(f"  • {epoch1} ↔ {epoch2}: {similarity:.4f}")
            
            # Key results  
            keys = results["keys"]
            print(f"🗝️  Common Keys: {len(keys)} pairs found")
            for epoch1, epoch2, similarity in keys:
                print(f"  • {epoch1} ↔ {epoch2}: {similarity:.4f}")
        
        # Summary statistics
        total_common_queries = sum(len(results["queries"]) for results in common_weights.values())
        total_common_keys = sum(len(results["keys"]) for results in common_weights.values())
        
        print(f"\n📈 SUMMARY:")
        print(f"Total common query pairs across all layers: {total_common_queries}")
        print(f"Total common key pairs across all layers: {total_common_keys}")
    
    def save_results(self, common_weights: Dict[str, Dict[str, List[Tuple[str, str, float]]]], 
                    output_file: str = "common_weights_results.json"):
        """Save results to a JSON file."""
        # Convert tuples to lists for JSON serialization
        serializable_results = {}
        for layer_name, results in common_weights.items():
            serializable_results[layer_name] = {
                "queries": [{"epoch1": e1, "epoch2": e2, "similarity": sim} 
                           for e1, e2, sim in results["queries"]],
                "keys": [{"epoch1": e1, "epoch2": e2, "similarity": sim} 
                        for e1, e2, sim in results["keys"]]
            }
        
        output_path = self.folder_path / output_file
        with open(output_path, 'w') as f:
            json.dump({
                "metadata": {
                    "similarity_threshold": self.similarity_threshold,
                    "num_files_compared": len(self.attention_data),
                    "files": list(self.attention_data.keys())
                },
                "results": serializable_results
            }, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare attention weights across epochs")
    parser.add_argument("folder_path", help="Path to folder containing JSON files")
    parser.add_argument("--threshold", type=float, default=0.95, 
                       help="Cosine similarity threshold (default: 0.95)")
    parser.add_argument("--output", type=str, default="common_weights_results.json",
                       help="Output file name (default: common_weights_results.json)")
    
    args = parser.parse_args()
    
    try:
        # Initialize comparer
        comparer = AttentionWeightComparer(args.folder_path, args.threshold)
        
        # Load JSON files
        print("Loading attention weight files...")
        comparer.load_json_files()
        
        # Find common weights
        print("Finding common queries and keys...")
        common_weights = comparer.find_common_weights()
        
        # Display results
        #comparer.print_results(common_weights)
        
        # Save results
        comparer.save_results(common_weights, args.output)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
