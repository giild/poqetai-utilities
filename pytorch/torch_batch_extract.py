import os
import subprocess
import glob
from pathlib import Path
import torch_extract_attention

startIndex = 1

def process_checkpoints(checkpoint_folder, output_directory):
    """
    Process all PyTorch checkpoint files in a folder using torch_extract_attention.py
    
    Args:
        checkpoint_folder (str): Path to folder containing checkpoint files
        output_directory (str): Directory where output files will be saved
        script_path (str): Path to torch_extract_attention.py script
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Common PyTorch checkpoint file extensions
    checkpoint_patterns = [
        "*.pt", "*.pth"
    ]
    
    checkpoint_files = []
    
    # Find all checkpoint files
    for pattern in checkpoint_patterns:
        checkpoint_files.extend(glob.glob(os.path.join(checkpoint_folder, pattern)))
    
    if not checkpoint_files:
        print(f"No checkpoint files found in {checkpoint_folder}")
        print(f"Looked for extensions: {', '.join(checkpoint_patterns)}")
        return
    
    # sort the files
    checkpoint_files.sort()
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Process each checkpoint file
    for i, checkpoint_file in enumerate(checkpoint_files, start=startIndex):
        checkpoint_name = Path(checkpoint_file).name
        checkpoint_stem = Path(checkpoint_file).stem  # filename without extension
        
        print(f"\n[{i}/{len(checkpoint_files)}] Processing: {checkpoint_name}")
        
        try:
            torch_extract_attention.run(
                checkpoint1_path=checkpoint_file,
                outputdir=output_directory,
                outputfile=checkpoint_stem
            )
        except Exception as e:
            print(f"✗ Unexpected error processing {checkpoint_name}: {str(e)}")

def main():
    # Configuration - modify these paths as needed
    checkpoint_folder = "./checkpoints"  # Folder containing your .pt/.pth files
    output_directory = "./attention_outputs"  # Where to save extracted attention
    
    # You can also accept command line arguments
    import sys
    if len(sys.argv) >= 2:
        checkpoint_folder = sys.argv[1]
    if len(sys.argv) >= 3:
        output_directory = sys.argv[2]
    if len(sys.argv) >= 4:
        startIndex = int(sys.argv[3])
    
    print(f"Checkpoint folder: {checkpoint_folder}")
    print(f"Output directory: {output_directory}")
    
    if not os.path.exists(checkpoint_folder):
        print(f"Error: Checkpoint folder '{checkpoint_folder}' does not exist")
        return
    
    process_checkpoints(checkpoint_folder, output_directory)
    print("\nBatch processing completed!")

if __name__ == "__main__":
    main()