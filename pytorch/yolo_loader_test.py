import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics package not found.")
    print("Install it using: pip install ultralytics")
    sys.exit(1)


def load_yolo_checkpoint(checkpoint_path):
    """
    Load a YOLOv12 checkpoint model.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        
    Returns:
        YOLO: Loaded YOLO model
    """
    # Verify checkpoint file exists
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Check file extension
    valid_extensions = ['.pt', '.pth']
    if ckpt_file.suffix.lower() not in valid_extensions:
        print(f"Warning: Expected .pt or .pth file, got {ckpt_file.suffix}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the model
    model = YOLO(checkpoint_path)
    
    print(f"✓ Model loaded successfully!")
    print(f"  Model type: {model.task}")
    print(f"  Model names: {model.names}")
    
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Load a YOLOv12 checkpoint using Ultralytics YOLO"
    )
    parser.add_argument(
        "checkpoint",
        type=str,
        help="Path to the YOLOv12 checkpoint file (.pt or .pth)"
    )
    
    args = parser.parse_args()
    
    try:
        model = load_yolo_checkpoint(args.checkpoint)
        print("\nModel ready for inference or training!")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
