import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary
from ultralytics import YOLO

def main():
    """
    Main entry point for the script.
    """
    parser = argparse.ArgumentParser(
        description="Try to use torchsummary to print a pytorch model. If it fails print the ordered keys."
    )
    parser.add_argument(
        "filepath", 
        type=str, 
        help="Path to the filename"
    )
    args = parser.parse_args()
    print("PyTorch Version:", torch.__version__)
    model = torch.load(args.filepath, map_location=torch.device('cpu'), weights_only=False)
    printSummary(model, args.filepath)

def printSummary(model, filename):
    print(f"File: {filename}")
    model = model['ema']
    try:
        summary(model)
        statedict = model.state_dict()
        print(f"State Dict Keys: {statedict.keys()}")
    except Exception as e:
        print(f"Summary: {model.keys()}")

if __name__ == "__main__":
    main()