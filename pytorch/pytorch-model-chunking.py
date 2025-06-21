import os
import sys
import torch
import torch.nn as nn
import json
from collections import OrderedDict
import time
import json
from itertools import pairwise

def compareLayerName(currentlayer, nextlayer):
    crsplit = str(currentlayer).split('.')
    nxsplit = str(nextlayer).split('.')
    if crsplit[0] == nxsplit[0]:
        return True
    return False

def inferLayer(keyname):
    print(str(keyname))
    if str(keyname).startswith('conv'):
        return "Conv2D"
    elif str(keyname).startswith('fc'):
        return "Dense"
    elif str(keyname).startswith('linear'):
        return "Linear"
    elif str(keyname).startswith('pool'):
        return "Pool"
    return keyname

def chunk_model(model, output_dir='model_chunks'):
    """
    Load a PyTorch model and save each layer in a separate folder with its weights
    and a summary file.
    
    Args:
        model: A PyTorch model or path to a saved model file
        output_dir: Directory where the chunked model will be saved
    """
    modeldict = None
    # If model is a file path, load it
    if isinstance(model, str):
        try:
            modeldict = torch.load(model, weights_only=False)
        except Exception as e:
            raise ValueError(f"Failed to load model from {model}: {e}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    layerindex = 1
    for currentkey,nextkey in pairwise(modeldict.keys()):
        #print(currentkey, " - ", nextkey)
        inferredlayer = inferLayer(currentkey) 
        pmatch = compareLayerName(currentkey, nextkey)
        if pmatch:                
            subfolder = f"{output_dir}/{str(layerindex)}"
            # make the folder for the layer
            makeOutputDir(subfolder)
            value1 = modeldict[currentkey]
            value2 = modeldict[nextkey]
            tensor_value1 = torch.tensor(value1) if not isinstance(value1, torch.Tensor) else value1
            tensor_value2 = torch.tensor(value2) if not isinstance(value2, torch.Tensor) else value2
            weights_str = json.dumps(tensor_value1.tolist())
            bias_str = json.dumps(tensor_value2.tolist())
            weightfile = f"{subfolder}/weights.json"
            biasfile = f"{subfolder}/bias.json"
            saveData(weightfile, weights_str)
            saveData(biasfile, bias_str)
            print(' --- save layer ---')
        elif inferredlayer == "Dense":
            subfolder = f"{output_dir}/{str(layerindex)}"
            # make the folder for the layer
            makeOutputDir(subfolder)
            value1 = modeldict[currentkey]
            tensor_value1 = torch.tensor(value1) if not isinstance(value1, torch.Tensor) else value1
            weights_str = json.dumps(tensor_value1.tolist())
            weightfile = f"{subfolder}/weights.json"
            saveData(weightfile, weights_str)
        elif inferredlayer == "Linear":
            subfolder = f"{output_dir}/{str(layerindex)}"
            # make the folder for the layer
            makeOutputDir(subfolder)
            value1 = modeldict[currentkey]
            tensor_value1 = torch.tensor(value1) if not isinstance(value1, torch.Tensor) else value1
            weights_str = json.dumps(tensor_value1.tolist())
            weightfile = f"{subfolder}/weights.json"
            saveData(weightfile, weights_str)
        else:
            print(' -- do nothing --')
        # ddd
        print(layerindex)
        layerindex += 1
    
    print(f"Model chunked successfully. Chunks saved to {output_dir}")

def saveData(filename:str, jsonstr:str):
    writeout = open(filename,"w")
    writeout.write(jsonstr)
    writeout.close()
    
def makeOutputDir(directoryname:str):
    if os.path.exists(directoryname) == False:
        os.makedirs(directoryname)

def main():
    args = sys.argv[0:]
    """Example usage"""
    modelpath = args[1]
    outputpath = args[2]
    chunk_model(modelpath, outputpath)

if __name__ == '__main__':
    main()
