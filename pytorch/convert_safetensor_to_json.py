import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from safetensors import safe_open
import time
import sys
import collections
from torchsummary import summary
import time
import json
from itertools import pairwise

class SafetensorConversion:
    
    def __init__(self):
        return
        
    def loadSafetensorToDict(self, filename):
        modeldata = {}
        with safe_open(filename, framework="pt", device="cpu") as f:
            for key in f.keys():
                modeldata[key] = f.get_tensor(key)
        return modeldata
    
    def summarizeModel(self, modeldata):
        return []
    
    def printModelSummary(self, model:collections.OrderedDict):
        allkeys = model.keys()
        #print(allkeys)
        for key, value in model.items():
            print(key)
        return
    
    def run(self, mname, inputfile, outputfile):
        modeldict = self.loadSafetensorToDict(inputfile)
        #self.printModelSummary(modeldict)
        jsonStr = self.convertToJson(mname,0,inputfile,modeldict)
        #print(jsonStr)
        testout = open(outputfile,"w")
        testout.write(jsonStr)
        testout.close()
        return
    
    def convertToJson(self, modelname, paramcount, inputfile, modeldict):
        modeltype = "pytorch"
        jsonStr = '{'
        jsonStr += '"name":"' + modelname + '",'
        jsonStr += '"documentType":"' + modeltype + '",'
        jsonStr += '"dtype":"float32",'
        jsonStr += '"params":' + str(paramcount) + ','
        jsonStr += '"url":"' + inputfile + '",'
        jsonStr += '"layers":['
        i = 0
        skip = False
        for currentkey,nextkey in pairwise(modeldict.keys()):
            print("[" + currentkey, "]")
            if skip == False:
                pmatch = self.compareLayerNane(currentkey, nextkey)
                if pmatch:
                    jsonStr2 = ''
                    value1 = modeldict[currentkey]
                    value2 = modeldict[nextkey]
                    tensor_value1 = torch.tensor(value1) if not isinstance(value1, torch.Tensor) else value1
                    tensor_value2 = torch.tensor(value2) if not isinstance(value2, torch.Tensor) else value2
                    if i > 0:
                        jsonStr += ','
                    jsonStr2 += '{'
                    jsonStr2 += '"name":"' + self.keyName(nextkey) + '",'
                    jsonStr2 += '"classtype":"' + self.keyName(nextkey) + '",'
                    jsonStr2 += '"input_shape":"' + self.formatShape(tensor_value2.shape) + '",'
                    jsonStr2 += '"output_shape":"' + "0:0:0:0" + '",'
                    jsonStr2 += '"dtype":"' + self.formatDType(tensor_value2.dtype) + '",'
                    jsonStr2 += '"weights":['
                    jsonStr2 += '{"name":"' + self.keyName(nextkey) + '/kernel:0",'
                    jsonStr2 += '"shape":"' + self.formatShape(tensor_value2.shape) + '",'
                    jsonStr2 += '"array":' + json.dumps(tensor_value2.tolist())
                    jsonStr2 += "},"
                    jsonStr2 += '{"name":"' + self.keyName(currentkey) + '/bias:0",'
                    jsonStr2 += '"shape":"' + self.formatShape(tensor_value1.shape) + '",'
                    jsonStr2 += '"array":' + json.dumps(tensor_value1.tolist())
                    jsonStr2 += "}"
                    jsonStr2 += "]"
                    jsonStr += jsonStr2
                    jsonStr += '}'
                    print(f"  X match={pmatch}, classtype={nextkey}/{currentkey}, input_shape={self.formatShape(tensor_value2.shape)}")
                    i += 1
                    skip = True
                else:
                    jsonStr2 = ''
                    value1 = modeldict[currentkey]
                    tensor_value1 = torch.tensor(value1) if not isinstance(value1, torch.Tensor) else value1
                    if i > 0:
                        jsonStr += ','
                    jsonStr2 += '{'
                    jsonStr2 += '"name":"' + self.keyName(currentkey) + '",'
                    jsonStr2 += '"classtype":"' + self.keyName(currentkey) + '",'
                    jsonStr2 += '"input_shape":"' + self.formatShape(tensor_value1.shape) + '",'
                    jsonStr2 += '"output_shape":"' + "0:0:0:0" + '",'
                    jsonStr2 += '"dtype":"' + self.formatDType(tensor_value1.dtype) + '",'
                    jsonStr2 += '"weights":['
                    jsonStr2 += '{"name":"' + self.keyName(currentkey) + '/kernel:0",'
                    jsonStr2 += '"shape":"' + self.formatShape(tensor_value1.shape) + '",'
                    jsonStr2 += '"array":' + json.dumps(tensor_value1.tolist())
                    jsonStr2 += "}"
                    jsonStr2 += "]"
                    jsonStr += jsonStr2
                    jsonStr += '}'
                    print(f"  - classtype={self.keyName(currentkey)}, input_shape={self.formatShape(tensor_value1.shape)}, output_shape=0:0:0:0, dtype={self.formatDType(tensor_value1.dtype)}")
                    skip = False
            else:
                skip = False
        jsonStr += ']'
        jsonStr += '}'
        return jsonStr

    def formatDType(self, tensortype):
        split = str(tensortype).split('.')
        if len(split) > 1:
            return split[-1]
        return tensortype

    def formatShape(self, tensorshape):
        tensorshape = str(tensorshape).replace('torch.Size','').replace("[","").replace("]","")
        return tensorshape
    
    def compareLayerNane(self, currentlayer, nextlayer):
        """compare the string name of the layer to identify if they are the same layer

        Args:
            currentlayer (_type_): current layer name
            nextlayer (_type_): next layer name

        Returns:
            _type_: boolean indicating if the layers are the same
        """
        crsplit = str(currentlayer).split('.')
        nxsplit = str(nextlayer).split('.')
        if crsplit[0] == nxsplit[0] and crsplit[1] == nxsplit[1] and crsplit[2] == nxsplit[2]:
            return True
        elif len(crsplit) == len(nxsplit) and crsplit[0] == nxsplit[0]:
            return True
        return False
    
    def keyName(self, keyname):
        if str(keyname).startswith('conv'):
            return "Conv2D"
        elif str(keyname).startswith('fc'):
            return "Dense"
        else:
            if keyname.endswith('.weight'):
                return keyname[:-7]  # Remove '.weight' suffix
            elif keyname.endswith('.bias'):
                return keyname[:-5] # Remove '.bias' suffix
        return keyname

def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: python convert_torch_to_json.py input-model.pth output_model.json ')
    else:
        input = args[1]
        outputfile = args[2]
        modelname = args[3]
        convert = SafetensorConversion()
        convert.run(modelname, input, outputfile) 

if __name__ == "__main__":
    main()
