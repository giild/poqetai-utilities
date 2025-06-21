import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time
import sys
import collections
from torchsummary import summary
import time
import json
from itertools import pairwise

class TorchConversion:
    
    def __init__(self):
        return
    
    def loadTorchToDict(self, filename):
        modeldata = torch.load(filename, weights_only=False)
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
        modeldict = self.loadTorchToDict(inputfile)
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
            print(currentkey, " - ", nextkey)
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
                jsonStr2 += '"name":"' + self.formatName(currentkey) + '",'
                jsonStr2 += '"classtype":"' + self.keyName(currentkey) + '",'
                jsonStr2 += '"input_shape":"' + self.formatShape(tensor_value1.shape) + '",'
                jsonStr2 += '"output_shape":"' + "0:0:0:0" + '",'
                jsonStr2 += '"dtype":"' + self.formatDType(tensor_value1.dtype) + '",'
                jsonStr2 += '"weights":['
                jsonStr2 += '{"name":"' + currentkey + '/kernel:0",'
                jsonStr2 += '"shape":"' + self.formatShape(tensor_value1.shape) + '",'
                jsonStr2 += '"array":' + json.dumps(tensor_value1.tolist())
                jsonStr2 += "},"
                jsonStr2 += '{"name":"' + nextkey + '/bias:0",'
                jsonStr2 += '"shape":"' + self.formatShape(tensor_value2.shape) + '",'
                jsonStr2 += '"array":' + json.dumps(tensor_value2.tolist())
                jsonStr2 += "}"
                jsonStr2 += "]"
                jsonStr += jsonStr2
                jsonStr += '}'
                skip = True
                print(f"  X match={pmatch}, classtype={currentkey}/{nextkey}, input_shape={self.formatShape(tensor_value1.shape)}")
            else:
                skip = False
            i += 1
            
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
        if currentlayer.endswith('.weight'):
            currentlayer = currentlayer[:-7]
        if nextlayer.endswith('.bias'):
            nextlayer = nextlayer[:-5]
        if currentlayer == nextlayer:
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

    def formatName(self, keyname):
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
        convert = TorchConversion()
        convert.run(modelname, input, outputfile) 

if __name__ == "__main__":
    main()
