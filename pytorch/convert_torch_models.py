import time
import os
import sys
import json
import convert_torch_to_json

# Script will iterate over the directory and convert all HDF5 checkpoint models to JSON format
# it uses convert_keras_h5_to_json to read the binary format and save it as JSON
def main():
    args = sys.argv[0:]
    converter = convert_torch_to_json.TorchConversion()
    if len(sys.argv) == 1:
        print(' Example: python convert_torch_models.py input_folder output_folder modelname')
    else:
        input = args[1].replace("\\","/")
        outputdir = args[2].replace("\\", "/")
        modelname = args[3].replace("\\", "/")
        print(' converting safetensor models in folder: ', input)
        if os.path.exists(outputdir) == False:
            os.mkdir(outputdir)
        filelist = os.listdir(input)
        filelist = sorted(filelist)
        for f in filelist:
            if f.endswith('pth'):
                inputfile = input + '/' + f
                outputfile = outputdir + '/' + f.replace('pth','json')
                converter.run(modelname, inputfile, outputfile)

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
