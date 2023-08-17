import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import time
import os
import sys
import json
import convert_keras_h5_to_json

# Script will iterate over the directory and convert all HDF5 checkpoint models to JSON format
# it uses convert_keras_h5_to_json to read the binary format and save it as JSON
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: input_folder output_folder')
    else:
        input = args[1].replace("\\","/")
        outputdir = args[2].replace("\\", "/")
        print(' converting H5 models in folder: ', input)
        if os.path.exists(outputdir) == False:
            os.mkdir(outputdir)
        filelist = os.listdir(input)
        filelist = sorted(filelist)
        for f in filelist:
            if f.endswith('h5'):
                inputfile = input + '/' + f
                outputfile = outputdir + '/' + f.replace('h5','json')
                convert_keras_h5_to_json.run(inputfile, outputfile)

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
