import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import time
import os
import sys
import json
import cifar10_detailed_test

# Script will iterate over the directory and convert all HDF5 checkpoint models to JSON format
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: input_folder ')
    else:
        input = args[1].replace("\\","/")
        print(' validation report for H5 models in folder: ', input)
        filelist = os.listdir(input)
        filelist = sorted(filelist)
        for f in filelist:
            if f.endswith('h5'):
                epoch = parseFilename(f)
                inputfile = input + '/' + f
                outputfile = input + '/validation/epoch_' + str(epoch) + '_testresult.json'
                cifar10_detailed_test.run(inputfile, outputfile, epoch)

# the expected file format should have the epoch after weights
# weights.01-0.363-1.097-0.597-1.050.h5
def parseFilename(file:str):
    tokens = file.split(".")
    wtokens = tokens[1].split("-")
    return wtokens[0]

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
