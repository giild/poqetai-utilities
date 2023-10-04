from textwrap import indent
from numpy import ndarray
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import tensorflow_datasets as tfds
import validationresult
import cifar10_rename_h5cpt

print("tensorflow version ", tf.__version__)
print("TF dataset version", tfds.__version__)

def main(): 
    args = sys.argv[0:]
    print(args)
    if len(sys.argv) == 1:
        print(' Example: python cifar10_rename_checkpoints.py input_folder')
    else:
        input = args[1].replace("\\","/")
        print(' converting H5 models in folder: ', input)
        filelist = os.listdir(input)
        filelist = sorted(filelist)
        for f in filelist:
            if f.endswith('h5'):
                inputfile = input + '/' + f
                cifar10_rename_h5cpt.run(inputfile)
    
# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
