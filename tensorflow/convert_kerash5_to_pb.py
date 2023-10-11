import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import time
import os
import sys
import json

# print the version of tensorflow for sanity in case there's a version conflict
print(tf.__version__)
print(tfds.__version__)

# main expects one arguments: the input h5 file 
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: python convert_kerash5_to_pb.py input-model.h5')
    else:
        input = args[1]
        run(input)
        
def run(inputfile):
    print('Converting HDF5 to Protocal buffer')
    inputmodel = tf.keras.models.load_model(inputfile)
    newname = inputfile.replace('h5','pb')
    inputmodel.save(newname)
    
# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
