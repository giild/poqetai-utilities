import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import time
import os
import sys
import json
import coremltools

# Script will iterate over the directory and convert all HDF5 checkpoint models to JSON format
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: my_model.h5 my_model')
    else:
        input = args[1].replace("\\","/")
        output = args[2].replace("\\","/")
        author = args[3]
        license = args[4]
        mversion = args[5]
        sdesc = args[6]
        output = output + ".mlpackage"
        print(' converting H5 models to Apple coreml format ', input)
        output_labels = ["rock","paper","scissor"]
        h5model = tf.keras.models.load_model(input)
        # the shape needs to be changed to match your model
        image_input = coremltools.ImageType(name="conv2d_input", shape=(1,300,300,3))
        classifier_config = coremltools.ClassifierConfig(output_labels)
        ctmodel = coremltools.convert(h5model,
                                      inputs=[image_input],
                                      classifier_config=classifier_config,
                                      source='tensorflow')
        ctmodel.author = author
        ctmodel.version = mversion
        ctmodel.license = license
        ctmodel.short_description = sdesc
        ctmodel.save(output)

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
