import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import time
import os
import sys
import json
import coremltools

# Script will read HDF5 format and save it in CoreML format. You have to install
# coremltools from Apple to make this work. The github repo has details about
# the latest release https://github.com/apple/coremltools
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: my_model.h5 my_model')
    else:
        input = args[1].replace("\\","/")
        output_file = args[2].replace("\\","/")
        author = args[3]
        license = args[4]
        mversion = args[5]
        sdesc = args[6]
        output_file = output_file + ".mlpackage"
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
        ctmodel = coremltools.models.neural_network.quantization_utils.quantize_weights(ctmodel,8)
        ctmodel.save(output_file)

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
