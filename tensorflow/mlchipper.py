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

# main expects two arguments: the input h5 file, temp directory, epoch string
# Wood chippers take a tree and cut it up into little pices. ML Chipper takes a
# ML model and chunks it up into smaller pieces. This makes it easier to load
# parts of the model and reduce memory utilization
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: python convert_keras_h5_to_json.py input-model.h5 temp_dir epoch')
    else:
        input = args[1]
        tempdir = args[2]
        epochstr = args[3]
        if os.path.exists(tempdir) == False:
            os.makedirs(tempdir)
        run(input, tempdir, epochstr)

# Run will use Keras to load the model and break it up by layers
def run(inputfile, tempdir, epochstr):
    chipdir = tempdir + '/epoch.' + epochstr
    if os.path.exists(chipdir) == False:
        os.mkdir(chipdir)
    inputmodel = tf.keras.models.load_model(inputfile)
    print(inputmodel.summary())
    start_time = time.time()
    chunkModel(inputmodel, chipdir)
    end_time = time.time()
    print(' - convert time: ', (end_time - start_time), ' ms')
    print(' - Saved model: ', tempdir)

def chunkModel(model: keras.Sequential, tempdir):
    modeltype = "keras-weights"
    # iterate over the layers
    for i in range(len(model.layers)):
        jsonStr = ""
        outdir = tempdir + '/layer-' + str(i)
        if os.path.exists(outdir) == False:
            os.mkdir(outdir)
        outputfile = outdir + '/weights.json'
        if i > 0:
            jsonStr += ','
        layer = model.layers[i]
        layerType = type(layer).__name__
        weights = layer._trainable_weights
        jsonStr += '{'
        jsonStr += '"model":"' + model.name + '",'
        jsonStr += '"documentType":"' + modeltype + '",'
        jsonStr += '"name":"' + layer.name + '",'
        jsonStr += '"classtype":"' + layerType + '",'
        jsonStr += '"input_shape":"' + str(layer.input_shape) + '",'
        jsonStr += '"output_shape":"' + str(layer.output_shape) + '",'
        if layerType == 'Conv2D':
            jsonStr += '"kernel":"' + str(layer.kernel_size) + '",'
        jsonStr += '"dtype":"' + layer.dtype + '",'
        jsonStr += '"weights":'
        jsonStr +='['
        # iterate over the weights
        for w in range(len(weights)):
            weight = weights[w]
            if w > 0:
                jsonStr += ','
            jsonStr += '{'
            jsonStr += '"name":"' + weight.name + '",'
            jsonStr += '"shape":"' + str(weight.shape) + '",'
            jsonStr += '"array":' + json.dumps(weight.numpy().tolist())
            jsonStr += '}'
        jsonStr += ']'
        jsonStr += '}'
        testout = open(outputfile, "w")
        testout.write(jsonStr)
        testout.close()
    print('done chipping')

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
