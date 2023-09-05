import os
import sys
# NOTE:  it is important to set the OS environment KERAS_BACKEND before loading
#        the other libraries. Setting the backend after loading causes error loading models
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_cv
import keras_core as keras
import time
import json

# print the version of tensorflow for sanity in case there's a version conflict
print(tf.__version__)

# main expects two arguments: the input h5 file and the file to save the json model
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: input-model.keras output_model.json ')
    else:
        input = args[1]
        outputfile = args[2]
        run(input, outputfile)

# Run will use Keras to load the model and call convertToJson and save the JSON
def run(inputfile, outputfile):
    inputmodel = tf.keras.models.load_model(inputfile)
    print(inputmodel.summary())
    start_time = time.time()
    jsonString = convertToJson(inputmodel, inputfile)
    end_time = time.time()
    testout = open(outputfile,"w")
    testout.write(jsonString)
    testout.close()
    end_time2 = time.time()
    print(' - convert time: ', (end_time - start_time), ' ms')
    print(' - Saved model: ', outputfile)
    print(' - save time: ', (end_time2 - end_time), ' ms')

def convertToJson(model: keras.Sequential, inputfile):
    modeltype = "keras_core"
    jsonStr = '{'
    jsonStr += '"name":"' + model.name + '",'
    jsonStr += '"documentType":"' + modeltype + '",'
    jsonStr += '"dtype":"' + model.dtype + '",'
    jsonStr += '"params":' + str(model.count_params()) + ','
    jsonStr += '"url":"' + inputfile + '",'
    jsonStr += '"layers":['
    # iterate over the layers
    for i in range(len(model.layers)):
        if i > 0:
            jsonStr += ','
        layer = model.layers[i]
        print(layer)
        layerType = type(layer).__name__
        weights = layer.trainable_weights
        jsonStr += '{'
        jsonStr += '"name":"' + layer.name + '",'
        jsonStr += '"classtype":"' + layerType + '",'
        if hasattr(layer, "_input_tensor"):
            jsonStr += '"input_shape":"' + str(layer._input_tensor.shape) + '",'
        elif hasattr(layer, "input"):
            if hasattr(layer.input,"shape"):
                jsonStr += '"input_shape":"' + str(layer.input.shape) + '",'
            else:
                jsonStr += '"input_shape":"' + str(layer.input_shape) + '",'
        if hasattr(layer, "output"):
            if hasattr(layer.output, "shape"):
                jsonStr += '"output_shape":"' + str(layer.output.shape) + '",'
            else:
                jsonStr += '"output_shape":"' + str(layer.output) + '",'
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
    jsonStr += ']'
    jsonStr += '}'
    #print(jsonStr)
    return jsonStr

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
