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

# main expects two arguments: the input h5 file and the file to save the json model
# recommend saving the json version to a different folder
def main():
    args = sys.argv[0:]

    if len(sys.argv) == 1:
        print(' Example: python convert_keras_h5_to_json.py input-model.h5 output_model.json ')
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
    modeltype = "keras"
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
        layerType = type(layer).__name__
        weights = layer._trainable_weights
        jsonStr += '{'
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
    jsonStr += ']'
    jsonStr += '}'
    #print(jsonStr)
    return jsonStr

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
