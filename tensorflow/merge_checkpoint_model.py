from textwrap import indent
from numpy import ndarray
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import floatdelta
import modeldelta as md
import layerdelta
import json
import JsonWriter

print(tf.__version__)

# the basic idea is to take the weight corrections and apply them to the model.
# NOTE: depending on the parameter count, this process can take minutes. I've profiled
#       the code and the slowness is mostly python. We may need to rewrite this in C
#       to make it fast and not suck. Yes, python performance for IO/CPU intensive stuff
#       sucks like a rotting corpse. For now, this works, but it needs to be rewritten.
def main(args):
    if len(args) == 1:
        print('Example usage:')
        print('          python correct_checkpoint_model.py ./checkpoint_model.hdf5 corrections.json')
    else:
        # load all of the files
        print('Loading with args:  ', args)
        modelfile = args[1].replace("\\","/")
        weightregress = args[2]
        model = tf.keras.models.load_model(modelfile)
        corrections = json.load(open(weightregress))
        modified = modifyWeights(corrections, model)
        # save the model with a new filename
        modified.save('merged_checkpoint_model.h5')

def modifyWeights(wcorrections, model):
    mode = 'all'
    conv2dmode = 'all'
    mlayers = model.layers
    # iterate over the corrections and apply it to the model
    crarray = wcorrections['corrections']
    clen = len(crarray)
    convChangeCount = 0
    denseChangeCount=0
    starttime = time.time()
    for c in range(clen):
        # lookup the layer and apply the change
        crt = crarray[c]
        layeridx = crt["layerIndex"]
        wistr = crt["weightIndex"]
        wttype = crt["weightType"]
        cweight = float(crt["newvalue"])
        weightdelta = float(crt["weightDelta"])
        wtidxs = wistr.split(':')
        # get the layer by the layer index
        layr = mlayers[layeridx]
        #print(' ------ the layer weights: ', layr.weights)
        if isinstance(layr, tf.keras.layers.Dense) and wttype == "p":
            lweights = layr.weights[0].numpy()
            bweights = layr.weights[1].numpy()
            idx1 = int(wtidxs[1])
            idx2 = int(wtidxs[2])
            oldwt = lweights[idx1,idx2]
            #print( idx1, idx2, cweight, lweights)
            if mode == 'all':
                lweights[idx1,idx2] = cweight
                denseChangeCount += 1
            elif mode == 'pos' and cweight > 0.0:
                lweights[idx1,idx2] = cweight
                denseChangeCount += 1
            elif mode == 'neg' and cweight < 0.0:
                lweights[idx1,idx2] = cweight
                denseChangeCount += 1
            #print(lweights[idx1,idx2], ' - old: ', oldwt)
            # to change the weights of a layer, you have to call set_weights() 
            # https://github.com/keras-team/keras/blob/master/keras/engine/base_layer.py
            layr.set_weights(np.array([lweights, bweights], dtype=object))
        elif isinstance(layr, tf.keras.layers.Conv2D) and wttype == "p":
            # apply changes to Conv2D layer
            lweights = layr.weights[0].numpy()
            bweights = []
            if len(layr.weights) == 2:
                bweights = layr.weights[1].numpy()

            idx1 = int(wtidxs[1])
            idx2 = int(wtidxs[2])
            idx3 = int(wtidxs[3])
            idx4 = int(wtidxs[4])
            oldwt = lweights[idx1, idx2, idx3, idx4]
            if conv2dmode == 'all':
                lweights[idx1, idx2, idx3, idx4] = cweight
                convChangeCount += 1
            elif conv2dmode == 'pos' and cweight > 0.0:
                lweights[idx1, idx2, idx3,  idx4] = cweight
                convChangeCount += 1
            elif conv2dmode == 'neg' and cweight < 0.0:
                lweights[idx1, idx2, idx3,  idx4] = cweight
                convChangeCount += 1
            
            if len(layr.weights) == 2:
                layr.set_weights(np.array([lweights, bweights], dtype=object))
            else:
                layr.set_weights(np.array([lweights], dtype=object))

    endtime = time.time()
    totaltime = endtime - starttime
    # check layer type and get the layer weights
    print(' dense changes - ', denseChangeCount)
    print(' conv2d changes - ', convChangeCount)
    print('done - total time: ', totaltime/60)

    return model

def saveUpdatedModel(model):
    print('  - start saving updated model')

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main(sys.argv[0:])