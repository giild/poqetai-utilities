from textwrap import indent
from numpy import ndarray
import numpy as np
import tensorflow as tf
import time
import os
import sys
import floatdelta
import modeldelta as md
import layerdelta
import json
import JsonWriter

#print(tf.__version__)

def main(args):
    takevalue = "high"
    if len(args) == 0:
        print('Example usage:')
        print('          python selective_checkpoint_update.py ./checkpoint_model.hdf5 midpoint_analysis.json new_version.hdf5 high 1,2,4')
    else:
        # load all of the files
        #print('Loading with args:  ', args)
        outputname = 'modified_checkpoint_model.h5'
        modelfile = args[1].replace("\\","/")
        midpointfile = args[2].replace("\\","/")
        changelayers = []
        if len(args) >= 4:
            outputname = args[3].replace("\\","/")
        else:
            outputname = modelfile.replace(".h5","_modified.h5")
        if len(args) >= 5:
            takevalue = args[4]
        if len(args) == 6:
            argstring = args[5]
            changelayers = argstring.split(',')
        print(args)

        model = tf.keras.models.load_model(modelfile)
        mpanalysis = json.load(open(midpointfile))
        modified = modifyWeights(mpanalysis, model, changelayers, takevalue)
        # save the model with a new filename
        saveUpdatedModel(modified, outputname)

# check if the layer should be modified
def checkChangeLayer(index, changelyrs):
    if len(changelyrs) == 0:
        return True
    else:
        print('check if we should change the layer')
        for n in range(len(changelyrs)):
            if changelyrs[n] == str(index):
                return True
            
def modifyWeights(mpdata, model, changelyrs, takevalue):
    modelLayers = model.layers
    mplayers = mpdata["layerlist"]
    #print(mplayers)
    convChangeCount = 0
    denseChangeCount=0
    starttime = time.time()
    print('layers to change: ', changelyrs)
    print('layer count', len(mplayers))
    for l in range(len(mplayers)):
        mplyr = mplayers[l]
        li = mplyr["index"]
        if checkChangeLayer(li, changelyrs):
            # update the layer
            lwtarray = mplyr["weights"]
            lyrType = mplyr["layerType"]
            layr = modelLayers[li]
            for w in range(len(lwtarray)):
                wts = lwtarray[w]
                wtidxs = wts["weightIndex"].split(':')
                highval = float(wts["highValue"])
                midval = float(wts["midValue"])
                lowval = float(wts["lowValue"])
                # check the layer type
                if lyrType == "Conv2D":
                    lweights = layr.weights[0].numpy()
                    bweights = []
                    if len(layr.weights) == 2:
                        bweights = layr.weights[1].numpy()

                    idx1 = int(wtidxs[1])
                    idx2 = int(wtidxs[2])
                    idx3 = int(wtidxs[3])
                    idx4 = int(wtidxs[4])
                    # update the weight
                    if takevalue == "high":
                        lweights[idx1, idx2, idx3, idx4] = highval
                    elif takevalue == "mid":
                        lweights[idx1, idx2, idx3, idx4] = midval
                    elif takevalue == "low":
                        lweights[idx1, idx2, idx3, idx4] = lowval
                    convChangeCount += 1    
                    # update the layer
                    if len(layr.weights) == 2:
                        layr.set_weights(np.array([lweights, bweights], dtype=object))
                    else:
                        layr.set_weights(np.array([lweights], dtype=object))
                    
                elif lyrType == "Dense":
                    lweights = layr.weights[0].numpy()
                    bweights = layr.weights[1].numpy()
                    idx1 = int(wtidxs[1])
                    idx2 = int(wtidxs[2])
                    # update the weight
                    if takevalue == "high":
                        lweights[idx1, idx2] = highval
                    elif takevalue == "mid":
                        lweights[idx1, idx2] = midval
                    elif takevalue == "low":
                        lweights[idx1, idx2] = lowval
                    denseChangeCount += 1
                    # update the layer
                    layr.set_weights(np.array([lweights, bweights], dtype=object))
        
    endtime = time.time()
    totaltime = endtime - starttime
    duration = totaltime/60
    # check layer type and get the layer weights
    print(' dense changes - ', denseChangeCount)
    print(' conv2d changes - ', convChangeCount)
    print(f"done - Time: {duration:.2f} minutes")
    return model
        
def saveUpdatedModel(modified, outputfile):
    print(f"- saving updated model {outputfile}")
    modified.save(outputfile)

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main(sys.argv[0:])
