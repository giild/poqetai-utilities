from textwrap import indent
from numpy import ndarray
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import floatdelta
import layerdelta
import modeldelta as md
import layerdelta
import json
import JsonWriter

print(tf.__version__)

model_diff:md.ModelDelta = None

# main function expects 3 arguments. The two checkpoint file name and the output filename
def main(args):

    if len(args) == 1:
        print('Example usage:')
        print('               python checkpoint_diff.py ./mymodel1.hdf5 ./mymodel2.hdf5 result_output.json')
    else:
        print('Loading with args:  ', args)
        filename1 = args[1].replace("\\","/")
        filename2 = args[2].replace("\\","/")
        model1 = tf.keras.models.load_model(filename1)
        model2 = tf.keras.models.load_model(filename2)
        outputfile = args[3]
        epoch1 = args[4]
        epoch2 = args[5]
        model_diff = md.ModelDelta(model1.name,filename1, filename2)
        model_diff.epoch1 = epoch1
        model_diff.epoch2 = epoch2
        print(model_diff.name)
        print(model_diff.modelfile1)
        print(model_diff.modelfile2)
        #print(' deltas=', model_diff.layerdeltas)
        start_time = time.time()
        compare(model_diff, model1, model2)
        end_time = time.time();
        print('  - diff time: ', end_time - start_time, " seconds")
        print('  - saving diff to file: ', outputfile)
        write_start = time.time()
        teststring = JsonWriter.writeDiffModel(model_diff)
        testout = open(outputfile,"w")
        testout.write(teststring)
        testout.close()
        write_end = time.time()
        print('  write time: ', write_end - write_start, " seconds")
    # return the diff model
    return model_diff

""" diff is the entry point for comparing the weights of two checkpoint models
 For now diff will ignore the layer if it's the Input for the model. The reason
 for skipping the input layer is to reduce noise. The assumption might be
 wrong and the filters in the input layer could be significant.
"""
def compare(diff, model1, model2):
    print(' ---------- comparing the checkpoints ----------')
    # iterate over a sequential model and do diff
    for index, item in enumerate(model1.layers):
        m1layer = item
        m2layer = model2.layers[index]
        # switch statement to handle each layer type properly
        if isinstance(item, tf.keras.layers.Conv2D):
            print('Conv2D layer')
            diffConv2D(diff, index, m1layer.weights, m2layer.weights)
        elif isinstance(item, tf.keras.layers.MaxPooling2D):
            print('MaxPooling2D layer')
            diffMaxPool(diff, index, m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Flatten):
            diffFlatten(diff,index, m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Dropout):
            diffDropout(diff, index, m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Dense):
            diffDense(diff, index, m1layer, m2layer)
        elif isinstance(item, tf.keras.layers.Conv3D):
            print('Conv3D layer')
        else:
            print(item.__class__.__name__)

        #print(diff)
    print(' --- done with diff')

""" If the layer is not input layer, we compare the weights.
 Layer definition: Conv2D(256, (2, 2), strides=(1,1), activation='relu', name='L2_conv2d')
 Weight shape: shape(2, 2, 256, 256)
 The first 2 number is the kernel (2,2), the third number is channels (aka previous layer filter size),
 forth number is the layer filters. Keras source for third number has input_channel // self.groups
 https://github.com/keras-team/keras/blob/master/keras/layers/convolutional/base_conv.py line 212. 
 The operator is floor division, which means most of the time the value is divided by default group 1.

 The input is equal to the output from the previous layer
 the last is the output filter. Note the kernel may be different, so the function has to look
 at the shape.

 TODO - for now it's a bunch of nested for loops. Needs to be refactored and clean it up
"""
def diffConv2D(diff, index, weights1, weights2):
    if index > 0:
        # We should always have weights for Conv2D, but check to be safe
        if len(weights1) > 0:
            kheight = weights1[0].shape[0]
            kwidth = weights1[0].shape[1]
            prevchannels = weights1[0].shape[2]
            filters = weights1[0].shape[3]
            #print(' kernel shape=', weights1[0].shape)
            #print(' channels=', prevchannels)
            #print(' filter =', filters)
            # Conv2D layers weights have kernel and bias. By default bias is true. It is optional
            lydelta = layerdelta.Conv2dLayerDelta(index, weights1[0].name, kheight, kwidth, prevchannels, filters)
            diff.addLayerDelta(lydelta)
            # weights are at index 0
            wgts1 = weights1[0].numpy()
            wgts2 = weights2[0].numpy()
            # the height array for deltas based on kernel height
            wtarray = lydelta.deltaarray
            #print(wgts1)
            hlen = len(wgts1)
            print('  weight shape: ', wgts1.shape)
            print('  - height len: ', hlen)
            for h in range(hlen):
                print(' h index: ', h)
                h1 = wgts1[h]
                h2 = wgts2[h]
                #print(h1)
                hty = []
                wtarray.append(hty)
                wlen = len(h1)
                print('    - width len: ', wlen)
                for w in range(wlen):
                    wy1 = h1[w]
                    wy2 = h2[w]
                    wty = []
                    hty.append(wty)
                    clen = len(wy1)
                    print('     - chan len: ', clen)
                    for c in range(clen):
                        # channel array
                        cy1 = wy1[c]
                        cy2 = wy2[c]
                        chry = []
                        wty.append(chry)
                        flen = len(cy1)
                        #print('    filter len: ', flen)
                        for f in range(flen):
                            # the filters weights
                            lydelta.incrementParamCount()
                            wt1 = cy1[f]
                            wt2 = cy2[f]
                            delta = abs(wt2 - wt1)
                            lydelta.AddDelta(delta)
                            float_diff = floatdelta.FloatDelta(wt1, wt2, delta)
                            chry.append(float_diff)
                            #print(' diff : ', wt1, wt2, delta, end=' ')
                            if delta > 0:
                                lydelta.incrementDeltaCount()

                else:
                    #print(wdarray1)
                    print('')
            print(' layer diff count: ', lydelta.diffcount, " - total: ", lydelta.paramcount, " deltaSum: ", lydelta.deltasum)

            if len(weights1) == 2:
                # bias is just 1 array of floats
                arraylen = weights1[1].shape[0]
                print('  shape =', arraylen)
                bw1 = weights1[1].numpy()
                bw2 = weights2[1].numpy()
                deltas = []
                lydelta.biasarray = deltas
                for ix in range(arraylen):
                    w1 = bw1[ix]
                    w2 = bw2[ix]
                    delta = abs(w1 - w2)
                    float_diff = floatdelta.FloatDelta(w1, w2, delta)
                    deltas.append(float_diff)
                    lydelta.AddBiasDelta(delta)
                    lydelta.incrementBiasParamCount()
                    if delta > 0:
                        lydelta.incrementBiasDeltaCount()
            print(' bias diff count: ', lydelta.biasdiffcount, " - total: ", lydelta.biasparamcount, " deltaSum: ", lydelta.biasdeltasum)

            # for x in range(len(weights1)):
            #     print('  shape=', weights1[x].shape, '\n')
            #     nw1 = weights1[x].numpy()
            #     for y in range(len(nw1)):
            #         yarr = nw1[y]
            #         inspectArray(yarr,'  ')

    else:
        print('input layer - no need to diff')

# not sure there's a benefit to calculating diff for max pooling
# the purpose of pooling is to reduce the dimension, which in theory
# filters out noise and improves accuracy. The parameter count for
# maxpooling and flatten both are zero. The official Keras documentation
# page is https://keras.io/api/layers/pooling_layers/max_pooling2d/
def diffMaxPool(diff, index, layer1, layer2):
    print(" - maxpool size: ", layer1.pool_size)

def inspectArray(narrayobj, sep):
    if hasattr(narrayobj, "__len__"):
        print('[',end='')
        print(len(narrayobj),sep,end='')
        for z in range(len(narrayobj)):
            charray = narrayobj[z]
            inspectArray(charray,'')
        print('] ',end='')

""" Keras dense layer has weights and bias. Depending on the model configuration, the layer
might not have bias.
"""
def diffDense(diff, index, layer1, layer2):
    print(layer1.name)
    print(' - calculate diff for dense layer')
    # #print(layer1.weights)
    shape = layer1.weights[0].shape
    print('  - dense shape: ', shape)
    weights1 = layer1.weights
    weights2 = layer2.weights
    wlen = len(weights1)
    print('  weights len: ', wlen)
    denseDelta = layerdelta.DenseLayerDelta(index, layer1.name)
    denseDelta.height = shape[0]
    denseDelta.width = shape[1]
    diff.addLayerDelta(denseDelta)
    # dense layer weights has kernel and bias
    kshape = weights1[0].shape
    inputs = kshape[0]
    outputs = kshape[1]
    knarray1 = weights1[0]
    knarray2 = weights2[0]
    deltaarray = []
    denseDelta.AddArray(deltaarray)
    #print('  weights: ', weights1)
    #print('  kernarray: ', knarray1)
    for x in range(inputs):
        #print(' x: ', x, end=' ')
        dimarray1 = knarray1[x]
        dimarray2 = knarray2[x]
        dimensions = []
        deltaarray.append(dimensions)
        # defensive code to make sure it's an array
        if hasattr(dimarray1, "__len__"):
            nestlen = len(dimarray1)
            #print(' weights length: ', nestlen)
            for y in range(nestlen):
                wt1 = tf.get_static_value(dimarray1[y])
                wt2 = tf.get_static_value(dimarray2[y])
                dval = abs(wt1 - wt2)
                fldelta = floatdelta.FloatDelta(wt1, wt2, dval)
                dimensions.append(fldelta)
                #print('  -- delta: ', dval)
                denseDelta.incrementParamCount()
                denseDelta.AddDelta(dval)
                if dval > 0.0:
                    denseDelta.incrementDeltaCount()
    # the bias
    if len(weights1) > 1:
        arraylen = weights1[1].shape[0]
        bw1 = weights1[1].numpy()
        bw2 = weights2[1].numpy()
        deltas = []
        denseDelta.biasarray = deltas
        print('    - bias length: ', arraylen)
        for ix in range(arraylen):
            w1 = bw1[ix]
            w2 = bw2[ix]
            delta = abs(w1 - w2)
            float_diff = floatdelta.FloatDelta(w1, w2, delta)
            deltas.append(float_diff)
            denseDelta.AddBiasDelta(delta)
            denseDelta.incrementBiasParamCount()
            if delta > 0:
                denseDelta.incrementBiasDeltaCount()

    # for x in range(wlen):
    #     print('  shape=', weights1[x].shape, '\n')
    #     nw1 = weights1[x].numpy()
    #     for y in range(len(nw1)):
    #         yarr = nw1[y]
    #         inspectArray(yarr,'  ')

    #print('  dense delta: ', len(denseDelta.deltaarray), ' diffcount: ', denseDelta.diffcount)

def diffDropout(diff, index, layer1, layer2):
    print(' - calculate diff for dropout')
    print(' layer name: ', layer1.name)

def diffFlatten(diff, index, layer1, layer2):
    print(' - calculate diff for Flatten')
    print('  layer name: ', layer1.name)
    print('  flat weights: ', layer1.weights)
    

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main(sys.argv[0:])
