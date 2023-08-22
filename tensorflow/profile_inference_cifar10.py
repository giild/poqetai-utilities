import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import numpy as np
import ActivationDetails as ad
import LayerActivation as la
import time
import JsonWriter

# to be safe, we purposely set CUDA devices to -1 and force it to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

print(tf.__version__)

details:ad.Activationdetails = None

def main(args):
    testImage = None
    dataset = "None"
    imagename = ''
    modelfile = None
    resultfile = None
    tfmodel = None
    activations = np.array([])

    if len(args) == 1:
        print('Example usage:')
        print('               python profile_activation.py mymode.h5 cifar10 test_09590 activation_result.json')
    else:
        print('Start the extract process from tensorflow dataset:')
        modelfile = args[1].replace("\\","/")
        dsname = "cifar10"
        imagename = args[2]
        resultfile = args[3]
        epoch = args[4]
        dataset = loadDataset(dsname)
        print(' args: ', args)
        extimg = extractImage(dataset, imagename)
        testImage = extimg[0]
        imglabel = extimg[1]
        tfmodel = loadModel(modelfile)
        # run inference and save the activations
        start_time = time.perf_counter()
        activations = runInference(tfmodel, testImage)
        end_time = time.perf_counter()
        print(f' --- inference time: {end_time - start_time:0.4f} seconds')
        print('done - diff: ', resultfile)
        details = ad.Activationdetails()
        details.modeFile = modelfile
        details.modelName = tfmodel.name
        details.epoch = epoch
        details.recordName = imagename
        details.recordLabel = imglabel
        details.documentType = "ProfileInference"
        details.modelType = "keras"
        tic = time.perf_counter()
        createReport(tfmodel, details, activations)
        toc = time.perf_counter();
        print(f' --- create report time: {toc - tic:0.4f} seconds')
        jsonstring = JsonWriter.writeActivationDetails(details)
        print(' -- saving file: ', resultfile)
        testout = open(resultfile,"w")
        testout.write(jsonstring)
        testout.close()


    return

def extractImage(tfdataset, testimg):
    if tfdataset != None:
        print(tfdataset)
        dset = tfdataset['test']
        # iterate through the dataset and find the image by the filename
        for data in dset:
            # get the image
            image = data['image']
            # get the label
            label = data['label'].numpy()
            # get the filename
            filename = data["id"].numpy().decode("utf-8")
            # convert the image to numpy array
            image = image.numpy()
            # check if the filename is the same as the imagefile
            if filename == testimg:
                print('image label: ', label)
                return [image, label]

def loadDataset(datasetname):
    # load the dataset
    print('  load the dataset ')
    return tfds.load(datasetname, shuffle_files=False)

def loadModel(modelfile):
    # load the model so we can test
    print('  load the model - ', modelfile)
    return tf.keras.models.load_model(modelfile)

def runInference(model, imagedata):
    # run inference on the 
    print('  iterate over the layers to run prediction ')
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

    inputdata = imagedata.reshape(1,32,32,3)
    activations = activation_model.predict(inputdata)
    first_layer = activations[0]
    last_layer = activations[13]
    #print(last_layer)
    return activations

def createReport(model, details, activations):
    print(' create ActivationDetails object and populate it')
    for l in range(len(activations)):
        layer = model.layers[l]
        act = activations[l]
        lsh = str(layer.output.shape)
        #lsh = lsh.replace(" ","")
        layeractivation = la.LayerActivation()
        layeractivation.layerName = layer.name
        layeractivation.shape = lsh
        layeractivation.layerType = layer.__class__.__name__
        layeractivation.layerIndex = l
        layeractivation.activations = act
        details.addLayerActivation(layeractivation)

if __name__ == '__main__':
    main(sys.argv)