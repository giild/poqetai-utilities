from textwrap import indent
from numpy import ndarray
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import tensorflow_datasets as tfds
import validationresult
import JsonWriter

print(tf.__version__)

# iterate over cifar10 test data and record the positive and false positive
# it takes a checkpoint file, iterates over the TFDataset for cifar10.
# it isn't generalized yet and  only works for Cifar10
def main(): 
    args = sys.argv[0:]
    print(args)
    if len(args) == 1:
        print('Example usage:')
        print('               python cifar_detailed_test.py ./checkpoint_model.hdf5 rps_test_result.json 2')
    else:
        print(' start validation')
        # use cifar10 dataset
        modelfilename = args[1].replace("\\","/")
        resultfile = args[2]
        epoch = "1"
        if len(args) == 4:
            epoch = args[3]
        run(modelfilename, resultfile, epoch)

def run(modelfilename:str, resultfile:str, epochstr:str):
        result = validationresult.ValidationResult()
        result.epoch = epochstr
        dataset = tfds.load('cifar10', shuffle_files=False)
        print(dataset)
        ds_train = dataset['train']
        model = tf.keras.models.load_model(modelfilename)
        result.checkpointfile = modelfilename
        #print(ds_test)
        testcount = len(ds_train)
        #print(' test dataset count: ', testcount)
        ds_train.cache()
        counter = 0
        result.starttime = time.time()
        # iterate over tensorflow dataset
        for element in ds_train:
            #print(element)
            filename = element["id"].numpy().decode("utf-8")
            label = element["label"].numpy()
            #print(' ------------ filename --- ', filename)
            #print(' ------------ label ------', label)
            image = element["image"].numpy()
            #print(image)
            # get the prediction
            prediction = model.predict(image.reshape(1,32,32,3))
            im_class = tf.argmax(prediction[0], axis=-1).numpy()
            ##print(' ------------ the prediction  ----', im_class)
            # compare the prediction to the label
            if (im_class == label):
                result.addPositive(filename, label)
            else:
                result.addFalsePositive(filename, label, im_class)
            counter = counter + 1
            # if counter > 100:
            #     break
            #print(counter)
        result.endtime = time.time()
        elapsed = result.endtime - result.starttime
        print(' ------------------- done -------------------')
        print(' time: ', elapsed/60, ' minutes')
        print(' file: ', resultfile)
        #print(result.getSummary())
        # save the results to json file
        output = open(resultfile,"w")
        jsonstring = JsonWriter.writeValidationResult(result)
        output.write(jsonstring)
        output.close()

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
