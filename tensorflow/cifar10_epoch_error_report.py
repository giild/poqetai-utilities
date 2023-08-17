from textwrap import indent
from numpy import ndarray
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys
import tensorflow_datasets as tfds
import csv

print(tf.__version__)

# iterate over rockpaperscissor test data and record the positive and false positive
# it takes a checkpoint file, iterates over the TFDataset for rockpaperscissor.
# it isn't generalized yet and  only works for rockpaperscissor
def main(): 
    args = sys.argv[0:]
    print(args)
    if len(args) == 1:
        print('Example usage:')
        print('               python cifar10_epochs_report.py ./checkpoint_model.hdf5 results.csv')
    else:
        print(' start validation')
        # use cifar10 dataset
        modelfilename = args[1].replace("\\","/")
        resultfile = args[2].replace("\\","/")
        run(modelfilename, resultfile)

def run(modelfilename:str, resultfile:str):
    dataset = tfds.load('cifar10', shuffle_files=False)
    print(dataset)
    ds_test = dataset['test']
    model = tf.keras.models.load_model(modelfilename)
    ds_test.cache()
    counter = 0
    correct = 0
    errorpercent = 0.0
    # iterate over tensorflow dataset
    start = time.time()
    with open(resultfile, 'w', newline='') as csvfile:
        header = ["record", "label", "prediction", "accuracy"]
        csvwriter = csv.DictWriter(csvfile, fieldnames=header)
        csvwriter.writeheader()
        for element in ds_test:
            filename = element["id"].numpy().decode('utf-8')
            label = element["label"].numpy()
            acc = "0"
            image = element["image"].numpy()
            # get the prediction
            prediction = model.predict(image.reshape(1,32,32,3), verbose=False)
            im_class = tf.argmax(prediction[0], axis=-1).numpy()
            # compare the prediction to the label
            if (im_class == label):
                acc = "1"
                correct = correct + 1
            counter = counter + 1
            csvwriter.writerow({'record':filename, 'label':label, 'prediction': im_class, 'accuracy':acc})
    end = time.time()
    elapsed = end - start
    errorpercent = ((counter - correct) / counter) * 100
    print(' ------------------- done -------------------')
    print(' correct prediction: ', correct, ' - total: ', counter)
    print(' time: ', elapsed/60, ' minutes')
    print(' file: ', resultfile)
    return [counter, correct, errorpercent]

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
