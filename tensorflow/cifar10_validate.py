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

result = validationresult.ValidationResult()
batchsize = 16

# iterate over cifar10 test data and record the positive and false positive
# it takes a checkpoint file, iterates over the TFDataset for cifar10.
# it isn't generalized yet and  only works for Cifar10
def main(): 
    args = sys.argv[0:]
    print(args)
    if len(args) == 1:
        print('Example usage:')
        print('               python cifar_rest_report.py ./mymodel.hdf5')
    else:
        print(' start validation')
        # use cifar10 dataset
        (ds_train, ds_test), ds_info = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=False,
        as_supervised=True,
        with_info=True)

        print(ds_test)
        ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_test = ds_test.batch(batchsize)
        ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)
        modelfilename = args[1].replace("\\","/")
        model = tf.keras.models.load_model(modelfilename)
        result.checkpointfile = modelfilename
        print(ds_test)
        ds_test.cache()
        print(' validate against TEST')
        result.starttime = time.time()
        # iterate over tensorflow dataset
        validate = model.evaluate(ds_test, batch_size=128)
        result.endtime = time.time()
        elapsed = result.endtime - result.starttime
        print(' time: ', elapsed/60, ' minutes')
        print(validate)

        ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
        ds_train = ds_train.batch(batchsize)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        print(' validate against TRAIN')
        ds_train.cache()
        result.starttime = time.time()
        # iterate over tensorflow dataset
        validate = model.evaluate(ds_train, batch_size=128)
        result.endtime = time.time()
        elapsed = result.endtime - result.starttime
        print(' time: ', elapsed/60, ' minutes')
        print(validate)
        print(' ------------------- done -------------------')

        # save the results to json file

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

# this is the recommended approach of handling main function
if __name__ == "__main__":
    main()
