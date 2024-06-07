import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys

def main(args):
    if len(args) == 1:
        print('To download Tensorflow Datasets, call the script with the dataset name')
        print('    python download_tensorflowdataset.py cifar100')
    else:
        print(' -- start downloading cifar10 ---')
        createDataset(args[1])
    
def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def createDataset(dsname):
        # create the dataset
        # the benchmark loads the CIFAR10 dataset from tensorflow datasets
    (train,test),info  = tfds.load(
        dsname,
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (train, test), info

if __name__ == "__main__":
    main(sys.argv)