import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
import sys

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

def createDataset(batchsize):
        # create the dataset
        # the benchmark loads the CIFAR10 dataset from tensorflow datasets
    (train,test),info  = tfds.load(
        'cifar10',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    train = train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.cache()
    train = train.shuffle(info.splits['train'].num_examples)
    train = train.batch(batchsize)
    train = train.prefetch(tf.data.experimental.AUTOTUNE)


    test = test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test.batch(batchsize)
    test = test.cache()
    test = test.prefetch(tf.data.experimental.AUTOTUNE)
    return (train, test), info
