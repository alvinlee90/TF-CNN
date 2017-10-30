import os
import tensorflow as tf
import mnist_model

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets('mnist_data', one_hot=True)

