from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

data_path = 'tfrecord/train.tfrecords'

# All classes for the dataset
classes = ['Blue Cube',
           'Blue Hollow Triangle',
           'Green Cube',
           'Green Hollow Cube',
           'Green Hollow Cylinder',
           'Orange Hollow Cross',
           'Orange Star',
           'Purple Hollow Cross',
           'Purple Star',
           'Red Hollow Cube',
           'Red Hollow Cylinder',
           'Red Sphere',
           'Yellow Cube',
           'Yellow Sphere']

with tf.Session() as sess:
    feature = {'image_raw': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.int64)}

    # Create a list of filenames and pass it to a queue
    filename_queue = tf.train.string_input_producer([data_path], num_epochs=10, shuffle=True)

    # Define a reader and read the next record
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # Decode the record read by the reader
    features = tf.parse_single_example(serialized_example, features=feature)

    # Convert the image data from string back to the numbers
    image = tf.decode_raw(features['image_raw'], tf.float32)

    # Cast label data into int32
    label = tf.cast(features['label'], tf.int32)

    # Reshape image data into the original shape
    image = tf.reshape(image, [64, 64, 3])

    # Data argumentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.adjust_brightness(image, 0.25)

    # Creates batches by randomly shuffling tensors
    images, labels = tf.train.shuffle_batch([image, label],
                                            batch_size=100,
                                            capacity=1300,
                                            num_threads=2,
                                            min_after_dequeue=1000)

    # Initialize all global and local variables
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess.run(init_op)

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for step in range(3):
        img, label = sess.run([images, labels])
        img = img.astype(np.uint8)

        for j in range(10):
            plt.subplot(2, 5, j + 1)
            plt.imshow(img[j, ...])
            plt.axis('off')
            for i, shape in enumerate(classes):
                if i == label[j]:
                    plt.title(shape)
        plt.show()

    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()
