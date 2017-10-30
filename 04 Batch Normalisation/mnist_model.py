""" CNN model based from the TensorFlow tutorial for the MNIST dataset
    (https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def conv2d(x, weight, bias):
    """ Function for the convolutional layers.
    CNN layers with 1 x 1 stride and zero padding

    Args:
        x: input tensor for the CNN layer
        weight: tensor for the filter of the CNN
        bias: tensor for the bias of the CNN

    Returns:
        Tensor of the output of the layer
    """
    return tf.nn.conv2d(x, filter=weight, strides=[1,1,1,1], padding='SAME') + bias

def max_pool_2x2(x):
    """ Function for the max pool layer
    Max pool layer with 2 x 2 kernel size, 2 x 2 stride and zero padding.

    Args:
        x: input tensor for the max pool layer

    Returns:
        Tensor of the output of the layer
    """
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def variable_summaries(var):
    """ Function for the TensorBoard summaries
    Adds the mean, standard deviation and histogram to the TensorBoard summaries

    Args:
        var: tensor of variable to add to summaries
    """
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.histogram('histogram', var)


def inference(image, img_size, num_classes, keep_prob, train_phase):
    # Create graph
    with tf.variable_scope('conv1') as scope:
        image = tf.reshape(image, shape=[-1, img_size, img_size, 1])

        # Placeholder for weights and bias
        with tf.variable_scope('weight'):
            w = tf.get_variable('weights',
                                shape=[5, 5, 1, 32],
                                initializer=tf.truncated_normal_initializer())
            variable_summaries(w)

        with tf.variable_scope('bias'):
            b = tf.get_variable('bias',
                                shape=[32],
                                initializer=tf.truncated_normal_initializer())
            variable_summaries(b)

        conv1a = tf.nn.relu(conv2d(image, w, b), name=scope.name)
        conv1b = tf.contrib.layers.batch_norm(conv1a,
                                              center=True,
                                              scale=True,
                                              is_training=train_phase)

    with tf.variable_scope('pool1'):
        pool1 = max_pool_2x2(conv1b)
        # output dimension = 14 x 14 x 32

    with tf.variable_scope('conv2'):
        # Placeholder for weights and bias
        with tf.variable_scope('weight'):
            w = tf.get_variable('weights',
                                shape=[5, 5, 32, 64],
                                initializer=tf.truncated_normal_initializer())
            variable_summaries(w)

        with tf.variable_scope('bias'):
            b = tf.get_variable('bias',
                                shape=[64],
                                initializer=tf.truncated_normal_initializer())
            variable_summaries(b)

        conv2a = tf.nn.relu(conv2d(pool1, w, b), name=scope.name)
        conv2b = tf.contrib.layers.batch_norm(conv2a,
                                              center=True,
                                              scale=True,
                                              is_training=train_phase)

    with tf.variable_scope('pool2'):
        pool2 = max_pool_2x2(conv2b)
        # output dimension = 7 x 7 x 64

    # Flatten layer (for fully_connected layer)
    with tf.name_scope('flatten'):
        flat_dim = pool2.get_shape()[1].value * pool2.get_shape()[2].value * pool2.get_shape()[3].value
        flat = tf.reshape(pool2, [-1, flat_dim])

    with tf.variable_scope('fc1') as scope:
        # Placeholder for weights and bias
        with tf.variable_scope('weight'):
            w = tf.get_variable('weights', shape=[flat_dim, 1024],
                                initializer=tf.truncated_normal_initializer())
            variable_summaries(w)

        with tf.variable_scope('bias'):
            b = tf.get_variable('bias', shape=[1024],
                                initializer=tf.random_normal_initializer())
            variable_summaries(b)

        fc1a = tf.nn.relu(tf.matmul(flat, w) + b, name=scope.name)

        fc1b = tf.contrib.layers.batch_norm(fc1a,
                                            center=True,
                                            scale=True,
                                            is_training=train_phase)
        # Apply dropout
        fc1_drop = tf.nn.dropout(fc1b, keep_prob)

    with tf.variable_scope('softmax'):
        # Placeholder for weights and bias
        with tf.variable_scope('weight'):
            w = tf.get_variable('weights', shape=[1024, num_classes],
                                initializer=tf.truncated_normal_initializer())
            variable_summaries(w)

        with tf.variable_scope('bias'):
            b = tf.get_variable('bias', shape=[num_classes],
                                initializer=tf.random_normal_initializer())
            variable_summaries(b)

        y_conv = tf.add(tf.matmul(fc1_drop, w), b)
    return y_conv


def loss(labels, logits):
    with tf.name_scope('loss'):
        # Define loss
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='loss')
        tf.summary.scalar('loss', loss)
    return loss


def train(loss):
    # Define training method
    with tf.name_scope('train'):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss)
    return train_op


def evaluate(logits, labels):
    # Evaluate the model
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        with tf.name_scope('evaluate'):
            # Evaluate accuracy
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy
