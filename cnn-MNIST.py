""" Using convolutional neural networks (CNN) on the MNIST dataset

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(0)

#  Download MNIST images and labels
mnist = input_data.read_data_sets('mnist_data', one_hot=True, validation_size=0)

# Define parameters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
DROP_OUT = 0.75
N_EPOCHS = 1

# Data parameters
N_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

# Function for the convolutional layers (1 x 1 stride with padding)
def conv2d(x, weight, bias):
    return tf.nn.conv2d(x, filter=weight, strides=[1,1,1,1], padding='SAME') + bias

# Function for the max pool layer (2 x 2 filter with 2 x 2 stride and padding)
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# Function for the Tensorboard summaries
def variable_summaries(var):
    mean = tf.reduce_mean(var)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('mean', mean)
    tf.summary.scalar('stddev', stddev)
    tf.summary.histogram('histogram', var)

# Place holders for the input data and labels
# Input image node: 2D tensor; batch size x flattened 28 x 28 MNIST image
# Target output classes node: 2D tensor; batch size x number of classes
with tf.name_scope("data"):
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name="image")
    y_ = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name="label")

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# Create graph
with tf.variable_scope('conv1') as scope:
    image = tf.reshape(x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 1])

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

    conv1 = tf.nn.relu(conv2d(image, w, b), name=scope.name)

with tf.variable_scope('pool1'):
    pool1 = max_pool_2x2(conv1)
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

    conv2 = tf.nn.relu(conv2d(pool1, w, b), name=scope.name)

with tf.variable_scope('pool2'):
    pool2 = max_pool_2x2(conv2)
    # output dimension = 7 x 7 x 64

with tf.variable_scope('fc1') as scope:
    input_dim = 7 * 7 * 64

    # Flatten last convolutional layer
    pool2_flat = tf.reshape(pool2, [-1, input_dim])

    # Placeholder for weights and bias
    with tf.variable_scope('weight'):
        w = tf.get_variable('weights', shape=[input_dim, 1024],
                            initializer=tf.truncated_normal_initializer())
        variable_summaries(w)

    with tf.variable_scope('bias'):
        b = tf.get_variable('bias', shape=[1024],
                            initializer=tf.random_normal_initializer())
        variable_summaries(b)

    fc1 = tf.nn.relu(tf.matmul(pool2_flat, w) + b, name=scope.name)

    # Apply dropout
    fc1_drop = tf.nn.dropout(fc1, keep_prob)

with tf.variable_scope('softmax'):
    # Placeholder for weights and bias
    with tf.variable_scope('weight'):
        w = tf.get_variable('weights', shape=[1024, N_CLASSES],
                            initializer=tf.truncated_normal_initializer())
        variable_summaries(w)

    with tf.variable_scope('bias'):
        b = tf.get_variable('bias', shape=[N_CLASSES],
                            initializer=tf.random_normal_initializer())
        variable_summaries(b)

    y_conv = tf.matmul(fc1_drop, w) + b

with tf.name_scope('loss'):
    # Define loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name='loss')

# Evaluate the model
with tf.name_scope('evaluate'):
    # Evaluate accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define training method
optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE).minimize(loss, global_step=global_step)

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # To visualise training on Tensorboard
    tensorboard_path = 'tmp'
    os.makedirs(tensorboard_path, exist_ok=True)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_path, sess.graph)

    initial_step = global_step.eval()
    steps_per_epoch = int(mnist.train.num_examples / BATCH_SIZE)
    training_steps = steps_per_epoch * N_EPOCHS + initial_step

    # Train the model n_epochs times
    for i in range(initial_step, training_steps):
        x_batch, y_batch = mnist.train.next_batch(BATCH_SIZE)

        # Evaluate training accuracy
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: x_batch, y_: y_batch, keep_prob: 1.0})
            print('Training Step %d of %d | Training Accuracy: %g' % (i, training_steps, train_accuracy))

        # Train the model, and write summaries.
        _, summary = sess.run([optimizer, merged], feed_dict={x: x_batch, y_: y_batch, keep_prob: DROP_OUT})

        # Add summary to writer for Tensorboard
        writer.add_summary(summary, global_step=i)

    # Test the model
    print('Test | Accuracy: %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    sess.close()
