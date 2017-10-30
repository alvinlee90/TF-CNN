""" Using convolutional neural networks (CNN) on the MNIST dataset

    Based on the tutorial from TensorFlow (https://www.tensorflow.org/get_started/mnist/pros)
    and Stanford tutorials for the class of CS20SI: "TensorFlow for Deep Learning Research"
    (https://github.com/chiphuyen/stanford-tensorflow-tutorials/tree/master/examples)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#  Download MNIST images and labels
mnist = input_data.read_data_sets('mnist_data', one_hot=True, validation_size=0)

# Model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-3,
                   'Initial learning rate.')
flags.DEFINE_integer('n_epochs', 10,
                     'Number of training epochs.')
flags.DEFINE_float('drop_out', 0.75,
                   'Keep rate for drop out')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Batch size for training.')
flags.DEFINE_string('tensorboard_path', 'tmp',
                    'Directory to put TensorBoard summaries of the training data.')
flags.DEFINE_string('checkpoint_path', 'ckpt',
                    'Directory to save/load the checkpoints of the model')

# Data parameters
N_CLASSES = 10
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

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

with tf.name_scope("data"):
    # Place holders for the input data and labels
    # Input image node (x): 2D tensor; batch size x flattened 28 x 28 MNIST image
    # Target output classes node (y_): 2D tensor; batch size x number of classes
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name="image")
    y_ = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name="label")

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
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
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

    y_conv = td.add(tf.matmul(fc1_drop, w), b)

with tf.name_scope('loss'):
    # Define loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), name='loss')
    tf.summary.scalar('loss', loss)

# Evaluate the model
with tf.name_scope('evaluate'):
    # Evaluate accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Define training method
with tf.name_scope('train'):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    # Set checkpoint path and restore checkpoint if exists
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, save_path=ckpt.model_checkpoint_path)
        print('Loaded model from latest checkpoint')

    # To visualise training on TensorBoard
    os.makedirs(FLAGS.tensorboard_path, exist_ok=True)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(FLAGS.tensorboard_path, sess.graph)

    initial_step = global_step.eval()
    steps_per_epoch = int(mnist.train.num_examples / FLAGS.batch_size)
    training_steps = steps_per_epoch * FLAGS.n_epochs + initial_step

    # Train the model n_epochs times
    for i in range(initial_step, training_steps):
        x_batch, y_batch = mnist.train.next_batch(FLAGS.batch_size)

        # Evaluate training accuracy
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x: x_batch, y_: y_batch, keep_prob: 1.0})
            print('Training Step %d of %d | Training Accuracy: %g' % (i, training_steps, train_accuracy))
            saver.save(sess, FLAGS.checkpoint_path + "/mnist_cnn", i)

        # Train the model, and write summaries.
        _, summary = sess.run([optimizer, merged], feed_dict={x: x_batch, y_: y_batch, keep_prob: FLAGS.drop_out})

        # Add summary to writer for TensorBoard
        writer.add_summary(summary, global_step=i)

    # Test the model
    print('Test | Accuracy: %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    sess.close()
