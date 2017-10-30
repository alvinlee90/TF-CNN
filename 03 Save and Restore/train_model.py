""" Train and export a convolutional neural networks (CNN)
    CNN model is from the TensorFlow tutorial for the MNIST dataset

    Based on the tutorial from TensorFlow
    (https://github.com/tensorflow/serving/blob/master/tensorflow_serving/example/mnist_saved_model.py)
"""

import os
import tensorflow as tf
import mnist_model

from tensorflow.examples.tutorials.mnist import input_data

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Download MNIST images and labels
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# Model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_epochs', 2,
                     'Number of training epochs.')
flags.DEFINE_float('drop_out', 0.5,
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


with tf.name_scope("data"):
    # Place holders for the input data and labels
    # Input image node (x): 2D tensor; batch size x flattened 28 x 28 MNIST image
    # Target output classes node (y_): 2D tensor; batch size x number of classes
    x = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS], name="image")
    y_ = tf.placeholder(tf.float32, shape=[None, N_CLASSES], name="label")

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

logits = mnist_model.inference(x, IMAGE_SIZE, N_CLASSES, keep_prob)

loss = mnist_model.loss(y_, logits)

train_op = mnist_model.train(loss)

accuracy = mnist_model.evaluate(y_, logits)

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

            validation_accuracy = accuracy.eval(feed_dict={
                x: mnist.validation.images,
                y_: mnist.validation.labels,
                keep_prob: 1.0
            })

            print('Training Step %d of %d | Training Accuracy: %g | Validation Accuracy: %g'
                  % (i, training_steps, train_accuracy, validation_accuracy))
            saver.save(sess, FLAGS.checkpoint_path + "/mnist_cnn", i)

        # Train the model, and write summaries.
        _, summary = sess.run([train_op, merged],
                              feed_dict={x: x_batch, y_: y_batch, keep_prob: FLAGS.drop_out})

        # Add summary to writer for TensorBoard
        writer.add_summary(summary, global_step=i)

    # Test the model
    print('Test | Accuracy: %g' % accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

    sess.close()
