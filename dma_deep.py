# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf
import generate_data as gd
import numpy as np

FLAGS = None

learning_rate = 0.1
EPOCHS = 1000
batch_size = 1000 
dropout_rate = 0.5
num_classes = 3
display_step = 2

def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 14, 14, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  #h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])

  h_pool2_flat = tf.reshape(h_conv2, [-1, 7*7*64])
  #h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 3])
  b_fc2 = bias_variable([3])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  #mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  train = np.load('train.npy')
  test = np.load('test.npy')
  train_labels = tf.one_hot(np.load('train_labels.npy'),3)
  test_labels = tf.one_hot(np.load('test_labels.npy'),3)

  #create dataset objects
  tr_data = tf.data.Dataset.from_tensor_slices((train, train_labels))
  val_data = tf.data.Dataset.from_tensor_slices((test, test_labels))


  iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                       tr_data.output_shapes)
  next_batch = iterator.get_next()

  # Ops for initializing the two different iterators
  training_init_op = iterator.make_initializer(tr_data)
  validation_init_op = iterator.make_initializer(val_data)
  


  # create TensorFlow placeholders
  x = tf.placeholder(tf.float32, shape=[None,14,14])
  y =  tf.placeholder(tf.float32, shape=[None,3])

  # shuffle the first `buffer_size` elements of the dataset
  tr_data = tr_data.shuffle(buffer_size=100000)

  # create a new dataset with batches of images
  tr_data = tr_data.batch(batch_size)


  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x)

  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  saver = tf.train.Saver()
  train_batches_per_epoch = int(np.floor(len(train)/batch_size))
  val_batches_per_epoch = int(np.floor(len(test) / batch_size))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tr_labels = train_labels.eval()
    for epoch in range(EPOCHS):
	  #Initialize iterator with training dataset
      sess.run(training_init_op)
      ind = 0
      for step in range(train_batches_per_epoch):
        #img_batch, label_batch = sess.run(next_batch)
        img_batch = train[ind:ind+batch_size]
        label_batch = tr_labels[ind:ind+batch_size]
        ind = ind +batch_size 
        #img_batch = np.reshape(img_batch, (1,14,14))
        #label_batch = np.reshape(label_batch, (1,3))
        sess.run(train_step, feed_dict={x: img_batch, y: label_batch, 
            keep_prob: dropout_rate})
        if step % display_step == 0:

            s = accuracy.eval( feed_dict={x: img_batch,
                y: label_batch, keep_prob: 1.} )
            print('step %d, training accuracy %g' % 
                (epoch*train_batches_per_epoch+step, s))
            #writer.add_summary(s, epoch*train_batches_per_epoch + step)

      print('test accuracy %g' % accuracy.eval(feed_dict={
          x: test, y: test_labels.eval(), keep_prob: 1.0}))
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)

if __name__ == '__main__':
  tf.app.run(main=main)
