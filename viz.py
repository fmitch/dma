from __future__ import division

# import numpy as np
import tensorflow as tf
import numpy as np
import imageio
import sys
import os
import math
from IPython import embed
import time
import matplotlib.pyplot as plt
import tf_cnnvis

learning_rate = 1e-4
training_epochs = 1000
display_step = 5
test_step = 10
batch_size = 250
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

def global_average_pool_6x6(x):
    """ Global average layer
    """

    # average pooling on 6*6 block (full size of the input feature map), for each input (first 1), for each feature map (last 1)
    return tf.nn.avg_pool(x, ksize=[1, 6, 6, 1],
                        strides=[1, 6, 6, 1], padding='SAME')

def conv_relu(x, kernel_shape, bias_shape, stride=1, padding="SAME"):
    """ Convolutional layer
    """

    # Create variable named "weights".
    weights = tf.get_variable("weights",
        shape=kernel_shape,
        initializer=tf.random_normal_initializer(mean = 0, stddev = 0.01, seed = seed))
    # Create variable named "biases".
    biases = tf.get_variable("biases",
        shape=bias_shape,
        initializer=tf.constant_initializer(0.0))
    conv = tf.nn.conv2d(x, filter=weights,
        strides=[1, stride, stride, 1],
        padding=padding,
        data_format="NHWC")

    return tf.nn.relu(tf.add(conv, biases)), weights

def fcl(x, input_size, output_size, dropout=0.0):
    """ Fully connected layer
    """

    # Create variable named "weights".
    weights = tf.get_variable("weights",
        # [input_size, output_size],
        input_size,
        initializer = tf.random_normal_initializer(seed = seed))
    # Create variable named "biases".
    biases = tf.get_variable("biases",
        output_size,
        initializer=tf.constant_initializer(0.0))
    return tf.nn.relu(tf.add(tf.matmul(x, weights), biases))



# load data
ds = {}
ds['test_data'] = np.reshape(np.kron( np.load('data/step2_10000.npy'), np.ones((2,2))), (-1,32,32,1))
ds['test_labels'] = np.load('data/res16_10000.npy')
length = ds['test_labels'].shape[0]
tmp = np.zeros((length, 3))
tmp[np.arange(length), ds['test_labels']] = 1
ds['test_labels'] = tmp
X = tf.placeholder("float", [None,32,32,1], name="input")
Y = tf.placeholder("int64", [None,3], name="labels")
labels = Y
#labels = tf.one_hot(Y, 3, axis=-1, name="targets", dtype="int64")
keep_prob = tf.placeholder("float")



with tf.name_scope("All-CNN"):
     with tf.variable_scope("conv1"):
         conv1, weights1 = conv_relu(X, kernel_shape=[3, 3, 1, 96], bias_shape=[96],
                                     stride=1)  # # 3*3 filter, 3 input channel, 96 filters (output channel)
         # conv1: ?,32,32,96
     with tf.variable_scope("conv2"):
         conv2, weights2 = conv_relu(conv1, kernel_shape=[3, 3, 96, 96], bias_shape=[96],
                                     stride=1)  # # 3*3 filter, 96 input channel, 96 filters (output channel)
         # conv2: ?,16,16,96
     with tf.variable_scope("conv3"):
         conv3, weights3 = conv_relu(conv2, kernel_shape=[3, 3, 96, 192], bias_shape=[192],
                                     stride=2)  # # 3*3 filter, 96 input channel, 192 filters (output channel)
         # conv3: ?,16,16,192
     with tf.variable_scope("conv4"):
         conv4, weights4 = conv_relu(conv3, kernel_shape=[3, 3, 192, 192], bias_shape=[192],
                                     stride=1)  # # 3*3 filter, 192 input channel, 192 filters (output channel)
         # conv4: ?,8,8,192
     with tf.variable_scope("conv5"):
         conv5, weights5 = conv_relu(conv4, kernel_shape=[3, 3, 192, 192], bias_shape=[192],
                                     stride=1)  # # 3*3 filter, 192 input channel, 192 filters (output channel)
         # conv5: ?,6,6,192
     with tf.variable_scope("conv6"):
         conv6, weights6 = conv_relu(conv5, kernel_shape=[3, 3, 192, 192], bias_shape=[192],
                                     stride=2)  # # 1*1 filter, 192 input channel, 192 filters (output channel)


         # conv6: ?,6,6,192
     with tf.variable_scope("conv7"):
         conv7, weights7 = conv_relu(conv6, kernel_shape=[3, 3, 192, 192], bias_shape=[192], stride=1,
                                     padding="VALID")  # # 1*1 filter, 192 input channel, 192 filters (output channel)
         # conv7: ?,6,6,10

     with tf.variable_scope("conv8"):
         conv8, weights8 = conv_relu(conv7, kernel_shape=[1, 1, 192, 192], bias_shape=[192], stride=1,
                                     padding="VALID")  # # 1*1 filter, 192 input channel, 192 filters (output channel)
         # conv8: ?,6,6,10

     with tf.variable_scope("conv9"):
         conv9, weights9 = conv_relu(conv8, kernel_shape=[1, 1, 192, 3], bias_shape=[3], stride=1,
                                     padding="VALID")  # # 1*1 filter, 192 input channel, 10 filters (output channel)
         # conv9: ?,6,6,10


     with tf.variable_scope("gap"):
         gap = global_average_pool_6x6(conv9);

     # with tf.variable_scope("fcl1"):
     #     conv1_flat = tf.reshape(conv1, [-1, 32 * 32 * 96])
     #     output = fcl(conv1_flat, [32*32*96, 10], [10])
     #     softmax = tf.nn.softmax(output)

     with tf.variable_scope("softmax"):
         gap_flat = tf.reshape(gap, [-1, 3])  # change the shape from ?,1,1,10 to ?,10
         softmax = tf.nn.softmax(gap_flat)

with tf.name_scope('cost'):
    # tf.nn.softmax_cross_entropy_with_logits internally applies the softmax on the model's unnormalized model prediction and sums across all classes,
    # and tf.reduce_mean takes the average over these sums.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax, labels=labels))

with tf.name_scope('optimizer'):
    #optimizer = tf.train.RMSPropOptimizer(0.1, decay=0.001, momentum=0.0, epsilon=1e-8,).minimize(cost) # train
     optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(softmax, 1), tf.argmax(labels, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
accuracy = tf.reduce_mean(correct_prediction)

# start the session
#sess = tf.InteractiveSession()
sess = tf.Session(graph=tf.get_default_graph())
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()
if len(sys.argv) > 1:
    saver.restore(sess, sys.argv[1])
else:
    saver.restore(sess, 'step2.ckpt')
# train the model
i = 3
img = np.reshape(ds['test_data'][i], (1,32,32,1))
lab = np.reshape(ds['test_labels'][i], (1,3))
acc = sess.run(accuracy, feed_dict={X: img, Y: lab, keep_prob: 1.})

# open a session and initialize graph variables
# CAVEAT: trained alexnet weights have been set as initialization values in the graph nodes.
#         For this reason visualization can be performed just after initialization
# activation visualization
layers = ['c']

start = time.time()
with sess.as_default():
# with sess_graph_path = None, the default Session will be used for visualization.
#    is_success = tf_cnnvis.activation_visualization(sess_graph_path = None, value_feed_dict = {X : img}, 
#                                          layers=layers, path_logdir=os.path.join("Log","AlexNet"), 
#                                          path_outdir=os.path.join("Output","AlexNet"))
    is_success = tf_cnnvis.deconv_visualization(sess_graph_path = None, value_feed_dict = {X : img}, 
                                      layers=layers, path_logdir=os.path.join("Log","AlexNet"), 
                                      path_outdir=os.path.join("Output","AlexNet"))
