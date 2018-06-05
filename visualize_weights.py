from math import sqrt
import tensorflow as tf
import sys
seed = 0

def put_kernels_on_grid (kernel, pad = 1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size, height, width, channels],
  #   where in this case batch_size == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x


#
# ... and somewhere inside "def train():" after calling "inference()"
#

# Visualize conv1 kernels
#with tf.variable_scope('conv1'):
#  tf.get_variable_scope().reuse_variables()
#  weights = tf.get_variable('weights')
#  grid = put_kernels_on_grid (weights)
#  tf.image.summary('conv1/kernels', grid, max_outputs=1)


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

X = tf.placeholder("float", [None,32,32,1], name="input")
Y = tf.placeholder("int64", [None], name="labels")
labels = tf.one_hot(Y, 3)
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

# start the session
#sess = tf.InteractiveSession()
sess = tf.Session()

# LSUV normalization

saver = tf.train.Saver()
loss_bin = []
test_batch = 0
# train the model
sess.run(tf.global_variables_initializer()) # has to be here
saver.restore(sess, sys.argv[1])
with tf.variable_scope(sys.argv[2]):
  tf.get_variable_scope().reuse_variables()
  weights = tf.get_variable('weights')
  grid = put_kernels_on_grid (weights)
  writer = tf.summary.FileWriter('./')
  summary_op = tf.summary.image('conv1/kernels', grid, max_outputs=1)
  summary = sess.run(summary_op)
  writer.add_summary(summary)
