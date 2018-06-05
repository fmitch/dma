from __future__ import division

# import numpy as np
import tensorflow as tf
import numpy as np
import sys
from IPython import embed
import time
import matplotlib.pyplot as plt

learning_rate = 1e-4
training_epochs = 1000
display_step = 5
test_step = 10
batch_size = 250
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)


def shuffle(imgs, labels, seed = 1):
    np.random.seed(seed)
    permutation_index = np.random.permutation(range(0, len(imgs)))
    imgs = imgs[permutation_index]
    labels = labels[permutation_index]

    return imgs, labels

def next_batch(imgs, labels, batch, batch_size):
    print "batch: ", batch
    batch_xs = imgs[batch * batch_size : (batch + 1) * batch_size]
    batch_ys = labels[batch * batch_size: (batch + 1) * batch_size]
    return batch_xs, batch_ys

def next_test_batch(imgs, labels, batch, batch_size):
    print "test batch: ", batch
    batch_xs = imgs[batch * batch_size : (batch + 1) * batch_size]
    batch_ys = labels[batch * batch_size: (batch + 1) * batch_size]
    return batch_xs, batch_ys

def summary(labels):
    """ Output the summary infomation of the prediction to the standard output

    Args:
        prediction (list): prediction values

    Returns:
        summary: the summary of labels
    """

    summary = {}
    for label in labels:
        if label in summary:
            summary[label] = summary[label] + 1
        else:
            summary[label] = 1

    return summary

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
ds['training_data'] = np.reshape(np.kron( np.load('data/step4_40000.npy'), np.ones((2,2))), (-1,32,32,1))
ds['test_data'] = np.reshape(np.kron( np.load('data/step4_10000.npy'), np.ones((2,2))), (-1,32,32,1))
#ds['training_data'] = np.reshape(np.kron( np.load('data/batch16_40000.npy'), np.ones((2,2))), (-1,32,32,1))
ds['training_labels'] = np.load('data/res16_40000.npy')
#ds['test_data'] = np.reshape(np.kron( np.load('data/batch16_10000.npy'), np.ones((2,2))), (-1,32,32,1))
ds['test_labels'] = np.load('data/res16_10000.npy')
X = tf.placeholder("float", [None,32,32,1], name="input")
Y = tf.placeholder("int64", [None], name="labels")
labels = tf.one_hot(Y, 3)
#labels = tf.one_hot(Y, 3, axis=-1, name="targets", dtype="int64")
keep_prob = tf.placeholder("float")

# condigure the model
#with tf.name_scope("Strided-CNN"):
#    with tf.variable_scope("conv1"):
#        conv1, weights1 = conv_relu(X, kernel_shape=[3, 3, 3, 96], bias_shape=[96], stride=1) # # 3*3 filter, 3 input channel, 96 filters (output channel)
#        # conv1: ?,32,32,96
#    with tf.variable_scope("conv2"):
#        conv2, weights2 = conv_relu(conv1, kernel_shape=[3, 3, 96, 96], bias_shape=[96], stride=2) # # 3*3 filter, 96 input channel, 96 filters (output channel)
#        # conv2: ?,16,16,96
#    with tf.variable_scope("conv3"):
#        conv3, weights3 = conv_relu(conv2, kernel_shape=[3, 3, 96, 192], bias_shape=[192], stride=1) # # 3*3 filter, 96 input channel, 192 filters (output channel)
#        # conv3: ?,16,16,192
#    with tf.variable_scope("conv4"):
#        conv4, weights4 = conv_relu(conv3, kernel_shape=[3, 3, 192, 192], bias_shape=[192], stride=2) # # 3*3 filter, 192 input channel, 192 filters (output channel)
#        # conv4: ?,8,8,192
#
#    with tf.variable_scope("conv5"):
#        conv5, weights5 = conv_relu(conv4, kernel_shape=[3, 3, 192, 192], bias_shape=[192], stride=1, padding="VALID") # # 3*3 filter, 192 input channel, 192 filters (output channel)
#        # conv5: ?,6,6,192
#    with tf.variable_scope("conv6"):
#        conv6, weights6 = conv_relu(conv5, kernel_shape=[1, 1, 192, 192], bias_shape=[192], stride=1, padding="VALID") # # 1*1 filter, 192 input channel, 192 filters (output channel)
#        # conv6: ?,6,6,192
#    with tf.variable_scope("conv7"):
#        conv7, weights7 = conv_relu(conv6, kernel_shape=[1, 1, 192, 10], bias_shape=[10], stride=1, padding="VALID") # # 1*1 filter, 192 input channel, 10 filters (output channel)
#        # conv7: ?,6,6,10
#    with tf.variable_scope("gap"):
#        gap = global_average_pool_6x6(conv7);
#
#    # with tf.variable_scope("fcl1"):
#    #     conv1_flat = tf.reshape(conv1, [-1, 32 * 32 * 96])
#    #     output = fcl(conv1_flat, [32*32*96, 10], [10])
#    #     softmax = tf.nn.softmax(output)
#
#    with tf.variable_scope("softmax"):
#        gap_flat = tf.reshape(gap, [-1, 10]) # change the shape from ?,1,1,10 to ?,10
#        softmax = tf.nn.softmax(gap_flat)


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
sess = tf.Session()
sess.run(tf.global_variables_initializer()) # has to be here

# LSUV normalization
margin = 1e-6
max_iter = 10
layers_cnt = 0
layers = range(1, 8)
training_data_shuffled, training_labels_shuffled = shuffle(ds["training_data"], ds["training_labels"])
training_data_shuffled_normalized = training_data_shuffled / 255.0  # normalized
batch_LSUV = training_data_shuffled_normalized[0:batch_size]
for layerCnt in layers:
    layerName = "conv" + str(layerCnt)
    weightsName = "weights" + str(layerCnt)

    print('LSUV initializing', layerName)
    layers_cnt += 1

    # pre-initialize with orthonormal matrices
    s, u, v = tf.svd(tf.transpose(locals()[weightsName], [2,3,0,1]))
    assign_weight = (locals()[weightsName]).assign(tf.transpose(u, [2,3,0,1])) # update orthonormal weights
    output = sess.run(locals()[layerName], feed_dict={X: batch_LSUV, Y: training_labels_shuffled})  ## run one forward pass
    # mean, var = tf.nn.moments(locals()[layerName], axes=[0, 1, 2, 3])  # get the variance
    var = np.var(output)
    print("starting var is: ", var)

    iter = 0
    target_var = 1.0 # the targeted variance

    while (abs(target_var - var) > margin):
        # update weights based on the variance of the output
        weights_update = tf.assign(locals()[weightsName], tf.div(locals()[weightsName], tf.sqrt(var)))
        sess.run(weights_update, feed_dict={X: batch_LSUV})

        # mean, var = tf.nn.moments(locals()[layerName], axes=[0,1,2,3]) # get the variance

        output = sess.run(locals()[layerName],
                          feed_dict={X: batch_LSUV, Y: training_labels_shuffled})  ## run one forward pass
        # mean, var = tf.nn.moments(locals()[layerName], axes=[0, 1, 2, 3])  # get the variance
        var = np.var(output)
        print("cur var is: ", var)

        iter = iter + 1
        if iter > max_iter:
            break

print('LSUV: total layers initialized', layers_cnt)

total_batch = int(40000/batch_size)
saver = tf.train.Saver()
loss_bin = []
test_batch = 0
# train the model
while True:
  try:
    if len(sys.argv) > 2:
        saver.restore(sess, sys.argv[2])
    for epoch in range(training_epochs):
        training_data_shuffled, training_labels_shuffled = shuffle(ds["training_data"], ds["training_labels"], seed=epoch)

        # normalization
        training_data_shuffled_normalized = training_data_shuffled.astype('float32')
        training_data_shuffled_normalized = training_data_shuffled / 255.0  # normalized

        # loop over all batches
        print"epoch: ", epoch
        epoch_time = time.time()
        for batch in range(0, total_batch):
            batch_time = time.time()
            batch_xs, batch_ys = next_batch(training_data_shuffled_normalized, training_labels_shuffled, batch, batch_size) # Get data
            # print summary(batch_ys)

            # batch_ys_onehot = tf.one_hot(batch_ys, 10, axis=-1, name="targets", dtype="int64")
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
            # batch_ys = tf.one_hot(batch_ys, 10, axis=-1, name="targets", dtype="int64") # covert to one-hot
            if batch % display_step == 0:
                # conv1_activation = sess.run(conv1, feed_dict={X: batch_xs, Y: batch_ys})
                # print "conv1 activation: ", conv1_activation
                # print "conv5 activation: ", sess.run(conv5, feed_dict={X: batch_xs, Y: batch_ys})

                # conv1_weights = locals()["weights1"]
                # print conv1_weights.eval()[0][0][0]

                # conv6_weights = locals()["weights6"]
                # print conv6_weights.eval()[0][0][0]

                # pred = sess.run(softmax, feed_dict={X: batch_xs, Y: batch_ys})
                # print np.argmax(pred, axis = 1)
                loss, acc = sess.run([cost, accuracy], feed_dict={X: batch_xs, Y: batch_ys})
                loss_bin.append(loss)
                print("Epoch " + str(epoch) + ", Batch " + str(batch) + ", Minibatch Loss = " + str(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))

            if batch % test_step == 0:
                # Calculate accuracy on test data
                test_xs, test_ys = next_test_batch(ds['test_data'], ds['test_labels'], test_batch, 500)
                test_batch = (test_batch+1) % 20
                print("Testing Accuracy:",
                      sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys, keep_prob: 1.}))
            #print("Batch time: ", time.time() - batch_time)
        print("Epoch time: ", time.time() - epoch_time)
    print("Optimization Finished!")
    save_path = saver.save(sess, './' + str(epoch)+"epochs.ckpt")
    # Calculate accuracy on test data
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={X: ds["test_data"], Y: ds["test_labels"], keep_prob: 1.}))
    plt.plot(loss_bin)
    plt.show()
  except KeyboardInterrupt:
    if len(sys.argv) > 1:
        save_path = saver.save(sess, './' + sys.argv[1])
        print("Cut the training, saved as " + sys.argv[1])
    else:
        save_path = saver.save(sess, './model.ckpt')
        print("Cut the training, saved as model.ckpt")
    plt.plot(loss_bin)
    plt.show()
    sys.exit(0)
