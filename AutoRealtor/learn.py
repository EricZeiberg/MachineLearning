import os
import numpy as np
import tensorflow as tf

# Load them!
cwd = os.getcwd()
loadpath = cwd + "/processedData.npz"
l = np.load(loadpath)

# See what's in here
l.files

# Parse data
trainimg = l['trainimg']
trainlabel = l['trainlabel']
testimg = l['testimg']
testlabel = l['testlabel']
ntrain = trainimg.shape[0]
nclass = trainlabel.shape[1]
dim    = trainimg.shape[1]
ntest  = testimg.shape[0]

print ("%d train images loaded" % (ntrain))
print ("%d test images loaded" % (ntest))
print ("%d dimensional input" % (dim))
print ("%d classes" % (nclass))

# Define convolutional neural network architecture

# Parameters
learning_rate   = 0.001
training_epochs = 100
batch_size      = 100
display_step    = 10

# Network
n_input  = dim
n_output = nclass
weights  = {
    'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.1)),
    'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.1)),
    'wd1': tf.Variable(tf.random_normal([8*8*128, 1024], stddev=0.1)),
    'wd2': tf.Variable(tf.random_normal([1024, n_output], stddev=0.1))
}
biases   = {
    'bc1': tf.Variable(tf.random_normal([64], stddev=0.1)),
    'bc2': tf.Variable(tf.random_normal([128], stddev=0.1)),
    'bd1': tf.Variable(tf.random_normal([1024], stddev=0.1)),
    'bd2': tf.Variable(tf.random_normal([n_output], stddev=0.1))
}

def conv_basic(_input, _w, _b, _keepratio):
    # Input
    _input_r = tf.reshape(_input, shape=[-1, 30, 30, 1])

    # Conv1
    _conv1 = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(_input_r, _w['wc1'], strides=[1, 1, 1, 1], padding='SAME')
            , _b['bc1']))
    _pool1 = tf.nn.max_pool(_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    mean, var = tf.nn.moments(_pool1, [0, 1, 2])
    _pool1 = tf.nn.batch_norm_with_global_normalization(_pool1, mean, var, 1., 0., 1e-7, 0)
    _pool_dr1 = tf.nn.dropout(_pool1, _keepratio)

    # Conv2
    _conv2 = tf.nn.relu(tf.nn.bias_add(
            tf.nn.conv2d(_pool_dr1, _w['wc2'], strides=[1, 1, 1, 1], padding='SAME')
            , _b['bc2']))
    _pool2 = tf.nn.max_pool(_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    mean, var = tf.nn.moments(_pool2, [0, 1, 2])
    _pool2 = tf.nn.batch_norm_with_global_normalization(_pool2, mean, var, 1., 0., 1e-7, 0)
    _pool_dr2 = tf.nn.dropout(_pool2, _keepratio)

    # Vectorize
    _dense1 = tf.reshape(_pool_dr2, [-1, _w['wd1'].get_shape().as_list()[0]])

    # Fc1
    _fc1 = tf.nn.relu(tf.add(tf.matmul(_dense1, _w['wd1']), _b['bd1']))
    _fc_dr1 = tf.nn.dropout(_fc1, _keepratio)

    # Fc2
    _out = tf.add(tf.matmul(_fc_dr1, _w['wd2']), _b['bd2'])

    # Return everything
    out = {
        'input_r': _input_r,
        'conv1': _conv1,
        'pool1': _pool1,
        'pool1_dr1': _pool_dr1,
        'conv2': _conv2,
        'pool2': _pool2,
        'pool_dr2': _pool_dr2,
        'dense1': _dense1,
        'fc1': _fc1,
        'fc_dr1': _fc_dr1,
        'out': _out
    }
    return out


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_output])
keepratio = tf.placeholder(tf.float32)

# Functions!
_pred = conv_basic(x, weights, biases, keepratio)['out']
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(_pred, y))
optm = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
_corr = tf.equal(tf.argmax(_pred,1), tf.argmax(y,1)) # Count corrects
accr = tf.reduce_mean(tf.cast(_corr, tf.float32)) # Accuracy
init = tf.initialize_all_variables()

print ("Network Ready to Go!")

# Launch the graph
sess = tf.Session()
sess.run(init)

# Training cycle
for epoch in range(training_epochs):
    avg_cost = 0.
    num_batch = int(ntrain/batch_size)+1
    # Loop over all batches
    for i in range(num_batch):
        randidx = np.random.randint(ntrain, size=batch_size)
        batch_xs = trainimg[randidx, :]
        batch_ys = trainlabel[randidx, :]
        # Fit training using batch data
        sess.run(optm, feed_dict={x: batch_xs, y: batch_ys, keepratio:0.7})
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})/num_batch

    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        train_acc = sess.run(accr, feed_dict={x: batch_xs, y: batch_ys, keepratio:1.})
        print (" Training accuracy: %.3f" % (train_acc))
        test_acc = sess.run(accr, feed_dict={x: testimg, y: testlabel, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

print ("Optimization Finished!")
