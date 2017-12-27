'''
This file defines the architecture for the neural network model.
'''
import tensorflow as tf
from layers import *

image_h = 28
image_w = 28
image_ch = 1
output_cls = 10

# Input Layer:
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, shape=[None, image_h, image_w, image_ch])
    y_ = tf.placeholder(tf.float32, shape=[None, output_cls])

# Convolutional Layer 1::
with tf.name_scope('Conv1'):
    conv1 = Conv1(x)
    out_conv1 = conv1.forward()

# Convolutional Layer 2:
with tf.name_scope('Conv2'):
    conv2 = Conv2(out_conv1)
    out_conv2 = conv2.forward()

# Fully Connection Layer:
with tf.name_scope('fc'):
    fc = FC(out_conv2)
    out_fc = fc.forward()

# Readout Layer:
with tf.name_scope('Readout'):
    readout = Readout(out_fc, output_cls)
    out_readout = readout.forward()

# Loss and evaluation:
loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=out_readout)
        )
prediction = tf.argmax(out_readout, 1)
correct_prediction = tf.equal(tf.argmax(out_readout, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
