'''
This file defines each layers for the neural network.
'''
import tensorflow as tf

# Convolutional Layer 1::
class Conv1:
    def __init__(self, input_data):
        _, _, _, input_ch = input_data.shape
        self.weights = tf.Variable(
                tf.truncated_normal([5, 5, int(input_ch), 32], stddev=0.1))
        self.conv_out = tf.nn.conv2d(
                input_data, self.weights, strides=[1,2,2,1], padding='SAME')
        self.relu_out = tf.maximum(self.conv_out, 0.0) # relu
        self.activation = tf.nn.max_pool(
                self.relu_out,
                ksize=[1,2,2,1],
                strides=[1,2,2,1],
                padding='SAME')

    def forward(self):
        return self.activation

# Convolutional Layer 2:
class Conv2:
    def __init__(self, input_data):
        _, _, _, input_ch = input_data.shape
        self.weights = tf.Variable(
                tf.truncated_normal([3, 3, int(input_ch), 32], stddev=0.1))
        self.conv_out = tf.nn.conv2d(
                input_data, self.weights, strides=[1,2,2,1], padding='SAME')
        self.relu_out = tf.maximum(self.conv_out, 0.0) # relu
        self.activation = tf.nn.max_pool(self.relu_out, ksize=[1,2,2,1],
                strides=[1,2,2,1], padding='SAME')

    def forward(self):
        return self.activation

# Fully Connection Layer:
class FC:
    def __init__(self, input_data):
        _, input_height, input_width, input_ch = input_data.shape
        d1 = int(input_height*input_width*input_ch)
        self.weights = tf.Variable(
                tf.truncated_normal([d1, 4096],
                    stddev=0.1)
        )
        self.input_flat = tf.reshape(input_data, [-1, d1])
        self.out_fc = tf.matmul(self.input_flat, self.weights)
        self.activation = tf.maximum(self.out_fc, 0.0) # relu

    def forward(self):
        return self.activation

# Readout Layer:
class Readout:
    def __init__(self, input_data, output_dim):
        _, input_dim = input_data.shape
        self.weights = tf.Variable(
            tf.truncated_normal([int(input_dim), output_dim], stddev=0.1)
        )
        self.out_readout = tf.matmul(input_data, self.weights)
        #self.activation = tf.nn.softmax(self.out_readout)
        self.activation = self.out_readout

    def forward(self):
        return self.activation
