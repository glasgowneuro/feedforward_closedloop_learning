from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
import ops as my_ops
import os
import re
import itertools as it


class visPredictor:
    def __init__(self, sess, args):

        self.sess = sess
        self.n_ffnet_hidden = args['n_ffnet_hidden']
        self.imgs_shape[0] = args['imgs_shape'][0]
        self.imgs_shape[1] = args['imgs_shape'][1]
        self.imgs_shape[2] = args['imgs_shape'][2]
        self.output = np.zeros(args['imgs_shape'][0] * args['imgs_shape'][1])
        self.actions = np.zeros(args['n_actions'])
        self.game_vars = np.zeros(args['n_game_vars'])

    def make_ffnet(self):
        n_ffnet_inputs = self.imgs_shape[0] * self.imgs_shape[1] * self.state_imgs_shape[2] + 16 + 6
        n_ffnet_outputs = self.state_imgs_shape[1] * self.state_imgs_shape[2]

        self.ffnet_input = tf.placeholder(tf.float32, shape=[None, n_ffnet_inputs])
        self.ffnet_output = tf.placeholder(tf.float32, shape=[None, n_ffnet_outputs])
        self.ffnet_target = tf.placeholder(tf.float32, shape=[None, n_ffnet_outputs])

        W_layer1 = self.weight_variable([n_ffnet_inputs, self.n_ffnet_hidden[0]])
        b_layer1 = self.bias_variable([self.n_ffnet_hidden[0]])

        W_layer2 = self.weight_variable([self.n_ffnet_hidden[0], self.n_ffnet_hidden[1]])
        b_layer2 = self.bias_variable([self.n_ffnet_hidden[1]])

        W_layer3 = self.weight_variable([self.n_ffnet_hidden[1], n_ffnet_outputs])
        b_layer3 = self.bias_variable([n_ffnet_outputs])

        h_1 = tf.nn.relu(tf.matmul(self.ffnet_input, W_layer1) + b_layer1)
        h_2 = tf.nn.relu(tf.matmul(h_1, W_layer2) + b_layer2)

        # dropout
        self.keep_prob = tf.placeholder(tf.float32)
        my_drop = tf.nn.dropout(h_2, self.keep_prob)
        print("output shape: ", self.ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        self.ffnet_output = tf.matmul(h_2, W_layer3) + b_layer3
        print("output shape: ", self.ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        self.loss = tf.squared_difference(self.ffnet_output, self.ffnet_target)

        self.ffnet_train_step = tf.train.AdamOptimizer(0).minimize(self.loss)

        self.accuracy = tf.reduce_mean(self.loss)
        #        sess.run(tf.global_variables_initializer())





