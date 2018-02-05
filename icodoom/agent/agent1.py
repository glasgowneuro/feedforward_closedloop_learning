from __future__ import print_function
import numpy as np
import time
import tensorflow as tf
import ops as my_ops
import os
import re
import itertools as it

class Agent:

    def __init__(self, sess, args):
        '''Agent - powered by neural nets, can infer, act, train, test.
        '''
        self.sess = sess
        
        # input data properties
        self.state_imgs_shape = args['state_imgs_shape']
        self.state_meas_shape = args['state_meas_shape']
        self.meas_for_net = args['meas_for_net']
        self.meas_for_manual = args['meas_for_manual']
        self.resolution = args['resolution']

        # preprocessing
        self.preprocess_input_images = args['preprocess_input_images']

        # net parameters
        self.conv_params = args['conv_params']
        self.fc_img_params = args['fc_img_params']
        self.fc_meas_params = args['fc_meas_params']
        self.fc_joint_params = args['fc_joint_params']      
        self.target_dim = args['target_dim']

        self.n_ffnet_input = args['n_ffnet_input']
        self.n_ffnet_hidden = args['n_ffnet_hidden']
        self.n_ffnet_output = args['n_ffnet_output']
        self.learning_rate = args['learning_rate']
        self.momentum = args['momentum']
        self.ext_ffnet_output = np.zeros(self.n_ffnet_output)
        self.ext_covnet_output = np.zeros(self.n_ffnet_output)
        self.ext_fcnet_output = np.zeros(self.n_ffnet_output)
        print ("ffnet_inputs: ", args['n_ffnet_input'])
        print ("ffnet_hidden: ", args['n_ffnet_hidden'])
        print ("ext_ffnet_output: ", self.ext_ffnet_output.shape)

        self.ffnet_input = tf.placeholder(tf.float32, shape=[None, self.n_ffnet_input])
        self.ffnet_target = tf.placeholder(tf.float32, shape=[None, self.n_ffnet_output])
        self.covnet_input = tf.placeholder(tf.float32, shape=[None, self.n_ffnet_input])
        self.covnet_target = tf.placeholder(tf.float32, shape=[None, self.n_ffnet_output])
        self.fcnet_input = tf.placeholder(tf.float32, shape=[None, self.n_ffnet_input])
        self.fcnet_target = tf.placeholder(tf.float32, shape=[None, self.n_ffnet_output])

        self.build_model()
        self.epoch = 20
        self.iter = 1
        
    def make_convnet(self):
        n_ffnet_inputs = self.n_ffnet_input
        n_ffnet_outputs = self.n_ffnet_output

        print("COVNET: Inputs: ", n_ffnet_inputs, " outputs: ", n_ffnet_outputs)

        with tf.name_scope('reshape'):
            x_image = tf.reshape(self.covnet_input, [-1, self.resolution[0], self.resolution[1], 1])

        with tf.name_scope('conv1'):
            W_conv1 = my_ops.weight_variable([5, 5, 1, 32])
            b_conv1 = my_ops.bias_variable([32])
            h_conv1 = tf.nn.relu(my_ops.conv2d(x_image, W_conv1) + b_conv1)

        with tf.name_scope('pool1'):
            h_pool1 = my_ops.max_pool_2x2(h_conv1)

        with tf.name_scope('conv2'):
            W_conv2 = my_ops.weight_variable([5, 5, 32, 64])
            b_conv2 = my_ops.bias_variable([64])
            h_conv2 = tf.nn.relu(my_ops.conv2d(h_pool1, W_conv2) + b_conv2)

        with tf.name_scope('pool2'):
            h_pool2 = my_ops.max_pool_2x2(h_conv2)

        with tf.name_scope('fc1'):
            W_fc1 = my_ops.weight_variable([int(self.resolution[0]/4) * int(self.resolution[1]/4) * 64, 64])
            b_fc1 = my_ops.bias_variable([64])

            h_pool2_flat = tf.reshape(h_pool2, [-1, int(self.resolution[0]/4) * int(self.resolution[1]/4) * 64])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # single output:
        with tf.name_scope('fc2'):
            W_fc2 = my_ops.weight_variable([64, 1])
            b_fc2 = my_ops.bias_variable([1])

        self.y_conv = tf.tanh(tf.matmul(h_fc1, W_fc2) + b_fc2)
        self.covloss = tf.squared_difference(self.y_conv, self.covnet_target)
        self.covnet_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.covloss)
        self.covaccuracy = tf.reduce_mean(self.covloss)

    def make_fcnet(self):
        n_ffnet_inputs = self.n_ffnet_input
        n_ffnet_outputs = self.n_ffnet_output

        print("FCNET: Inputs: ", n_ffnet_inputs, " outputs: ", n_ffnet_outputs)

        W_fc1 = my_ops.weight_variable([n_ffnet_inputs, 8], 0.003)
        b_fc1 = my_ops.bias_variable([8])

        W_fc2 = my_ops.weight_variable([8, 2], 0.003)
        b_fc2 = my_ops.bias_variable([2])

        W_fc3 = my_ops.weight_variable([2, 1], 0.003)
        b_fc3 = my_ops.bias_variable([1])

        h1 = tf.tanh(tf.matmul(self.fcnet_input, W_fc1) + b_fc1)
        h2 = tf.tanh(tf.matmul(h1, W_fc2) + b_fc2)
        self.y_fc = tf.tanh(tf.matmul(h2, W_fc3) + b_fc3)

        self.fcloss = tf.squared_difference(self.y_fc, self.fcnet_target)
        self.fcnet_train_step = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.fcloss)
        self.fcaccuracy = tf.reduce_mean(self.fcloss)

    def make_ffnet(self):


        n_ffnet_inputs = self.n_ffnet_input
        n_ffnet_outputs = self.n_ffnet_output
        print ("FFNET: in: ", n_ffnet_inputs, " hid: ", self.n_ffnet_hidden, " out: ", n_ffnet_outputs)


        W_layer1 = my_ops.weight_variable([n_ffnet_inputs, self.n_ffnet_hidden[0]])
        b_layer1 = my_ops.bias_variable([self.n_ffnet_hidden[0]])

        W_layer2 = my_ops.weight_variable([self.n_ffnet_hidden[0], self.n_ffnet_hidden[1]])
        b_layer2 = my_ops.bias_variable([self.n_ffnet_hidden[1]])

        W_layer3 = my_ops.weight_variable([self.n_ffnet_hidden[1], n_ffnet_outputs])
        b_layer3 = my_ops.bias_variable([n_ffnet_outputs])

        h_1 = tf.nn.relu(tf.matmul(self.ffnet_input, W_layer1) + b_layer1)
        h_2 = tf.nn.relu(tf.matmul(h_1, W_layer2) + b_layer2)

        # dropout
        #print("output shape: ", self.ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        #print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        self.ffnet_output = tf.matmul(h_2, W_layer3) + b_layer3
        #print("output shape: ", self.ffnet_output.get_shape(), "target shape: ", self.ffnet_target.get_shape())
        #print("W3: ", W_layer3.get_shape(), " bias3: ", b_layer3.get_shape())

        self.loss = tf.squared_difference(self.ffnet_output, self.ffnet_target)

        self.ffnet_train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.accuracy = tf.reduce_mean(self.loss)
#        sess.run(tf.global_variables_initializer())

    def make_net(self, input_images, input_measurements, input_actions, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        
        self.fc_val_params = np.copy(self.fc_joint_params)
        self.fc_val_params['out_dims'][-1] = self.target_dim
        self.fc_adv_params = np.copy(self.fc_joint_params)
        self.fc_adv_params['out_dims'][-1] = len(self.net_discrete_actions) * self.target_dim
        print(len(self.net_discrete_actions) * self.target_dim)
        p_img_conv = my_ops.conv_encoder(input_images, self.conv_params, 'p_img_conv', msra_coeff=0.9)
        print ("Conv Params: ", self.conv_params)

        p_img_fc = my_ops.fc_net(my_ops.flatten(p_img_conv), self.fc_img_params, 'p_img_fc', msra_coeff=0.9)
        print ("img_params", self.fc_img_params)
        p_meas_fc = my_ops.fc_net(input_measurements, self.fc_meas_params, 'p_meas_fc', msra_coeff=0.9)
        print ("meas_params", self.fc_meas_params)
        p_val_fc = my_ops.fc_net(tf.concat(1, [p_img_fc,p_meas_fc]), self.fc_val_params, 'p_val_fc', last_linear=True, msra_coeff=0.9)
        print ("val_params", self.fc_val_params)
        p_adv_fc = my_ops.fc_net(tf.concat(1, [p_img_fc,p_meas_fc]), self.fc_adv_params, 'p_adv_fc', last_linear=True, msra_coeff=0.9)
        print ("adv_params", self.fc_adv_params)

        p_adv_fc_nomean = p_adv_fc - tf.reduce_mean(p_adv_fc, reduction_indices=1, keep_dims=True)  
        
        self.pred_all_nomean = tf.reshape(p_adv_fc_nomean, [-1, len(self.net_discrete_actions), self.target_dim])
        self.pred_all = self.pred_all_nomean + tf.reshape(p_val_fc, [-1, 1, self.target_dim])
        self.pred_relevant = tf.boolean_mask(self.pred_all, tf.cast(input_actions, tf.bool))
        print ("make_net: input_actions: ", input_actions)
        print ("make_net: pred_all: ", self.pred_all)
        print ("make_net: pred_relevant: ", self.pred_relevant)

    def build_model(self):

        #self.make_ffnet()
        #self.make_convnet()
        self.make_fcnet()
        self.saver = tf.train.Saver()
        tf.initialize_all_variables().run(session=self.sess)
    
    def act(self, state_imgs, state_meas, objective):
        return self.postprocess_actions(self.act_net(state_imgs, state_meas, objective), self.act_manual(state_meas)), None # last output should be predictions, but we omit these for now

    def act_ffnet(self, in_image, in_meas, target):

        net_inputs = in_image
        net_targets = target

        with self.sess.as_default():
            self.ext_ffnet_output, hack = self.sess.run([self.ffnet_output, self.ffnet_train_step], feed_dict={
                self.ffnet_input: net_inputs,
                self.ffnet_target: net_targets
            })

            if self.iter % self.epoch == 0:
                print ("LOSS: ", self.accuracy.eval(feed_dict={
                    self.ffnet_input: net_inputs,
                    self.ffnet_target: net_targets
                }))

        self.iter = self.iter+1

    def act_covnet(self, in_image, in_meas, target):

        net_inputs = in_image
        net_targets = target

        with self.sess.as_default():
            self.sess.run([self.covnet_train_step], feed_dict={
                self.covnet_input: net_inputs,
                self.covnet_target: net_targets
            })
            self.ext_covnet_output = self.sess.run([self.y_conv], feed_dict={
                self.covnet_input: net_inputs,
                self.covnet_target: net_targets
            })[0]

            if self.iter % self.epoch == 0:
                print ("LOSS: ", self.covaccuracy.eval(feed_dict={
                    self.covnet_input: net_inputs,
                    self.covnet_target: net_targets
                }))

        self.iter = self.iter+1

    def act_fcnet(self, in_image, in_meas, target):

        net_inputs = in_image
        net_targets = target

        if (in_meas[1] > 0):

            with self.sess.as_default():
                self.sess.run([self.fcnet_train_step], feed_dict={
                    self.fcnet_input: net_inputs,
                    self.fcnet_target: net_targets
                })
                self.ext_fcnet_output = self.sess.run([self.y_fc], feed_dict={
                    self.fcnet_input: net_inputs,
                    self.fcnet_target: net_targets
                })[0]

                if self.iter % self.epoch == 0:
                    print ("LOSS: ", self.fcaccuracy.eval(feed_dict={
                        self.fcnet_input: net_inputs,
                        self.fcnet_target: net_targets
                    }))

            self.iter = self.iter+1


    def act_net(self, state_imgs, state_meas, objective):
        #Act given a state and objective
        predictions = self.sess.run(self.pred_all, feed_dict={self.input_images: state_imgs, 
                                                            self.input_measurements: state_meas[:,self.meas_for_net]})
        #print (predictions)

        objectives = np.sum(predictions[:,:,objective[0]]*objective[1][None,None,:], axis=2)    
        curr_action = np.argmax(objectives, axis=1)
#        print (" ** ACTION ", curr_action)
        return curr_action

    # act_manual is a purely hard-coded method to handle weapons-switching
    def act_manual(self, state_meas):
        if len(self.meas_for_manual) == 0:
            return []
        else:
            assert(len(self.meas_for_manual) == 13) # expected to be [AMMO2 AMMO3 AMMO4 AMMO5 AMMO6 AMMO7 WEAPON2 WEAPON3 WEAPON4 WEAPON5 WEAPON6 WEAPON7 SELECTED_WEAPON]
            assert(self.num_manual_controls == 6) # expected to be [SELECT_WEAPON2 SELECT_WEAPON3 SELECT_WEAPON4 SELECT_WEAPON5 SELECT_WEAPON6 SELECT_WEAPON7]

            curr_act = np.zeros((state_meas.shape[0],self.num_manual_controls), dtype=np.int)
            for ns in range(state_meas.shape[0]):
                # always pistol
                #if not state_meas[ns,self.meas_for_manual[12]] == 2:
                    #curr_act[ns, 0] = 1
                # best weapon
                curr_ammo = state_meas[ns,self.meas_for_manual[:6]]
                curr_weapons = state_meas[ns,self.meas_for_manual[6:12]]
                #print(curr_ammo,curr_weapons)
                available_weapons = np.logical_and(curr_ammo >= np.array([1,2,1,1,1,40]), curr_weapons)
                if any(available_weapons):
                    best_weapon = np.nonzero(available_weapons)[0][-1]
                    if not state_meas[ns,self.meas_for_manual[12]] == best_weapon+2:
                        curr_act[ns, best_weapon] = 1
            return curr_act

    def save(self, checkpoint_dir, iter):
        self.save_path = self.saver.save(self.sess, checkpoint_dir, global_step=iter)
        print ("saving model file: ", self.save_path)



    def load(self, checkpoint_dir):
        self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))
        return True
