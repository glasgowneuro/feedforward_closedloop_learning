import numpy as np 
import tensorflow as tf

get_stddev = lambda x, k_h, k_w: 1/np.sqrt(0.5*k_w*k_h*x.get_shape().as_list()[-1])

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, sd):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=sd)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def linear(input_, output_size, name='linear', msra_coeff=1):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable("w", [shape[1], output_size], tf.float32,
                                tf.random_normal_initializer(stddev=msra_coeff * get_stddev(input_, 1, 1)))
        b = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + b
    
def conv_encoder(data, params, name, msra_coeff=1):
    layers = []
    for nl, param in enumerate(params):
        print ("nl", nl)
        print ("conv_encoder: nl: ", nl, " param out_channels: ", param['out_channels'], " param kernel: ", param['kernel'], " param stride: ", param['stride'])
        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
            
        layers.append(lrelu(conv2d(curr_inp, param['out_channels'], k_h=param['kernel'], k_w=param['kernel'], d_h=param['stride'], d_w=param['stride'], name=name + str(nl), msra_coeff=msra_coeff)))
        
    return layers[-1]

#Fully-connected network
def fc_net(data, params, name, last_linear = False, return_layers = [-1], msra_coeff=1):
    layers = []

    for nl, param in enumerate(params):

        if len(layers) == 0:
            curr_inp = data
        else:
            curr_inp = layers[-1]
        
        if nl == len(params) - 1 and last_linear:
            layers.append(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff))
        else:
            layers.append(lrelu(linear(curr_inp, param['out_dims'], name=name + str(nl), msra_coeff=msra_coeff)))

        print ("Ok, here's the ", nl, "th layer: ", layers[-1].get_shape(), " param[", nl, "]: ", param, "linear: ", last_linear)

            
    if len(return_layers) == 1:
        return layers[return_layers[0]]
    else:
        return [layers[nl] for nl in return_layers]

def flatten(data):
    return tf.reshape(data, [-1, np.prod(data.get_shape().as_list()[1:])])

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)
