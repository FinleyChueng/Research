import tensorflow as tf


def parametric_relu(x, name_space):
    r'''
        Parametric ReLU activation function. Additional parameters.

    :param x:
    :return:
    '''

    alphas = tf.get_variable(name_space+'_alpha', x.get_shape()[-1],
                             initializer=tf.constant_initializer(0.0),
                             dtype=tf.float32)
    pos = tf.nn.relu(x)
    neg = alphas * (x - tf.abs(x)) * 0.5

    return pos + neg


# Define the function used to select activation function.
def activate(x, activation, name_space):
    r'''
        Activate the input tensor according to the method.

    :param x:
    :param activation:
    :param name_space:
    :return:
    '''

    if activation == 'W/O' or activation is None:
        y = x
    elif activation == 'relu':
        y = tf.nn.relu(x, name=name_space + '/relu')
    elif activation == 'lrelu':
        y = tf.nn.leaky_relu(x, name=name_space + '/lrelu')
    elif activation == 'prelu':
        y = parametric_relu(x, name_space=name_space + '/prelu')
    elif activation == 'elu':
        y = tf.nn.elu(x, name=name_space + '/elu')
    elif activation == 'selu':
        y = tf.nn.selu(x, name=name_space + '/selu')
    else:
        raise ValueError('Unknown activation function !!!')

    return y


