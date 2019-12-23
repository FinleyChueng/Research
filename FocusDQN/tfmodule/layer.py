import tensorflow as tf
import tfmodule.normalization as tf_norl
import tfmodule.activation as tf_act


def base_activate_block(input_tensor,
                        name_space,
                        activation,
                        keep_prob,
                        feature_normalization,
                        reuse=False,
                        bn_decay=None,
                        bn_training=None):
    r'''
        The basic activation procedure block. The order likes below:
            Feature Normalization -> Activation -> Dropout
        It's the most common order.

    ------------------------------------------------------------------------
    Parameters:
        input_tensor: Input tensors.
        name_space: The name space for conv2d block.
        activation: The activation function.
        keep_prob: Whether enable the "Dropout" or not.
        feature_normalization: The feature normalization method.
        reuse: Whether to reuse the conv kernel (and bias) or not.
        bn_decay: The decay rate for "Batch Normalization". (This parameter only
            works when feature normalization method is "batch")
        bn_training: The training flag for "Batch Normalization". This parameter only
            works when feature normalization method is "batch")

    ------------------------------------------------------------------------
    Return:
        The tensor of same type as input tensor.
    '''

    # 1. Feature Normalize.
    output_tensor = tf_norl.feature_normalize(input_tensor,
                                              feature_normalization,
                                              name_space,
                                              reuse=reuse,
                                              bn_training=bn_training,
                                              bn_decay=bn_decay)
    # 2. Activate.
    output_tensor = tf_act.activate(output_tensor, activation, name_space)
    # 3. Dropout.
    if keep_prob != 1.0:
        output_tensor = tf.nn.dropout(output_tensor, keep_prob=keep_prob, name=name_space + '/dropout')

    return output_tensor


def base_conv2d(input_tensor,
                output_channels,
                kernel_size,
                stride,
                name_space,
                pre_activate=True,
                activation='relu',
                dilate_rate=1,
                use_bias=False,
                reuse=False,
                regularizer=None,
                keep_prob=1.0,
                feature_normalization='instance',
                bn_decay=0.9,
                bn_training=True):
    r'''
        Basic convolution block. The whole structure includes
            feature normalization, convolution and activation.
        Specially, the normal procedure is:
            Convolution -> Feature Normalization -> Activation
        If enable the "Pre-activate", the procedure changes like below:
            Feature Normalization -> Activation -> Convolution

    ------------------------------------------------------------------------
    Parameters:
        input_tensor: Input tensors.
        output_channels: The dimension (channels) of output tensors.
        kernel_size: The kernel size of convolution.
        stride: The stride for convolution.
        name_space: The name space for 2D convolution block.
        pre_activate: Whether enable "Pre-activate" or not.
        activation: The activation function.
        dilate_rate: The dilate rate for the convolution.
        use_bias: Whether use bias or not for convolution.
        reuse: Whether to reuse the conv kernel (and bias) or not.
        regularizer: Whether to add regularization to convolution or not.
        keep_prob: Whether enable the "Dropout" or not.
        feature_normalization: The feature normalization method.
        bn_decay: The decay rate for "Batch Normalization". (This parameter only
            works when feature normalization method is "batch")
        bn_training: The training flag for "Batch Normalization". This parameter only
            works when feature normalization method is "batch")

    ------------------------------------------------------------------------
    Return:
        The tensor of same type as input tensor.
    '''

    # Expand name space.
    name_space += '/conv'

    # ---------------------------- Sub-function Definition -------------------------
    # Define the sub-function used to activate neuron.
    def activate(x):
        y = base_activate_block(x, name_space,
                                activation=activation,
                                keep_prob=keep_prob,
                                feature_normalization=feature_normalization,
                                bn_decay=bn_decay,
                                bn_training=bn_training)
        return y

    # Define the sub-function used to conduct 2D-conv.
    def convolution_2d(x):
        y = tf.layers.conv2d(x, output_channels, kernel_size, stride, 'same',
                             dilation_rate=dilate_rate,
                             use_bias=use_bias,
                             reuse=reuse,
                             kernel_regularizer=regularizer,
                             name=name_space + '/conv')
        return y
    # -----------------------------------------------------------------------------

    # Pre-activate or not.
    if pre_activate:
        # "Feature Normalization" -> "Activation" -> "Convolution".
        output_tensor = convolution_2d(activate(input_tensor))
    else:
        # "Convolution" -> "Feature Normalization" -> "Activation".
        output_tensor = activate(convolution_2d(input_tensor))

    return output_tensor


def base_fc(input_tensor,
            output_channels,
            name_space,
            pre_activate=True,
            use_bias=True,
            regularizer=None,
            activation='relu',
            keep_prob=1.0,
            feature_normalization='instance',
            bn_decay=0.9,
            bn_training=True):
    r'''
        Generate the fully connected layer according to parameters.

    ------------------------------------------------------------------------
    Parameters:
        input_tensor: The input tensor. Must be [-1, channels] like.
        output_channels: A integer indicating the output channels.
        name_space: The name space.
        pre_activate: Whether enable "Pre-activate" or not.
        activation: The activation function.
        use_bias: Whether use bias or not for convolution.
        regularizer: Whether to add regularization to convolution or not.
        keep_prob: Whether enable the "Dropout" or not.
        feature_normalization: The feature normalization method.
        bn_decay: The decay rate for "Batch Normalization". (This parameter only
            works when feature normalization method is "batch")
        bn_training: The training flag for "Batch Normalization". This parameter only
            works when feature normalization method is "batch")

    ------------------------------------------------------------------------
    Return:
        The output tensor of FC layer.
    '''

    # Expand name space.
    name_space += '/fc'

    # ---------------------------- Sub-function Definition ----------------------
    # Define the sub-function used to activate neuron.
    def activate(x):
        y = base_activate_block(x, name_space,
                                activation=activation,
                                keep_prob=keep_prob,
                                feature_normalization=feature_normalization,
                                bn_decay=bn_decay,
                                bn_training=bn_training)
        return y

    # Define the sub-function used to fully connect neurons.
    def fully_connect(x):
        # Matmul
        initialW = tf.truncated_normal([int(x.shape[-1]), output_channels], stddev=0.01, name=name_space + '/var_init')
        W = tf.get_variable(name_space + '/var', initializer=initialW, regularizer=regularizer)
        y = tf.matmul(x, W, name=name_space + '/matmul')  # [None, output_channels]
        # Add bias if needed.
        if use_bias:
            initialb = tf.constant(0.01, shape=[output_channels], name=name_space+'/bias_init')
            b = tf.get_variable(name=name_space+'/bias', initializer=initialb)
            y = tf.add(y, b, name=name_space + '/add')  # [None, output_channels]
        return y
    # -----------------------------------------------------------------------------

    # Check validity.
    if len(input_tensor.shape) != 2:
        raise Exception('The input tensor of FC layer must be 2D tensor !!!')

    # Pre-activate or not.
    if pre_activate:
        # "Feature Normalization" -> "Activation" -> "Convolution".
        output_tensor = fully_connect(activate(input_tensor))
    else:
        # "Convolution" -> "Feature Normalization" -> "Activation".
        output_tensor = activate(fully_connect(input_tensor))

    return output_tensor


def base_deconv2d(input_tensor,
                  output_channels,
                  kernel_size,
                  stride,
                  name_space,
                  pre_activate=True,
                  activation='relu',
                  use_bias=False,
                  reuse=False,
                  regularizer=None,
                  keep_prob=1.0,
                  feature_normalization='instance',
                  bn_decay=0.9,
                  bn_training=True):
    r'''
        Basic deconvolution block. The whole structure includes
            feature normalization, deconvolution and activation.
        Specially, the normal procedure is:
            Deconvolution -> Feature Normalization -> Activation
        If enable the "Pre-activate", the procedure changes like below:
            Feature Normalization -> Activation -> Deconvolution

    ------------------------------------------------------------------------
    Parameters:
        input_tensor: Input tensors.
        output_channels: The dimension (channels) of output tensors.
        kernel_size: The kernel size of convolution.
        stride: The stride for convolution.
        name_space: The name space for conv2d block.
        pre_activate: Whether enable "Pre-activate" or not.
        activation: The activation function.
        use_bias: Whether use bias or not for convolution.
        reuse: Whether to reuse the conv kernel (and bias) or not.
        regularizer: Whether to add regularization to convolution or not.
        keep_prob: Whether enable the "Dropout" or not.
        feature_normalization: The feature normalization method.
        bn_decay: The decay rate for "Batch Normalization". (This parameter only
            works when feature normalization method is "batch")
        bn_training: The training flag for "Batch Normalization". This parameter only
            works when feature normalization method is "batch")

    ------------------------------------------------------------------------
    Return:
        The tensor of same type as input tensor.
    '''

    # Expand name space.
    name_space += '/deconv'

    # ---------------------------- Sub-function Definition -------------------------
    # Define the sub-function used to activate neuron.
    def activate(x):
        y = base_activate_block(x, name_space,
                                activation=activation,
                                keep_prob=keep_prob,
                                feature_normalization=feature_normalization,
                                bn_decay=bn_decay,
                                bn_training=bn_training)
        return y

    # Define the sub-function used to conduct 2D-conv.
    def deconvolution_2d(x):
        y = tf.layers.conv2d_transpose(x, output_channels, kernel_size, stride, 'same',
                                       use_bias=use_bias,
                                       reuse=reuse,
                                       kernel_regularizer=regularizer,
                                       name=name_space + '/deconv')
        return y
    # -----------------------------------------------------------------------------

    # Pre-activate or not.
    if pre_activate:
        # "Feature Normalization" -> "Activation" -> "Convolution".
        output_tensor = deconvolution_2d(activate(input_tensor))
    else:
        # "Convolution" -> "Feature Normalization" -> "Activation".
        output_tensor = activate(deconvolution_2d(input_tensor))

    return output_tensor


