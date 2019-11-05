import tensorflow as tf
import tfmodule.layer as tf_layer
import tfmodule.normalization as tf_norl
import tfmodule.activation as tf_act


############################# Residual Part ##############################
def __residual_v2(input_tensor,
                  output_channels,
                  feature_normalization,
                  activation,
                  keep_prob,
                  name_space,
                  bottleneck=True,
                  layer_type='unit',
                  kernel_size=3,
                  scale_factor=2,
                  regularizer=None):
    r'''
        Residual V2. It use the "Pre-activate" structure.
        The "Bottle Structure" likes below:
            (0.25*OC, 1, 1) -> (0.25*OC, 3, 1) -> (out_chan, 1, 1)

    ------------------------------------------------------------------------
    Parameters:
        input_tensor: Input tensors.
        output_channels: The output channels number.
        feature_normalization: The feature normalization method.
        activation: The activation function.
        keep_prob: Whether to enable the "Dropout" or not.
        name_space: The name space.
        bottleneck: Whether to use the "Bottleneck" structure or not.
        layer_type: Determine the layer's usage. Default is "Unit",
            "Down-sample", "Up-sample" is under options.
        kernel_size: The kernel size for convolution.
        scale_factor: Determine the feature scale. It works only when
            layer type is "Down-sample" or "Up-sample".
        regularizer: The regularizer for weights variables.

    ------------------------------------------------------------------------
    Return:
        The same type as input tensors.
    '''

    # Add suffix for name space.
    name_space += '/residualV2_layer'

    # Cast type into invalid parameter.
    if not isinstance(output_channels, int):
        output_channels = int(output_channels)

    # Determine the computation function we will use
    #   in corresponding phrase.
    if layer_type == 'unit':
        tf_conv_func = tf.layers.conv2d
        cus_conv_func = tf_layer.base_conv2d
        stride = 1
        name_space += '/unit'
    elif layer_type == 'down':
        tf_conv_func = tf.layers.conv2d
        cus_conv_func = tf_layer.base_conv2d
        stride = scale_factor
        name_space += '/down'
    elif layer_type == 'up':
        tf_conv_func = tf.layers.conv2d_transpose
        cus_conv_func = tf_layer.base_deconv2d
        stride = scale_factor
        name_space += '/up'
    else:
        raise ValueError('Unknown layer type !!!')

    # Pre-activate input tensors.
    preact = tf_layer.base_activate_block(input_tensor, name_space + '/preact',
                                          activation=activation,
                                          keep_prob=keep_prob,
                                          feature_normalization=feature_normalization,
                                          bn_decay=0.9,
                                          bn_training=True)     # The BN parameters keep same as base conv.

    # Translate to the same channel if needed.
    if input_tensor.get_shape()[-1] != output_channels or stride != 1:
        # Map method.
        short_cut = tf_conv_func(preact, output_channels, 1, stride, 'same',
                                 use_bias=False,
                                 kernel_regularizer=regularizer,
                                 name=name_space + '/mapdim')   # The parameters keep same as base conv.
    else:
        # Introduce the shortcut, which is directly the inputs.
        short_cut = input_tensor

    # Whether use "Bottleneck" or not.
    if bottleneck:
        # Pass the input through the "Bottle" structure.
        #   The kernel size and its procedure likes below:
        #   (0.25*OC, 1, 1) -> (0.25*OC, 3, 1) -> (out_chan, 1, 1)
        conv2d = tf.layers.conv2d(preact, int(0.25 * output_channels), 1, 1, 'same',
                                  use_bias=False,
                                  kernel_regularizer=regularizer,
                                  name=name_space + '/conv_1')
        conv2d = cus_conv_func(conv2d, int(0.25 * output_channels), kernel_size, stride,
                               name_space=name_space + '/conv_2',
                               activation=activation,
                               keep_prob=keep_prob,
                               feature_normalization=feature_normalization,
                               regularizer=regularizer)
        conv2d = tf_layer.base_conv2d(conv2d, output_channels, 1, 1, name_space=name_space + '/conv_3',
                                      activation=activation,
                                      keep_prob=keep_prob,
                                      feature_normalization=feature_normalization,
                                      regularizer=regularizer)
    else:
        # Pass through the "Plain" Structure.
        #   The kernel size and its procedure likes below:
        #   (out_chan, 3, 2) -> (out_chan, 3, 1)
        conv2d = tf_conv_func(preact, output_channels, kernel_size, stride, 'same',
                              use_bias=False,
                              kernel_regularizer=regularizer,
                              name=name_space + '/conv_1')
        conv2d = tf_layer.base_conv2d(conv2d, output_channels, kernel_size, 1,
                                      name_space=name_space + '/conv_2',
                                      activation=activation,
                                      keep_prob=keep_prob,
                                      feature_normalization=feature_normalization,
                                      regularizer=regularizer)

    # Finally Add the conv features and the shortcut outputs.
    return tf.add(conv2d, short_cut, name=name_space + '/res_out')


def __residual_v1(input_tensor,
                  output_channels,
                  feature_normalization,
                  activation,
                  keep_prob,
                  name_space,
                  bottleneck=True,
                  layer_type='unit',
                  kernel_size=3,
                  scale_factor=2,
                  regularizer=None):
    r'''
        Residual V1. It use the common activation structure.
        The "Bottle Structure" likes below:
            (0.25*OC, 1, 1) -> (0.25*OC, 3, 1) -> (out_chan, 1, 1)

    ------------------------------------------------------------------------
    Parameters:
        input_tensor: Input tensors.
        output_channels: The output channels number.
        feature_normalization: The feature normalization method.
        activation: The activation function.
        keep_prob: Whether to enable the "Dropout" or not.
        name_space: The name space.
        kernel_size: The kernel size for convolution.
        regularizer: The regularizer for weights variables.

    ------------------------------------------------------------------------
    Return:
        The same type as input tensors.
    '''

    # Add suffix for name space.
    name_space += '/residualV1_layer'

    # Cast type into invalid parameter.
    if not isinstance(output_channels, int):
        output_channels = int(output_channels)

    # Determine the computation function we will use
    #   in corresponding phrase.
    if layer_type == 'unit':
        tf_conv_func = tf.layers.conv2d
        cus_conv_func = tf_layer.base_conv2d
        stride = 1
        name_space += '/unit'
    elif layer_type == 'down':
        tf_conv_func = tf.layers.conv2d
        cus_conv_func = tf_layer.base_conv2d
        stride = scale_factor
        name_space += '/down'
    elif layer_type == 'up':
        tf_conv_func = tf.layers.conv2d_transpose
        cus_conv_func = tf_layer.base_deconv2d
        stride = scale_factor
        name_space += '/up'
    else:
        raise ValueError('Unknown layer type !!!')

    # Translate to the same channel if needed.
    if input_tensor.get_shape()[-1] != output_channels or stride != 1:
        # Map method.
        short_cut = cus_conv_func(input_tensor, output_channels, 1, stride,
                                  name_space=name_space + '/mapdim',
                                  pre_activate=False,
                                  activation=activation,
                                  keep_prob=keep_prob,
                                  feature_normalization=feature_normalization,
                                  regularizer=regularizer)
    else:
        # Introduce the shortcut, which is directly the inputs.
        short_cut = input_tensor

    # Whether use "Bottleneck" or not.
    if bottleneck:
        # Pass the input through the "Bottle" structure.
        #   The kernel size and its procedure likes below:
        #   (0.25*OC, 1, 1) -> (0.25*OC, 3, 1) -> (out_chan, 1, 1)
        conv2d = tf_layer.base_conv2d(input_tensor, int(0.25 * output_channels), 1, 1,
                                      name_space=name_space + '/conv_1',
                                      pre_activate=False,
                                      activation=activation,
                                      keep_prob=keep_prob,
                                      feature_normalization=feature_normalization,
                                      regularizer=regularizer)
        conv2d = cus_conv_func(conv2d, int(0.25 * output_channels), kernel_size, stride,
                               name_space=name_space + '/conv_2',
                               pre_activate=False,
                               activation=activation,
                               keep_prob=keep_prob,
                               feature_normalization=feature_normalization,
                               regularizer=regularizer)
        # Coz ResNetV1. The procedure likes below:
        #   conv -> feature normalize -> addition -> ReLU
        conv2d = tf.layers.conv2d(conv2d, output_channels, 1, 1, 'same',
                                  use_bias=False,
                                  kernel_regularizer=regularizer,
                                  name=name_space + '/conv_3_conv')   # The parameters keep same as base conv.
        conv2d = tf_norl.feature_normalize(conv2d, feature_normalization,
                                           name_space=name_space + '/conv_3_bn',
                                           bn_decay=0.9,
                                           bn_training=True)    # The parameters keep same as base conv.
        conv2d = tf.add(conv2d, short_cut, name=name_space + '/conv_3_add')
    else:
        # Pass the input through the "Plain" structure.
        #   The kernel size and its procedure likes below:
        #   (out_chan, 3, 2) -> (out_chan, 3, 1)
        conv2d = cus_conv_func(input_tensor, output_channels, kernel_size, stride,
                               name_space=name_space + '/conv_1',
                               pre_activate=False,
                               activation=activation,
                               keep_prob=keep_prob,
                               feature_normalization=feature_normalization,
                               regularizer=regularizer)
        # Coz ResNetV1. The procedure likes below:
        #   conv -> feature normalize -> addition -> ReLU
        conv2d = tf_conv_func(conv2d, output_channels, kernel_size, 1, 'same',
                              use_bias=False,
                              kernel_regularizer=regularizer,
                              name=name_space + '/conv_2_conv')  # The parameters keep same as base conv.
        conv2d = tf_norl.feature_normalize(conv2d, feature_normalization,
                                           name_space=name_space + '/conv_2_bn',
                                           bn_decay=0.9,
                                           bn_training=True)  # The parameters keep same as base conv.
        conv2d = tf.add(conv2d, short_cut, name=name_space + '/conv_2_add')

    # Finally Add the conv features and the shortcut outputs.
    return tf_act.activate(conv2d, activation, name_space + '/res_out')


# --- Residual Block. ---
def residual_block(input_tensor,
                   output_channels,
                   layer_num,
                   feature_normalization,
                   activation,
                   keep_prob,
                   name_space,
                   bottleneck=True,
                   kernel_size=3,
                   regularizer=None,
                   structure='V2'):
    r'''
        Residual Block. Which consists of multiple residual structure.
            The scale depends on layer number.
        Generally speaking, a residual block corresponding to a specific
            feature map size.

    ------------------------------------------------------------------------
    Parameters:
        input_tensor: Input tensors.
        output_channels: The output channels number.
        layer_num: Determine the layer number of the block unit.
        feature_normalization: The feature normalization method.
        activation: The activation function.
        keep_prob: Whether to enable the "Dropout" or not.
        name_space: The name space.
        bottleneck: Whether use the "Bottle" or "Plain" structure.
        kernel_size: The kernel size for convolution.
        regularizer: The regularizer for weights variables.
        structure: Indicating the version of residual structure.

    ------------------------------------------------------------------------
    Return:
        The same type as input tensors.
    '''

    # Add suffix for name space.
    name_space += '/block_unit'

    # Choose the residual structure.
    if structure == 'V1':
        residual = __residual_v1
    elif structure == 'V2':
        residual = __residual_v2
    else:
        raise ValueError('Unknown residual structure !!!')

    # Assign for convenient iteration.
    conv2d = input_tensor
    # Iteratively add the residual layer to construct residual block.
    for l in range(layer_num):
        conv2d = residual(conv2d, output_channels,
                          bottleneck=bottleneck,
                          feature_normalization=feature_normalization,
                          activation=activation,
                          keep_prob=keep_prob,
                          name_space=name_space + '_' + str(l+1),
                          kernel_size=kernel_size,
                          regularizer=regularizer)

    # Finish the construction of residual block, and return the output of residual block.
    return conv2d


############################# Transition Layer Part ##############################
def transition_layer(input_tensor,
                     output_channels,
                     feature_normalization,
                     activation,
                     keep_prob,
                     name_space,
                     scale_down=True,
                     scale_factor=2,
                     regularizer=None,
                     structure='ResV2'):
    r'''
        A wrapper of transition method.

    ------------------------------------------------------------------------
    Parameters:
        input_tensor: Input tensors.
        output_channels: The output channels number.
        feature_normalization: The feature normalization method.
        activation: The activation function.
        keep_prob: Whether to enable the "Dropout" or not.
        name_space: The name space.
        scale_down: The flag indicating whether is "Down-sample" phrase or not.
        scale_factor: The scale factore for feature maps.
        regularizer: The regularizer for weights variables.
        structure: Indicating the detailed transition down method.

    ------------------------------------------------------------------------
    Return:
        The same type as input tensors.
    '''

    # Add suffix for name space.
    name_space += '/transition_layer'

    # Deal with the residual transition tensor, independently.
    if structure.startswith('Res'):
        # Determine layer type.
        if scale_down:
            layer_type = 'down'
        else:
            layer_type = 'up'
        # Determine the residual function and layer structure.
        if structure == 'ResV2':
            # "Bottleneck" structure with pre-activation.
            res_func = __residual_v2
            bottle_neck = True
        elif structure == 'ResV2-Plain':
            # "Plain" structure with pre-activation.
            res_func = __residual_v2
            bottle_neck = False
        elif structure == 'ResV1':
            # "Bottleneck" structure with post-activation.
            res_func = __residual_v1
            bottle_neck = True
        elif structure == 'ResV1-Plain':
            # "Plain" structure with pre-activation.
            res_func = __residual_v1
            bottle_neck = False
        else:
            raise ValueError('Unknown residual structure !!!')
        # Execute the residual function.
        trans_tensor = res_func(input_tensor, output_channels,
                                layer_type=layer_type,
                                scale_factor=scale_factor,
                                bottleneck=bottle_neck,
                                feature_normalization=feature_normalization,
                                activation=activation,
                                keep_prob=keep_prob,
                                regularizer=regularizer,
                                name_space=name_space)
        # Return the residual transition tensor.
        return trans_tensor

    # Not common method. Deal with it respectively.
    if scale_down:
        if structure == 'MP':
            # Rectify the dimension
            if input_tensor.get_shape()[-1] != output_channels:
                trans_tensor = tf_layer.base_conv2d(input_tensor, output_channels, 1, scale_factor, name_space)
            else:
                trans_tensor = input_tensor
            # Use max pooling.
            trans_tensor = tf.layers.max_pooling2d(trans_tensor, scale_factor, scale_factor, 'same',
                                                   name=name_space + '/MP')
        else:
            raise ValueError('Unknown transition down method !!!')
    else:
        if structure == 'PURE':
            trans_tensor = tf_layer.base_deconv2d(input_tensor, output_channels, 3, scale_factor, name_space)
        else:
            raise ValueError('Unknown transition up method !!!')

    return trans_tensor

