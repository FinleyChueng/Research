import tensorflow as tf
import numpy as np

from net.base import *



class DenseNet(FeatureExtractionNetwork):
    r'''
        The definition of DenseNet.
    '''

    def build(self, input_layer, weights_regularizer, name_scope, need_AFPP=False, feats_proc_func=None, arg=None):
        r'''
            The definition of DenseNet.

        Return:
            The "input" layer (Note that, this input layer is consisted of "DenseNet" input
                layer and the placeholder of action history list) and output layer of
                this CNN, which is actually the placeholder(input) and tensorflow
                operation(output) of model base. That is in the term of
                (input_layer, output_layer)
        '''

        # Some hyper-parameters used to control the architecture of DenseNet.
        reduction = 0.5     # 0.0
        nb_grouth_rate = 32

        # Reset the "Need AFPP" flag.
        self._need_AFPP = need_AFPP

        # Get the additional parameters.
        feats_pro, self._train_flag = arg

        # Do not support that.
        if feats_pro != 1.0:
            raise Exception('DenseNet do not support AFPP !!!')

        # Specify the keep prob for Dropout.
        self._keep_prob = tf.constant(1.0, name='DenseNet_keep_prob')
        # self._keep_prob = tf.cond(
        #     self._train_flag,
        #     lambda: tf.constant(0.8),
        #     lambda: tf.constant(1.0),
        #     name='DenseNet_keep_prob'
        # )

        # Feature stride.
        feature_stride = 1  # The feature map stride
        # The feature dictionary.
        feats_dict = {}

        # Add a dimension reduce layer.
        with tf.variable_scope(name_scope + '_Init'):
            # The initial conv block to decrease the dimension.
            inputs = tf.contrib.layers.batch_norm(input_layer,
                                                  is_training=True,
                                                  renorm=True,
                                                  scope='BN')
            inputs = tf.nn.leaky_relu(inputs, name='Relu')
            init_base_conv = tf.layers.conv2d(inputs, 32, 7, 2, 'same',
                                              use_bias=False,
                                              kernel_regularizer=weights_regularizer,
                                              name='Base_conv')  # 120,120,32
            # Increase the stride
            feature_stride *= 2
            feats_dict['feats-'+str(feature_stride)] = init_base_conv
        # Dense block 1
        with tf.variable_scope(name_scope + '_Dense_block1'):
            # First block with (60, 60, 64).
            db1, nbfilt1 = self.__dense_block(init_base_conv, nb_layers=3, nb_filter=int(init_base_conv.shape[-1]),
                                              growth_rate=nb_grouth_rate)
            trans1 = self.__transition_block(db1, nb_filter=nbfilt1, reduction=reduction)  # 60, 60, (32 + 32*3) * 0.5
            # Increase the stride
            feature_stride *= 2
            feats_dict['feats-' + str(feature_stride)] = trans1
        # Dense block 2
        with tf.variable_scope(name_scope + '_Dense_block2'):
            # Second block with (30, 30, 128).
            db2, nbfilt2 = self.__dense_block(trans1, nb_layers=6, nb_filter=int(trans1.shape[-1]),
                                              growth_rate=nb_grouth_rate)
            trans2 = self.__transition_block(db2, nb_filter=nbfilt2, reduction=reduction)  # 30, 30, (64 + 32*6) * 0.5
            # Increase the stride
            feature_stride *= 2
            feats_dict['feats-' + str(feature_stride)] = trans2
        # Dense block 3
        with tf.variable_scope(name_scope + '_Dense_block3'):
            # Third block with (15, 15, 256).
            db3, nbfilt3 = self.__dense_block(trans2, nb_layers=12, nb_filter=int(trans2.shape[-1]),
                                              growth_rate=nb_grouth_rate)
            dense_out = self.__transition_block(db3, nb_filter=nbfilt3, reduction=reduction) # 15, 15, (128 + 32*12) * 0.5
            # Increase the stride
            feature_stride *= 2
            feats_dict['feats-' + str(feature_stride)] = dense_out

        # Return the output of the DenseNet, feature stride and feats dict.
        return dense_out, feature_stride, feats_dict



    #### --------------------- DenseNet Related -----------------------------
    def __dense_block(self, inputs, nb_layers, nb_filter, growth_rate, grow_nb_filters=True):
        """
        Build a dense_block where the output of each conv_block is fed to subsequent ones
        # Arguments
            x:               input tensor
            nb_layers:       the number of layers of conv_block to append to the model.
            nb_filter:       number of filters
            grow_nb_filters: flag to decide to allow number of filters to grow
        """

        # The raw feature maps.
        concat_feat = inputs

        # Iteratively concate the feature maps.
        for i in range(nb_layers):
            # The conv block (BN-ReLU-Conv-Dropout).
            conv_output = self.__conv_block(concat_feat, growth_rate, name='Conv_block_' + str(i+1))
            # Concate the feature maps.
            concat_feat = tf.concat([concat_feat, conv_output], axis=-1, name='Concat_' + str(i+1))

            # Grow the output channels.
            if grow_nb_filters:
                nb_filter += growth_rate

        # Return the concated feature maps and output channels
        return concat_feat, nb_filter


    def __conv_block(self, inputs, nb_filter, name):
        """
        Apply BatchNorm --> Relu --> bottleneck 1x1 Conv2D, 3x3 Conv2D, and option dropout
        # Arguments
            x:             input tensor
            nb_filter:     number of filters
        """
        eps = 1.1e-5

        # " 1*1 convolutional (Bottleneck layer)"
        inter_channel = 4 * nb_filter

        # Pass through BN-ReLU-Conv-Dropout.
        inputs = tf.contrib.layers.batch_norm(inputs, epsilon=eps,
                                              is_training=True,
                                              renorm=True,
                                              scope=name + '_bottle_bn')
        inputs = tf.nn.relu(inputs, name=name + '_bottle_relu')
        bottle = tf.layers.conv2d(inputs, filters=inter_channel,
                                  kernel_size=1, strides=1, padding='same',
                                  use_bias=False,
                                  name=name + '_bottle_conv')
        bottle = tf.nn.dropout(bottle, keep_prob=self._keep_prob, name=name + '_bottle_dropout')

        # " 3*3 convolutional"
        conv33 = tf.contrib.layers.batch_norm(bottle, epsilon=eps,
                                              is_training=True,
                                              renorm=True,
                                              scope=name + '_conv3x3_bn')
        conv33 = tf.nn.relu(conv33, name=name + '_conv3x3_relu')
        conv33 = tf.layers.conv2d(conv33, filters=nb_filter,
                                  kernel_size=3, strides=1, padding='same',
                                  use_bias=False,
                                  name=name + '_conv3x3_conv')
        conv33 = tf.nn.dropout(conv33, keep_prob=self._keep_prob, name=name + '_conv3x3_dropout')

        # Return the conv block output.
        return conv33


    def __transition_block(self, inputs, nb_filter, reduction):
        """
        Apply BatchNorm --> 1x1 Convolution --> averagePooling, optional compression, dropout
        # Arguments
            x:             input tensor
            nb_filter:     number of filters
            compression:   calculated as 1 - reduction. Reduces the number of feature maps in the transition block.
        """

        # Epsilon
        eps = 1.1e-5
        # Compression.
        compression = 1.0 - reduction

        # First pass through the 1*1 conv block.
        inputs = tf.contrib.layers.batch_norm(inputs, epsilon=eps,
                                              is_training=True,
                                              renorm=True,
                                              scope='transition_1x1conv_bn')
        # inputs = tf.nn.relu(inputs, name='transition_1x1conv_relu')
        conv1x1 = tf.layers.conv2d(inputs, filters=int(nb_filter * compression),
                                   kernel_size=1, strides=1, padding='same',
                                   use_bias=False,
                                   name='transition_1x1conv_conv')
        conv1x1 = tf.nn.dropout(conv1x1, keep_prob=self._keep_prob, name='transition_1x1conv_dropout')

        # Finally pass through a average layer.
        avg = tf.layers.average_pooling2d(conv1x1, pool_size=2, strides=2, name='transition_avgPool')

        # Return the final output tensor.
        return avg

    # def __batch_norm(self, x_tensor, name=None):
    #     mean, variance = tf.nn.moments(x_tensor, axes=[0])
    #     L = tf.nn.batch_normalization(x_tensor, mean, variance, 0.01, 1, 0.001, name=name)
    #     return L
    #
    # def __ds_conv2d(self, x_tensor, conv_num_outputs, conv_ksize, conv_strides, regularizer, name):
    #     """
    #     Apply convolution then max pooling to x_tensor
    #     :param x_tensor: TensorFlow Tensor
    #     :param conv_num_outputs: Number of outputs for the convolutional layer
    #     :param conv_strides: Stride 2-D Tuple for convolution
    #     :param pool_ksize: kernal size 2-D Tuple for pool
    #     :param pool_strides: Stride 2-D Tuple for pool
    #     : return: A tensor that represents convolution and max pooling of x_tensor
    #     """
    #     # TODO: Implement Function
    #     x_shape = x_tensor.get_shape().as_list()
    #     # regularizer = tf.contrib.layers.l2_regularizer(scale=0.0001)
    #     # regularizer = None
    #     n = conv_ksize[0] * conv_ksize[1] * conv_num_outputs
    #     with tf.variable_scope(name):
    #         weights = tf.get_variable('conv_weights',
    #                                   shape=[conv_ksize[0], conv_ksize[1], x_shape[3], conv_num_outputs],
    #                                   initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0 / n)),
    #                                   regularizer=regularizer)
    #         L = self.__batch_norm(x_tensor, 'bn')
    #         L = tf.nn.relu(L)
    #         L = tf.nn.conv2d(L, weights, strides=[1, conv_strides[0], conv_strides[1], 1], padding='SAME')
    #     return L
    #
    # def __conv_concate(self, x, growth_rate, regularizer, name):
    #     shape = x.get_shape().as_list()
    #     with tf.variable_scope(name):
    #         l = self.__ds_conv2d(x, conv_num_outputs=growth_rate, conv_ksize=(3, 3), conv_strides=(1, 1),
    #                              regularizer=regularizer, name='conv')
    #         l = tf.concat([l, x], 3)
    #     return l
    #
    # def __dense_block(self, l, regularizer, layers=3, growth_rate=12):
    #     for i in range(layers):
    #         l = self.__conv_concate(l, growth_rate=growth_rate, regularizer=regularizer,
    #                                 name='dense_blcok_{}.'.format(i))
    #         print('The dense_block_{} is {}'.format(i, l.shape))
    #     return l
    #
    # def __transition(self, l, regularizer):
    #     l = self.__ds_conv2d(l, 16, (1, 1), (1, 1), regularizer=regularizer, name='transition_conv_1x1')
    #     l = tf.nn.avg_pool(l, ksize=[1, 2, 2, 1],
    #                        strides=[1, 2, 2, 1], padding='VALID', name="transition_avg_pool")
    #     return l

