import tensorflow as tf

from net.base import *



class ResNet(FeatureExtractionNetwork):
    r'''
        The definition of ResNet.
    '''

    def build(self, input_layer, weights_regularizer, name_scope, need_AFPP=False, feats_proc_func=None, arg=None):
        r'''
            The definition of the ResNet.

        Return:
            The "input" layer (Note that, this input layer is consisted of "ResNet" input
                layer and the placeholder of action history list) and output layer of
                this CNN, which is actually the placeholder(input) and tensorflow
                operation(output) of model base. That is in the term of
                (input_layer, output_layer)
        '''

        # Reset the "Need AFPP" flag.
        self._need_AFPP = need_AFPP

        # Get the proportion for output channels.
        feats_pro, self._training_flag = arg

        # The enable "Fusion of Low and High Frequency" flag.
        enable_FLHF = feats_pro != 1.

        # Fake low frequency tensor.
        LF_tensor = None

        # The feature dictionary. Which maybe used in up-sample network.
        feats_dict = {}

        # # Check the validity of the AFPP-related parameters.
        # if not need_AFPP and enable_FLHF:
        #     raise Exception('The feats_pro should be 1. when disable the AFPP !!!')

        self._keep_prob = tf.constant(1.0, name='ResNet_keep_prob')
        # self._keep_prob = tf.cond(
        #     self._training_flag,
        #     lambda: tf.constant(0.8),
        #     lambda: tf.constant(1.0),
        #     name='ResNet_keep_prob'
        # )

        # Define the feature stride.
        feature_stride = 1  # The feature map stride

        with tf.variable_scope(name_scope):
            # # The input first pass through a "Batch Normalization".
            # bn_input = tf.layers.batch_normalization(input_layer, training=self._training_flag, name='ResNet_BN')
            # # bn_input = input_layer
            #
            # # The basic conv-layer to scale the feature maps to 120, 120.
            # base_conv = tf.layers.conv2d(bn_input, 64, 7, 2, 'same',
            #                              # activation=tf.nn.relu,
            #                              activation=tf.nn.leaky_relu,
            #                              name='ResNet_base_conv')  # 120,120,64
            # feature_stride *= 2  # Increase the stride

            # The basic conv-layer to scale the feature maps to 120, 120.
            # The output first pass through a "Batch Normalization".
            base_conv = tf.contrib.layers.batch_norm(input_layer,
                                                     # is_training=self._training_flag,
                                                     is_training=True,
                                                     # activation_fn=tf.nn.leaky_relu,
                                                     decay=0.9,
                                                     # zero_debias_moving_mean=True,
                                                     # renorm=True,
                                                     # updates_collections=None,
                                                     scope='ResNet_BN')
            # And then pass the leaky ReLU.
            base_conv = tf.nn.leaky_relu(base_conv, name='ResNet_ReLU')
            base_conv = tf.layers.conv2d(base_conv, 64, 7, 2, 'same',
                                         use_bias=False,
                                         # activation=tf.nn.relu,
                                         name='ResNet_base_conv')  # 120,120,64
            feature_stride *= 2  # Increase the stride

            # Only pass through the fusion function to generate the high and low frequency.
            if enable_FLHF:
                base_conv, LF_tensor = self.__fusion_func(LF_tensor=None,
                                                          HF_tensor=base_conv,
                                                          whole_chans=64,
                                                          low_Fp=(1.-feats_pro),
                                                          high_Fp=feats_pro,
                                                          arg=weights_regularizer)  # 120,120,64    * fp

            # Package the feature into dictionary. Feats-2
            self.__pack_downsample_feats(feats_dict=feats_dict,
                                         feature_stride=feature_stride,
                                         HF_tensor=base_conv,
                                         LF_tensor=LF_tensor,
                                         weights_regularizer=weights_regularizer)   # feats-2

            # # The basic conv-layer to scale the feature maps to 120, 120.
            # base_conv = tf.layers.conv2d(bn_input, round(64 * feats_pro), 7, 2, 'same',
            #                              activation=tf.nn.relu,
            #                              name='ResNet_base_conv')  # 120,120,64     * fp
            # feature_stride *= 2  # Increase the stride

            # # Pass through the additional process function.
            # base_conv = self._proc_afpool(base_conv,
            #                               proc_func=feats_proc_func,
            #                               feats_stride=feature_stride,
            #                               arg=arg)  # 120,120,64     * fp

            # A max pooling layer to down-sample the feature maps to 60, 60.
            base_mp = tf.layers.max_pooling2d(base_conv, 2, 2, 'same',
                                              name=name_scope + '_ResNet_base_maxpool')  # 60,60,64     * fp
            feature_stride *= 2  # Increase the stride
            # Shape with 60, 60.
            resnet_res_block1 = self.__residual_block(base_mp, 3, round(64 * feats_pro), weights_regularizer,
                                                      kernel_size=3, dilation_rate=1,
                                                      name='ResNet_residual_layer_1')  # 60,60,64  * fp
            # Pass through the function to generate the low frequency if specified.
            #   What's more, pass through the additional process function.
            if enable_FLHF:
                # Pass through the function to generate the low frequency if specified.
                LF_tensor = self.__gen_low_freq(tensor=LF_tensor,
                                                whole_chans=64,
                                                feats_pro=(1.-feats_pro),
                                                block_num=3,
                                                weights_regularizer=weights_regularizer)    # 30,30,64     * fp
                # Pass through the additional process function.
                resnet_res_block1, LF_tensor = self.__fusion_func(LF_tensor=LF_tensor,
                                                                  HF_tensor=resnet_res_block1,
                                                                  whole_chans=64,
                                                                  low_Fp=(1. - feats_pro),
                                                                  high_Fp=feats_pro,
                                                                  arg=weights_regularizer)  # 60,60,64     * fp

            # Package the feature into dictionary. Feats-4
            self.__pack_downsample_feats(feats_dict=feats_dict,
                                         feature_stride=feature_stride,
                                         HF_tensor=resnet_res_block1,
                                         LF_tensor=LF_tensor,
                                         weights_regularizer=weights_regularizer)  # feats-4

            # A down transition layer to down scale the feature maps to 30, 30.
            resnet_down_trans1 = self.__transition_down(resnet_res_block1, resnet_res_block1.get_shape()[-1],
                                                        name='ResNet_trans_down_1')     # 30,30,64     * fp
            feature_stride *= 2  # Increase the stride
            # Shape with 30, 30.
            resnet_res_block2 = self.__residual_block(resnet_down_trans1, 4, round(128 * feats_pro),
                                                      weights_regularizer, kernel_size=3, dilation_rate=1,
                                                      name='ResNet_residual_layer_2')  # 30,30,128  * fp
            # resnet_res_block2 = self.__residual_block(resnet_down_trans1, 3, round(128 * feats_pro),
            # Pass through the function to generate the low frequency if specified.
            #   What's more, pass through the additional process function.
            if enable_FLHF:
                # Pass through the function to generate the low frequency if specified.
                LF_tensor = self.__gen_low_freq(tensor=LF_tensor,
                                                whole_chans=128,
                                                feats_pro=(1. - feats_pro),
                                                block_num=4,
                                                weights_regularizer=weights_regularizer)    # 15,15,128 * fp
                # Pass through the additional process function. Note that, here is the
                #   last fusion layer. So we will fuse all channels to high frequency
                #   and no more low frequency.
                resnet_res_block2, LF_tensor = self.__fusion_func(LF_tensor=LF_tensor,
                                                                  HF_tensor=resnet_res_block2,
                                                                  whole_chans=128,
                                                                  low_Fp=0.,
                                                                  high_Fp=1.,
                                                                  arg=weights_regularizer)  # 30,30,128

            # Package the feature into dictionary. Feats-8
            self.__pack_downsample_feats(feats_dict=feats_dict,
                                         feature_stride=feature_stride,
                                         HF_tensor=resnet_res_block2,
                                         LF_tensor=LF_tensor,
                                         weights_regularizer=weights_regularizer)  # feats-8

            # A down transition layer to down scale the feature maps to 15, 15.
            resnet_down_trans2 = self.__transition_down(resnet_res_block2, resnet_res_block2.get_shape()[-1],
                                                        name='ResNet_trans_down_2')  # 15,15,128
            feature_stride *= 2  # Increase the stride
            # Shape with 15, 15.
            resnet_res_block3 = self.__residual_block(resnet_down_trans2, 6, 256,
                                                      weights_regularizer, kernel_size=3, dilation_rate=1,
                                                      name="ResNet_residual_layer_3")  # 15,15,256
            # resnet_res_block3 = self.__residual_block(resnet_down_trans2, 3, 256,

            # Package the feature into dictionary. Feats-16
            self.__pack_downsample_feats(feats_dict=feats_dict,
                                         feature_stride=feature_stride,
                                         HF_tensor=resnet_res_block3,
                                         LF_tensor=LF_tensor,
                                         weights_regularizer=weights_regularizer)  # feats-16

            # # Add additional block. (15, 15)
            # resnet_res_block3 = self.__residual_block(resnet_res_block3, 3, 256,
            #                                           weights_regularizer, kernel_size=3, dilation_rate=1,
            #                                           name="ResNet_residual_layer_4")  # 15,15,256
            #
            # # Package the feature into dictionary. Feats-16
            # self.__pack_downsample_feats(feats_dict=feats_dict,
            #                              feature_stride=feature_stride,
            #                              HF_tensor=resnet_res_block3,
            #                              LF_tensor=LF_tensor,
            #                              weights_regularizer=weights_regularizer)  # feats-16

        # Finish the definition of "Feature Extractor".
        return resnet_res_block3, feature_stride, feats_dict

    #### --------------------- ResNet Related -----------------------------
    def __residual(self, input_layer, output_channels, kernel_regularizer,
                   kernel_size=3, dilation_rate=1, name="residual_layer"):

        preact = tf.contrib.layers.batch_norm(input_layer,
                                              # is_training=self._training_flag,
                                              is_training=True,
                                              decay=0.9,
                                              # zero_debias_moving_mean=True,
                                              # renorm=True,
                                              # updates_collections=None,
                                              scope=name + "_batch_norm_1")
        preact = tf.nn.relu(preact, name=name + "_relu_1")

        # Translate to the same channel if needed.
        if input_layer.get_shape()[-1] != output_channels:
            # Map method.
            short_cut = tf.layers.conv2d(preact, output_channels, kernel_size, 1, 'same',
                                      dilation_rate=dilation_rate, name=name + '_res_map_dim')
        else:
            # Introduce the shortcut, which is directly the inputs.
            short_cut = input_layer

        # # Pass the input through the "Bottle Structure".
        # # -- first.
        # conv2d = tf.contrib.layers.batch_norm(input_layer, is_training=self._training_flag,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       renorm=True,
        #                                       # updates_collections=None,
        #                                       scope=name + "_batch_norm_1")
        # conv2d = tf.nn.relu(conv2d, name=name + "_relu_1")
        # conv2d = tf.layers.conv2d(conv2d, 64, kernel_size, 1, 'same',
        #                           # activation=tf.nn.relu,
        #                           use_bias=False,
        #                           kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
        #                           name=name + "_conv_1")
        # conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_1")
        # # -- second.
        # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training=self._training_flag,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       renorm=True,
        #                                       # updates_collections=None,
        #                                       scope=name + "_batch_norm_2")
        # conv2d = tf.nn.relu(conv2d, name=name + "_relu_2")
        # conv2d = tf.layers.conv2d(conv2d, 64, kernel_size, 1, 'same',
        #                           # activation=tf.nn.relu,
        #                           use_bias=False,
        #                           kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
        #                           name=name + "_conv_2")
        # conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_2")
        # # -- third.
        # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training=self._training_flag,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       renorm=True,
        #                                       # updates_collections=None,
        #                                       scope=name + "_batch_norm_3")
        # conv2d = tf.nn.relu(conv2d, name=name + "_relu_3")
        # conv2d = tf.layers.conv2d(conv2d, output_channels, kernel_size, 1, 'same',
        #                           # activation=tf.nn.relu,
        #                           use_bias=False,
        #                           kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
        #                           name=name + "_conv_3")
        # conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_3")

        # # Pass the input through the "Traditional Structure".
        # # -- first.
        # conv2d = tf.contrib.layers.batch_norm(input_layer, is_training=self._training_flag,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       renorm=True,
        #                                       # updates_collections=None,
        #                                       scope=name + "_batch_norm_1")
        # conv2d = tf.nn.relu(conv2d, name=name + "_relu_1")
        # conv2d = tf.layers.conv2d(conv2d, output_channels, kernel_size, 1, 'same',
        #                           # activation=tf.nn.relu,
        #                           use_bias=False,
        #                           kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
        #                           name=name + "_conv_1")
        # conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_1")
        # # -- third.
        # conv2d = tf.contrib.layers.batch_norm(conv2d, is_training=self._training_flag,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       renorm=True,
        #                                       # updates_collections=None,
        #                                       scope=name + "_batch_norm_2")
        # conv2d = tf.nn.relu(conv2d, name=name + "_relu_2")
        # conv2d = tf.layers.conv2d(conv2d, output_channels, kernel_size, 1, 'same',
        #                           # activation=tf.nn.relu,
        #                           use_bias=False,
        #                           kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
        #                           name=name + "_conv_2")
        # conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_2")

        # Pass the input through the "Bottle Structure".
        # -- first.
        conv2d = tf.layers.conv2d(preact, 64, 1, 1, 'same',
                                  # activation=tf.nn.relu,
                                  use_bias=False,
                                  kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
                                  name=name + "_conv_1")
        conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_1")
        # -- second.
        conv2d = tf.contrib.layers.batch_norm(conv2d,
                                              # is_training=self._training_flag,
                                              is_training=True,
                                              decay=0.9,
                                              # zero_debias_moving_mean=True,
                                              # renorm=True,
                                              # updates_collections=None,
                                              scope=name + "_batch_norm_2")
        conv2d = tf.nn.relu(conv2d, name=name + "_relu_2")
        conv2d = tf.layers.conv2d(conv2d, 64, 3, 1, 'same',
                                  # activation=tf.nn.relu,
                                  use_bias=False,
                                  kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
                                  name=name + "_conv_2")
        conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_2")
        # -- third.
        conv2d = tf.contrib.layers.batch_norm(conv2d,
                                              # is_training=self._training_flag,
                                              is_training=True,
                                              decay=0.9,
                                              # zero_debias_moving_mean=True,
                                              # renorm=True,
                                              # updates_collections=None,
                                              scope=name + "_batch_norm_3")
        conv2d = tf.nn.relu(conv2d, name=name + "_relu_3")
        conv2d = tf.layers.conv2d(conv2d, output_channels, 1, 1, 'same',
                                  # activation=tf.nn.relu,
                                  use_bias=False,
                                  kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
                                  name=name + "_conv_3")
        conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_3")

        # Finally Add the conv features and the shortcut outputs.
        return tf.add(conv2d, short_cut, name=name + '_res_out')

        # # Firstly pass the input layer through a Batch-Normalization.
        # inputs = tf.layers.batch_normalization(input_layer, training=self._training_flag, name=name+"_batch_norm")
        # # inputs = input_layer
        #
        # # Introduce the shortcut, which is directly the inputs.
        # short_cut = inputs
        #
        # # Pass the input through the "Bottle Structure".
        # conv2d = tf.layers.conv2d(inputs, 64, kernel_size, 1, 'same', activation=tf.nn.relu,
        #                           kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
        #                           name=name+"_conv_1")
        # conv2d = tf.layers.conv2d(conv2d, 64, kernel_size, 1, 'same', activation=tf.nn.relu,
        #                           kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
        #                           name=name+"_conv_2")
        # conv2d = tf.layers.conv2d(conv2d, output_channels, kernel_size, 1, 'same', activation=tf.nn.relu,
        #                           kernel_regularizer=kernel_regularizer, dilation_rate=dilation_rate,
        #                           name=name + "_conv_3")
        #
        # # Finally add the conv features and the shortcut outputs.
        # return tf.add(conv2d, short_cut, name=name+'_res_out')

    def __residual_block(self, input_layer, layer_num, output_channels, kernel_regularizer, kernel_size=3,
                           dilation_rate=1, name="residual_block"):
        # Assign for convenient iteration.
        conv2d = input_layer

        # Iteratively add the residual layer to construct residual block.
        for l in range(layer_num):
            conv2d = self.__residual(conv2d, output_channels, kernel_regularizer, kernel_size,
                                     dilation_rate=dilation_rate, name=name+'_'+str(l))
        # Finish the construction of residual block, and return the output of residual block.
        return conv2d

    def __transition_down(self, input_layer, output_channels, name='transition'):

        # Pass the input through the "Bottle Structure".
        # -- first.
        conv2d = tf.contrib.layers.batch_norm(input_layer,
                                              # is_training=self._training_flag,
                                              is_training=True,
                                              decay=0.9,
                                              # zero_debias_moving_mean=True,
                                              # renorm=True,
                                              # updates_collections=None,
                                              scope=name + "_batch_norm_1")
        conv2d = tf.nn.relu(conv2d, name=name + "_relu_1")
        conv2d = tf.layers.conv2d(conv2d, 64, 3, 1, 'same',
                                  # activation=tf.nn.relu,
                                  use_bias=False,
                                  name=name + "_conv_1")
        conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_1")
        # -- second.
        conv2d = tf.contrib.layers.batch_norm(conv2d,
                                              # is_training=self._training_flag,
                                              is_training=True,
                                              decay=0.9,
                                              # zero_debias_moving_mean=True,
                                              # renorm=True,
                                              # updates_collections=None,
                                              scope=name + "_batch_norm_2")
        conv2d = tf.nn.relu(conv2d, name=name + "_relu_2")
        conv2d = tf.layers.conv2d(conv2d, 64, 3, 2, 'same',
                                  # activation=tf.nn.relu,
                                  use_bias=False,
                                  name=name + "_conv_2")
        conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_2")

        # Finally Add the conv features and the shortcut outputs.
        return conv2d

        # # Introduce the shortcut, which is directly the inputs.
        # short_cut = tf.layers.max_pooling2d(input_layer, 2, 2, 'same',
        #                                     name=name + '_trans_maxpool')  # 60,60,64     * fp
        #
        # # Pass the input through the "Bottle Structure".
        # # -- first.
        # conv2d = tf.contrib.layers.batch_norm(input_layer,
        #                                       # is_training=self._training_flag,
        #                                       is_training=True,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       renorm=True,
        #                                       # updates_collections=None,
        #                                       scope=name + "_batch_norm_1")
        # conv2d = tf.nn.relu(conv2d, name=name + "_relu_1")
        # conv2d = tf.layers.conv2d(conv2d, 64, 1, 1, 'same',
        #                           # activation=tf.nn.relu,
        #                           use_bias=False,
        #                           name=name + "_conv_1")
        # conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_1")
        # # -- second.
        # conv2d = tf.contrib.layers.batch_norm(conv2d,
        #                                       # is_training=self._training_flag,
        #                                       is_training=True,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       renorm=True,
        #                                       # updates_collections=None,
        #                                       scope=name + "_batch_norm_2")
        # conv2d = tf.nn.relu(conv2d, name=name + "_relu_2")
        # conv2d = tf.layers.conv2d(conv2d, 64, 3, 2, 'same',
        #                           # activation=tf.nn.relu,
        #                           use_bias=False,
        #                           name=name + "_conv_2")
        # conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_2")
        # # -- third.
        # conv2d = tf.contrib.layers.batch_norm(conv2d,
        #                                       # is_training=self._training_flag,
        #                                       is_training=True,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       renorm=True,
        #                                       # updates_collections=None,
        #                                       scope=name + "_batch_norm_3")
        # conv2d = tf.nn.relu(conv2d, name=name + "_relu_3")
        # conv2d = tf.layers.conv2d(conv2d, output_channels, 1, 1, 'same',
        #                           # activation=tf.nn.relu,
        #                           use_bias=False,
        #                           name=name + "_conv_3")
        # conv2d = tf.nn.dropout(conv2d, keep_prob=self._keep_prob, name=name + "_dropout_3")
        #
        # # Finally Add the conv features and the shortcut outputs.
        # return tf.add(conv2d, short_cut, name=name + '_res_out')

        # # Then pass through a Batch-Normalization and a ReLU activation.
        # inputs = tf.contrib.layers.batch_norm(input_layer, is_training=self._training_flag,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       # updates_collections=None,
        #                                       renorm=True,
        #                                       scope=name + '_bn_1')
        # # inputs = tf.nn.relu(inputs, name=name + '_relu_1')
        # # Pass through a conv layer.
        # inputs = tf.layers.conv2d(inputs, output_channels,
        #                           1, 1, 'same',
        #                           use_bias=False, name=name + '_conv2d_1x1')
        # inputs = tf.nn.dropout(inputs, keep_prob=self._keep_prob, name=name + "_dropout_1")
        #
        # # Then pass through a Batch-Normalization and a ReLU activation.
        # inputs = tf.contrib.layers.batch_norm(inputs, is_training=self._training_flag,
        #                                       # decay=0.9, zero_debias_moving_mean=True,
        #                                       # updates_collections=None,
        #                                       renorm=True,
        #                                       scope=name + '_bn_2')
        # # inputs = tf.nn.relu(inputs, name=name + '_relu_2')
        # # A conv with stride 2.
        # inputs = tf.layers.conv2d(inputs, output_channels,
        #                           3, 2, 'same', use_bias=False,
        #                           name=name + '_conv2d_3x3')
        # inputs = tf.nn.dropout(inputs, keep_prob=self._keep_prob, name=name + "_dropout_2")
        #
        # # Finish the transition layer.
        # return inputs
        #
        # # # Firstly pass the input layer through a Batch-Normalization.
        # # inputs = tf.layers.batch_normalization(input_layer, training=self._training_flag, name=name+'_bn')
        # # # inputs = input_layer
        # #
        # # # Then pass through a ReLU activation.
        # # inputs = tf.nn.relu(inputs)
        # # # Pass through a conv layer.
        # # inputs = tf.layers.conv2d(inputs, output_channels, 1, 1, 'same', name=name + '_conv2d_1x1')
        # # # A conv with stride 2.
        # # inputs = tf.layers.conv2d(inputs, output_channels, 3, 2, 'same', name=name + '_conv2d_3x3')
        # # # Finish the transition layer.
        # # return inputs

    #### --------------------- Low-High Fusion Related -----------------------------
    def __gen_low_freq(self, tensor, whole_chans, feats_pro, block_num, weights_regularizer):
        r'''
            Generate the low frequency channels.

        :param name_scope:
        :param tensor:
        :return:
        '''

        # A down transition layer to down scale the feature maps to half shape.
        resnet_LF_down_trans = self.__transition_down(tensor, tensor.get_shape()[-1],
                                                      name='ResNet_LF_trans_down_' + str(whole_chans))

        # Pass through the residual blocks.
        resnet_LF_res_block = self.__residual_block(resnet_LF_down_trans, block_num, round(whole_chans * feats_pro),
                                                    weights_regularizer,
                                                    kernel_size=3, dilation_rate=1,
                                                    name='ResNet_residual_layer_' + str(whole_chans))

        # Finish generating the low frequency.
        return resnet_LF_res_block


    def __fusion_func(self, LF_tensor, HF_tensor, whole_chans, low_Fp=None, high_Fp=None, arg=None):
        r'''
            Generate the selective tensor. Main idea is to weight and sum the
                feature maps of high frequency and low frequency.

        :param tensor:
        :param feats_stride:
        :param arg:
        :return:
        '''

        # Check validity.
        if high_Fp == 0.:
            raise Exception('The high frequency proportion cannot be 0. !!!')
        if high_Fp + low_Fp != 1.:
            raise Exception('The sum of HF_pro and LF_pro should be 1. !!!')

        # Get weights regularizer from arg.
        if arg is not None:
            weights_regularizer = arg
        else:
            weights_regularizer = None

        # Get the shape level.
        shape_level = HF_tensor.get_shape().as_list()[1]

        # The smallest shape of low frequency feature maps is 15 !!! Directly return.
        if shape_level < 30:
            raise Exception('Can not generate smaller shape tensor !!!')

        # Indicating it is the first layer.
        if LF_tensor is None:
            # ----------------------------------- High Frequency Part -----------------------------
            Hout_channels = round(whole_chans * high_Fp)
            # Add a conv-layer to high freq as the transition layer.
            freq_H2H = tf.layers.conv2d(HF_tensor, Hout_channels, 3, 1, 'same',
                                        activation=tf.nn.relu,
                                        kernel_regularizer=weights_regularizer,
                                        dilation_rate=1, name='freq_H2H_' + str(shape_level))
            # Only H2H part.
            HF_fusion_tensor = freq_H2H

            # ----------------------------------- Low Frequency Part -----------------------------
            Lout_channels = round(whole_chans * low_Fp)
            # A average pooling layer to down-sample the feature maps to
            #   the same shape as low frequency.
            high_2low_freq = tf.layers.average_pooling2d(HF_tensor, 2, 2, 'same',
                                                         name='high_2low_freq_' + str(shape_level))
            # Pass through a conv-layer to translate the shape to the same as low frequency.
            freq_H2L = tf.layers.conv2d(high_2low_freq, Lout_channels, 3, 1, 'same',
                                        activation=tf.nn.relu,
                                        kernel_regularizer=weights_regularizer,
                                        dilation_rate=1, name='freq_H2L_' + str(shape_level))
            # Only H2L part.
            LF_fusion_tensor = freq_H2L

        # Indicating it's in the mediate layers.
        else:
            # ----------------------------------- High Frequency Part -----------------------------
            Hout_channels = round(whole_chans * high_Fp)
            # Pass through a conv-layer to translate the shape to the same as high frequency.
            low_2high_freq = tf.layers.conv2d(LF_tensor, Hout_channels, 3, 1, 'same',
                                              activation=tf.nn.relu,
                                              kernel_regularizer=weights_regularizer,
                                              dilation_rate=1, name='low_2high_freq_' + str(shape_level))
            # A average pooling layer to up-sample the feature maps to
            #   the same shape as high frequency.
            freq_L2H = tf.layers.conv2d_transpose(low_2high_freq, Hout_channels, 3, 2, 'same',
                                                  name='freq_L2H_' + str(shape_level))
            # Add a conv-layer to high freq as the transition layer.
            freq_H2H = tf.layers.conv2d(HF_tensor, Hout_channels, 3, 1, 'same',
                                        activation=tf.nn.relu,
                                        kernel_regularizer=weights_regularizer,
                                        dilation_rate=1, name='freq_H2H_' + str(shape_level))
            # Add the two part tensor to generate the fusion tensor of "Low and High".
            HF_fusion_tensor = tf.add(freq_H2H, freq_L2H, name='HF_fusion_feats_' + str(shape_level))

            # ----------------------------------- Low Frequency Part -----------------------------
            # Indicating it's the last layer.
            if low_Fp == 0.:
                LF_fusion_tensor = None

            # Indicating it is in the mediate layers.
            else:
                Lout_channels = round(whole_chans * low_Fp)
                # A average pooling layer to down-sample the feature maps to
                #   the same shape as low frequency.
                high_2low_freq = tf.layers.average_pooling2d(HF_tensor, 2, 2, 'same',
                                                             name='high_2low_freq_' + str(shape_level))
                # Pass through a conv-layer to translate the shape to the same as low frequency.
                freq_H2L = tf.layers.conv2d(high_2low_freq, Lout_channels, 3, 1, 'same',
                                            activation=tf.nn.relu,
                                            kernel_regularizer=weights_regularizer,
                                            dilation_rate=1, name='freq_H2L_' + str(shape_level))
                # Add a conv-layer to low frequency as the transition layer.
                freq_L2L = tf.layers.conv2d(LF_tensor, Lout_channels, 3, 1, 'same',
                                            activation=tf.nn.relu,
                                            kernel_regularizer=weights_regularizer,
                                            dilation_rate=1, name='freq_L2L_' + str(shape_level))
                # Add the two part tensor to generate the fusion tensor of "Low and High".
                LF_fusion_tensor = tf.add(freq_L2L, freq_H2L, name='LF_fusion_feats_' + str(shape_level))

        print('### Finish the fusion of high and low frequency. '
              'The high frequency part: {}, the low frequency part: {}'.format(
            HF_fusion_tensor, LF_fusion_tensor
        ))

        # Return the selective tensor.
        return HF_fusion_tensor, LF_fusion_tensor


    def __pack_downsample_feats(self, feats_dict, feature_stride, HF_tensor, LF_tensor, weights_regularizer):
        r'''
            Package the down-sample feature tensors into the feature dictionary.
                Which maybe used to up-sample network.

        :param feats_dict:
        :param tensor:
        :param FLHF_flag:
        :return:
        '''

        # Assign the feature dictionary according to the flag.
        if LF_tensor is None:
            key = 'feats-' + str(feature_stride)
            if key in feats_dict:
                key += '-dup'
            # Directly use the high frequency tensor.
            feats_dict[key] = HF_tensor
            # # Directly use the high frequency tensor.
            # feats_dict['feats-'+str(feature_stride)] = HF_tensor

        # Enable the FLHF.
        else:
            # Calculate the whole channels of output.
            Hout_chans = HF_tensor.get_shape().as_list()[-1]
            Lout_chans = LF_tensor.get_shape().as_list()[-1]
            whole_chans = int(Hout_chans + Lout_chans)

            # Pass through a conv-layer to translate the shape to the same as high frequency.
            low_2high_freq = tf.layers.conv2d(LF_tensor, whole_chans, 3, 1, 'same',
                                              activation=tf.nn.relu,
                                              kernel_regularizer=weights_regularizer,
                                              dilation_rate=1, name='fd_l2h_freq_' + str(feature_stride))
            # A average pooling layer to up-sample the feature maps to
            #   the same shape as high frequency.
            freq_L2H = tf.layers.conv2d_transpose(low_2high_freq, whole_chans, 3, 2, 'same',
                                                  name='fd_l2h_tensor_' + str(feature_stride))
            # Add a conv-layer to high freq as the transition layer.
            freq_H2H = tf.layers.conv2d(HF_tensor, whole_chans, 3, 1, 'same',
                                        activation=tf.nn.relu,
                                        kernel_regularizer=weights_regularizer,
                                        dilation_rate=1, name='fd_h2h_tensor_' + str(feature_stride))
            # Add the two part tensor to generate the fusion tensor of "Low and High".
            HF_fusion_tensor = tf.add(freq_H2H, freq_L2H, name='feats_dict_' + str(feature_stride))

            # Use the fusion tensor.
            feats_dict['feats-'+str(feature_stride)] = HF_fusion_tensor

        # Finish assignment.
        return
