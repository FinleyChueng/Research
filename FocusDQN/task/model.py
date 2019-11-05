import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import tfmodule.layer as cus_layers
import tfmodule.residual as cus_res
import tfmodule.util as net_util
import util.config as conf_util


# import os
# config_file = '/FocusDQN/config.ini'
# config_file = os.path.abspath(os.path.dirname(os.getcwd())) + config_file


class DqnAgent:
    r'''
        The task-specific neural network model, which incorporate the feature extractor and up-sample
            network. It's used to predict the action-value function for the DQN agent deal with the
            image segmentation task.
    '''

    def __init__(self, config, name_space):
        r'''
            The initialization method of implementation.
        '''

        # Assign
        self._config = config
        self._name_space = name_space

        # Get detailed configuration.
        conf_base = self._config['Base']
        conf_train = self._config['Training']
        conf_dqn = self._config['DQN']

        # Normal Initialization.
        self._input_shape = conf_base.get('input_shape')
        self._clz_dim = conf_base.get('classification_dimension')

        # Score maps holder.
        self._score_maps = None

        # Determine the action dimension according to the config.
        if conf_dqn.get('restriction_action'):
            self._action_dim = 9
        else:
            self._action_dim = 17

        # The name scope pair specified to support the "Double DQN".
        self._DoubleDQN_scope = conf_dqn.get('double_dqn', ['ORG', 'TAR'])
        # Check whether enable "Prioritized Replay" or not.
        self._prioritized_replay = conf_dqn.get('prioritized_replay', True)
        # Check whether enable "Dueling Network" or not.
        self._dueling_network = conf_dqn.get('dueling_network', True)







        # -------------- The holders for inference. --------------
        # The public holders.
        self._train_phrase = None
        self._input = None
        # The DRQN holders.
        self._drqn_output = None
        # UN holders.
        self._un_output = None

        # -------------- The holders for loss-related operations. --------------
        # The DRQN related holders.
        self._target_drqn_output = None

        # Regularization Related.
        regularize_coef = conf_train.get('regularize_coef', 0.0)
        self._regularizer = tf_layers.l2_regularizer(regularize_coef)

        # The final holder, loss and summary dictionary.
        self._ios_dict = None
        self._loss_dict = None
        self._summary_dict = None

        # Finish initialization
        return




    def definition(self,
                   fuse_RF_feats=True,
                   dqn_name_scope_pair=None,
                   prioritized_replay=False,
                   dueling_network=False,
                   stepwise_training_name_scope=None):
        r'''
            Definition of the whole model. Including build architecture and define loss function.

        :param dqn_name_scope_pair:
        :param prioritized_replay:
        :param dueling_network:
        :param info_interact:
        :param fuse_RF_feats:
        :param weight_PN_loss:
        :param weight_FTN_loss:
        :param stepwise_training_name_scope
        :return:
        '''



        # Then get the name scope pair of "Double DQN".
        if dqn_name_scope_pair is None or isinstance(dqn_name_scope_pair, tuple):
            # Assign the value.
            self._DDQN_namescpoe_pair = dqn_name_scope_pair
        else:
            raise TypeError('The dqn_name_scope_pair must be a tuple of str or NoneType !!!')


        self.__architecture(action_dim=self._action_dim,
                            training_phrase=None,
                            name_space='Test')






        ##################

        # print('------ The begin of the definition of whole model ------')
        #
        # # Define the placeholder (Scalar) for the training phrase flag, which will be used
        # #   to control the reshape size of DRQN tensor.
        # self._train_phrase = tf.placeholder(tf.bool, name=self._base_name_scope + 'Training_Phrase')  # scalar
        #
        # # Build the "Feature Extract Network".
        # if dqn_name_scope_pair is None:
        #     # Means only need raw FEN.
        #     print('Raw FEN !')
        #     # Building...
        #     self._image, \
        #     self._fe_output, \
        #     self._feature_stride, \
        #     feats_dict = self.__build_Feature_Extract_Net(name_scope=self._base_name_scope + 'FEN',
        #                                                   fuse_RF_feats=fuse_RF_feats)
        # else:
        #     # Means duplicate the FEN.
        #     print('Double FEN !')
        #     origin_scope, target_scope = dqn_name_scope_pair
        #     # Build the raw.
        #     self._image, \
        #     self._fe_output, \
        #     self._feature_stride, \
        #     feats_dict = self.__build_Feature_Extract_Net(name_scope=self._base_name_scope + origin_scope + '/FEN',
        #                                                   fuse_RF_feats=fuse_RF_feats)
        #     # Build the duplication.
        #     self._target_image, \
        #     target_fe_output, \
        #     _3, \
        #     _4 = self.__build_Feature_Extract_Net(name_scope=self._base_name_scope + target_scope + '/FEN',
        #                                           fuse_RF_feats=fuse_RF_feats)
        #
        # # Build the "Deep Recurrent Q Network" (DQN part).
        # if dqn_name_scope_pair is None:
        #     # Means do not use the "Double DQN" architecture.
        #     print('Not enable "Double DQN"')
        #     # Building the whole network...
        #     self._drqn_in, \
        #     self._drqn_gru_Sin, \
        #     drqn_conv_flat, \
        #     self._drqn_gru_state, \
        #     self._drqn_output = self.__build_Deep_Recurrent_Q_Net(
        #         FEN_output=self._fe_output,
        #         name_scope=self._base_name_scope + 'Agent',
        #         dueling_network=dueling_network
        #     )
        # else:
        #     # Means enable the "Double DQN" architecture.
        #     print('Define the model in #" Double DQN "# mode !!!')
        #     origin_scope, target_scope = dqn_name_scope_pair
        #     # Building the "Origin" network...
        #     self._drqn_in, \
        #     self._drqn_gru_Sin, \
        #     drqn_conv_flat, \
        #     self._drqn_gru_state, \
        #     self._drqn_output = self.__build_Deep_Recurrent_Q_Net(
        #         FEN_output=self._fe_output,
        #         name_scope=self._base_name_scope + origin_scope + '/Agent',
        #         dueling_network=dueling_network
        #     )
        #     # Building the "Target" network...
        #     self._target_drqn_in, \
        #     self._target_drqn_gru_Sin, \
        #     _3, \
        #     _4, \
        #     self._target_drqn_output = self.__build_Deep_Recurrent_Q_Net(
        #         FEN_output=target_fe_output,
        #         name_scope=self._base_name_scope + target_scope + '/Agent',
        #         dueling_network=dueling_network
        #     )
        #
        # # Build the "Up-sample Network". That is, "Segmentation Network".
        # self._selective_mask, upfe_trans, self._un_output, self._un_score_maps = self.__build_Upsample_Net(
        #     name_scope=self._base_name_scope + 'UN',
        #     feats_dict=feats_dict
        # )
        #
        # print('------ The end of the definition of whole model ------')
        #
        # # Package the network holders.
        # model_ios_dict = {
        #     # public holders.
        #     'Training_phrase': self._train_phrase,
        #     'FEATURE_Stride': self._feature_stride,
        #     'image': self._image,
        #     'FEN/output': self._fe_output,
        #     'FEN/feats_dict': feats_dict,
        #     # DRQN holders.
        #     'DRQN/input': self._drqn_in,
        #     'DRQN/GRU_sin': self._drqn_gru_Sin,
        #     'DRQN/GRU_sout': self._drqn_gru_state,
        #     'DRQN/output': self._drqn_output,
        #     # UN holders.
        #     'UN/select_mask': self._selective_mask,
        #     'UN/up_fet': upfe_trans,
        #     'UN/output': self._un_output
        # }
        #
        # # Finish the network construction and return the holders in map form.
        # return model_ios_dict
        #
        #
        # ####################################
        #
        #
        #
        #
        #
        #
        # # Define the loss function and summaries of whole model.
        # self._loss_dict, self._summary_dict = self.__loss_summary(prioritized_replay=prioritized_replay)

        return self._ios_dict, self._loss_dict, self._summary_dict


    def notify_copy2_DDQN(self, tf_sess, only_head):
        r'''
            Copy the parameters from Origin DQN to the Target DQN. To support the "Double DQN".

        :param tf_sess: The tensorflow session supplied by caller method.
        :return:
        '''

        # Only execute when specify the name scope pair.
        if self._DDQN_namescpoe_pair is not None:
            # Get name scope pair.
            from_namespace, to_namespace = self._DDQN_namescpoe_pair
            # Concat the base name scope.
            from_namespace = self._base_name_scope + from_namespace
            to_namespace = self._base_name_scope + to_namespace
            # Only copy the parameters of head if specified.
            if only_head:
                from_namespace += '/Agent'
                to_namespace += '/Agent'
            # Operation to copy parameters.
            ops_base = net_util.copy_model_parameters(from_namespace, to_namespace)
            # Execute the operation.
            tf_sess.run(ops_base)

        return



    #### --------------------- Model Definition Related ----------------------------
    def __architecture(self,
                       action_dim,
                       training_phrase,
                       name_space,
                       with_segmentation=False,
                       classification_dim=-1):
        r'''
            Construct the whole architecture. Specify by the parameters.

        --------------------------------------------------------------------------------
        Parameters:
            action_dim: Determine the quantity of "DQN" action branch.
            training_phrase: A tensor indicating whether is "Training Phrase" or not.
            name_space: The name space for this model.
            with_segmentation: Whether should build the "Segmentation" branch or not.
            classification_dim: Determine the dimension for "Segmentation" branch.

        ---------------------------------------------------------------------------------
        Return:
            The holders for each inputs and outputs.
        '''

        # Get detailed configuration.
        conf_base = self._config['Base']
        conf_train = self._config['Training']
        conf_cus = self._config['Custom']
        # Input shape.
        input_shape = conf_base.get('input_shape')
        # Feature normalization method and activation function.
        fe_norm = conf_base.get('feature_normalization', 'batch')
        activation = conf_base.get('activation', 'relu')
        # Dropout (keep probability) for convolution and fully-connect operation.
        conv_kprob = conf_base.get('convolution_dropout', 1.0)
        fc_kprob = conf_base.get('fully_connect_dropout', 0.5)
        # Regularization Related.
        regularize_coef = conf_train.get('regularize_coef', 0.0)
        regularizer = tf_layers.l2_regularizer(regularize_coef)


        # Score maps holder.
        score_maps = None

        # The input tensor holder.
        input_name = name_space+'/input'
        input_tensor = tf.placeholder(tf.float32, input_shape, name=input_name)  # [?, 224, 224, ?]

        # Crop or resize the input image into suitable size.


        # Determine the introduction method of "Position Information".
        pos_method = conf_cus.get('position_info')
        # Define the "Position Information" placeholder.
        pos_name = name_space+'/position_info'
        if pos_method == 'map':
            pos_info = tf.placeholder(tf.float32, input_shape[:-1], name=pos_name)   # [?, w, h]
        elif pos_method == 'coord':
            pos_info = tf.placeholder(tf.float32, [None, 4], name=pos_name)     # [?, 4]
        elif pos_method == 'sight':
            pos_info = tf.placeholder(tf.float32, [None, 4], name=pos_name)     # [?, 4]
        elif pos_method == 'none':
            pass
        else:
            raise TypeError('Unknown position information fusion method !!!')

        # --------------------------------- "Feature Extraction" backbone. ------------------------------------
        # Get configuration for ResNet
        conf_res = self._config['ResNet']
        # Layer number and kernel number of blocks for ResNet.
        kernel_numbers = conf_res.get('kernel_numbers')
        layer_units = conf_res.get('layer_units')

        # Start definition.
        FE_name = name_space + '/FeatExt'
        with tf.variable_scope(FE_name):
            # Base conv to reduce the feature map size.
            base_conv = cus_layers.base_conv2d(input_tensor, kernel_numbers[0], 7, 2,
                                               feature_normalization=fe_norm,
                                               activation='lrelu',
                                               keep_prob=conv_kprob,
                                               regularizer=regularizer,
                                               name_space='ResNet_bconv')     # [?, 112, 112, ?]

            # Recursively build the block part.
            block_tensor = base_conv
            for idx in range(len(layer_units)):
                # Scale down the feature maps size. Use idx to judge
                #   whether use "Max Pooling" or "Transition".
                if idx == 0:
                    # A max pooling layer to down-sample the feature maps to 56, 56.
                    block_tensor = tf.layers.max_pooling2d(block_tensor, 3, 2, 'same',
                                                           name='ResNet_mp')
                else:
                    # A transition layer to down-sample the feature maps to 28, 28.
                    block_tensor = cus_res.transition_layer(block_tensor, kernel_numbers[idx+1],
                                                            feature_normalization=fe_norm,
                                                            activation=activation,
                                                            keep_prob=conv_kprob,
                                                            regularizer=regularizer,
                                                            name_space='ResNet_Trans0'+str(idx+1))
                # Pass through the residual block.
                block_tensor = cus_res.residual_block(block_tensor, kernel_numbers[idx+1], layer_units[idx]-1,
                                                      feature_normalization=fe_norm,
                                                      activation=activation,
                                                      keep_prob=conv_kprob,
                                                      regularizer=regularizer,
                                                      name_space='ResNet_Block0'+str(idx+1))

            # For conveniently usage.
            FE_tensor = block_tensor    # [?, 7, 7, ?]  default: 2048

            # Print some information.
            print('### Finish "Feature Extract Network" (name scope: {}). The output shape: {}'.format(
                name_space, FE_tensor.shape))

        # --------------------------------- "Region Selection" (DQN) branch. ------------------------------------
        # Get configuration for DQN.
        conf_dqn = self._config['DQN']
        # Get the dimension reduction method.
        reduce_dim = conf_dqn['reduce_dim']
        # Check whether enable "Dueling Network" or not.
        dueling_network = conf_dqn.get('dueling_network', True)

        # Start definition.
        DQN_name = name_space + '/DQN'
        with tf.variable_scope(DQN_name):
            # Scale down the feature maps according to the specific method.
            if reduce_dim == 'conv':
                # Use "Convolution" to reduce dimension.
                redc_tensor = cus_layers.base_conv2d(FE_tensor, 1024, 3, 2,
                                                     feature_normalization=fe_norm,
                                                     activation=activation,
                                                     keep_prob=conv_kprob,
                                                     regularizer=regularizer,
                                                     name_space='reduce_dim01')     # [?, 4, 4, 1024]
                redc_tensor = cus_layers.base_conv2d(redc_tensor, 512, 3, 2,
                                                     feature_normalization=fe_norm,
                                                     activation=activation,
                                                     keep_prob=conv_kprob,
                                                     regularizer=regularizer,
                                                     name_space='reduce_dim02')     # [?, 2, 2, 512]
            elif reduce_dim == 'residual':
                # Use "residual" structure to reduce dimension.
                redc_tensor = cus_res.transition_layer(FE_tensor, 1024,
                                                       feature_normalization=fe_norm,
                                                       activation=activation,
                                                       keep_prob=conv_kprob,
                                                       regularizer=regularizer,
                                                       name_space='reduce_dim01')  # [?, 4, 4, 1024]
                redc_tensor = cus_res.transition_layer(redc_tensor, 512,
                                                       feature_normalization=fe_norm,
                                                       activation=activation,
                                                       keep_prob=conv_kprob,
                                                       regularizer=regularizer,
                                                       name_space='reduce_dim02')  # [?, 2, 2, 512]
            else:
                raise TypeError('Unknown reduce dimension method for DQN !!!')

            # Flatten the tensor to 1-D vector.
            fdim = 1
            for d in redc_tensor.shape[1:]:
                fdim *= int(d)
            flat_tensor = tf.reshape(redc_tensor, [-1, fdim], name='flatten')     # [?, OC]   default: 2048

            # Pass through two fully connected layers.
            fc01_tensor = cus_layers.base_fc(flat_tensor, 1024,
                                             feature_normalization=fe_norm,
                                             activation=activation,
                                             keep_prob=fc_kprob,
                                             regularizer=regularizer,
                                             name_space='FC01')     # [?, 1024]
            fc02_tensor = cus_layers.base_fc(fc01_tensor, 1024,
                                             feature_normalization=fe_norm,
                                             activation=activation,
                                             keep_prob=fc_kprob,
                                             regularizer=regularizer,
                                             name_space='FC02')     # [?, 1024]

            # Build the DRQN header according to the "Dueling Network" mode or not.
            if dueling_network:
                # Separate the feature map produced by "FC-layer" into the "State" and "Action" branches.
                fc02vec_shape = int(fc02_tensor.shape[-1])
                bch_shape = fc02vec_shape // 2
                state_bch, action_bch = tf.split(fc02_tensor,
                                                 [bch_shape, fc02vec_shape - bch_shape],
                                                 axis=-1,
                                                 name='branch_split')  # [-1, h_size/2]
                # Build the "State" branch.
                state_tensor = cus_layers.base_fc(state_bch, 1,
                                                  feature_normalization=fe_norm,
                                                  activation=activation,
                                                  keep_prob=fc_kprob,
                                                  regularizer=regularizer,
                                                  name_space='state_value')     # [?, 1]
                # Build the "Action" branch.
                action_tensor = cus_layers.base_fc(action_bch, action_dim,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=fc_kprob,
                                                   regularizer=regularizer,
                                                   name_space='action_value')  # [?, act_dim]
                # Mean the "Action" (Advance) branch.
                norl_Adval = tf.subtract(action_tensor,
                                         tf.reduce_mean(action_tensor, axis=-1, keep_dims=True),
                                         name='advanced_value')   # [?, act_dim]
                # Add the "State" value and "Action" value to obtain the final output.
                dqn_output = tf.add(state_tensor, norl_Adval, name='DQN_output')    # [?, act_dim]
            # Not enable "Dueling" structure. Directly pass through a FC-layer.
            else:
                # The normal mode, do not need to split into two branches.
                dqn_output = cus_layers.base_fc(fc02_tensor, action_dim,
                                                feature_normalization=fe_norm,
                                                activation=activation,
                                                keep_prob=fc_kprob,
                                                regularizer=regularizer,
                                                name_space='DQN_output')    # [?, act_dim]

            # Print some information.
            print('### Finish "DQN Head" (name scope: {}). The output shape: {}'.format(
                name_space, dqn_output.shape))










        # --------------------------------- "Segmentation" branch. ------------------------------------






        return


    def __loss_summary(self, prioritized_replay):
        r'''
            The definition of the Loss Function of the whole model.

        :param prioritized_replay:
        :param weight_PN_loss:
        :param weight_FTN_loss:
        :return:
        '''

        print('------ The begin of definition of the loss function of whole model. ------')

        # The Loss-related holder dictionary of whole model.
        model_loss_dict = {}

        # A minimal value to avoid "NAN".
        epsilon = 1e-10

        ############################ Definition of UN cross-entropy loss ############################
        # Placeholder of "Visited Flag".
        self._visit = tf.placeholder(tf.float32, [None, None],
                                     name=self._base_name_scope + 'UN_visit_flag')  # [b, t]

        # Placeholder of ground truth segmentation.
        self._GT_segmentation = tf.placeholder(tf.float32,
                                               [None, None, self._input_size[0], self._input_size[1], self._cls_dim],
                                               name=self._base_name_scope + 'UN_GT_segmentation')   # [b, t, w, h, cls]

        # Use the ground truth (label) to calculate weights for each clazz.
        self._clazz_weights = tf.placeholder(tf.float32, [None, None, self._cls_dim],
                                             name=self._base_name_scope + 'UN_clazz_weights')  # [cls]
        # clazz_weights = tf.constant([1, 5, 3, 4, 10], dtype=tf.float32,
        #                             name=self._base_name_scope + 'UN_clazz_weights')    # [cls]
        # clazz_weights = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32,
        #                             name=self._base_name_scope + 'UN_clazz_weights')  # [cls]
        # clazz_statistic = tf.reduce_sum(self._GT_segmentation, axis=(0, 1, 2, 3),
        #                                 name=self._base_name_scope + 'UN_clazz_statistic')  # [cls]
        # total_pixels = tf.reduce_sum(clazz_statistic,
        #                              name=self._base_name_scope + 'UN_total_pixels')     # scalar
        # clazz_prop = tf.divide(clazz_statistic, total_pixels,
        #                        name=self._base_name_scope + 'UN_clazz_proportion')  # [cls]
        # clazz_weights = tf.divide(1., tf.log(1. + clazz_prop),
        #                           name=self._base_name_scope + 'UN_raw_weights')  # [cls]
        # clazz_weights = tf.where(
        #     tf.not_equal(clazz_prop, 0.),
        #     clazz_weights,
        #     tf.zeros_like(clazz_weights),
        #     name=self._base_name_scope + 'UN_clazz_weights'
        # )   # [cls]

        # # The cross-entropy loss for "Up-sample Network", which is actually the "Classification" network.
        # UN_cross_entropy_loss = - self._GT_segmentation * tf.log(
        #     tf.clip_by_value(self._un_output, clip_value_min=epsilon, clip_value_max=1.0),
        #     name=self._base_name_scope + 'UN_cross_entropy_loss'
        # )  # [b, t, w, h, cls_dim]
        # UN_image_CEloss = tf.reduce_mean(UN_cross_entropy_loss, axis=(2, 3),
        #                                  name=self._base_name_scope + 'UN_image_CEloss')  # [b, t, cls]
        # # Multiply the visit flag.
        # UN_visit_filt = tf.divide(UN_image_CEloss,
        #                           tf.expand_dims(self._visit, axis=-1),
        #                           name=self._base_name_scope + 'UN_visit_filt')  # [b, t, cls]
        # # Add the clazz-related weights.
        # UN_cls_CEloss = tf.reduce_mean(UN_visit_filt, axis=(0, 1),
        #                                name=self._base_name_scope + 'UN_cls_CEloss')  # [cls_dim]
        # UN_Wcls_CEloss = tf.multiply(UN_cls_CEloss, clazz_weights,
        #                              name=self._base_name_scope + 'UN_Wcls_CEloss')     # [cls_dim]
        # self._un_loss = tf.reduce_sum(UN_Wcls_CEloss, name=self._base_name_scope + 'UN_loss')   # scalar

        # Iteratively expend the loss.
        self._un_loss = 0.
        for prob in self._un_score_maps:
            # Translate the raw value to real probability.
            probability = tf.nn.softmax(prob)
            # Reshape the flatten output tensor to the time-related shape tensor.
            probability = tf.cond(
                self._train_phrase,
                lambda: tf.reshape(probability, shape=[self._train_batch_size, self._train_track_len,
                                                     self._input_size[0], self._input_size[1], self._cls_dim]),
                lambda: tf.reshape(probability, shape=[self._infer_batch_size, self._infer_track_len,
                                                     self._input_size[0], self._input_size[1], self._cls_dim]),
            )  # [b, t, w, h, cls_dim]
            # The cross-entropy loss for "Up-sample Network", which is actually the "Classification" network.
            UN_cross_entropy_loss = - self._GT_segmentation * tf.log(
                tf.clip_by_value(probability, clip_value_min=epsilon, clip_value_max=1.0),
                name=self._base_name_scope + 'UN_cross_entropy_loss'
            )  # [b, t, w, h, cls_dim]
            UN_image_CEloss = tf.reduce_mean(UN_cross_entropy_loss, axis=(2, 3),
                                             name=self._base_name_scope + 'UN_image_CEloss')  # [b, t, cls]
            # Multiply the visit flag.
            UN_visit_filt = tf.divide(UN_image_CEloss,
                                      tf.expand_dims(self._visit, axis=-1),
                                      name=self._base_name_scope + 'UN_visit_filt')  # [b, t, cls]
            # Add the clazz-related weights.
            UN_Wcls_CEloss = tf.multiply(UN_visit_filt, self._clazz_weights,
                                         name=self._base_name_scope + 'UN_Wcls_CEloss')  # [b, t, cls_dim]
            self._un_loss += tf.reduce_sum(UN_Wcls_CEloss, name=self._base_name_scope + 'UN_loss')  # scalar

        print('### Finish the definition of UN loss, Shape: {}'.format(self._un_loss.shape))

        ################################## Definition of DRQN L2 loss ##################################
        # Placeholder of input actions. Indicates which Q value (output of DRQN) used to calculate cost.
        self._drqn_input_act = tf.placeholder(
            tf.float32, [self._train_batch_size, self._train_track_len, self._mask_dim, self._act_dim],
            name=self._base_name_scope + 'DRQN_input_actions'
        )
        # Placeholder of target q values. That is, the "Reward + Future Q values".
        self._drqn_target_q_val = tf.placeholder(
            tf.float32, [self._train_batch_size, self._train_track_len, self._mask_dim],
            name=self._base_name_scope + 'DRQN_target_q_values'
        )

        # Only use selected Q values for DRQN. Coz in the "Future Q values" we use the max value.
        drqn_act_q_val = tf.reduce_sum(
            tf.multiply(self._drqn_output, self._drqn_input_act),
            axis=-1)    # [batch_size, track_len, mask_dim]

        # According to the "Doom" paper. We only train the latter half track length sample.
        #   Coz the front half will introduce inaccuracy due to the "Zero Initial State".
        front_half_len = self._train_track_len // 2
        track_notr_M = tf.zeros((self._train_batch_size, front_half_len, self._mask_dim), dtype=tf.float32)
        track_2tr_M = tf.ones((self._train_batch_size, self._train_track_len - front_half_len, self._mask_dim),
                              dtype=tf.float32)
        track_mask = tf.concat((track_notr_M, track_2tr_M), axis=1,
                               name=self._base_name_scope + 'DRQN_track_mask')  # [batch_size, track_len, mask_dim]
        # The time-relared Q diff.
        drqn_time_q_diff = tf.subtract(self._drqn_target_q_val, drqn_act_q_val)     # [batch_size, track_len, mask_dim]
        # Multiply the raw time-related Q diff with track mask to filter the train sample.
        drqn_mask_T_Qdiff = tf.multiply(drqn_time_q_diff, track_mask,
                                        name=self._base_name_scope + 'DRQN_masked_time_Q_diff')  # [b, t, mask]

        # Calculate the difference between "Origin" Q values and "Target" Q values for DRQN.
        drqn_q_diff = tf.reduce_mean(drqn_mask_T_Qdiff, axis=(1, 2),
                                     name=self._base_name_scope + 'DRQN_Q_diff')    # [batch_size]

        # Define placeholder for IS weights if use "Prioritized Replay".
        if prioritized_replay:
            # Notification in the console.
            print('Enable the #" Prioritized Replay "# training mode !!!')
            # Define the weights for experience.
            self._exp_priority = tf.abs(drqn_q_diff,
                                        name=self._base_name_scope + 'DRQN_priority')   # used to update Sumtree
            self._ISWeights = tf.placeholder(tf.float32, [None, 1], self._base_name_scope + 'DRQN_IS_weights')
            # Package the abs_error and the ISWeights.
            model_loss_dict['DRQN/exp_priority'] = self._exp_priority
            model_loss_dict['DRQN/ISWeights'] = self._ISWeights
            # Construct the prioritized PN loss.
            self._drqn_loss = tf.reduce_mean(
                tf.multiply(self._ISWeights, tf.square(drqn_q_diff)),   # [batch_size, 1]
                name=self._base_name_scope + 'DRQN_priority_loss'
            )
        else:
            # Only define the simple DRQN loss.
            self._drqn_loss = tf.reduce_mean(
                tf.square(drqn_q_diff),     # [batch_size]
                name=self._base_name_scope + 'DRQN_simple_loss'
            )

        print('### Finish the definition of DRQN loss, Shape: {}'.format(self._drqn_loss.shape))

        ####################### Definition of Whole Loss and Target holders #######################
        # Define the whole loss for model. This consists of three parts:
        #   1) The "Up-sample Network" cross-entropy loss.
        #   2) The "Deep Recurrent Q Network" L2 loss.
        self._whole_loss = tf.add(self._un_loss, self._drqn_loss,
                                  name=self._base_name_scope + 'NET_Whole_loss')
        # self._whole_loss = tf.add(self._un_loss, 0.5 * self._drqn_loss,
        #                           name=self._base_name_scope + 'NET_Whole_loss')

        # Add the regularization loss if enabled.
        if self._l2_regularizer is not None:
            print('Enable Regularization !!!')
            org_namespace, _2 = self._DDQN_namescpoe_pair
            for var in tf.trainable_variables('kernel'):
                vname = var.name
                org_FEN_kernel = vname.startswith(org_namespace + '/FEN') and ('kernel' in vname or 'bias' in vname)
                org_Agent_kernel_var = vname.startswith(org_namespace + '/Agent') and \
                                       ('kernel' in vname or 'var' in vname or 'bias' in vname)
                org_UN_kernel = vname.startswith('UN') and ('kernel' in vname or 'bias' in vname)
                if org_FEN_kernel or org_Agent_kernel_var or org_UN_kernel:
                    self._whole_loss += 1e-7 * tf.nn.l2_loss(var)

                # vname = var.name
                # org_FEN_kernel = vname.startswith(org_namespace + '/FEN') and 'kernel' in vname
                # org_Agent_kernel_var = vname.startswith(org_namespace + '/Agent') and \
                #                        ('kernel' in vname or 'var' in vname)
                # org_UN_kernel = vname.startswith('UN') and 'kernel' in vname
                # if org_FEN_kernel or org_Agent_kernel_var or org_UN_kernel:
                #     self._whole_loss += 1e-4 * tf.nn.l2_loss(var)
                #     # self._whole_loss += 1e-3 * tf.nn.l2_loss(var)

            # var_list = tf.trainable_variables('kernel')
            # print(var_list)

        print('### Finish the definition of Whole loss, Shape: {}'.format(self._whole_loss.shape))

        # Package the Loss-related holders for Whole network.
        model_loss_dict['UN/clazz_weights'] = self._clazz_weights
        model_loss_dict['UN/visit_flag'] = self._visit
        model_loss_dict['UN/GT_segmentation'] = self._GT_segmentation
        model_loss_dict['DRQN/input_act'] = self._drqn_input_act
        model_loss_dict['DRQN/target_q_val'] = self._drqn_target_q_val
        model_loss_dict['UN/loss'] = self._un_loss
        model_loss_dict['DRQN/loss'] = self._drqn_loss
        model_loss_dict['NET/Whole_loss'] = self._whole_loss

        # Only record target holders when specify the name scope pair.
        if self._DDQN_namescpoe_pair is not None:
            # Target DRQN holders.
            model_loss_dict['DRQN/target_image'] = self._target_image
            model_loss_dict['DRQN/target_input'] = self._target_drqn_in
            model_loss_dict['DRQN/target_GRU_sin'] = self._target_drqn_gru_Sin
            model_loss_dict['DRQN/target_output'] = self._target_drqn_output

        print('------ The end of the definition of the loss function of whole model. ------')

        ############################# Definition of some summaries of Whole model #############################
        print('------ The begin of definition of the summaries of whole model. ------')

        # Record some summaries information for whole network.
        tf.summary.scalar('NET_Whole_Loss', self._whole_loss)
        # tf.summary.scalar('UN_Loss',
        #                   tf.reduce_sum(tf.multiply(tf.reduce_mean(UN_image_CEloss, axis=(0, 1)), clazz_weights)))
        tf.summary.scalar('UN_Loss', self._un_loss)
        tf.summary.scalar('DRQN_Loss', self._drqn_loss)
        tf.summary.scalar('DRQN_Q_vals', tf.reduce_mean(drqn_act_q_val))
        # Merge list.
        merge_list = [tf.get_collection(tf.GraphKeys.SUMMARIES, 'NET_Whole_Loss'),
                      tf.get_collection(tf.GraphKeys.SUMMARIES, 'UN_Loss'),
                      tf.get_collection(tf.GraphKeys.SUMMARIES, 'DRQN_Loss'),
                      tf.get_collection(tf.GraphKeys.SUMMARIES, 'DRQN_Q_vals')
                      ]

        # Define the reward placeholders for RecN.
        rewards = tf.placeholder(tf.float32, [None, None, self._mask_dim],
                                       name=self._base_name_scope + 'DRQN_reward')
        DICE = tf.placeholder(tf.float32, [None, None, self._cls_dim],
                                    name=self._base_name_scope + 'DRQN_dice')
        BRATS = tf.placeholder(tf.float32, [None, None, 3],
                                    name=self._base_name_scope + 'DRQN_dice')
        # with tf.device('/cpu:0'):
        statis_rewards = tf.reduce_mean(rewards, axis=(0, 1))   # [cls]
        statis_dice = tf.reduce_mean(DICE, axis=(0, 1))     # [cls]
        statis_brats = tf.reduce_mean(BRATS, axis=(0, 1))     # [3]
        for m in range(self._mask_dim):
            tf.summary.scalar('DRQN_rewards_M' + str(m+1), statis_rewards[m])
            merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'DRQN_rewards_M' + str(m+1)))
        for c in range(self._cls_dim):
            tf.summary.scalar('DRQN_dice_C' + str(c+1), statis_dice[c])
            merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'DRQN_dice_C' + str(c+1)))
        for b in range(3):
            tf.summary.scalar('DRQN_brats_' + str(b + 1), statis_brats[b])
            merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'DRQN_brats_' + str(b + 1)))

        # Merge all the summaries.
        self._summaries = tf.summary.merge(merge_list)

        # Package the summaries into dictionary.
        model_summaries_dict = {
            'DRQN/statis_rewards': rewards,
            'NET/DICE': DICE,
            'NET/BRATS': BRATS,
            'NET/summaries': self._summaries
        }

        print('### The summary dict is {}'.format(model_summaries_dict))

        print('------ The end of the summaries of whole model. ------')

        return model_loss_dict, model_summaries_dict


    def __build_Deep_Recurrent_Q_Net(self, FEN_output, name_scope, dueling_network):
        r'''
            The definition of "Deep Recurrent Q" network, Which is actually the DQN agent.
                Note that, we use the "Dueling Network" Architecture.

        Return:
            The "input" and "output" layer of this DQN, which is actually the tensorflow
                operation(output) of tensorflow model.
        '''

        # Check the validity.
        if FEN_output is None:
            raise Exception('One should build the Feature Extract Network first before'
                            'building the Deep Recurrent Q Network !!!')

        # Calculate the input size of the DRQN. It's actually the size (shape)
        #   of the output feature maps of FEN.
        DRQN_inw = self._input_size[0] // self._feature_stride
        DRQN_inh = self._input_size[1] // self._feature_stride
        # DRQN_inc = FEN_output.shape.as_list()[-1]
        DRQN_inc = 32
        # Check validity.
        assert DRQN_inw == FEN_output.shape.as_list()[1]
        assert DRQN_inh == FEN_output.shape.as_list()[2]

        # Build the detailed architecture.
        with tf.variable_scope(name_scope):

            # # Pass through a 1*1 conv-layer as an transition layer. (Meanwhile
            # #   decrease the channels number.)
            # drqn_1x1_trans = tf.layers.conv2d(self._fe_output, DRQN_inc, 1, 1, 'same',
            #                                   activation=tf.nn.relu,
            #                                   kernel_regularizer=self._l2_regularizer,
            #                                   dilation_rate=1, name="DRQN_1x1_trans")  # 15,15,32
            # # Resize to the proper size.
            # resize_fe_outs = tf.image.resize_bilinear(drqn_1x1_trans, (30, 30),
            #                                           name=name_scope + "_bilinear_fe_outs")  # 30,30,32

            # The input of DRQN.
            drqn_in = tf.multiply(FEN_output, 1., name=name_scope + '_DRQN_in')

            # Pass through a 1*1 conv-layer as an transition layer. (Meanwhile
            #   decrease the channels number.)
            DRQN_inc = FEN_output.shape.as_list()[-1]
            drqn_1x1_trans = tf.reshape(drqn_in, [-1, 2 * DRQN_inw, 2 * DRQN_inh, DRQN_inc // 4],
                                        name=name_scope + "_DRQN_rein_30")    # 30,30,64
            resize_fe_outs = tf.contrib.layers.batch_norm(drqn_1x1_trans,
                                                          # is_training=self._train_phrase,
                                                          is_training=True,
                                                          decay=0.9,
                                                          # zero_debias_moving_mean=True,
                                                          # renorm=True,
                                                          # updates_collections=None,
                                                          scope='DRQN_1x1_trans_BN')
            resize_fe_outs = tf.nn.relu(resize_fe_outs, name='DRQN_1x1_trans_Relu')
            resize_fe_outs = tf.layers.conv2d(resize_fe_outs, DRQN_inc // 4, 1, 1, 'same',
                                              use_bias=False,
                                              kernel_regularizer=self._l2_regularizer,
                                              dilation_rate=1, name="DRQN_1x1_trans")  # 30,30,64

            # Pass the re-sized feature maps through two conv-layers to decrease dimension.
            # -- first
            drqn_desc_dim1 = tf.contrib.layers.batch_norm(resize_fe_outs,
                                                          is_training=True,
                                                          # is_training=self._train_phrase,
                                                          # renorm=True,
                                                          decay=0.9,
                                                          # zero_debias_moving_mean=True,
                                                          # updates_collections=None,
                                                          scope='DRQN_desc_dim1_BN')
            drqn_desc_dim1 = tf.nn.relu(drqn_desc_dim1, name='DRQN_desc_dim1_Relu')
            drqn_desc_dim1 = tf.layers.conv2d(drqn_desc_dim1, 64, 4, 2, 'valid',
                                              use_bias=False,
                                              kernel_regularizer=self._l2_regularizer,
                                              dilation_rate=1, name="DRQN_desc_dim1")  # 14,14,64
            # -- second
            drqn_desc_dim2 = tf.contrib.layers.batch_norm(drqn_desc_dim1,
                                                          is_training=True,
                                                          # is_training=self._train_phrase,
                                                          decay=0.9,
                                                          # zero_debias_moving_mean=True,
                                                          # renorm=True,
                                                          # updates_collections=None,
                                                          scope='DRQN_desc_dim2_BN')
            drqn_desc_dim2 = tf.nn.relu(drqn_desc_dim2, name='DRQN_desc_dim2_Relu')
            drqn_desc_dim2 = tf.layers.conv2d(drqn_desc_dim2, 64, 4, 2, 'valid',
                                              use_bias=False,
                                              kernel_regularizer=self._l2_regularizer,
                                              dilation_rate=1, name="DRQN_desc_dim2")  # 6,6,64
            # -- third
            drqn_desc_dim3 = tf.contrib.layers.batch_norm(drqn_desc_dim2,
                                                          is_training=True,
                                                          # is_training=self._train_phrase,
                                                          decay=0.9,
                                                          # zero_debias_moving_mean=True,
                                                          # renorm=True,
                                                          # updates_collections=None,
                                                          scope='DRQN_desc_dim3_BN')
            drqn_desc_dim3 = tf.nn.relu(drqn_desc_dim3, name='DRQN_desc_dim3_Relu')
            drqn_desc_dim3 = tf.layers.conv2d(drqn_desc_dim3, self._gru_hsize // 4, 4, 2, 'valid',
                                              use_bias=False,
                                              kernel_regularizer=self._l2_regularizer,
                                              dilation_rate=1, name="DRQN_desc_dim3")  # 2,2,h_size/4 (256)

            # Reshape (in fact it's flatten) the final conv feature maps.
            drqn_conv_flat = tf.reshape(drqn_desc_dim3, shape=[-1, self._gru_hsize],
                                        name=name_scope + '_DRQN_conv_flat')    # [-1, h_size]

            # -------------------------- Now Build the GRU ----------------------------

            # Add a basic cell of GRU.
            gru_cell = tf.contrib.rnn.GRUCell(num_units=self._gru_hsize)

            # The initial hidden state for GRU cell.
            gru_hstate = tf.cond(self._train_phrase,
                                 lambda: gru_cell.zero_state(self._train_batch_size, tf.float32),
                                 lambda: gru_cell.zero_state(self._infer_batch_size, tf.float32),
                                 name=name_scope + '_DRQN_gru_ini_hstate')  # [batch_size, h_size]

            # Reshape the flatten DRQN conv feature maps to the time-related shape
            #   as the input for RNN.
            drqn_gru_featsin = tf.cond(
                self._train_phrase,
                lambda: tf.reshape(drqn_conv_flat,
                                   shape=[self._train_batch_size, self._train_track_len, self._gru_hsize]),
                lambda: tf.reshape(drqn_conv_flat,
                                   shape=[self._infer_batch_size, self._infer_track_len, self._gru_hsize]),
                name=name_scope + '_DRQN_gru_feats_in'
            )  # [batch_size, track_len, h_size]

            # Pass through a GRU layer to get the time-related output.
            drqn_gru_out, drqn_gru_state = tf.nn.dynamic_rnn(
                inputs=drqn_gru_featsin, cell=gru_cell, dtype=tf.float32, initial_state=gru_hstate,
                scope=name_scope + '_DRQN_gru_net')  # out: [b, t, h], state: [b, h]

            # Reshape (flatten) the output of DRQN GRU layer to 2-D tensor.
            drqn_gru_out_2d = tf.reshape(drqn_gru_out, shape=[-1, self._gru_hsize],
                                         name=name_scope + '_DRQN_gru_out_2D')  # [-1, h_size]

            # -------------------------- The Head of DRQN ----------------------------

            # Build the DRQN header according to the "Dueling Network" mode or not.
            if dueling_network:
                # Separate the feature map produced by "GRU-layer" into the "State" and "Action"
                #   branches. Coz we are using the "Dueling Network" Architecture.
                drqn_state_bch, drqn_act_bch = tf.split(drqn_gru_out_2d,
                                                        [self._gru_hsize // 2, self._gru_hsize - self._gru_hsize // 2],
                                                        axis=1,
                                                        name=name_scope + '_DRQN_branch_split')  # [-1, h_size/2]

                # Build the "State" branch.
                drqn_state_value = self.__fully_connected_layer(input_tensor=drqn_state_bch,
                                                                output_channels=self._mask_dim,
                                                                var_name='DRQN_state_value',
                                                                name_scope=name_scope)  # [b, t, mask]
                # drqn_state_val4D = tf.cond(
                #     self._train_phrase,
                #     lambda: tf.reshape(drqn_state_value,
                #                        shape=[self._train_batch_size, self._train_track_len, self._mask_dim, 1]),
                #     lambda: tf.reshape(drqn_state_value,
                #                        shape=[self._infer_batch_size, self._infer_track_len, self._mask_dim, 1]),
                #     name=name_scope + '_DRQN_state_val4D'
                # )  # [b, t, mask, 1]
                drqn_state_val4D = tf.expand_dims(drqn_state_value, axis=-1,
                                                  name=name_scope + '_DRQN_state_val4D')    # [b, t, mask, 1]

                # Build the "Action" branch.
                drqn_act_value = self.__fully_connected_layer(input_tensor=drqn_act_bch,
                                                              output_channels=self._mask_dim * self._act_dim,
                                                              var_name='DRQN_action_value',
                                                              name_scope=name_scope)  # [b, t, mask * act]
                drqn_act_val4D = tf.cond(
                    self._train_phrase,
                    lambda: tf.reshape(drqn_act_value, shape=[self._train_batch_size, self._train_track_len, self._mask_dim, self._act_dim]),
                    lambda: tf.reshape(drqn_act_value, shape=[self._infer_batch_size, self._infer_track_len, self._mask_dim, self._act_dim]),
                    name=name_scope + '_DRQN_action_val4D'
                )  # [b, t, mask, act_dim]

                # Mean the "Action" (Advance) branch.
                drqn_nom_Adval = tf.subtract(drqn_act_val4D,
                                             tf.reduce_mean(drqn_act_val4D, axis=-1, keep_dims=True),
                                             name=name_scope + '_DRQN_nom_Adval')  # [b, t, mask, act]

                # Add the "State" value and "Action" value to obtain the final output.
                drqn_Coutput = tf.add(drqn_state_val4D, drqn_nom_Adval,
                                      name=name_scope + '_DRQN_head_Coutput')  # [b, t, mask, act_dim]

            # Not enable "Dueling" structure. Directly pass through a FC-layer.
            else:
                # The normal mode, do not need to split into two branches.
                drqn_output = self.__fully_connected_layer(input_tensor=drqn_gru_out_2d,
                                                           output_channels=self._mask_dim * self._act_dim,
                                                           var_name='DRQN_head_out',
                                                           name_scope=name_scope)  # [-1, mask * act_dim]

                print('### Finish the definition of DRQN in total (name scope: {}). The output shape: {}'.format(
                    name_scope, drqn_output.shape))

                # Reshape the flatten output tensor to the time-related shape tensor.
                #   Coz this will be convenient for the "Priority Update".
                drqn_Coutput = tf.cond(
                    self._train_phrase,
                    lambda: tf.reshape(drqn_output, shape=[self._train_batch_size, self._train_track_len, self._mask_dim, self._act_dim]),
                    lambda: tf.reshape(drqn_output, shape=[self._infer_batch_size, self._infer_track_len, self._mask_dim, self._act_dim]),
                    name=name_scope + '_DRQN_time_output'
                )  # [b, t, act_dim]

        print('### Finish the definition of DRQN in class-form (name scope: {}). The output shape: {}'.format(
            name_scope, drqn_Coutput.shape))

        # ### DEBUG
        # if name_scope.startswith('ORG'):
        #     self.DEBUG_drqn_raw_out = drqn_output
        #     self.DEBUG_t_CA_prob = t_CA_prob
        # ### DEBUG

        # Finish the definition of DRQN. Meanwhile return the
        #   1) Input holders: DRQN input tensor, DRQN GRU initial hidden state.
        #   2) Output: DRQN conv features, DRQN GRU output hidden state, DRQN output
        return drqn_in, gru_hstate, drqn_conv_flat, drqn_gru_state, drqn_Coutput


    def __build_backbone(self, name_scope):
        r'''
            Build the "Backbone", which is actually the whole "Feature Extract Network"
                followed by the "Segmentation" branch (head).

        :param name_scope:
        :return:
        '''

        # with tf.scope
        UN_inc = 512
        selective_mask = None
        feats_dict = {}




        # Define the visualization tensor.
        vis_tensor = None

        # Build the detailed architecture.
        with tf.variable_scope(name_scope):

            # -------------------------- The Up-sample Blocks ----------------------------

            # Add a BN and ReLU.
            upfe_trans = tf.contrib.layers.batch_norm(self._fe_output,
                                                      is_training=True,
                                                      # is_training=self._train_phrase,
                                                      decay=0.9,
                                                      # zero_debias_moving_mean=True,
                                                      #     updates_collections=None,
                                                      # renorm=True,
                                                      scope='Up_tensor_trans_BN')
            upfe_trans = tf.nn.relu(upfe_trans, name='Up_tensor_trans_Relu')
            # Add a 1x1 conv as the transition layer.
            upfe_trans = tf.layers.conv2d(upfe_trans, UN_inc, 1, 1, 'same',
                                          use_bias=False,
                                          kernel_regularizer=self._l2_regularizer,
                                          dilation_rate=1, name='Up_tensor_trans')  # 15, 15, 256

            # Fuse attention to (15, 15).
            upfe_trans = self.__fuse_attention_block(input_tensor=upfe_trans,
                                                     selective_mask=selective_mask,
                                                     name_scope=name_scope)     # 15,15,256
            # The first MFB with size (15, 15).
            score_map15, up15_trans = self.__score_map_block(down_tensor=upfe_trans,
                                                             output_chans=16)  # 240, 240, 16


            # The down-sample tensor with size (30, 30).
            ds_tensor30 = feats_dict['feats-8']
            # The first MFB with size (30, 30).
            MF_block1 = self.__mask_fusion_block(up_tensor=upfe_trans,
                                                     down_tensor=ds_tensor30,
                                                     output_chans=128,
                                                     )  # 30, 30, 128
            # Fuse attention to (30, 30).
            MF_block1 = self.__fuse_attention_block(input_tensor=MF_block1,
                                                    selective_mask=selective_mask,
                                                    name_scope=name_scope)     # 30,30,128
            # The first MFB with size (30, 30).
            score_map30, up30_trans = self.__score_map_block(down_tensor=MF_block1,
                                                             output_chans=16)  # 240, 240, 16

            # The down-sample tensor with size (60, 60).
            ds_tensor60 = feats_dict['feats-4']
            # The second MFB with size (60, 60).
            MF_block2 = self.__mask_fusion_block(up_tensor=MF_block1,
                                                     down_tensor=ds_tensor60,
                                                     output_chans=64,
                                                     )   # 60, 60, 64
            # Fuse attention to (60, 60).
            MF_block2 = self.__fuse_attention_block(input_tensor=MF_block2,
                                                    selective_mask=selective_mask,
                                                    name_scope=name_scope)  # 60,60,64
            # The second MFB with size (60, 60).
            score_map60, up60_trans = self.__score_map_block(down_tensor=MF_block2,
                                                             output_chans=16)  # 240, 240, 16
                                                             # output_chans=32)  # 240, 240, 32

            # The down-sample tensor with size (120, 120).
            ds_tensor120 = feats_dict['feats-2']
            # The third MFB with size (120, 120).
            MF_block3 = self.__mask_fusion_block(up_tensor=MF_block2,
                                                     down_tensor=ds_tensor120,
                                                     output_chans=32,
                                                     # output_chans=64
                                                     )  # 120, 120, 64
            # Fuse attention to (120, 120).
            MF_block3 = self.__fuse_attention_block(input_tensor=MF_block3,
                                                    selective_mask=selective_mask,
                                                    name_scope=name_scope)  # 120,120,64
            # The third MFB with size (120, 120).
            score_map120, up120_trans = self.__score_map_block(down_tensor=MF_block3,
                                                               output_chans=16)  # 240, 240, 16
                                                               # output_chans=32)  # 240, 240, 32

            # -------------------------- The Multi-layer Information Fusion ----------------------------

            # Firstly concat the tensor of different layer.
            fusion_tensor = tf.concat([up15_trans, up30_trans, up60_trans, up120_trans], axis=-1,
                                      name='MLIF_tensor')
            # fusion_tensor = tf.concat([up60_trans, up120_trans], axis=-1,
            # Pass through a 1*1 conv.
            fusion_score = tf.layers.conv2d(fusion_tensor, self._cls_dim, 1, 1, 'same',
                                            kernel_regularizer=self._l2_regularizer,
                                            dilation_rate=1, name='Fusion_score')  # 240, 240, cls_dim

            # -------------------------- The Final Clazz Probability ----------------------------

            # Add a BN and ReLU.
            upsam_orgs_tensor = tf.contrib.layers.batch_norm(MF_block3,
                                                             is_training=True,
                                                             decay=0.9,
                                                             # zero_debias_moving_mean=True,
                                                             # renorm=True,
                                                             # updates_collections=None,
                                                             scope='Usam_OS_tensor_BN')
            upsam_orgs_tensor = tf.nn.relu(upsam_orgs_tensor, name='Usam_OS_tensor_Relu')
            # Add the final transpose-conv layer to upsample it to the original size.
            upsam_orgs_tensor = tf.layers.conv2d_transpose(upsam_orgs_tensor, MF_block3.get_shape().as_list()[-1],
                                                           3, 2, 'same',
                                                           use_bias=False,
                                                           kernel_regularizer=self._l2_regularizer,
                                                           name='Usam_org_size_tensor')  # [-1, 240, 240, 64]

            # Add a final conv-layer to translate it to the same channels as clazz dimension.
            upsam_prob_map = tf.layers.conv2d(upsam_orgs_tensor, self._cls_dim, 1, 1, 'same',
                                           kernel_regularizer=self._l2_regularizer,
                                           dilation_rate=1, name='Upsam_prob_map')  # 240, 240, cls_dim

            un_prob_map = tf.reduce_mean(tf.stack([fusion_score, upsam_prob_map], axis=-1), axis=-1,
                                         name='Probability_map')

            # Final softmax layer to convert the probability to the final segmentation result.
            un_output = tf.nn.softmax(un_prob_map, name=name_scope + '_Segmentation_result')  # 240, 240, cls_dim

            # Add the score maps into the list.
            score_maps = [score_map15, score_map30, score_map60, score_map120, fusion_score, upsam_prob_map]
            # score_maps = [score_map60, score_map120, fusion_score, upsam_prob_map]

            # Assign the visualization tensor.
            # vis_tensor = MF_block3
            vis_tensor = MF_block2
            ### DEBUG
            # self._DEBUG_att_60 = MF_block2
            ### DEBUG

            # Reshape the flatten output tensor to the time-related shape tensor.
            un_output = tf.cond(
                self._train_phrase,
                lambda: tf.reshape(un_output, shape=[self._train_batch_size, self._train_track_len,
                                                     self._input_size[0], self._input_size[1], self._cls_dim]),
                lambda: tf.reshape(un_output, shape=[self._infer_batch_size, self._infer_track_len,
                                                     self._input_size[0], self._input_size[1], self._cls_dim]),
                name=name_scope + '_Time_segmentation'
            )  # [b, t, w, h, cls_dim]

        print('### Finish the definition of UN in class-form (name scope: {}). The output shape: {}'.format(
            name_scope, un_output.shape))

        # Finish the definition of UN. Meanwhile return the
        #   1) Input holders: Selective mask, Operation Conv Kernel.
        #   2) Output: Up-sample Transition Features, UN output (Segmentation result), prob tensors
        return selective_mask, vis_tensor, un_output, score_maps
        # return selective_mask, optconv_kernel, vis_tensors[:, :, :, :, 1:], un_output


    def __fuse_attention_block(self, input_tensor, selective_mask, name_scope, reuse=None):
        r'''
            Fuse the attention. Which is apply the operation-conv to the features map.

        :param input_tensor:
        :param select_mask:
        :param optconv_kernel:
        :param out_chans:
        :param name_scope:
        :return:
        '''

        # Get the shapes that will be used in while-loop for shape invariant.
        cmt_w, cmt_h, cmt_c, = input_tensor.get_shape()[1:].as_list()

        # Declare the initial class indicator and the mask tensor list.
        clazz_ind = tf.constant(0, dtype=tf.int64, name=name_scope + '_clazz_ind_' + str(cmt_w))
        # Cmask_tensors = tf.zeros([1, cmt_w, cmt_h, cmt_c], dtype=tf.float32,
        #                          name=name_scope + '_Cmask_tensors_init_' + str(cmt_w))
        Cmask_tensors = input_tensor

        # # Declare the shape invariants here.
        # shape_invariants = [input_tensor.get_shape(),
        #                     selective_mask.get_shape(),
        #                     optconv_kernel.get_shape(),
        #                     clazz_ind.get_shape(),
        #                     tf.TensorShape([None, cmt_w, cmt_h, cmt_c])]

        # Introduce the while loop to fill (generate) the class-based tensor list.
        _1, _2, _3, Cmask_tensors = tf.while_loop(
            cond=lambda _1, _2, clazz_ind, _4:
                tf.less(clazz_ind, self._mask_dim, name=name_scope + '_Attention_loop_cond_' + str(cmt_w)),
            body=self.__attention_body_func,
            loop_vars=[input_tensor, selective_mask,
                       clazz_ind, Cmask_tensors],
            # shape_invariants=shape_invariants,
            name=name_scope + '_Cmask_generator_' + str(cmt_w)
        )

        # # Fuse the class-based mask tensors for further processing.
        # FCmask_tensor = tf.divide(Cmask_tensors, tf.constant(self._mask_dim, dtype=tf.float32),
        #                           name=name_scope + '_Fusion_Cmask_tensor_' + str(cmt_w))   # w, h, c

        # Use the raw Cmask tensor.
        FCmask_tensor = Cmask_tensors

        # Add a BN and ReLU.
        output_tensor = tf.contrib.layers.batch_norm(FCmask_tensor,
                                                     is_training=True,
                                                     decay=0.9,
                                                     # renorm=True,
                                                     scope='Fusion_output_BN_' + str(cmt_w))
        output_tensor = tf.nn.relu(output_tensor, name='Fusion_output_Relu_' + str(cmt_w))

        # Pass through a 1*1 conv as a transition layer.
        output_tensor = tf.layers.conv2d(output_tensor, FCmask_tensor.get_shape().as_list()[-1],
                                         1, 1, 'same',
                                         use_bias=False,
                                         kernel_regularizer=self._l2_regularizer,
                                         dilation_rate=1, name='Fusion_Cmask_1x1trans_' + str(cmt_w))   # w, h, c

        # Finish. And return the output tensor.
        return output_tensor


    def __attention_body_func(self,
                             input_tensor,
                             select_Cmasks,
                             clazz_ind,
                             Cmask_tensors):
        r'''
            The body function used in tensorflow loop for generating class-based mask tensors.

        :param up_tensor:
        :param feats_dict:
        :param name_scope:
        :param select_Cmasks:
        :param optConvs:
        :param clazz_ind:
        :param Cmask_tensors:
        :param vis_tensors:
        :return:
        '''

        # Check validity.
        if not isinstance(input_tensor, tf.Tensor) or \
                not isinstance(select_Cmasks, tf.Tensor) or \
                not isinstance(clazz_ind, tf.Tensor) or \
                not isinstance(Cmask_tensors, tf.Tensor):
            raise TypeError('The up_tensor, select_Cmasks, clazz_ind, '
                            'Cmask_tensors must be of tensorflow.Tensor !!!')

        # Get the shape of input tensor.
        inw, inh, inc = input_tensor.get_shape().as_list()[1:]

        # Get the class-specific selective mask and operation-conv kernel.
        selective_mask = select_Cmasks[:, :, :, clazz_ind]      # [b*t, w, h]

        # Expand the selective mask.
        expa_select_mask = tf.expand_dims(selective_mask, axis=-1, name='expa_mask_' + str(inw))
        # Resize the select mask to the same shape as input tensor.
        resize_mask = tf.image.resize_bilinear(expa_select_mask, (inw, inh),
                                               name='resize_mask_' + str(inw))  # b*t, w, h, 1
        binary_mask = tf.to_float(tf.greater(resize_mask, 0.),
                                  name='binary_mask_' + str(inw))  # b*t, w, h, 1

        # Filter and "Operate" the smallest FEN feature maps.
        filt_upfet = tf.multiply(input_tensor, binary_mask,
                                 name='filt_upfet_' + str(inw))  # 15,15,oc

        # Add a BN and Relu.
        filt_upfet = tf.contrib.layers.batch_norm(filt_upfet,
                                                    is_training=True,
                                                    # is_training=self._train_phrase,
                                                  decay=0.9,
                                                  # zero_debias_moving_mean=True,
                                                    #    updates_collections=None,
                                                    # renorm=True,
                                                    scope='filtUpFet_BN_' + str(inw))
        # Add a "ReLU" layer as activation. (Filter the negative value)
        filt_upfet = tf.nn.relu(filt_upfet, name='filtUpFet_ReLU_' + str(inw))  # [b*t, w, h, c]
        # optUpfet_att = tf.nn.leaky_relu(optUpfet_att, name='transUpAtt_ReLU_' + str(inw))  # [b*t, w, h, c]

        # Firstly pass a 1*1 conv as transition.
        upfet_att1x1_conv1 = tf.layers.conv2d(filt_upfet, inc, 1, 1, 'same',
                                              activation=tf.nn.relu,
                                              kernel_regularizer=None,
                                              dilation_rate=1, reuse=tf.AUTO_REUSE,
                                              name='Upfet_att1x1_conv1'+str(inw))    # 15,15,oc

        # Use sigmoid convert the tensor into the weights tensor.
        upfet_att_sigmoid = tf.sigmoid(upfet_att1x1_conv1,
                                       name='raw_upfet_att_weights_' + str(inw))  # [-1, w, h, c]

        # Add a BN and Relu.
        upfet_att_bn2 = tf.contrib.layers.batch_norm(upfet_att_sigmoid,
                                                  is_training=True,
                                                  # is_training=self._train_phrase,
                                                     decay=0.9,
                                                     # zero_debias_moving_mean=True,
                                                  #    updates_collections=None,
                                                  # renorm=True,
                                                  scope='filtUpFet2_BN_' + str(inw))
        # # Add a "ReLU" layer as activation. (Filter the negative value)
        # upfet2_att_relu = tf.nn.relu(upfet2_att_bn, name='filtUpFet2_ReLU_' + str(inw))  # [b*t, w, h, c]
        # # optUpfet_att = tf.nn.leaky_relu(optUpfet_att, name='transUpAtt_ReLU_' + str(inw))  # [b*t, w, h, c]

        # Firstly pass a 1*1 conv as transition.
        upfet_att1x1_conv2 = tf.layers.conv2d(upfet_att_bn2, inc, 1, 1, 'same',
                                              activation=tf.nn.relu,
                                              kernel_regularizer=None,
                                              dilation_rate=1, reuse=tf.AUTO_REUSE,
                                              name='Upfet_att1x1_conv2' + str(inw))  # 15,15,oc


        # # Add the "Attention".
        # output_tensor = tf.multiply(input_tensor,
        #                             1. + upfet_att_weights,
        #                             name='Att_upfet_' + str(inw))  # [-1, w, h, c]
        #
        # # Stack the class-based tensors for further processing.
        # Cmask_tensors = tf.cond(
        #     tf.equal(clazz_ind, 0),
        #     lambda: output_tensor,  # w, h, c
        #     lambda: tf.add(Cmask_tensors, output_tensor)  # w, h, c
        # )

        # Use the raw features after 1*1 conv as weights.
        upfet_att_weights = upfet_att1x1_conv2
        # Add the "Attention".
        output_tensor = tf.multiply(input_tensor, upfet_att_weights,
                                    name='Att_upfet_' + str(inw))  # [-1, w, h, c]

        # # Stack the class-based tensors for further processing.
        # Cmask_tensors = tf.cond(
        #     tf.equal(clazz_ind, 0),
        #     lambda: tf.maximum(input_tensor, output_tensor),    # w, h, c
        #     lambda: tf.maximum(Cmask_tensors, output_tensor)    # w, h, c
        # )

        # Use the max response.
        Cmask_tensors = tf.maximum(Cmask_tensors, output_tensor)  # w, h, c

        # Increase the class indicator.
        clazz_ind += 1

        # Finish. And return the input arguments for next loop.
        return input_tensor, select_Cmasks, clazz_ind, Cmask_tensors


    # Used in "Traditional"
    def __mask_fusion_block(self, up_tensor, down_tensor, output_chans, reuse=None):
        r'''
            Generate the Mask-Fusion block, which is used to mask and fuse the down-sample
                and up-sample tensor of same level.

        :param up_tensor:
        :param down_tensor:
        :param select_mask:
        :param output_chans:
        :param name_scope:
        :return:
        '''

        # Check the validity.
        assert up_tensor.get_shape().as_list()[1] * 2 == down_tensor.get_shape().as_list()[1]
        assert up_tensor.get_shape().as_list()[2] * 2 == down_tensor.get_shape().as_list()[2]
        # assert up_tensor.get_shape().as_list()[3] == down_tensor.get_shape().as_list()[3] * 2     # coz ResNet

        # Get the shape of raw down-sample tensor.
        fw, fh, fc = down_tensor.get_shape().as_list()[1:]

        # -------------------------- The Down-sample Tensor Part ----------------------------

        # with tf.variable_scope('DS'):

        # Add a batch normalization and ReLU.
        ds_tensor_trans = tf.contrib.layers.batch_norm(down_tensor,
                                                       is_training=True,
                                                       decay=0.9,
                                                       # zero_debias_moving_mean=True,
                                                       # renorm=True,
                                                       # updates_collections=None,
                                                       scope='Ds_tensor_BN_' + str(fw))
        ds_tensor_trans = tf.nn.relu(ds_tensor_trans, name='Ds_tensor_Relu_' + str(fw))
        # Pass through a 1*1 conv layer as the transition layer.
        ds_tensor_trans = tf.layers.conv2d(ds_tensor_trans, fc, 1, 1, 'same',
                                           # activation=tf.nn.relu,
                                           use_bias=False,
                                           kernel_regularizer=self._l2_regularizer,
                                           dilation_rate=1, reuse=reuse,
                                           name='Ds_tensor_1x1_trans_' + str(fw))  # fw,fh,fc

            # ds_tensor_trans = down_tensor

        # -------------------------- The Up-sample Tensor Part ----------------------------

        # with tf.variable_scope('US'):

        # Add a batch normalization and ReLU.
        us_tensor_trans = tf.contrib.layers.batch_norm(up_tensor,
                                                       is_training=True,
                                                       decay=0.9,
                                                       # zero_debias_moving_mean=True,
                                                       # renorm=True,
                                                       # updates_collections=None,
                                                       scope='Us_tensor_BN_' + str(fw))
        us_tensor_trans = tf.nn.relu(us_tensor_trans, name='Us_tensor_Relu_' + str(fw))
        # Pass through the transpose-conv to up-sample the tensor.
        us_tensor_trans = tf.layers.conv2d_transpose(us_tensor_trans, fc, 3, 2, 'same',
                                                     # activation=tf.nn.relu,
                                                     use_bias=False,
                                                     kernel_regularizer=self._l2_regularizer,
                                                     reuse=reuse,
                                                     name='Us_tensor_conv_trans' + str(fw))  # fw, fh, fc

        # -------------------------- The Fusion Part ----------------------------

        # Add the Up-sample tensor and Down-sample tensor as the fusion tensor.
        fusion_tensor = tf.add(ds_tensor_trans, us_tensor_trans,
                               name='UD_fuse_tensor_' + str(fw))  # [-1, fw, fh, 2fc]

        # # Concatenate the Up-sample tensor and Down-sample tensor as the fusion tensor.
        # fusion_tensor = tf.concat((ds_tensor_trans, us_tensor_trans), axis=-1,
        #                           name='UD_fuse_tensor_' + str(fw))  # [-1, fw, fh, 2fc]

        # Pass through two-conv layer to generate the final mask-fusion output.
        MF_conv1 = tf.contrib.layers.batch_norm(fusion_tensor,
                                                is_training=True,
                                                decay=0.9,
                                                # zero_debias_moving_mean=True,
                                                # renorm=True,
                                                # updates_collections=None,
                                                scope='MF_conv1_BN_' + str(fw))
        MF_conv1 = tf.nn.relu(MF_conv1, name='MF_conv1_Relu_' + str(fw))
        MF_conv1 = tf.layers.conv2d(MF_conv1, output_chans, 3, 1, 'same',
                                    # activation=tf.nn.relu,
                                    use_bias=False,
                                    kernel_regularizer=self._l2_regularizer,
                                    dilation_rate=1, reuse=reuse,
                                    name='MF_conv1_' + str(fw))  # fw,fh,oc
        # Second
        MF_output = tf.contrib.layers.batch_norm(MF_conv1,
                                                 is_training=True,
                                                 decay=0.9,
                                                 # zero_debias_moving_mean=True,
                                                 # renorm=True,
                                                 # updates_collections=None,
                                                 scope='MF_output_BN_' + str(fw))
        MF_output = tf.nn.relu(MF_output, name='MF_output_Relu_' + str(fw))
        MF_output = tf.layers.conv2d(MF_output, output_chans, 3, 1, 'same',
                                     # activation=tf.nn.relu,
                                     use_bias=False,
                                     kernel_regularizer=self._l2_regularizer,
                                     dilation_rate=1, reuse=reuse,
                                     name='MF_output_' + str(fw))  # fw,fh,oc

        # Finish. Return the final fusion output and the raw up trans tensor.
        return MF_output


    # Used in "MLIF".
    def __score_map_block(self, down_tensor, output_chans):
        r'''
            Generate the Score Map block, which is used to mask and fuse the down-sample
                and up-sample tensor of same level.

        :param up_tensor:
        :param down_tensor:
        :param select_mask:
        :param output_chans:
        :param name_scope:
        :return:
        '''

        # Get the shape of raw down-sample tensor.
        fw, _2, _3 = down_tensor.get_shape().as_list()[1:]

        # -------------------------- The Down-sample Tensor Part ----------------------------

        # Add a batch normalization and ReLU.
        ds_tensor_trans = tf.contrib.layers.batch_norm(down_tensor,
                                                       is_training=True,
                                                       decay=0.9,
                                                       # zero_debias_moving_mean=True,
                                                       # renorm=True,
                                                       # updates_collections=None,
                                                       scope='Score_Dtensor_BN_' + str(fw))
        ds_tensor_trans = tf.nn.relu(ds_tensor_trans, name='Score_Dtensor_Relu_' + str(fw))

        # Pass through a 1*1 conv layer as the transition layer.
        ds_tensor_trans = tf.layers.conv2d(ds_tensor_trans, output_chans, 1, 1, 'same',
                                           use_bias=False,
                                           kernel_regularizer=self._l2_regularizer,
                                           dilation_rate=1,
                                           name='Score_Dtensor_1x1_trans_' + str(fw))  # fw,fh,fc

        # -------------------------- The Up-sample Tensor Part ----------------------------

        # Up-sample (Bi-linear) to the original image size.
        upsam_tensor = tf.image.resize_bilinear(ds_tensor_trans, size=self._input_size,
                                                name='Score_Upsam_tensor_Bi_' + str(fw))

        # -------------------------- The Score Map Part ----------------------------

        # Pass through two-conv layer to generate the final mask-fusion output.
        score_map = tf.layers.conv2d(upsam_tensor, self._cls_dim, 1, 1, 'same',
                                     kernel_regularizer=self._l2_regularizer,
                                     dilation_rate=1,
                                     name='Score_map_' + str(fw))  # fw,fh,oc
        # score_map = tf.layers.batch_normalization(score_map, training=self._train_flag, name='Score_map_BN_' + str(fw))
        # score_map = tf.nn.relu(score_map, name='Score_map_Relu_' + str(fw))

        # Finish. Return the final fusion output and the raw up trans tensor.
        return score_map, upsam_tensor


