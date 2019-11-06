import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np
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

        # The inputs holder.
        self._inputs = {}
        # The outputs holder.
        self._outputs = {}

        # Get detailed configuration.
        # conf_base = self._config['Base']
        # conf_train = self._config['Training']
        conf_dqn = self._config['DQN']

        # # Normal Initialization.
        # self._input_shape = conf_base.get('input_shape')
        # self._clz_dim = conf_base.get('classification_dimension')

        # # Score maps holder.
        # self._score_maps = None

        # Determine the action dimension according to the config.
        if conf_dqn.get('restriction_action'):
            self._action_dim = 9
        else:
            self._action_dim = 17

        # # The name scope pair specified to support the "Double DQN".
        # self._DoubleDQN_scope = conf_dqn.get('double_dqn', ['ORG', 'TAR'])
        # # Check whether enable "Prioritized Replay" or not.
        # self._prioritized_replay = conf_dqn.get('prioritized_replay', True)
        # # Check whether enable "Dueling Network" or not.
        # self._dueling_network = conf_dqn.get('dueling_network', True)







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

        # # Regularization Related.
        # regularize_coef = conf_train.get('regularize_coef', 0.0)
        # self._regularizer = tf_layers.l2_regularizer(regularize_coef)

        # The final holder, loss and summary dictionary.
        self._ios_dict = None
        self._loss_dict = None
        self._summary_dict = None

        # Finish initialization
        return




    def definition(self):
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

        # Specify the name space if enable the "Double DQN".
        conf_dqn = self._config['DQN']
        double_dqn = conf_dqn.get('double_dqn', None)

        # Start build the model.
        if double_dqn is None:
            DQN_output, CEloss_tensors = self._architecture(self._action_dim, self._name_space)
        else:
            ORG_name, TAR_name = double_dqn
            DQN_output, CEloss_tensors = self._architecture(self._action_dim, ORG_name)
            self._architecture(self._action_dim, TAR_name, with_segmentation=False)

        # Construct the loss function after building model.







        ##################

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
    def _architecture(self, action_dim, name_space, with_segmentation=True):
        r'''
            Construct the whole architecture. Specify by the parameters.

        --------------------------------------------------------------------------------
        Parameters:
            action_dim: Determine the quantity of "DQN" action branch.
            name_space: The name space for this model.
            with_segmentation: Whether should build the "Segmentation" branch or not.

        ---------------------------------------------------------------------------------
        Return:
            The tensors used to construct loss function.
        '''

        # Indicating.
        print('* ---> Start build the model ... (name scope: {})'.format(name_space))

        # Generate the input for model.
        input_tensor = self.__input_justify(name_space)
        # Extract feature maps.
        FE_tensor, DS_feats = self.__feature_extract(input_tensor, name_space)
        # Select next region to deal with. (Generate the DQN output)
        DQN_output = self.__region_selection(FE_tensor, action_dim, name_space)
        # Segment current image (patch). (Generate the segmentation branch)
        if with_segmentation:
            SEG_output, CEloss_tensors = self.__segmentation(FE_tensor, DS_feats, self._name_space)
        else:
            CEloss_tensors = None

        # Show the input holders.
        print('| ---> Inputs holder (name scope: {}): {}'.format(name_space, self._inputs.keys()))

        # Package some outputs.
        self._outputs[name_space+'/DQN_output'] = DQN_output
        if with_segmentation:
            self._outputs['SEG_output'] = SEG_output

        # Only return the tensors that will be used in "Loss Construction".
        return DQN_output, CEloss_tensors


    def __input_justify(self, name_space):
        r'''
            Define and justify the input tensor. Mainly to justify the
                input size and fuse some information.
        '''

        # Get detailed configuration.
        conf_base = self._config['Base']
        # Input shape.
        input_shape = conf_base.get('input_shape')

        # --------------------------------- "Input Justification" part. ------------------------------------
        # Get configuration for justification.
        conf_cus = self._config['Custom']
        # Determine the introduction method of "Position Information".
        pos_method = conf_cus.get('position_info', 'map')

        # Start definition.
        IJ_name = name_space + '/InputJustify'
        with tf.variable_scope(IJ_name):
            # The input image holder.
            raw_image = net_util.placeholder_wrapper(self._inputs, tf.float32, input_shape,
                                                     name='image')  # [?, 240, 240, ?]

            # The input previous segmentation holder.
            prev_result = net_util.placeholder_wrapper(self._inputs, tf.float32, input_shape[:-1],
                                                       name='prev_result')  # [?, 240, 240]
            # Expand additional dimension for conveniently processing.
            prev_result = tf.expand_dims(prev_result, axis=-1, name='expa_Pres')  # [?, 240, 240, 1]

            # Define the "Position Information" placeholder.
            pos_name = 'position_info'
            if pos_method == 'map':
                pos_info = net_util.placeholder_wrapper(self._inputs, tf.float32, input_shape[:-1],
                                                        name=pos_name)  # [?, h, w]
                # Expand additional dimension for conveniently processing.
                pos_info = tf.expand_dims(pos_info, axis=-1, name='expa_Pinfo')  # [?, h, w, 1]
            elif pos_method == 'coord':
                pos_info = net_util.placeholder_wrapper(self._inputs, tf.float32, [None, 4],
                                                        name=pos_name)  # [?, 4]
            elif pos_method == 'sight':
                pos_info = net_util.placeholder_wrapper(self._inputs, tf.float32, [None, 4],
                                                        name=pos_name)  # [?, 4]
            elif pos_method == 'w/o':
                pos_info = None
            else:
                raise ValueError('Unknown position information fusion method !!!')

            # Crop or resize the input image into suitable size if size not matched.
            suit_w = conf_base.get('suit_width')
            suit_h = conf_base.get('suit_height')
            if suit_w != input_shape[1] or suit_h != input_shape[2]:
                crop_method = conf_cus.get('size_matcher', 'crop')
                if crop_method == 'crop':
                    # Crop to target size.
                    raw_image = tf.image.resize_image_with_crop_or_pad(raw_image, suit_h, suit_w)
                    prev_result = tf.image.resize_image_with_crop_or_pad(prev_result, suit_h, suit_w)
                    if pos_method == 'map':
                        pos_info = tf.image.resize_image_with_crop_or_pad(pos_info, suit_h, suit_w)
                elif crop_method == 'bilinear':
                    # Bilinear resize to target size.
                    raw_image = tf.image.resize_bilinear(raw_image, [suit_w, suit_h], name='bi_image')
                    prev_result = tf.image.resize_nearest_neighbor(prev_result, [suit_w, suit_h], name='nn_prev')
                    if pos_method == 'map':
                        pos_info = tf.image.resize_nearest_neighbor(pos_info, [suit_w, suit_h], name='nn_pos')
                else:
                    raise ValueError('Unknown size match method !!!')

            # Concat the tensors to generate input for whole model.
            input_tensor = tf.concat([raw_image, prev_result], axis=-1, name='2E_input')
            if pos_method == 'map':
                input_tensor = tf.concat([input_tensor, pos_info], axis=-1, name='3E_input')

            # After declare the public input part, deal with the input tensor according to
            #   the different stage (branch). That is:
            # 1. Focus on the given region (bounding-box) if it's "Segmentation" stage.
            # 2. Introduce the position info if it's "Region Selection" stage with the
            #       "sight" position info.
            segment_stage = net_util.placeholder_wrapper(self._inputs, tf.bool, None, name='Segment_Stage')
            focus_bbox = net_util.placeholder_wrapper(self._inputs, tf.float32, [None, 4], name='Focus_Bbox')
            def region_crop(x, bbox, size):
                bbox_ids = tf.range(tf.reduce_sum(tf.ones_like(bbox, dtype=tf.int32)[:,0]))
                y = tf.image.crop_and_resize(x, bbox, bbox_ids, size, name='focus_crop')
                return y
            input_tensor = tf.where(segment_stage,
                                    region_crop(input_tensor, focus_bbox, [suit_h, suit_w]),
                                    input_tensor if pos_method != 'sight'
                                    else region_crop(input_tensor, pos_info, [suit_h, suit_w]))

            # Print some information.
            print('### Finish "Input Justification" (name scope: {}). The output shape: {}'.format(
                name_space, input_tensor.shape))

            # Return the justified input tensor.
            return input_tensor


    def __feature_extract(self, input_tensor, name_space):
        r'''
            Feature extraction backbone. Which is used to generate the deep
                feature maps for successive processing.

            The input is the raw input tensor generated by "Input Justification" module.
        '''

        # Get detailed configuration.
        conf_base = self._config['Base']
        conf_train = self._config['Training']
        # Feature normalization method and activation function.
        fe_norm = conf_base.get('feature_normalization', 'batch')
        activation = conf_base.get('activation', 'relu')
        # Dropout (keep probability) for convolution and fully-connect operation.
        conv_kprob = conf_base.get('convolution_dropout', 1.0)
        # Regularization Related.
        regularize_coef = conf_train.get('regularize_coef', 0.0)
        regularizer = tf_layers.l2_regularizer(regularize_coef)

        # --------------------------------- "Feature Extraction" backbone. ------------------------------------
        # Get configuration for ResNet
        conf_res = self._config['ResNet']
        # Layer number and kernel number of blocks for ResNet.
        kernel_numbers = conf_res.get('kernel_numbers')
        layer_units = conf_res.get('layer_units')

        # Start definition.
        FE_name = name_space + '/FeatExt'
        with tf.variable_scope(FE_name):
            # The feature maps dictionary, which is lately used in up-sample part.
            DS_feats = []

            # Base conv to reduce the feature map size.
            base_conv = cus_layers.base_conv2d(input_tensor, kernel_numbers[0], 7, 2,
                                               feature_normalization=fe_norm,
                                               activation='lrelu',
                                               keep_prob=conv_kprob,
                                               regularizer=regularizer,
                                               name_space='ResNet_bconv')  # [?, 112, 112, ?]

            # Record the 2x - feature maps.
            DS_feats.append(base_conv)

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
                    block_tensor = cus_res.transition_layer(block_tensor, kernel_numbers[idx + 1],
                                                            feature_normalization=fe_norm,
                                                            activation=activation,
                                                            keep_prob=conv_kprob,
                                                            regularizer=regularizer,
                                                            name_space='ResNet_Trans0' + str(idx + 1))
                # Pass through the residual block.
                block_tensor = cus_res.residual_block(block_tensor, kernel_numbers[idx + 1], layer_units[idx] - 1,
                                                      feature_normalization=fe_norm,
                                                      activation=activation,
                                                      keep_prob=conv_kprob,
                                                      regularizer=regularizer,
                                                      name_space='ResNet_Block0' + str(idx + 1))

                # Record each scale feature maps of block.
                DS_feats.append(block_tensor)

            # For conveniently usage.
            FE_tensor = block_tensor  # [?, 7, 7, ?]  default: 2048

            # Print some information.
            print('### Finish "Feature Extract Network" (name scope: {}). The output shape: {}'.format(
                name_space, FE_tensor.shape))

            # Return the feature maps extracted by the backbone. What's more,
            #   return the feature maps dictionary.
            return FE_tensor, DS_feats


    def __region_selection(self, FE_tensor, action_dim, name_space):
        r'''
            Region selection branch. Which is actually a DQN head.
                It's used to select the next region for precisely processing.

            The input is the deep feature maps extracted by
                "Feature Extraction" backbone.
        '''

        # Get detailed configuration.
        conf_base = self._config['Base']
        conf_train = self._config['Training']
        # Feature normalization method and activation function.
        fe_norm = conf_base.get('feature_normalization', 'batch')
        activation = conf_base.get('activation', 'relu')
        # Dropout (keep probability) for convolution and fully-connect operation.
        conv_kprob = conf_base.get('convolution_dropout', 1.0)
        fc_kprob = conf_base.get('fully_connect_dropout', 0.5)
        # Regularization Related.
        regularize_coef = conf_train.get('regularize_coef', 0.0)
        regularizer = tf_layers.l2_regularizer(regularize_coef)

        # --------------------------------- "Region Selection" (DQN) branch. ------------------------------------
        # Get configuration for DQN.
        conf_dqn = self._config['DQN']
        # Get the dimension reduction method.
        reduce_dim = conf_dqn.get('reduce_dim', 'residual')
        # Check whether enable "Dueling Network" or not.
        dueling_network = conf_dqn.get('dueling_network', True)

        # Check for the introduction method of "Position Information".
        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')

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
                                                     name_space='reduce_dim01')  # [?, 4, 4, 1024]
                redc_tensor = cus_layers.base_conv2d(redc_tensor, 512, 3, 2,
                                                     feature_normalization=fe_norm,
                                                     activation=activation,
                                                     keep_prob=conv_kprob,
                                                     regularizer=regularizer,
                                                     name_space='reduce_dim02')  # [?, 2, 2, 512]
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
                raise ValueError('Unknown reduce dimension method for DQN !!!')

            # Flatten the tensor to 1-D vector.
            fdim = 1
            for d in redc_tensor.shape[1:]:
                fdim *= int(d)
            flat_tensor = tf.reshape(redc_tensor, [-1, fdim], name='flatten')  # [?, OC]   default: 2048

            # Fuse (concat) the position vector if use "coord"-like position information.
            if pos_method == 'coord':
                pos_info = self._inputs['position_info']
                flat_tensor = tf.concat([flat_tensor, pos_info], axis=-1, name='fuse_pos_coord')  # [?, OC+4]

            # Pass through two fully connected layers.
            fc01_tensor = cus_layers.base_fc(flat_tensor, 1024,
                                             feature_normalization=fe_norm,
                                             activation=activation,
                                             keep_prob=fc_kprob,
                                             regularizer=regularizer,
                                             name_space='FC01')  # [?, 1024]
            fc02_tensor = cus_layers.base_fc(fc01_tensor, 1024,
                                             feature_normalization=fe_norm,
                                             activation=activation,
                                             keep_prob=fc_kprob,
                                             regularizer=regularizer,
                                             name_space='FC02')  # [?, 1024]

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
                                                  name_space='state_value')  # [?, 1]
                # Build the "Action" branch.
                action_tensor = cus_layers.base_fc(action_bch, action_dim,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=fc_kprob,
                                                   regularizer=regularizer,
                                                   name_space='action_value')  # [?, act_dim]
                # Mean the "Action" (Advance) branch.
                norl_Adval = tf.subtract(action_tensor,
                                         tf.reduce_mean(action_tensor, axis=-1, keepdims=True),
                                         name='advanced_value')  # [?, act_dim]
                # Add the "State" value and "Action" value to obtain the final output.
                DQN_output = tf.add(state_tensor, norl_Adval, name='DQN_output')  # [?, act_dim]
            # Not enable "Dueling" structure. Directly pass through a FC-layer.
            else:
                # The normal mode, do not need to split into two branches.
                DQN_output = cus_layers.base_fc(fc02_tensor, action_dim,
                                                feature_normalization=fe_norm,
                                                activation=activation,
                                                keep_prob=fc_kprob,
                                                regularizer=regularizer,
                                                name_space='DQN_output')  # [?, act_dim]

            # Print some information.
            print('### Finish "DQN Head" (name scope: {}). The output shape: {}'.format(
                name_space, DQN_output.shape))

            # Return the outputs of DQN, which is the result for region value.
            return DQN_output


    def __segmentation(self, FE_tensor, DS_feats, name_space):
        r'''
            Segmentation branch. Which is actually a up-sample head.
                It's used to segment the current input image (patch).

            The input is the deep feature maps (each level) extracted by
                "Feature Extraction" backbone.
        '''

        # Get detailed configuration.
        conf_base = self._config['Base']
        conf_train = self._config['Training']
        # Suitable width and height.
        suit_w = conf_base.get('suit_width')
        suit_h = conf_base.get('suit_height')
        # Classification categories.
        classification_dim = conf_base.get('classification_dimension')
        # Feature normalization method and activation function.
        fe_norm = conf_base.get('feature_normalization', 'batch')
        activation = conf_base.get('activation', 'relu')
        # Dropout (keep probability) for convolution and fully-connect operation.
        conv_kprob = conf_base.get('convolution_dropout', 1.0)
        # Regularization Related.
        regularize_coef = conf_train.get('regularize_coef', 0.0)
        regularizer = tf_layers.l2_regularizer(regularize_coef)

        # Declare the scores fusion block. Pass through a 1x1 conv and bilinear.
        def gen_scores(x, out_chans, name_idx):
            y = cus_layers.base_conv2d(x, out_chans, 1, 1,
                                       feature_normalization=fe_norm,
                                       activation=activation,
                                       keep_prob=conv_kprob,
                                       regularizer=regularizer,
                                       name_space='gen_score_conv1x1_0' + str(name_idx))
            y = tf.image.resize_bilinear(y, size=[suit_w, suit_h],
                                         name='gen_score_bi_0' + str(name_idx))
            return y

        # --------------------------------- "Segmentation" branch. ------------------------------------
        # Get configuration for ResNet
        conf_res = self._config['ResNet']
        conf_up = self._config['UpSample']
        # Layer number and kernel number of blocks for ResNet.
        kernel_numbers = conf_res.get('kernel_numbers')
        layer_units = conf_res.get('layer_units')
        # Get parameters.
        up_structure = conf_up.get('upsample_structure', 'raw-U')
        up_method = conf_up.get('scale_up', 'ResV2')
        up_fuse = conf_up.get('upsample_fusion', 'concat')
        upres_layers = conf_up.get('upres_layers', 3)

        # Get the scale parameters.
        scale_exp = conf_up.get('scale_exp', 1)
        if not isinstance(scale_exp, int) or scale_exp not in range(1, 3):
            raise ValueError('Invalid up-sample scale up exponent value !!!')
        scale_factor = int(np.exp2(scale_exp))

        # Get Score flag and score channels.
        score_chan = conf_up.get('score_chans', 16)
        fuse_score = conf_up.get('fuse_scores', False)

        # Declare the loss holder (list).
        CEloss_tensors = []

        # Start definition.
        SEG_name = name_space + '/Segmentation'
        with tf.variable_scope(SEG_name):
            # Just the traditional structure. Gradually build the block is okay.
            if up_structure in ['raw', 'raw-U', 'conv-U', 'res-U']:
                US_tensor = FE_tensor
                for idx in range(len(layer_units) // scale_exp):
                    # Compute the index for corresponding down-sample tensor
                    #   and layer units.
                    cor_idx = - (idx + 1) * scale_exp - 1  # -1 for reverse. idx+1 for skip last one.

                    # Scale up the feature maps. Whether use the pure de-conv
                    #   or the "residual" de-conv.
                    US_tensor = cus_res.transition_layer(US_tensor, kernel_numbers[cor_idx],
                                                         scale_down=False,
                                                         scale_factor=scale_factor,
                                                         structure=up_method,
                                                         feature_normalization=fe_norm,
                                                         activation=activation,
                                                         keep_prob=conv_kprob,
                                                         regularizer=regularizer,
                                                         name_space='Scale_Up0' + str(idx + 1))  # [?, 2x, 2x, 0.5c]

                    # Add "Skip Connection" if specified.
                    if up_structure == 'raw-U':
                        # Use the raw down-sampled feature maps.
                        skip_conn = DS_feats[cor_idx]
                    elif up_structure == 'conv-U':
                        # Pass through a 1x1 conv to get the skip connection features.
                        raw_ds = DS_feats[cor_idx]
                        skip_conn = cus_layers.base_conv2d(raw_ds, raw_ds.shape[-1], 1, 1,
                                                           feature_normalization=fe_norm,
                                                           activation=activation,
                                                           keep_prob=conv_kprob,
                                                           regularizer=regularizer,
                                                           name_space='skip_conv0' + str(idx + 1))
                    elif up_structure == 'res-U':
                        # Pass through a residual layer to get the skip connection features.
                        raw_ds = DS_feats[cor_idx]
                        skip_conn = cus_res.residual_block(raw_ds, raw_ds.shape[-1], 1,
                                                           kernel_size=1,
                                                           feature_normalization=fe_norm,
                                                           activation=activation,
                                                           keep_prob=conv_kprob,
                                                           regularizer=regularizer,
                                                           name_space='skip_conv0' + str(idx + 1))
                    elif up_structure == 'raw':
                        # Do not use "Skip Connection".
                        skip_conn = None
                    else:
                        raise ValueError('Unknown "Skip Connection" method !!!')

                    # Now determine the features fusion method.
                    if skip_conn is not None:
                        if up_fuse == 'add':
                            US_tensor = tf.add(US_tensor, skip_conn, name='skip_add0' + str(idx + 1))  # [2x, 2x, 0.5c]
                        elif up_fuse == 'concat':
                            US_tensor = tf.concat([US_tensor, skip_conn], axis=-1,
                                                  name='skip_concat0' + str(idx + 1))  # [2x, 2x, c]
                        else:
                            raise ValueError('Unknown "Skip Feature" fusion method !!!')

                    # After up-sample, pass through the residual blocks to convolution.
                    if isinstance(upres_layers, int):
                        layer_num = upres_layers
                    elif upres_layers == 'same':
                        # if idx != len(layer_units)-1:
                        if idx != len(layer_units) // scale_exp - 1:
                            layer_num = layer_units[cor_idx] - 1
                        else:
                            # The last layer is special. Coz the corresponding layer
                            #   is max pooling, and it don't use any convolution.
                            layer_num = conf_up.get('last_match', 3)
                    else:
                        raise ValueError('Unknown up-sample block layers !!!')
                    # Specify the layer for residual block. Note that, we may use 1x1 conv
                    #   if @{layer_num} is a string.
                    if isinstance(layer_num, int) and layer_num > 0:
                        US_tensor = cus_res.residual_block(US_tensor, US_tensor.shape[-1], layer_num,
                                                           feature_normalization=fe_norm,
                                                           activation=activation,
                                                           keep_prob=conv_kprob,
                                                           regularizer=regularizer,
                                                           name_space='UpResidual_conv0' + str(idx + 1))
                    elif layer_num == '1x1conv':
                        US_tensor = cus_layers.base_conv2d(US_tensor, US_tensor.shape[-1], 1, 1,
                                                           feature_normalization=fe_norm,
                                                           activation=activation,
                                                           keep_prob=conv_kprob,
                                                           regularizer=regularizer,
                                                           name_space='Up1x1_conv0' + str(idx + 1))
                    elif layer_num == 'w/o':
                        pass
                    else:
                        raise ValueError('Unknown last layer match method !!!')

                    # Pass through the scores block if enable "Fuse Score".
                    if fuse_score:
                        score_tensor = gen_scores(US_tensor, score_chan, idx + 1)
                        score_tensor = cus_layers.base_conv2d(score_tensor, classification_dim, 1, 1,
                                                              feature_normalization=fe_norm,
                                                              activation=activation,
                                                              keep_prob=conv_kprob,
                                                              regularizer=regularizer,
                                                              name_space='Score_tensor0' + str(idx + 1))
                        CEloss_tensors.append(score_tensor)

                # For conveniently usage.
                half_UST = US_tensor  # [?, half, half, OC]

                # Scale up to the original size.
                org_UST = cus_layers.base_deconv2d(half_UST, classification_dim, 3, 2,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='WS_tensor')
                # Then translate the value into probability.
                SEG_output = tf.nn.softmax(org_UST, name='SEG_output')  # [?, h, w, cls]
                # Add the output tensor to the CE loss tensors for train.
                CEloss_tensors.append(org_UST)

            # Refer the paper's structure.
            elif up_structure == 'MLIF':
                score_maps = []
                for idx in range(len(DS_feats) // scale_exp):
                    cor_idx = scale_exp * idx
                    score_tensor = gen_scores(DS_feats[cor_idx], score_chan, idx + 1)
                    score_maps.append(score_tensor)
                    # Pass through 1x1 conv to generate score map
                    seg_map = cus_layers.base_conv2d(score_tensor, classification_dim, 1, 1,
                                                     feature_normalization=fe_norm,
                                                     activation=activation,
                                                     keep_prob=conv_kprob,
                                                     regularizer=regularizer,
                                                     name_space='Score_Maps0' + str(idx + 1))
                    CEloss_tensors.append(seg_map)
                # Generate MLIF tensor.
                MLIF_tensor = tf.concat(score_maps, axis=-1, name='MLIF_concat')
                MLIF_tensor = cus_layers.base_conv2d(MLIF_tensor, MLIF_tensor.shape[-1], 1, 1,
                                                     feature_normalization=fe_norm,
                                                     activation=activation,
                                                     keep_prob=conv_kprob,
                                                     regularizer=regularizer,
                                                     name_space='MLIF_tensor')
                MLIF_map = cus_layers.base_conv2d(MLIF_tensor, classification_dim, 1, 1,
                                                  feature_normalization=fe_norm,
                                                  activation=activation,
                                                  keep_prob=conv_kprob,
                                                  regularizer=regularizer,
                                                  name_space='MLIF_Map')
                CEloss_tensors.append(MLIF_map)
                # Fuse (mean) all score maps to get the final segmentation.
                SMs_tensor = tf.stack(CEloss_tensors, axis=-1, name='SMs_stack')
                SMs_tensor = tf.reduce_mean(SMs_tensor, axis=-1, name='SMs_tensor')
                SEG_output = tf.nn.softmax(SMs_tensor, name='SEG_output')  # [?, h, w, cls]

            else:
                raise ValueError('Unknown up-sample structure !!!')

            # Print some information.
            print('### Finish "Segmentation Head" (name scope: {}). The output shape: {}'.format(
                name_space, SEG_output.shape))

            # Return the segmentation results, and the loss tensors for "Cross-Entropy" calculation.
            return SEG_output, CEloss_tensors


    def _loss_summary(self, DQN_output, CEloss_tensors, prioritized_replay):
        r'''
            The definition of the Loss Function of the whole model.

        :param prioritized_replay:
        :param weight_PN_loss:
        :param weight_FTN_loss:
        :return:
        '''

        # Indicating.
        print('* ---> Start construct the loss function ...')


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


    def __upsample_loss(self, CEloss_tensors, epsilon):
        r'''
            Generate the loss for "Segmentation" branch.
        '''

        # Get configuration.
        conf_base = self._config['Base']

        # Get detailed parameters.
        input_shape = conf_base.get('input_shape')
        classification_dim = conf_base.get('classification_dimension')

        # ---------------------- Definition of UN cross-entropy loss ------------------------
        LOSS_name = self._name_space + '/SegLoss'
        with tf.variable_scope(LOSS_name):
            # Placeholder of ground truth segmentation.
            if not isinstance(input_shape, list):
                raise TypeError('The input shape parameter must be list !!!')
            label_shape = [s for s in input_shape[:-1]]
            label_shape.append(classification_dim)
            GT_label = net_util.placeholder_wrapper(self._inputs, tf.float32, label_shape,
                                                    name='GT_label')  # [?, h, w, cls]

            # The class weights is used to deal with the "Sample Imbalance" problem.
            clazz_weights = net_util.placeholder_wrapper(self._inputs, tf.float32, [None, classification_dim],
                                                         name='clazz_weights')  # [?, cls]

            ################################################################################

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


            return


    def __reinforcement_loss(self):


        return




