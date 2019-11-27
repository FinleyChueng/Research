import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np
import tfmodule.layer as cus_layers
import tfmodule.residual as cus_res
import tfmodule.util as net_util
import tfmodule.metrics as cus_metric



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
        # The losses holder.
        self._losses = {}
        # The summary holder.
        self._summary = {}
        # The visualization holder. (Option)
        self._visual = {}

        # The mediate holders used in coding.
        self.__mediates = {}

        # Determine the action dimension according to the config.
        conf_dqn = self._config['DQN']
        if conf_dqn.get('restriction_action'):
            self._action_dim = 9
        else:
            self._action_dim = 17

        # Finish initialization
        return



    def definition(self):
        r'''
            Definition of the whole model. Including build architecture and define loss function.
        '''

        # Specify the name space if enable the "Double DQN".
        conf_dqn = self._config['DQN']
        double_dqn = conf_dqn.get('double_dqn', None)

        # Start build the model.
        if double_dqn is None:
            DQN_output, SEG_logits, CEloss_tensors = self._architecture(self._action_dim, self._name_space)
        else:
            ORG_name, TAR_name = double_dqn
            DQN_output, SEG_logits, CEloss_tensors = self._architecture(self._action_dim, ORG_name)
            self._architecture(self._action_dim, TAR_name, with_segmentation=False)

        # Construct the loss function after building model.
        self._loss_summary(DQN_output, SEG_logits, CEloss_tensors)

        # Transfer some holders if enable "Double DQN".
        if double_dqn is not None:
            # Indicating.
            print('/-' * 50 + '/')
            print('* ---> Transfer some holders ... ')
            tar_name = double_dqn[1]
            # Inputs -> Losses.
            inputs_keys = list(self._inputs.keys())
            for k in inputs_keys:
                if k.startswith(tar_name):
                    v = self._inputs.pop(k)
                    self._losses[k] = v
            # Outputs -> Losses.
            outputs_keys = list(self._outputs.keys())
            for k in outputs_keys:
                if k.startswith(tar_name):
                    v = self._outputs.pop(k)
                    self._losses[k] = v
            # Show the all holders.
            print('| ---> Finish holders transferring !')
            print('|   ===> Inputs holder: {}'.format(self._inputs.keys()))
            print('|   ===> Outputs holder: {}'.format(self._outputs.keys()))
            print('|   ===> Losses holder: {}'.format(self._losses.keys()))
            print('|   ===> Summary holder: {}'.format(self._summary.keys()))
            print('|   ===> Visual holder: {}'.format(self._visual.keys()))
            print('\-' * 50 + '\\')

        # Return the inputs, outputs, losses, summary and visual holder.
        return self._inputs, self._outputs, self._losses, self._summary, self._visual


    def notify_copy2_DDQN(self, tf_sess, only_head=False):
        r'''
            Copy the parameters from Origin DQN to the Target DQN. To support the "Double DQN".

        :param tf_sess: The tensorflow session supplied by caller method.
        :return:
        '''
        # Get the name space if enable the "Double DQN".
        conf_dqn = self._config['DQN']
        double_dqn = conf_dqn.get('double_dqn', None)
        # Only execute when specify the name scope pair.
        if double_dqn is not None:
            # Get name scope pair.
            from_namespace, to_namespace = double_dqn
            # Only copy the parameters of head if specified.
            if only_head:
                from_namespace += '/DQN'
                to_namespace += '/DQN'
            # Operation to copy parameters.
            ops_base = net_util.copy_model_parameters(from_namespace, to_namespace)
            # Execute the operation.
            tf_sess.run(ops_base)
        # Plain return.
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
        print('/-' * 50 + '/')
        print('* ---> Start build the model ... (name scope: {})'.format(name_space))

        # Generate the input for model.
        input_tensor = self.__input_justify(name_space)
        # Extract feature maps.
        FE_tensor, DS_feats = self.__feature_extract(input_tensor, name_space)
        # Segment current image (patch). (Generate the segmentation branch)
        if with_segmentation:
            SEG_tensor, SEG_logits, CEloss_tensors = self.__segmentation(FE_tensor, DS_feats, self._name_space)
            # Fuse the previous (complete) and current (region) segmentation result.
            SEG_output, FUSE_result = self.__result_fusion(SEG_tensor, self._name_space)
        else:
            # No need the two holders.
            SEG_logits = CEloss_tensors = None
        # Select next region to deal with. (Generate the DQN output)
        SUIT_result = self.__mediates[self._name_space + '/SEG_suit_output']
        DQN_output = self.__region_selection(FE_tensor, SUIT_result, action_dim, name_space)

        # Package some outputs.
        net_util.package_tensor(self._outputs, DQN_output)
        if with_segmentation:
            net_util.package_tensor(self._outputs, SEG_output)
            net_util.package_tensor(self._outputs, FUSE_result)

        # Show the input and output holders.
        print('| ---> Finish model architecture (name scope: {}) !'.format(name_space))
        print('|   ===> Inputs holder (name scope: {}): {}'.format(name_space, self._inputs.keys()))
        print('|   ===> Outputs holder (name scope: {}): {}'.format(name_space, self._outputs.keys()))
        print('\-' * 50 + '\\')

        # Only return the tensors that will be used in "Loss Construction".
        return DQN_output, SEG_logits, CEloss_tensors


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
        # Determine the input type.
        input_type = conf_cus.get('input_type', 'whole')
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

            # The bounding-box indicating whether to focus on.
            focus_bbox = net_util.placeholder_wrapper(self._inputs, tf.float32, [None, 4], name='Focus_Bbox')

            # The flag that indicates whether it's "Segment" or "Focus" stage.
            segment_stage = net_util.placeholder_wrapper(self._inputs, tf.bool, [None, ], name='Segment_Stage')

            # Declare the function used to crop the specific bbox region of input tensor.
            def region_crop(x, bbox, size, name):
                bbox_ids = tf.range(tf.reduce_sum(tf.ones_like(bbox, dtype=tf.int32)[:, 0]))
                y = tf.image.crop_and_resize(x, bbox, bbox_ids, size, name=name + '_focus_crop')
                return y

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
                pos_info = net_util.placeholder_wrapper(self._inputs, tf.float32, [None,],
                                                        name=pos_name)  # [?]
            else:
                raise ValueError('Unknown position information fusion method !!!')

            # Crop or resize the input image into suitable size if size not matched.
            #   What's more, scale the "Focus Bounding-box" if needed.
            suit_h = conf_base.get('suit_height')
            suit_w = conf_base.get('suit_width')
            if suit_h != input_shape[1] or suit_w != input_shape[2]:
                crop_method = conf_cus.get('size_matcher', 'crop')
                if crop_method == 'crop':
                    # Crop to target size.
                    raw_image = tf.image.resize_image_with_crop_or_pad(raw_image, suit_h, suit_w)
                    prev_result = tf.image.resize_image_with_crop_or_pad(prev_result, suit_h, suit_w)
                    if pos_method == 'map':
                        pos_info = tf.image.resize_image_with_crop_or_pad(pos_info, suit_h, suit_w)
                elif crop_method == 'bilinear':
                    # Bilinear resize to target size.
                    raw_image = tf.image.resize_bilinear(raw_image, [suit_h, suit_w], name='bi_image')
                    prev_result = tf.image.resize_nearest_neighbor(prev_result, [suit_h, suit_w], name='nn_prev')
                    if pos_method == 'map':
                        pos_info = tf.image.resize_nearest_neighbor(pos_info, [suit_h, suit_w], name='nn_pos')
                else:
                    raise ValueError('Unknown size match method !!!')
                # Scale the "Focus Bounding-box".
                focus_bbox = net_util.scale_bbox(focus_bbox,
                                                 src_height=input_shape[1],
                                                 src_width=input_shape[2],
                                                 dst_height=suit_h,
                                                 dst_width=suit_w,
                                                 name='Scale_FoBbox')
                # Package the scaled "Focus Bbox" into the mediate holders.
                net_util.package_tensor(self.__mediates, focus_bbox)

            # Different fusion method for input according to input type.
            if input_type == 'region':
                # After declaring the public input part, it shall deal with the input tensor
                #   according to the different stage (branch). That is:
                # 1. Focus on the given region (bounding-box) if it's "Segmentation" stage.
                # 2. Introduce the position info if it's "Region Selection" stage with the
                #       "sight" position info.
                input_tensor = tf.where(segment_stage,
                                        region_crop(raw_image, focus_bbox, [suit_h, suit_w], 'SE'),
                                        raw_image if pos_method != 'sight'
                                        else region_crop(raw_image, pos_info, [suit_h, suit_w], 'FO'),
                                        name='1E_foInput')
            elif input_type == 'WR':
                # Concatenate the "Whole Perspective" with "Region Perspective".
                input_tensor = region_crop(raw_image, focus_bbox, [suit_h, suit_w], '1E_foInput')
                input_tensor = tf.concat([raw_image, input_tensor], axis=-1, name='WR_input')
            elif input_type == 'whole':
                # Do not need do something, use raw image is okay.
                input_tensor = raw_image
            else:
                raise ValueError('Unknown input type !!!')

            # Concat the tensors to generate input for whole model.
            input_tensor = tf.concat([input_tensor, prev_result], axis=-1, name='2E_input')

            # Concatenate the "Position Information" after "Focus Region" of whole tensor
            #   if the position information supplied in the "map"-form.
            if pos_method == 'map':
                input_tensor = tf.concat([input_tensor, pos_info], axis=-1, name='3E_input')

            # Print some information.
            print('### Finish "Input Justification" (name scope: {}). '
                  'The raw input shape: {}, the justified shape: {}'.format(
                name_space, tuple(input_shape[1:3]), input_tensor.shape))

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


    def __segmentation(self, FE_tensor, DS_feats, name_space):
        r'''
            Segmentation branch. Which is actually a up-sample head.
                It's used to segment the current input image (patch).

            ** Note that, this function do not directly generate the
                segmentation result, coz the output may not match the
                origin size. The real segmentation result is generated
                by the "Result Fusion" function. This function only
                provide the tensors before "Softmax".

            The input is the deep feature maps (each level) extracted by
                "Feature Extraction" backbone.
        '''

        # Get detailed configuration.
        conf_base = self._config['Base']
        conf_train = self._config['Training']
        # Suitable height and width.
        suit_h = conf_base.get('suit_height')
        suit_w = conf_base.get('suit_width')
        # Classification categories.
        classification_dim = conf_base.get('classification_dimension')
        # Feature normalization method and activation function.
        fe_norm = conf_base.get('feature_normalization', 'batch')
        activation = conf_base.get('activation', 'relu')
        # Dropout (keep probability) for convolution and fully-connect operation.
        conv_kprob = conf_base.get('convolution_dropout', 1.0)
        fc_kprob = conf_base.get('fully_connect_dropout', 0.5)
        # Regularization Related.
        regularize_coef = conf_train.get('regularize_coef', 0.0)
        regularizer = tf_layers.l2_regularizer(regularize_coef)

        # Get the "Focus Bbox" first.
        dqn_names = self._config['DQN'].get('double_dqn', None)
        if dqn_names is not None:
            stage_prefix = dqn_names[0]
        else:
            stage_prefix = self._name_space
        focus_bbox = self.__mediates[stage_prefix + '/Scale_FoBbox']

        # Declare the scores fusion block. Pass through a 1x1 conv and bilinear.
        def gen_scores(x, out_chans, name_idx):
            y = cus_layers.base_conv2d(x, out_chans, 1, 1,
                                       feature_normalization=fe_norm,
                                       activation=activation,
                                       keep_prob=conv_kprob,
                                       regularizer=regularizer,
                                       name_space='gen_score_conv1x1_0' + str(name_idx))
            y = tf.image.resize_bilinear(y, size=[suit_h, suit_w],
                                         name='gen_score_bi_0' + str(name_idx))
            return y

        # Declare the weights generation function. Which used to generate different
        #   weights (method) function according to structure.
        def FPN_weights(x, cor_size, structure):
            if structure == 'FPN-pW':
                w = tf.reshape(x, [-1, cor_size[0], cor_size[1] * classification_dim, x.shape[-1]],
                               name='FPN_pW_reshape')  # [?, h, w*cls, scales]
                w = cus_layers.base_conv2d(w, x.shape[-1], 1, 1,
                                           feature_normalization=fe_norm,
                                           activation=activation,
                                           keep_prob=conv_kprob,
                                           regularizer=regularizer,
                                           name_space='FPN_pW_1x1conv')
                w = tf.nn.softmax(w, axis=-1, name='FPN_pW_softmax')
                w = tf.reshape(w, [-1, cor_size[0], cor_size[1], classification_dim, x.shape[-1]],
                               name='FPN_pW_weights')  # [?, h, w, cls, scales]
            elif structure == 'FPN-fW' or structure == 'FPN-S':
                w = tf.expand_dims(tf.reduce_mean(x, axis=(0, 1, 2, 3)), axis=0)  # [1, scales]
                w = cus_layers.base_fc(w, int(x.shape[-1]),
                                       feature_normalization=fe_norm,
                                       activation=activation,
                                       keep_prob=fc_kprob,
                                       regularizer=regularizer,
                                       name_space='FPN_wt')  # [1, scales]
                w = tf.nn.softmax(w, axis=-1, name='FPN_fW_weights')  # [1, scales]
                if up_structure == 'FPN-S':
                    w = tf.one_hot(tf.argmax(w, axis=-1), depth=x.shape[-1],
                                   name='FPN_S_weights')  # [1, scales]
                w = tf.expand_dims(tf.expand_dims(tf.expand_dims(w, axis=0), axis=0),
                                   axis=0, name='FPN_flat_w')  # [1, 1, 1, 1, scales]
                w = tf.multiply(w, tf.ones_like(x), name='FPN_weights')     # [?, h, w, cls, scales]
            elif structure == 'FPN-bbox':
                p = cor_size[0] // FE_tensor.get_shape().as_list()[1]
                p = int(np.log2(p))
                p = [1/np.exp2(r+1) for r in range(p)]
                p.append(0.)
                p = tf.constant([p], dtype=tf.float32)  # [1, scales]
                bh = tf.expand_dims(focus_bbox[:, 2] - focus_bbox[:, 0], axis=-1)   # [?, 1]
                bw = tf.expand_dims(focus_bbox[:, 3] - focus_bbox[:, 1], axis=-1)   # [?, 1]
                ph = tf.greater_equal(bh, p)    # [?, scales]
                pw = tf.greater_equal(bw, p)    # [?, scales]
                pr = tf.to_int32(tf.logical_or(ph, pw))     # [?, scales]
                w = tf.one_hot(tf.argmin(pr, axis=-1), depth=p.shape[-1])   # [?, scales]
                w = tf.expand_dims(tf.expand_dims(tf.expand_dims(w, axis=1), axis=1),
                                   axis=1, name='FPN_bbox_w')  # [?, 1, 1, 1, scales]
                w = tf.multiply(w, tf.ones_like(x), name='FPN_weights')  # [?, h, w, cls, scales]
            else:
                raise ValueError('Unknown FPN up-sample structure !!!')
            return w

        # --------------------------------- "Segmentation" branch. ------------------------------------
        # Get configuration for ResNet
        conf_res = self._config['ResNet']
        conf_up = self._config['UpSample']
        # Layer number and kernel number of blocks for ResNet.
        kernel_numbers = conf_res.get('kernel_numbers')
        layer_units = conf_res.get('layer_units')
        # Get parameters.
        skip_method = conf_up.get('skip_connection', 'raw')
        up_structure = conf_up.get('upsample_structure', 'U-Net')
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

        # Declare the function used to "Up-sample" and add "Skip Connection".
        def up_block(x, scale, index, ds_idx, thres_id):
            r'''
                Generate the U-structure network, including: gradually up-sample
                    and add "Skip Connection". Or, just "Up-sample".
            '''
            # Scale up the feature maps. Whether use the pure de-conv
            #   or the "residual" de-conv.
            y = cus_res.transition_layer(x, kernel_numbers[ds_idx],
                                         scale_down=False,
                                         scale_factor=scale,
                                         structure=up_method,
                                         feature_normalization=fe_norm,
                                         activation=activation,
                                         keep_prob=conv_kprob,
                                         regularizer=regularizer,
                                         name_space='Scale_Up0' + str(index + 1))  # [?, 2x, 2x, 0.5c]

            # Add "Skip Connection" if specified.
            if skip_method == 'raw':
                # Use the raw down-sampled feature maps.
                skip_conn = DS_feats[ds_idx]
            elif skip_method == 'conv':
                # Pass through a 1x1 conv to get the skip connection features.
                raw_ds = DS_feats[ds_idx]
                skip_conn = cus_layers.base_conv2d(raw_ds, raw_ds.shape[-1], 1, 1,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='skip_conv0' + str(index + 1))
            elif skip_method == 'res':
                # Pass through a residual layer to get the skip connection features.
                raw_ds = DS_feats[ds_idx]
                skip_conn = cus_res.residual_block(raw_ds, raw_ds.shape[-1], 1,
                                                   kernel_size=1,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='skip_conv0' + str(index + 1))
            elif skip_method == 'w/o':
                # Do not use "Skip Connection".
                skip_conn = None
            else:
                raise ValueError('Unknown "Skip Connection" method !!!')

            # Now determine the features fusion method.
            if skip_conn is not None:
                if up_fuse == 'add':
                    y = tf.add(y, skip_conn, name='skip_add0' + str(index + 1))  # [2x, 2x, 0.5c]
                elif up_fuse == 'concat':
                    y = tf.concat([y, skip_conn], axis=-1, name='skip_concat0' + str(index + 1))  # [2x, 2x, c]
                else:
                    raise ValueError('Unknown "Skip Feature" fusion method !!!')

            # After up-sample, pass through the residual blocks to convolution.
            if isinstance(upres_layers, int):
                layer_num = upres_layers
            elif upres_layers == 'same':
                if index != thres_id:
                    layer_num = layer_units[ds_idx] - 1
                else:
                    # The last layer is special. Coz the corresponding layer
                    #   is max pooling, and it don't use any convolution.
                    layer_num = conf_up.get('last_match', 3)
            else:
                raise ValueError('Unknown up-sample block layers !!!')
            # Specify the layer for residual block. Note that, we may use 1x1 conv
            #   if @{layer_num} is a string.
            if isinstance(layer_num, int) and layer_num > 0:
                y = cus_res.residual_block(y, y.shape[-1], layer_num,
                                           feature_normalization=fe_norm,
                                           activation=activation,
                                           keep_prob=conv_kprob,
                                           regularizer=regularizer,
                                           name_space='UpResidual_conv0' + str(index + 1))
            elif layer_num == '1x1conv':
                y = cus_layers.base_conv2d(y, y.shape[-1], 1, 1,
                                           feature_normalization=fe_norm,
                                           activation=activation,
                                           keep_prob=conv_kprob,
                                           regularizer=regularizer,
                                           name_space='Up1x1_conv0' + str(index + 1))
            elif layer_num == 'w/o':
                pass
            else:
                raise ValueError('Unknown last layer match method !!!')
            # Finish block.
            return y
        # ----------------------------- end of up-block -----------------------------

        # Declare the loss holder (list).
        CEloss_tensors = []

        # Start definition.
        SEG_name = name_space + '/SEG'
        with tf.variable_scope(SEG_name):
            # Just the traditional structure. Gradually build the block is okay.
            if up_structure == 'U-Net':
                # Build the up-sample branch until "half-size".
                US_tensor = FE_tensor
                for idx in range(len(layer_units) // scale_exp):
                    # Compute the index for corresponding down-sample tensor and layer units.
                    cor_idx = - (idx + 1) * scale_exp - 1  # -1 for reverse. idx+1 for skip last one.
                    # Scale up the feature maps. That is, gradually add up-sample blocks.
                    US_tensor = up_block(US_tensor, scale=scale_factor,
                                         index=idx, ds_idx=cor_idx,
                                         thres_id=len(layer_units) // scale_exp - 1)
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
                # Independently return the segmentation logits, so that we can flexibly deal with it.
                SEG_logits = org_UST
                # Rename to get the SEG tensor (before Softmax).
                SEG_tensor = tf.identity(org_UST, name='SEG_tensor')    # [?, h, w, cls]

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
                # Independently return the segmentation logits, so that we can flexibly deal with it.
                SEG_logits = MLIF_map
                # Fuse (mean) all score maps to get the final segmentation.
                SMs_tensor = tf.stack(CEloss_tensors, axis=-1, name='SMs_stack')
                SMs_tensor = tf.reduce_mean(SMs_tensor, axis=-1, name='SMs_tensor')
                # Rename to get the SEG tensor (before Softmax).
                SEG_tensor = tf.identity(SMs_tensor, name='SEG_tensor')     # [?, h, w, cls]

            # Each layer will be used to generate result. Gradually or skip to build blocks.
            elif up_structure in ['FPN-pW', 'FPN-fW', 'FPN-S', 'FPN-bbox']:
                # Build the up-sample branch until "half-size".
                FPN_tensors = []
                FPN_tensors.append(FE_tensor)
                for bl_idx in range(len(layer_units) // scale_exp):
                    BL_tensor = FPN_tensors[-1]
                    for la_idx in range(scale_exp):
                        # Compute the index for corresponding down-sample tensor and layer units.
                        cor_idx = - bl_idx * scale_exp - (la_idx + 1) - 1  # -1 for reverse. idx+1 for skip last one.
                        # Scale up the feature maps. That is, gradually add up-sample blocks.
                        US_tensor = up_block(BL_tensor, scale=int(np.exp2(la_idx + 1)),
                                             index=bl_idx * scale_exp + la_idx,
                                             ds_idx=cor_idx,
                                             thres_id=len(layer_units) - 1)
                        # Add into BLOCK tensors.
                        FPN_tensors.append(US_tensor)

                # Scale up to the original size.
                half_UST = FPN_tensors[-1]  # [?, half, half, OC]
                org_UST = cus_layers.base_deconv2d(half_UST, classification_dim, 3, 2,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='WS_tensor')
                FPN_tensors.append(org_UST)
                # Resize each feature maps of "FPN tensors" to the original size.
                temp_T = []
                for i, t in enumerate(FPN_tensors):
                    t_shape = t.get_shape().as_list()[1:3]
                    if t_shape[0] == suit_h or t_shape[1] == suit_w:
                        org_t = t
                    else:
                        t_cls = cus_layers.base_conv2d(t, classification_dim, 1, 1,
                                                       feature_normalization=fe_norm,
                                                       activation=activation,
                                                       keep_prob=conv_kprob,
                                                       regularizer=regularizer,
                                                       name_space='FPN_2cls' + str(i+1))
                        org_t = tf.image.resize_bilinear(t_cls, [suit_h, suit_w], name='FPN_2org0' + str(i+1))
                    temp_T.append(org_t)
                FPN_tensors = temp_T

                # Generate the weights for FPN tensors.
                FPN_W = FPN_weights(tf.stack(FPN_tensors, axis=-1), cor_size=[suit_h, suit_w],
                                    structure=up_structure)     # [?, h, w, cls, scales]
                FPN_W = tf.reshape(FPN_W, [-1, suit_h, suit_w, classification_dim * len(FPN_tensors)],
                                   name='FPN_weights_4D')   # [?, h, w, cls*scales]
                FPN_tensors.append(FPN_W)   # for conveniently process.

                # Fuse results (logits) of each scale into one logit.
                def FPN_fuse(x, bbox, cor_size):
                    temp = []
                    for i, xi in enumerate(x):
                        yi = net_util.pad_2up(xi, bbox, cor_size, name='FPN_pad0'+str(i+1))  # [1, h, w, cls]
                        temp.append(yi)
                    y = tf.stack(temp[:-1], axis=-1, name='FPN_stack')  # [1, h, w, cls, scales]
                    w = tf.reshape(temp[-1],
                                   [temp[-1].shape[0], cor_size[0], cor_size[1], classification_dim, len(x[:-1])],
                                   name='FPN_weights_5D')   # [1, h, w, cls, scales]
                    y = tf.reduce_sum(tf.multiply(y, w), axis=-1, name='FPN_logits')  # [1, h, w, cls]
                    return y
                # Crop and pad zero to get the FPN logits and then fuse them.
                SEG_tensor = net_util.batch_resize_to_bbox_for_op(
                    FPN_tensors, bbox=focus_bbox, cor_size=[suit_h, suit_w],
                    resize_method=['crop'] * len(FPN_tensors), op_func=FPN_fuse,
                    output_shape=org_UST.get_shape().as_list()[1:],
                    name='SEG_tensor')  # [?, h, w, cls]
                # Independently return the segmentation logits, so that we can flexibly deal with it.
                SEG_logits = SEG_tensor

            else:
                raise ValueError('Unknown up-sample structure !!!')

            # Check the shape validity of each output tensors of "Segmentation" branch.
            shape_validity = True
            os1 = SEG_tensor.get_shape().as_list()[1:3]
            if suit_h != os1[0] or suit_w != os1[1]:
                shape_validity = False
            os2 = SEG_logits.get_shape().as_list()[1:3]
            if suit_h != os2[0] or suit_w != os2[1]:
                shape_validity = False
            for t in CEloss_tensors:
                os3 = t.get_shape().as_list()[1:3]
                if suit_h != os3[0] or suit_w != os3[1]:
                    shape_validity = False
            if not shape_validity:
                raise ValueError('The output shape of "Segmentation" branch is invalid !!! '
                                 'it should be: ({}, {})'.format(suit_h, suit_w))

            # Print some information.
            print('### Finish "Segmentation Head" (name scope: {}). '
                  'The output shape: {}'.format(name_space, SEG_tensor.shape))

            # Return the segmentation tensor, logits, and the loss tensors for "Cross-Entropy" calculation.
            return SEG_tensor, SEG_logits, CEloss_tensors


    def __result_fusion(self, SEG_tensor, name_space):
        r'''
            Fuse the "Region Result" with the previous "Complete Result".
                The fusion method is abundant.

            ** Note that, this function do recover the "Suit" size of segmentation
                output to the original input image size (for convenient usage).
        '''

        # Get basic configuration.
        conf_base = self._config['Base']
        input_shape = conf_base.get('input_shape')
        origin_h = input_shape[1]
        origin_w = input_shape[2]
        classification_dim = conf_base.get('classification_dimension')
        # Get the shape of "Segmentation" branch. Actually is "Suit" size.
        up_h = conf_base.get('suit_height')
        up_w = conf_base.get('suit_width')

        # --------------------------------- "Result Fusion" part. ------------------------------------
        # Get fusion method.
        conf_cus = self._config['Custom']
        fusion_method = conf_cus.get('result_fusion', 'prob')
        logit_type = conf_cus.get('result_tensor', 'complete')

        # Get the "Focus Bounding-box" holder.
        conf_dqn = self._config['DQN']
        dqn_names = conf_dqn.get('double_dqn', None)
        if dqn_names is not None:
            stage_prefix = dqn_names[0]
        else:
            stage_prefix = self._name_space
        focus_bbox = self.__mediates[stage_prefix+'/Scale_FoBbox']

        # Start definition.
        RF_name = name_space + '/ResFuse'
        with tf.variable_scope(RF_name):
            # Declare the padding function used to pad the region-tensor to the original up-sample size.
            def restore_2up(x, bbox, cor_size):
                y = x[0]  # [1, h, w, c]
                y = net_util.pad_2up(y, bbox, cor_size, name='Restore_2up')
                return y

            # Different transition method according to the "Logit Tensor Type".
            if logit_type in ['bbox-bi', 'bbox-crop']:
                # Bi-resize when the logit tensor is "bbox-bi" type.
                if logit_type == 'bbox-crop':
                    resize_methods = ['crop']
                else:
                    resize_methods = ['bilinear']
                # Resize and pad zero to get the region tensor.
                REGION_tensor = net_util.batch_resize_to_bbox_for_op(
                    [SEG_tensor], bbox=focus_bbox, cor_size=[up_h, up_w],
                    resize_method=resize_methods, op_func=restore_2up,
                    output_shape=SEG_tensor.get_shape().as_list()[1:],
                    name='Region_tensor')   # [?, h, w, cls]
            elif logit_type == 'complete':
                REGION_tensor = tf.identity(SEG_tensor, name='Region_tensor')
            else:
                raise ValueError('Unknown segmentation logit type !!!')

            # Fusion according to different method. (Including declare the "Complete Result")
            if fusion_method == 'logit':
                # The placeholder of "Complete Result". --> Logit tensors.
                complete_result = net_util.placeholder_wrapper(self._inputs, tf.float32,
                                                               [None, up_h, up_w, classification_dim],
                                                               name='Complete_Result')  # [?, h, w, cls]
                # Fuse the result in "Logit"-level, and generate the final segmentation result.
                FUSE_result = tf.add(complete_result, REGION_tensor, name='RawF_result')    # [?, h, w, cls]
                FUSE_result = tf.where(tf.equal(tf.reduce_sum(complete_result), 0.0),
                                       REGION_tensor, FUSE_result,
                                       name='FUSE_result')  # filter the init.
                SEG_prob = tf.nn.softmax(FUSE_result, name='SEG_prob')  # [?, h, w, cls]
                SEG_output = tf.argmax(SEG_prob, axis=-1, name='SEG_suit_output')    # [?, h, w]
            elif fusion_method == 'prob':
                # The placeholder of "Complete Result". --> Probability.
                complete_result = net_util.placeholder_wrapper(self._inputs, tf.float32,
                                                               [None, up_h, up_w, classification_dim],
                                                               name='Complete_Result')  # [?, h, w, cls]
                # Fuse the result in "Probability"-level, and generate the final segmentation result.
                region_prob = tf.nn.softmax(REGION_tensor, name='region_prob')  # [?, h, w, cls]
                FUSE_result = tf.where(tf.equal(region_prob, 0.0), complete_result,
                                       tf.add(region_prob, complete_result) / 2.0,
                                       name='RawF_result')  # [?, h, w, cls]
                FUSE_result = tf.where(tf.equal(tf.reduce_sum(complete_result), 0.0),
                                       region_prob, FUSE_result,
                                       name='FUSE_result')  # filter the init.
                SEG_output = tf.argmax(FUSE_result, axis=-1, name='SEG_suit_output')  # [?, h, w]
            elif fusion_method == 'mask':
                # The placeholder of "Complete Result". --> Mask.
                complete_result = net_util.placeholder_wrapper(self._inputs, tf.int64,
                                                               [None, up_h, up_w],
                                                               name='Complete_Result')  # [?, h, w]
                # Fuse the result in "Mask"-level, and generate the final segmentation result.
                region_prob = tf.nn.softmax(REGION_tensor, name='region_prob')  # [?, h, w, cls]
                region_mask = tf.argmax(region_prob, axis=-1, name='region_mask')   # [?, h, w]
                # Simply strategy: Use the "Region Mask" as the ground truth expects for "Background" category.
                FUSE_result = tf.where(tf.equal(region_mask, 0), complete_result, region_mask,
                                       name='RawF_result')  # [?, h, w]
                FUSE_result = tf.where(tf.equal(tf.reduce_sum(complete_result), 0),
                                       region_mask, FUSE_result,
                                       name='FUSE_result')  # filter the init.
                # Fusion result is the final segmentation result.
                SEG_output = tf.identity(FUSE_result, name='SEG_suit_output')  # [?, h, w]
            else:
                raise ValueError('Unknown result fusion method !!!')

            # Package the suitable segmentation result in the mediate holders for lately usage.
            net_util.package_tensor(self.__mediates, SEG_output)    # [?, sh, sw]

            # Recover the segmentation output (suit) size to the original size if don't matched.
            if up_h != origin_h or up_w != origin_w:
                # Temporarily expand the dimension for conveniently processing.
                SEG_output = tf.expand_dims(SEG_output, axis=-1, name='expa_SEG')  # [?, h, w, 1]
                # Justify the size.
                conf_cus = self._config['Custom']
                crop_method = conf_cus.get('size_matcher', 'crop')
                if crop_method == 'crop':
                    # Crop to target size.
                    SEG_output = tf.image.resize_image_with_crop_or_pad(SEG_output, origin_h, origin_w)
                elif crop_method == 'bilinear':
                    # Nearest-neighbour resize to target size. (Coz it's segmentation result, integer type)
                    SEG_output = tf.image.resize_nearest_neighbor(SEG_output, [origin_h, origin_w])
                else:
                    raise ValueError('Unknown size match method !!!')
                # Recover to the original dimension.
                SEG_output = tf.reduce_mean(SEG_output, axis=-1, name='redc_SEG')  # [?, h, w]

            # Rename the tensor.
            SEG_output = tf.identity(SEG_output, name='SEG_output')  # [?, h, w]

            # Print some information.
            print('### Finish "Result Fusion" (name scope: {}). '
                  'The raw input shape: {}, the justified shape: {}'.format(
                name_space, SEG_tensor.shape, SEG_output.shape))

            # Return the segmentation result and the fusion value for next iteration.
            return SEG_output, FUSE_result


    def __region_selection(self, FE_tensor, SUIT_result, action_dim, name_space):
        r'''
            Region selection branch. Which is actually a DQN head.
                It's used to select the next region for precisely processing.

            The input is the deep feature maps extracted by
                "Feature Extraction" backbone.
        '''

        # Get detailed configuration.
        conf_base = self._config['Base']
        conf_train = self._config['Training']
        # Suitable height and width.
        suit_h = conf_base.get('suit_height')
        suit_w = conf_base.get('suit_width')
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
        # Get segmentation fusion method.
        seg_fuse = conf_dqn.get('fuse_segmentation', 'add')
        # Get the dimension reduction method.
        reduce_dim = conf_dqn.get('reduce_dim', 'residual')
        # Get the action history length.
        his_len = conf_dqn.get('actions_history', 10)
        # Check whether enable "Dueling Network" or not.
        dueling_network = conf_dqn.get('dueling_network', True)

        # Check for the introduction method of "Position Information".
        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')

        # Start definition.
        DQN_name = name_space + '/DQN'
        with tf.variable_scope(DQN_name):
            # Reshape the FUSE_result as a part of DQN input.
            FE_h, FE_w, FE_c = FE_tensor.get_shape().as_list()[1:]  # [fe_h, fe_w, fe_c]
            FE_prop = (suit_h // FE_h) * (suit_w // FE_w)
            DQN_in = tf.reshape(SUIT_result, [-1, FE_h, FE_w, FE_prop],
                                name='suit_res_4DQN')   # [?, fe_h, fe_w, prop^2]
            DQN_in = cus_layers.base_conv2d(tf.to_float(DQN_in), FE_c, 1, 1,
                                            feature_normalization=fe_norm,
                                            activation=activation,
                                            keep_prob=conv_kprob,
                                            regularizer=regularizer,
                                            name_space='suit_res_conv1x1')  # [?, fe_h, fe_w, fe_c]
            if seg_fuse == 'add':
                DQN_in = tf.add(FE_tensor, DQN_in, name='Fuse_IN')  # [?, fe_h, fe_w, fe_c]
            elif seg_fuse == 'concat':
                DQN_in = tf.concat([FE_tensor, DQN_in], axis=-1, name='Fuse_IN')    # [?, fe_h, fe_w, 2 * fe_c]
            else:
                raise ValueError('Unknown segmentation fusion method for DQN !!!')

            # Scale down the feature maps according to the specific method.
            if reduce_dim == 'conv':
                # Use "Convolution" to reduce dimension.
                redc_tensor = cus_layers.base_conv2d(DQN_in, 1024, 3, 2,
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
                redc_tensor = cus_res.transition_layer(DQN_in, 1024,
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

            # Fuse the actions history to the DQN input if enabled.
            if his_len > 0:
                # The input actions history holder.
                acts_history = net_util.placeholder_wrapper(self._inputs, tf.float32, [None, his_len],
                                                            name='actions_history')  # [?, his_len]
                flat_tensor = tf.concat([flat_tensor, acts_history], axis=-1, name='fuse_pos_coord')  # [?, OC+his_len]
            else:
                # For conveniently coding. Do nothing.
                _1 = net_util.placeholder_wrapper(self._inputs, tf.bool, [None,], name='actions_history')  # [?,]

            # Fuse (concat) the position vector if use "coord"-like position information.
            if pos_method == 'coord':
                pos_info = self._inputs[name_space + '/position_info']
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


    def _loss_summary(self, DQN_output, SEG_logits, CEloss_tensors):
        r'''
            Construct the loss function for each branch. Fuse each independent loss
                to get the final loss for the whole model. What's more, generate
                the summary for visualization.
        '''

        # Indicating.
        print('/-' * 50 + '/')
        print('* ---> Start construct the loss function ...')

        # Calculate the segmentation loss.
        SEG_loss = self.__upsample_loss(SEG_logits, CEloss_tensors, name_space=self._name_space)
        # Calculate the DQN loss.
        DQN_loss = self.__reinforcement_loss(DQN_output, name_space=self._name_space)

        # Add the two parts as the whole loss for model.
        conf_train = self._config['Training']
        dqn_loss_factor = conf_train.get('dqn_loss_factor', 1.0)
        WHOLE_loss = tf.add(SEG_loss, dqn_loss_factor * DQN_loss, name=self._name_space+'/WHOLE_loss')

        # Start to calculate regularization loss. -------------------------------------------------------
        conf_dqn = self._config['DQN']
        double_dqn = conf_dqn.get('double_dqn', None)
        # Different process depends on whether enable the "Double DQN" or not.
        if double_dqn is None:
            REG_loss = 0.
            for idx, reg_loss in enumerate(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)):
                REG_loss = tf.add(REG_loss, reg_loss, name=self._name_space+'/reg_loss'+str(idx+1))
        else:
            org_name, tar_name = double_dqn
            # Recursively add regularization depends on different situations.
            REG_loss = 0.
            for idx, reg_loss in enumerate(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)):
                if reg_loss.name.startswith(tar_name):
                    # Filter the target (backup) model.
                    continue
                else:
                    if reg_loss.name.startswith(self._name_space+'/SEG'):
                        # 'CUS/SEG', segmentation branch.
                        REG_loss = tf.where(tf.not_equal(SEG_loss, 0.),
                                            tf.add(REG_loss, reg_loss),
                                            REG_loss,
                                            name=self._name_space+'/seg_REG'+str(idx+1))
                    elif reg_loss.name.startswith(org_name+'/DQN'):
                        # 'ORG/DQN', DQN branch.
                        REG_loss = tf.where(tf.not_equal(DQN_loss, 0.),
                                            tf.add(REG_loss, reg_loss),
                                            REG_loss,
                                            name=self._name_space+'/dqn_REG'+str(idx+1))
                    else:
                        # 'ORG/FeatExt', public part.
                        REG_loss = tf.add(REG_loss, reg_loss, name=self._name_space+'/feats_REG'+str(idx+1))

        # Add regularization to raw loss.
        NET_loss = tf.add(WHOLE_loss, REG_loss, name=self._name_space + '/NET_loss')

        # Print some information.
        print('### Finish the definition of NET loss, Shape: {}'.format(NET_loss.shape))

        # Package the losses tensors into dictionary.
        net_util.package_tensor(self._losses, SEG_loss)
        net_util.package_tensor(self._losses, DQN_loss)
        net_util.package_tensor(self._losses, NET_loss)

        # Generate the summaries for visualization.
        self.__gen_summary(name_space=self._name_space)

        # Show the losses and summary holders.
        print('| ---> Finish the loss function construction !')
        print('|   ===> The loss holders: {}'.format(self._losses.keys()))
        print('|   ===> The summary holders: {}'.format(self._summary.keys()))
        print('\-' * 50 + '\\')

        # Plain return.
        return


    def __upsample_loss(self, SEG_logits, CEloss_tensors, name_space):
        r'''
            Generate (Cross-Entropy) loss for the "Segmentation" branch.
        '''

        # Get configuration.
        conf_base = self._config['Base']
        conf_train = self._config['Training']
        conf_cus = self._config['Custom']

        # Get detailed parameters.
        input_shape = conf_base.get('input_shape')
        suit_h = conf_base.get('suit_height')
        suit_w = conf_base.get('suit_width')
        classification_dim = conf_base.get('classification_dimension')
        score_factor = conf_train.get('score_loss_factor', 1.0)
        sample_share = conf_train.get('sample_share', False)
        crop_method = conf_cus.get('size_matcher', 'crop')
        logit_type = conf_cus.get('result_tensor', 'complete')
        supv_method = conf_cus.get('segmentation_supervision', 'dilate')
        loss_type = conf_cus.get('segmentation_loss', 'DICE')

        # Get the stage prefix for conveniently inferring inputs holders.
        conf_dqn = self._config['DQN']
        dqn_names = conf_dqn.get('double_dqn', None)
        if dqn_names is not None:
            stage_prefix = dqn_names[0]
        else:
            stage_prefix = self._name_space

        # ---------------------- Definition of UN cross-entropy loss ------------------------
        LOSS_name = name_space + '/SegLoss'
        with tf.variable_scope(LOSS_name):
            # Placeholder of ground truth segmentation.
            GT_label = net_util.placeholder_wrapper(self._losses, tf.int32, input_shape[:-1],
                                                    name='GT_label')  # [?, h, w]
            # Get the "Focus Bounding-box" used to crop "Focus" region in label.
            focus_bbox = self.__mediates[stage_prefix+'/Scale_FoBbox']
            # Also need to crop label if size do not match. Meanwhile re-assign
            #   the coordinates of "Focus Bounding-box".
            if suit_h != input_shape[1] or suit_w != input_shape[2]:
                # Temporarily expand a dimension for conveniently processing.
                GT_label = tf.expand_dims(GT_label, axis=-1, name='expa01_label')   # [?, h, w, 1]
                # Match the size.
                if crop_method == 'crop':
                    # Crop to target size.
                    GT_label = tf.image.resize_image_with_crop_or_pad(GT_label, suit_h, suit_w)
                elif crop_method == 'bilinear':
                    # Nearest-neighbour resize to target size. (Coz it's segmentation result, integer type)
                    GT_label = tf.image.resize_nearest_neighbor(GT_label, [suit_h, suit_w], name='bi_labels')
                else:
                    raise ValueError('Unknown size match method !!!')
            # Temporarily expand a dimension for conveniently processing if it's 3-D tensor.
            if len(GT_label.shape) != 4:
                GT_label = tf.expand_dims(GT_label, axis=-1, name='expa02_label')   # [?, h, w, 1]

            # Only if the SEGMENTATION logits is "bbox"-form that we need crop logit or label.
            if logit_type == 'bbox-bi':
                # Different operation procedure according to the "Supervision Method".
                if supv_method == 'dilate':
                    # Crop and resize (to suit size) the corresponding region in label with respects to image (patch).
                    bbox_ids = tf.range(tf.reduce_sum(tf.ones_like(focus_bbox, dtype=tf.int32)[:, 0]))
                    GT_label = tf.image.crop_and_resize(GT_label, focus_bbox, bbox_ids, [suit_h, suit_w],
                                                        method='nearest',
                                                        name='focus_label')     # [?, h, w, 1]
                elif supv_method == 'erode':
                    # Don't need special process for label here.
                    pass
                else:
                    raise ValueError('Unknown segmentation supervision method !!!')

            # The class weights is used to deal with the "Sample Imbalance" problem.
            clazz_weights = net_util.placeholder_wrapper(self._losses, tf.float32, [None, classification_dim],
                                                         name='clazz_weights')  # [?, cls]
            if loss_type == 'CE':
                cw_mask = tf.one_hot(tf.reduce_mean(tf.to_int32(GT_label), axis=-1),
                                     classification_dim, name='one_hot_label')    # [?, h, w, cls]
                clazz_weights = tf.expand_dims(tf.expand_dims(clazz_weights, axis=1), axis=1,
                                               name='expa_CW')      # [?, 1, 1, cls]
                clazz_weights = tf.multiply(clazz_weights, cw_mask, name='rect_weights')    # [?, h, w, cls]
                # Determine the loss function we used to calculate loss for segmentation.
                loss_func = tf.losses.sparse_softmax_cross_entropy
            elif loss_type == 'DICE':
                clazz_weights = tf.expand_dims(tf.expand_dims(clazz_weights, axis=1), axis=1,
                                               name='expa_CW')  # [?, 1, 1, cls]
                clazz_weights = tf.multiply(clazz_weights, tf.ones_like(SEG_logits),
                                            name='rect_weights')  # [?, h, w, cls]
                # Determine the loss function we used to calculate loss for segmentation.
                loss_func = cus_metric.DICE
            else:
                raise ValueError('Unknown segmentation loss type !!!')





            # # Multiply the mask (stage flag) to filter the losses (samples) that don't
            # #   belongs to "Segmentation" branch if not enable "Sample Share".
            # # Here we can multiply it with class weights instead of the final loss.
            # if not sample_share:
            #     # Generate filter mask.
            #     sample_4SEG = tf.to_float(self._inputs[stage_prefix+'/Segment_Stage'], name='sample_4SEG')  # [?]
            #     # Filter.
            #     sample_4SEG = tf.expand_dims(tf.expand_dims(
            #         tf.expand_dims(sample_4SEG, axis=-1), axis=-1), axis=-1)    # [?, 1, 1, 1]
            #     clazz_weights = tf.multiply(clazz_weights, sample_4SEG, name='Filtered_weights')    # [?, h, w, cls]

            # Different loss calculation according to the "Segmentation Logits Type".
            #   1) logits for "Bbox" region.
            if logit_type in ['bbox-bi', 'bbox-crop']:
                # Different loss operation procedure according to the "Supervision Method".
                # --> Dilate the label.
                if supv_method == 'dilate':
                    # The "bbox-crop" result tensor should use "Erode" supervision method.
                    if logit_type == 'bbox-crop':
                        raise NotImplementedError('The "dilate" supervision method for "bbox-crop"'
                                                  ' result type is not implemented !!!')
                    # Recover label to the original dimension.
                    GT_label = tf.reduce_mean(GT_label, axis=-1, name='redc_label')  # [?, h, w]
                    GT_label = tf.cast(GT_label, 'int32', name='Corr_label')  # [?, h, w]
                    # Reduce dimension of clazz weights.
                    if loss_type == 'CE':
                        clazz_weights = tf.reduce_sum(clazz_weights, axis=-1, name='Weights_map')  # [?, h, w]
                    # The single segmentation loss.
                    fix_loss = loss_func(
                        labels=GT_label, logits=SEG_logits, weights=clazz_weights, scope='FIX_loss')
                    # Recursively calculate the loss.
                    additional_loss = 0.
                    for idx, logits in enumerate(CEloss_tensors):
                        additional_loss += loss_func(
                            labels=GT_label, logits=logits, weights=clazz_weights, scope='addition_loss0' + str(idx + 1))
                # --> "Erode" the segmentation tensor.
                elif supv_method == 'erode':
                    # Declare the region loss function.
                    def region_loss(x, bbox, cor_size):
                        # Get candidates.
                        lab, logit, w = x
                        # Recover label to the original dimension and dtype.
                        lab = tf.reduce_mean(lab, axis=-1)  # [1, h, w]
                        lab = tf.cast(lab, 'int32')     # [1, h, w]
                        # Reduce dimension of clazz weights.
                        if loss_type == 'CE':
                            w = tf.reduce_sum(w, axis=-1)   # [1, h, w]
                        # Cross-Entropy loss.
                        y = loss_func(labels=lab, logits=logit, weights=w)
                        return y
                    # Decide the corresponding size for "Focus Bounding-box".
                    if suit_h != input_shape[1] or suit_w != input_shape[2]:
                        cor_size = [suit_h, suit_w]
                    else:
                        cor_size = input_shape[1:3]
                    # Decide the tensor resize methods tuple for loss calculation.
                    if logit_type == 'bbox-crop':
                        resize_methods = ['crop', 'crop', 'crop']
                    else:
                        resize_methods = ['crop', 'bilinear', 'crop']
                    # The single segmentation loss.
                    fix_loss = net_util.batch_resize_to_bbox_for_op([GT_label, SEG_logits, clazz_weights],
                                                                    bbox=focus_bbox, cor_size=cor_size,
                                                                    resize_method=resize_methods,
                                                                    op_func=region_loss,
                                                                    output_shape=None,
                                                                    name='FIX_loss')
                    # Recursively calculate the loss.
                    additional_loss = 0.
                    for idx, logits in enumerate(CEloss_tensors):
                        additional_loss += net_util.batch_resize_to_bbox_for_op(
                            [GT_label, logits, clazz_weights],
                            bbox=focus_bbox, cor_size=cor_size,
                            resize_method=resize_methods,
                            op_func=region_loss, output_shape=None,
                            name='addition_loss0'+str(idx+1)
                        )
                # Invalid value.
                else:
                    raise ValueError('Unknown segmentation supervision method !!!')

            # 2) use the complete logits.
            elif logit_type == 'complete':
                # Recover label to the original dimension.
                GT_label = tf.reduce_mean(GT_label, axis=-1, name='redc_label')  # [?, h, w]
                GT_label = tf.cast(GT_label, 'int32', name='Corr_label')  # [?, h, w]
                # Reduce dimension of clazz weights.
                if loss_type == 'CE':
                    clazz_weights = tf.reduce_sum(clazz_weights, axis=-1, name='Weights_map')  # [?, h, w]
                # The single segmentation loss.
                fix_loss = loss_func(
                    labels=GT_label, logits=SEG_logits, weights=clazz_weights, scope='FIX_loss')
                # Recursively calculate the loss.
                additional_loss = 0.
                for idx, logits in enumerate(CEloss_tensors):
                    additional_loss += loss_func(
                        labels=GT_label, logits=logits, weights=clazz_weights, scope='addition_loss0' + str(idx + 1))

            else:
                raise ValueError('Unknown segmentation logits type !!!')

            # Add the two parts as the final classification loss.
            SEG_loss = tf.add(fix_loss, score_factor * additional_loss, name='SEG_loss')

            # Print some information.
            print('### Finish the definition of SEG loss, Shape: {}'.format(SEG_loss.shape))

            # Return the segmentation loss.
            return SEG_loss


    def __reinforcement_loss(self, DQN_output, name_space):
        r'''
            Generate (L2-regression) loss for "DQN (Region Selection)" branch.
        '''

        # Get configuration.
        conf_dqn = self._config['DQN']
        conf_train = self._config['Training']
        # Get detailed parameters.
        prioritized_replay = conf_dqn.get('prioritized_replay', True)
        sample_share = conf_train.get('sample_share', False)

        # ---------------------- Definition of UN cross-entropy loss ------------------------
        LOSS_name = name_space + '/DQNLoss'
        with tf.variable_scope(LOSS_name):
            # Placeholder of input actions. Indicates which Q value (output of DQN) used to calculate cost.
            pred_action = net_util.placeholder_wrapper(self._losses, tf.int32, [None,], name='prediction_actions')
            pred_action = tf.one_hot(pred_action, self._action_dim, name='one_hot_Predacts')    # [?, acts]

            # Placeholder of target q values. That is, the "Reward + Future Q values".
            target_q_vals = net_util.placeholder_wrapper(self._losses, tf.float32, [None,], name='target_Q_values')

            # Only use selected Q values for DRQN. Coz in the "Future Q values" we use the max value.
            pred_q_vals = tf.reduce_sum(tf.multiply(DQN_output, pred_action), axis=-1,
                                        name='prediction_Q_values')    # [?]

            # The difference between prediction and target Q values.
            q_diff = tf.subtract(target_q_vals, pred_q_vals, name='Q_diff')  # [?]

            # Multiply the mask (stage flag) to filter the losses (samples) that don't
            #   belongs to "Segmentation" branch if not enable "Sample Share".
            if not sample_share:
                # Generate filter mask.
                dqn_names = conf_dqn.get('double_dqn', None)
                if dqn_names is not None:
                    stage_prefix = dqn_names[0]
                else:
                    stage_prefix = self._name_space
                sample_4DQN = tf.to_float(tf.logical_not(self._inputs[stage_prefix+'/Segment_Stage']),
                                          name='sample_4DQN')   # [?]
                # Filter.
                q_diff = tf.multiply(q_diff, sample_4DQN, name='Filtered_Q_diff')   # [?]

            # Define placeholder for IS weights if use "Prioritized Replay".
            if prioritized_replay:
                # Updated priority for input experience.
                EXP_priority = tf.abs(q_diff, name='EXP_priority')  # used to update Sumtree
                net_util.package_tensor(self._losses, EXP_priority)
                # Placeholder of the weights for experience.
                IS_weights = net_util.placeholder_wrapper(self._losses, tf.float32, [None,],
                                                          name='IS_weights')
                # Construct the prioritized loss.
                DQN_loss = tf.reduce_mean(
                    tf.multiply(IS_weights, tf.square(q_diff)),  # [?]
                    name='DQN_loss'
                )
            else:
                # Construct the simple loss.
                DQN_loss = tf.reduce_mean(tf.square(q_diff), name='DQN_loss')

            # Print some information.
            print('### Finish the definition of DQN loss, Shape: {}'.format(DQN_loss.shape))

            # Return the DQN loss.
            return DQN_loss


    def __gen_summary(self, name_space):
        r'''
            Generate the summaries.
        '''

        # Get the classification dimension.
        conf_base = self._config['Base']
        classification_dim = conf_base.get('classification_dimension')

        # Start generate summaries.
        SUMMARY_name = name_space + '/Summary'
        with tf.variable_scope(SUMMARY_name):
            # Summary merge list.
            merge_list = []

            # Add losses.
            tf.summary.scalar('NET_Loss', self._losses[name_space+'/NET_loss'])
            merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'NET_Loss'))
            tf.summary.scalar('SEG_Loss', self._losses[name_space+'/SEG_loss'])
            merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'SEG_Loss'))
            tf.summary.scalar('DQN_Loss', self._losses[name_space+'/DQN_loss'])
            merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'DQN_Loss'))

            # Custom define some metric to show.
            rewards = net_util.placeholder_wrapper(self._summary, tf.float32, None, name='Reward')
            DICE = net_util.placeholder_wrapper(self._summary, tf.float32, [classification_dim], name='DICE')
            BRATS = net_util.placeholder_wrapper(self._summary, tf.float32, [3], name='BRATS_metric')
            # Recursively add the summary.
            tf.summary.scalar('Rewards', rewards)
            merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'Rewards'))
            for c in range(classification_dim):
                tf.summary.scalar('DICE_'+str(c+1), DICE[c])
                merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'DICE_'+str(c+1)))
            for b in range(3):
                tf.summary.scalar('BRATS_'+str(b+1), BRATS[b])
                merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'BRATS_'+str(b+1)))

            # Merge all the summaries.
            summaries = tf.summary.merge(merge_list)
            net_util.package_tensor(self._summary, summaries)

            # Print some information.
            print('### The summary dict is {}'.format(summaries))

            # Plain return.
            return

