import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
import numpy as np
import cv2
import tfmodule.layer as cus_layers
import tfmodule.residual as cus_res
import tfmodule.util as net_util
import tfmodule.losses as cus_loss
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

        # Check the validity of configuration options.
        self._config_validate()

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



    #### --------------------- Configuration Validation Related ----------------------------
    def _config_validate(self):
        r'''
            Check some configuration options here. Coz some options are incompatible.
                We check them here for conveniently coding later.
        '''

        # Get detailed configuration options.
        conf_up = self._config['UpSample']
        up_part = conf_up.get('upsample_part', 'whole')
        up_structure = conf_up.get('upsample_structure', 'U-Net')
        conf_cus = self._config['Custom']
        logit_type = conf_cus.get('result_tensor', 'complete')
        fuse_part = conf_cus.get('result_part', 'complete')
        fusion_method = conf_cus.get('result_fusion', 'prob')
        clazz_imb = conf_cus.get('class_imbalance', 'bbox')

        # Check the validity of up-sample "part - structure - result" pair.
        USSP_valid_cond_0 = (up_part == 'whole') and (up_structure == 'U-Net' or up_structure == 'MLIF') \
                            and (logit_type == 'complete')
        USSP_valid_cond_1 = (up_part == 'bbox-region') and (up_structure == 'U-Net' or up_structure == 'MLIF') \
                            and (logit_type == 'bbox-bi')
        USSP_valid_cond_2 = (up_part == 'whole') and (up_structure in ['FPN-pW', 'FPN-fW', 'FPN-S', 'FPN-bbox']) \
                            and (logit_type == 'bbox-crop')
        if not (USSP_valid_cond_0 or USSP_valid_cond_1 or USSP_valid_cond_2):
            raise Exception('Invalid Up-sample Structure pair. - (up-part: {}) - '
                            '(up-structure: {}) - (logit-type: {})'.format(up_part, up_structure, logit_type))

        # Check the validity of result-fusion "logit type" and "fuse type" pair.
        LFP_invalid_cond = (logit_type in ['bbox-bi', 'bbox-crop']) and (fuse_part == 'complete')
        if LFP_invalid_cond:
            raise Exception('Invalid LFP pair. - (logit-type: {}) - (fuse_part: {})'.format(
                logit_type, fuse_part
            ))

        # Check the validity of "class imbalance solve" and "logit type" pair.
        CL_invalid_cond = (logit_type == 'bbox-bi' or logit_type == 'bbox-crop') and (clazz_imb == 'bbox')
        if CL_invalid_cond:
            raise Exception('Invalid CLS pair. - (logit-type: {}) - (clazz-imb: {})'.format(
                logit_type, clazz_imb
            ))

        # Print some information.
        print('| ---> The configuration validation check !')
        print('|   ===> Up-sample part: {}, structure: {}, logits-type: {}, '
              'fusion-part: {}, fusion-method: {}'.format(up_part, up_structure, logit_type, fuse_part, fusion_method))
        print('|   ===> Class Imbalance Solution: {}'.format(clazz_imb))
        print('\-' * 50 + '\\')

        # Finish check.
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
            SEG_output, REGION_result = self.__result_fusion(SEG_tensor, self._name_space)
        else:
            # No need the two holders.
            SEG_logits = CEloss_tensors = None
        # Select next region to deal with. (Generate the DQN output)
        SEG_info = self.__mediates[self._name_space + '/Region_tensor']
        DQN_output = self.__region_selection(FE_tensor, SEG_info, action_dim, name_space)

        # Package some outputs.
        net_util.package_tensor(self._outputs, DQN_output)
        if with_segmentation:
            net_util.package_tensor(self._outputs, SEG_output)
            net_util.package_tensor(self._outputs, REGION_result)

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
        conf_cus = self._config['Custom']
        conf_dqn = self._config['DQN']
        # Input shape.
        input_shape = conf_base.get('input_shape')
        # Determine the input type.
        input_type = conf_cus.get('input_type', 'whole')
        # Determine the introduction method of "Position Information".
        pos_method = conf_cus.get('position_info', 'map')
        # Determine the max length of "History Information".
        his_thres = conf_dqn.get('step_threshold', 10)

        # --------------------------------- "Input Justification" part. ------------------------------------
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

            # The "Bbox History" used to introduce "Past Local Information".
            bbox_his = net_util.placeholder_wrapper(self._inputs, tf.float32, [None, his_thres, 4],
                                                    name='Bbox_History')

            # The "Valid Length" used to indicating the whether elements of "Bbox History" should be
            #   considered in current state.
            his_len = net_util.placeholder_wrapper(self._inputs, tf.int32, [None,], name='History_Length')

            # Declare the function used to crop the specific bbox region of input tensor.
            def region_crop(x, bbox, size, name):
                bbox_ids = tf.range(tf.reduce_sum(tf.ones_like(bbox, dtype=tf.int32)[:, 0]))
                y = tf.image.crop_and_resize(x, bbox, bbox_ids, size, name=name + '_focus_crop')
                return y

            # Declare the function used to fuzzy the region outside the "Focus Bbox".
            def region_fuzzy(x, map, name):
                rs = [-1,]
                rs.extend(x.get_shape().as_list()[1:])
                def _gause_fuzzy(_inp1, _inp2):
                    _y = []
                    for _x1, _x2 in zip(_inp1, _inp2):
                        _y1 = cv2.GaussianBlur(_x1, (9, 9), 0)  # [h, w, c]
                        _yi = np.where(_x2, _x1, _y1)    # [h, w, c]
                        _y.append(_yi)
                    _y = np.asarray(_y)
                    return _y
                y, = tf.py_func(_gause_fuzzy, inp=[x, map], Tout=[tf.float32])
                y = tf.reshape(y, rs, name=name)    # [?, h, w, c]
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
            elif pos_method == 'W/O':
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
                # Generate "Focus Map".
                focus_map = net_util.gen_focus_maps_4bbox(focus_bbox, cor_size=[suit_h, suit_w], name='Focus_Map')
                # Generate and package the "Past Bbox Maps".
                bbox_his = tf.reshape(bbox_his, [-1, 4], name='flat_bbox_his')  # [b*his, 4]
                bbox_his = net_util.scale_bbox(bbox_his,
                                               src_height=input_shape[1],
                                               src_width=input_shape[2],
                                               dst_height=suit_h,
                                               dst_width=suit_w,
                                               name='Scale_Bbox_His')
                bbox_maps = net_util.gen_focus_maps_4bbox(bbox_his, cor_size=[suit_h, suit_w],
                                                          name='flat_bbox_maps')    # [b*his, h, w, 1]
                bbox_maps = tf.reshape(bbox_maps, [-1, his_thres, suit_h, suit_w, 1],
                                       name='Bbox_Maps')    # [b, his, h, w, 1]
            else:
                # Rename the "Focus Bbox" for conveniently usage.
                focus_bbox = tf.identity(focus_bbox, name='Scale_FoBbox')
                # Only generate the "Focus Map".
                focus_map = net_util.gen_focus_maps_4bbox(focus_bbox, cor_size=[input_shape[1], input_shape[2]],
                                                          name='Focus_Map')
                # Generate and package the "Past Bbox Maps".
                bbox_his = tf.reshape(bbox_his, [-1, 4], name='flat_bbox_his')  # [b*his, 4]
                bbox_maps = net_util.gen_focus_maps_4bbox(bbox_his, cor_size=[input_shape[1], input_shape[2]],
                                                          name='flat_bbox_maps')    # [b*his, h, w, 1]
                bbox_maps = tf.reshape(bbox_maps, [-1, his_thres, input_shape[1], input_shape[2], 1],
                                       name='Bbox_Maps')    # [b, his, 4, h, w, 1]

            # Package the "Scaled Focus Bbox", "Focus Map", "Bbox Maps" into the mediate holders.
            net_util.package_tensor(self.__mediates, focus_bbox)
            net_util.package_tensor(self.__mediates, focus_map)
            net_util.package_tensor(self.__mediates, bbox_maps)

            # Generate and package the "History Flag", which indicates the valid length of "History Information".
            his_elem = tf.expand_dims(tf.range(his_thres), axis=0, name='history_range')  # [1, his]
            rev_his_len = tf.expand_dims(tf.subtract(his_thres, his_len), axis=-1,
                                         name='reverse_his_len')     # [b, 1]
            his_flag = tf.greater_equal(his_elem, rev_his_len, name='History_Flag')     # [b, his]
            net_util.package_tensor(self.__mediates, his_flag)  # [b, his]

            # Different fusion method for input according to input type.
            if input_type == 'region':
                # Focus on the "Region Perspective".
                input_tensor = region_crop(raw_image, focus_bbox, [suit_h, suit_w], name='region_input')
            elif input_type == 'WR':
                # Concatenate the "Whole Perspective" with "Region Perspective".
                input_tensor = region_crop(raw_image, focus_bbox, [suit_h, suit_w], name='1E_foInput')
                input_tensor = tf.concat([raw_image, input_tensor], axis=-1, name='WR_input')
            elif input_type == 'whole':
                # Do not need do something, use raw image is okay.
                input_tensor = raw_image
            elif input_type == 'fuzzy':
                # Fuzzy the region outside the "Focus Region".
                input_tensor = region_fuzzy(raw_image, focus_map, name='Fuzzy_input')
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
        # Get up-sample part.
        conf_up = self._config['UpSample']
        up_part = conf_up.get('upsample_part', 'whole')
        # Get max length of "Bbox History".
        conf_dqn = self._config['DQN']
        his_thres = conf_dqn.get('step_threshold', 10)
        # Get the name stage prefix.
        dqn_names = conf_dqn.get('double_dqn', None)
        if dqn_names is not None:
            stage_prefix = dqn_names[0]
        else:
            stage_prefix = self._name_space

        # Declare the function used to fuse the "Past Local Information".
        def history_fuse(x, glo_info, bbox, valid_his, scales, s_id, his_flag):
            # generate flag.
            bh = tf.expand_dims(tf.abs(bbox[:, 2] - bbox[:, 0]), axis=-1)   # [b*his, 1]
            bw = tf.expand_dims(tf.abs(bbox[:, 3] - bbox[:, 1]), axis=-1)   # [b*his, 1]
            ph = tf.greater_equal(bh, scales)   # [b*his, s]
            pw = tf.greater_equal(bw, scales)   # [b*his, s]
            p = tf.logical_or(ph, pw)   # [b*his, s]
            pind = tf.expand_dims(tf.range(scales.get_shape().as_list()[-1]), axis=0)   # [1, s]
            pind = tf.multiply(tf.to_int32(p), pind)    # [b*his, s]
            pind = tf.reduce_min(pind, axis=-1)     # [b*his]
            flag = tf.equal(pind, s_id)     # [b*his]
            flag = tf.logical_and(flag, his_flag)   # [b*his]
            # generate the "local feature maps".
            lfm = tf.expand_dims(glo_info, axis=1)     # [b, 1, gh, gw, gc]
            lfm = tf.tile(lfm, multiples=[1, his_thres, 1, 1, 1])   # [b, his, gh, gw, gc]
            g_h, g_w, g_c = glo_info.get_shape().as_list()[1:]
            l_h, l_w, l_c = x.get_shape().as_list()[1:]
            lfm = tf.reshape(lfm, [-1, g_h, g_w, g_c])  # [b*his, gh, gw, gc]
            b_ids = tf.range(tf.reduce_sum(tf.ones_like(bbox, dtype=tf.int32)[:, 0]))    # [b*his]
            lfm = tf.image.crop_and_resize(lfm, bbox, b_ids, crop_size=(l_h, l_w))  # [b*his, lh, lw, gc]
            lfm = cus_layers.base_conv2d(lfm, l_c, 1, 1,
                                         reuse=tf.AUTO_REUSE,
                                         feature_normalization=fe_norm,
                                         activation=activation,
                                         keep_prob=conv_kprob,
                                         regularizer=regularizer,
                                         name_space='Local_Feats_'+str(s_id))   # [b*his, lh, lw, lc]
            # filter the invalid history.
            flag = tf.expand_dims(tf.expand_dims(tf.expand_dims(flag, axis=-1), axis=-1),
                                  axis=-1)  # [b*his, 1, 1, 1]
            flag = tf.logical_and(flag, tf.ones_like(lfm, dtype=tf.bool))  # [b*his, lh, lw, lc]
            lfm = tf.reshape(lfm, [-1, his_thres, l_h, l_w, l_c])   # [b, his, lh, lw, lc]
            flag = tf.reshape(flag, [-1, his_thres, l_h, l_w, l_c])     # [b, his, lh, lw, lc]
            lfm = tf.where(flag, lfm, tf.zeros_like(lfm))   # [b, his, lh, lw, lc]
            lfm = tf.reduce_sum(lfm, axis=1)  # [b, lh, lw, lc]
            # fuse the "local feature maps" into the raw input.
            y = tf.add(x, lfm, name='Local_Fusion_'+str(s_id))  # [b, lh, lw, lc]
            return y
        # ------------------------- end of sub-func ----------------------------

        # Start definition.
        FE_name = name_space + '/FeatExt'
        with tf.variable_scope(FE_name):
            # Generate the "Scales Ruler".
            scales_ruler = []
            for p in range(len(layer_units)):
                scales_ruler.append(1/np.exp2(p+1))
            scales_ruler.append(0.)
            scales_ruler = tf.constant([scales_ruler], name='Scales_Ruler')     # [1, scales]
            # Get the "History Length".
            his_len = self._inputs[stage_prefix + '/History_Length']    # [b]
            # Get the "History Flag", which indicates the valid length of "History Information".
            his_flag = self.__mediates[stage_prefix+'/History_Flag']    # [b, his]
            his_flag = tf.reshape(his_flag, [-1], name='flat_history_flag')     # [b*his]
            # Get "Global Information", that is, the raw image.
            glo_info = self._inputs[stage_prefix+'/image']  # [?, h, w, c]
            # Get "History Information".
            bbox_his = self._inputs[stage_prefix+'/Bbox_History']
            bbox_his = tf.reshape(bbox_his, [-1, 4], name='flat_bbox_his')  # [b*his, 4]

            # The feature maps dictionary, which is lately used in up-sample part.
            DS_feats = []

            # Base conv to reduce the feature map size.
            base_conv = cus_layers.base_conv2d(input_tensor, kernel_numbers[0], 7, 2,
                                               feature_normalization=fe_norm,
                                               activation='lrelu',
                                               keep_prob=conv_kprob,
                                               regularizer=regularizer,
                                               name_space='ResNet_bconv')  # [?, 112, 112, ?]
            # Fuse "History Information" of this scale level.
            base_conv = history_fuse(base_conv, glo_info, bbox=bbox_his, valid_his=his_len,
                                     scales=scales_ruler, s_id=0, his_flag=his_flag)

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
                # Fuse "History Information" of this scale level.
                block_tensor = history_fuse(block_tensor, glo_info, bbox=bbox_his, valid_his=his_len,
                                            scales=scales_ruler, s_id=idx+1, his_flag=his_flag)
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

            # Use "Whole Tensors" or "Bbox-region Tensors" according to the up-sample part.
            if up_part == 'whole':
                FE_tensor = tf.identity(FE_tensor, name='FE_tensors')
            elif up_part == 'bbox-region':
                dqn_names = self._config['DQN'].get('double_dqn', None)
                if dqn_names is not None:
                    stage_prefix = dqn_names[0]
                else:
                    stage_prefix = self._name_space
                focus_bbox = self.__mediates[stage_prefix + '/Scale_FoBbox']
                bbox_ids = tf.range(tf.reduce_sum(tf.ones_like(focus_bbox, dtype=tf.int32)[:, 0]))
                FE_tensor = tf.image.crop_and_resize(FE_tensor, focus_bbox, bbox_ids,
                                                     FE_tensor.get_shape().as_list()[1:3],
                                                     name='FE_tensors')
            else:
                raise ValueError('Unknown up-sample part !!!')

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

        # Get the "Scaled Focus Bbox" here.
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
                p = suit_h // FE_tensor.get_shape().as_list()[1]
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
        up_part = conf_up.get('upsample_part', 'whole')
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

            # Get the corresponding down-sample part.
            raw_ds = DS_feats[ds_idx]
            # Crop the "Bbox region" from raw "Down-sample Tensors" if specified.
            if up_part == 'whole':
                pass
            elif up_part == 'bbox-region':
                bbox_ids = tf.range(tf.reduce_sum(tf.ones_like(focus_bbox, dtype=tf.int32)[:, 0]))
                raw_ds = tf.image.crop_and_resize(raw_ds, focus_bbox, bbox_ids,
                                                  raw_ds.get_shape().as_list()[1:3],
                                                  name='up_part_crop0' + str(index + 1))
            else:
                raise ValueError('Unknown up-sample part !!!')

            # Add "Skip Connection" if specified.
            if skip_method == 'raw':
                # Use the raw down-sampled feature maps.
                skip_conn = raw_ds
            elif skip_method == 'conv':
                # Pass through a 1x1 conv to get the skip connection features.
                skip_conn = cus_layers.base_conv2d(raw_ds, raw_ds.shape[-1], 1, 1,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='skip_conv0' + str(index + 1))
            elif skip_method == 'res':
                # Pass through a residual layer to get the skip connection features.
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
                org_UST = cus_layers.base_deconv2d(half_UST, half_UST.get_shape().as_list()[-1], 3, 2,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='raw_WS_tensor')
                org_UST = cus_res.residual_block(org_UST, half_UST.get_shape().as_list()[-1], 1,
                                                 feature_normalization=fe_norm,
                                                 activation=activation,
                                                 keep_prob=conv_kprob,
                                                 regularizer=regularizer,
                                                 name_space='res_WS_tensor')
                org_UST = cus_layers.base_deconv2d(org_UST, classification_dim, 1, 1,
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
                # Generate the patches from feature maps of each level.
                FPN_patch_size = (suit_h // 16, suit_w // 16)   # [h/16, w/16]
                FPN_patch_chans = 256
                temp_T = []
                for i, t in enumerate(FPN_tensors):
                    bbox_ids = tf.range(tf.reduce_sum(tf.ones_like(focus_bbox, dtype=tf.int32)[:, 0]))
                    t_cls = tf.image.crop_and_resize(t, focus_bbox, bbox_ids, FPN_patch_size,
                                                     name='FPN_patch_crop' + str(i + 1))    # [?, h/16, w/16, c]
                    t_cls = cus_res.residual_block(t_cls, FPN_patch_chans, 1,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='FPN_patch_res1_' + str(i + 1))   # [?, h/16, w/16, c]
                    t_cls = cus_layers.base_deconv2d(t_cls, FPN_patch_chans, 3, 2,
                                                     feature_normalization=fe_norm,
                                                     activation=activation,
                                                     keep_prob=conv_kprob,
                                                     regularizer=regularizer,
                                                     name_space='FPN_patch_up' + str(i + 1))    # [?, h/8, w/8, c]
                    t_cls = cus_res.residual_block(t_cls, FPN_patch_chans, 1,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='FPN_patch_res2_' + str(i + 1))   # [?, h/8, w/8, c]
                    org_t = cus_layers.base_conv2d(t_cls, classification_dim, 1, 1,
                                                   feature_normalization=fe_norm,
                                                   activation=activation,
                                                   keep_prob=conv_kprob,
                                                   regularizer=regularizer,
                                                   name_space='FPN_patch_cls' + str(i + 1))     # [?, h/8, w/8, cls]
                    temp_T.append(org_t)
                FPN_tensors = temp_T

                # Generate the weights, and what's more weight tensors to generate FPN tensors.
                FPN_tensors = tf.stack(FPN_tensors, axis=-1, name='FPN_stack')  # [?, h/8, w/8, cls, scales]
                FPN_W = FPN_weights(FPN_tensors, cor_size=FPN_tensors.get_shape().as_list()[1:3],
                                    structure=up_structure)     # [?, h/8, w/8, cls, scales]
                SEG_tensor = tf.reduce_sum(tf.multiply(FPN_tensors, FPN_W), axis=-1,
                                           name='SEG_tensor')   # [?, h/8, w/8, cls]
                # Independently return the segmentation logits, so that we can flexibly deal with it.
                SEG_logits = SEG_tensor

            else:
                raise ValueError('Unknown up-sample structure !!!')

            # Check the shape validity of each output tensors of "Segmentation" branch when
            #   it's some specific logits types.
            logit_type = self._config['Custom'].get('result_tensor', 'complete')
            if logit_type in ['bbox-bi', 'complete']:
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
        fuse_part = conf_cus.get('result_part', 'complete')

        # Get the "Focus Bounding-box" holder.
        conf_dqn = self._config['DQN']
        dqn_names = conf_dqn.get('double_dqn', None)
        if dqn_names is not None:
            stage_prefix = dqn_names[0]
        else:
            stage_prefix = self._name_space
        focus_bbox = self.__mediates[stage_prefix+'/Scale_FoBbox']
        focus_map = self.__mediates[stage_prefix+'/Focus_Map']
        # Get the "History Flag", which indicates the valid length of history information.
        his_flag = self.__mediates[stage_prefix + '/History_Flag']  # [b, his]
        # Get the "Bbox Maps", which indicates the "Past Focus Regions".
        bbox_maps = self.__mediates[stage_prefix + '/Bbox_Maps']    # [b, his, h, w, 1]
        # Get the "History Threshold".
        his_thres = conf_dqn.get('step_threshold', 10)

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
                # Resize and pad zero to get the region tensor.
                region_shape = [up_h, up_w, SEG_tensor.get_shape().as_list()[-1]]  # [h, w, c]
                REGION_tensor = net_util.batch_resize_to_bbox_for_op(
                    [SEG_tensor], bbox=focus_bbox, cor_size=[up_h, up_w],
                    resize_method=['bilinear'], op_func=restore_2up,
                    output_shape=region_shape,
                    name='raw_region_tensor')   # [?, h, w, cls]
            elif logit_type == 'complete':
                # Directly use the segmentation tensors.
                REGION_tensor = SEG_tensor
            else:
                raise ValueError('Unknown segmentation logit type !!!')
            REGION_tensor = tf.identity(REGION_tensor, name='Region_tensor')    # [?, h, w, cls]
            # Package the "Region Tensor" for part of DQN input.
            net_util.package_tensor(self.__mediates, REGION_tensor)

            # Generate the "Flag Maps" used to filter the "Fusion Result".
            if fuse_part == 'region':
                bbox_maps = tf.logical_and(
                    tf.expand_dims(tf.expand_dims(tf.expand_dims(his_flag, axis=-1), axis=-1), axis=-1),
                    bbox_maps, name='filt_bbox_maps')  # [?, his, h, w, 1]
                flag_maps = tf.concat([bbox_maps, tf.expand_dims(focus_map, axis=1)], axis=1,
                                      name='flag_maps_sin')  # [?, his+1, h, w, 1]
                flag_maps = tf.tile(flag_maps, multiples=[1, 1, 1, 1, classification_dim],
                                    name='Flag_maps')  # [?, his+1, h, w, cls]
            elif fuse_part == 'complete':
                flag_maps = tf.concat([his_flag,
                                       tf.slice(tf.ones_like(his_flag, dtype=tf.bool), begin=(0, 0), size=(-1, 1))],
                                      axis=-1, name='flag_maps_2d')  # [?, his+1]
                his_ind = tf.expand_dims(tf.ones_like(REGION_tensor, tf.bool), axis=1)  # [?, 1, h, w ,cls]
                his_ind = tf.tile(his_ind, multiples=[1, his_thres+1, 1, 1, 1],
                                  name='his_shape_ind')     # [?, his+1, h, w, cls]
                flag_maps = tf.logical_and(
                    tf.expand_dims(tf.expand_dims(tf.expand_dims(flag_maps, axis=-1), axis=-1), axis=-1),
                    tf.ones_like(his_ind, dtype=tf.bool), name='Flag_maps')  # [?, his+1, h, w, cls]
            else:
                raise ValueError('Unknown result fusion part !!!')
            # Meanwhile calculate the "mean factor".
            fmean_factor = tf.reduce_sum(tf.to_float(flag_maps), axis=1,
                                         name='raw_fmean_factor')  # [?, h, w, cls]
            fmean_factor = tf.where(tf.not_equal(fmean_factor, 0.), fmean_factor, tf.ones_like(fmean_factor),
                                    name='fmean_factor')  # [?, h, w, cls]

            # Fusion according to different method. (Including declare the "Complete Result")
            if fusion_method == 'logit':
                # The placeholder of "Complete Result". --> Logit tensors.
                complete_result = net_util.placeholder_wrapper(self._inputs, tf.float32,
                                                               [None, his_thres, up_h, up_w, classification_dim],
                                                               name='Complete_Result')  # [?, his, h, w, cls]
                # The "Region Result".
                REGION_result = tf.identity(REGION_tensor, name='Region_Result')    # [?, h, w, cls]
                # Stack the segmentation logits of previous and current.
                STACK_result = tf.concat([complete_result, tf.expand_dims(REGION_result, axis=1)], axis=1,
                                         name='Stack_result')   # [?, his+1, h, w, cls]
                # Fuse the result in "Logit"-level, and generate the final segmentation result.
                FUSE_result = tf.where(flag_maps, STACK_result, tf.zeros_like(STACK_result),
                                       name='raw_fuse_result_5D')  # [?, his+1, h, w, cls]
                FUSE_result = tf.reduce_sum(FUSE_result, axis=1,
                                            name='raw_fuse_result_4D')  # [?, h, w, cls]
                FUSE_result = tf.divide(FUSE_result, fmean_factor, name='Fuse_result')  # [?, h, w, cls]
                # Translate to result.
                SEG_prob = tf.nn.softmax(FUSE_result, name='SEG_prob')  # [?, h, w, cls]
                SEG_output = tf.argmax(SEG_prob, axis=-1, name='SEG_suit_output')    # [?, h, w]
            elif fusion_method == 'prob':
                # The placeholder of "Complete Result". --> Probability.
                complete_result = net_util.placeholder_wrapper(self._inputs, tf.float32,
                                                               [None, his_thres, up_h, up_w, classification_dim],
                                                               name='Complete_Result')  # [?, his, h, w, cls]
                # The "Region Result".
                REGION_result = tf.nn.softmax(REGION_tensor, name='Region_Result')  # [?, h, w, cls]
                # Stack the segmentation probabilities of previous and current.
                STACK_result = tf.concat([complete_result, tf.expand_dims(REGION_result, axis=1)], axis=1,
                                         name='Stack_result')  # [?, his+1, h, w, cls]
                # Fuse the result in "Probability"-level, and generate the final segmentation result.
                FUSE_result = tf.where(flag_maps, STACK_result, tf.zeros_like(STACK_result),
                                       name='raw_fuse_result_5D')  # [?, his+1, h, w, cls]
                FUSE_result = tf.reduce_sum(FUSE_result, axis=1,
                                            name='raw_fuse_result_4D')  # [?, h, w, cls]
                FUSE_result = tf.divide(FUSE_result, fmean_factor, name='Fuse_result')  # [?, h, w, cls]
                # Translate to result.
                SEG_output = tf.argmax(FUSE_result, axis=-1, name='SEG_suit_output')  # [?, h, w]
            # elif fusion_method == 'mask':
            elif fusion_method in ['mask-lap', 'mask-vote']:
                # The placeholder of "Complete Result". --> Mask.
                complete_result = net_util.placeholder_wrapper(self._inputs, tf.int64,
                                                               [None, his_thres, up_h, up_w],
                                                               name='Complete_Result')  # [?, his, h, w]
                # The "Region Result".
                region_prob = tf.nn.softmax(REGION_tensor, name='region_prob')  # [?, h, w, cls]
                REGION_result = tf.argmax(region_prob, axis=-1, name='Region_Result')   # [?, h, w]
                # Stack the segmentation logits of previous and current.
                raw_STACK_result = tf.concat([complete_result, tf.expand_dims(REGION_result, axis=1)],
                                             axis=1, name='raw_stack_result')   # [?, his+1, h, w]
                STACK_result = tf.one_hot(raw_STACK_result, depth=classification_dim, dtype=tf.int64,
                                          name='clz_stack_result')  # [?, his+1, h, w, cls]
                STACK_result = tf.where(flag_maps, STACK_result, tf.zeros_like(STACK_result),
                                        name='Stack_result')    # [?, his+1, h, w, cls]
                # The "Pure Background Mask".
                BG_MASK = tf.zeros_like(REGION_result, name='BG_MASK')  # [?, h, w]
                # Deal with the one-hot mask according different fusion method.
                if fusion_method == 'mask-lap':
                    oh_posind = tf.range(his_thres+1, dtype=tf.int64)   # [his+1]
                    oh_posind = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(
                        oh_posind, axis=0), axis=-1), axis=-1), axis=-1)    # [1, his+1, 1, 1, 1]
                    oh_posind = tf.multiply(STACK_result, oh_posind)    # [?, his+1, h, w, cls]
                    oh_posind = tf.reduce_sum(oh_posind, axis=-1)   # [?, his+1, h, w]
                    oh_posind = tf.one_hot(tf.reduce_max(oh_posind, axis=1), depth=his_thres+1, dtype=tf.int64,
                                           axis=1, name='oh_posind')    # [?, his+1, h, w]
                    FUSE_laps = tf.multiply(raw_STACK_result, oh_posind, name='fuse_laps')  # [?, his+1, h, w]
                    FUSE_result = tf.reduce_sum(FUSE_laps, axis=1, name='raw_fuse_result')  # [?, h, w]
                    FUSE_result = tf.where(
                        tf.not_equal(tf.reduce_sum(FUSE_laps, axis=1), 0),
                        FUSE_result, BG_MASK,
                        name='Fuse_result')  # [?, h, w]
                else:
                    FUSE_votes = tf.reduce_sum(STACK_result, axis=1, name='fuse_votes')     # [?, h, w, cls]
                    FUSE_result = tf.argmax(FUSE_votes, axis=-1, name='raw_fuse_result')    # [?, h, w]
                    FUSE_result = tf.where(
                        tf.not_equal(tf.reduce_sum(FUSE_votes, axis=-1), 0),
                        FUSE_result, BG_MASK,
                        name='Fuse_result')  # [?, h, w]
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
            return SEG_output, REGION_result


    def __region_selection(self, FE_tensor, SEG_info, action_dim, name_space):
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
        clazz_dim = conf_base.get('classification_dimension')
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
        grad_DQN2SEG = conf_cus.get('grad_DQN2SEG', True)

        # Start definition.
        DQN_name = name_space + '/DQN'
        with tf.variable_scope(DQN_name):
            # Stop gradients from DQN-to-SEG if specified.
            if not grad_DQN2SEG:
                SEG_info = tf.stop_gradient(SEG_info, name='grad_nop')
            # Reshape the FUSE_result as a part of DQN input.
            FE_h, FE_w, FE_c = FE_tensor.get_shape().as_list()[1:]  # [fe_h, fe_w, fe_c]
            FE_prop = (suit_h // FE_h) * (suit_w // FE_w) * clazz_dim
            DQN_in = tf.reshape(SEG_info, [-1, FE_h, FE_w, FE_prop],
                                name='SEG_info_4DQN')   # [?, fe_h, fe_w, prop^2]
            DQN_in = cus_res.residual_block(DQN_in, FE_c, 1,
                                            feature_normalization=fe_norm,
                                            activation=activation,
                                            keep_prob=conv_kprob,
                                            regularizer=regularizer,
                                            name_space='SEG_info_res1')     # [?, fe_h, fe_w, fe_c]
            if seg_fuse == 'add':
                DQN_in = tf.add(FE_tensor, DQN_in, name='Fuse_IN')  # [?, fe_h, fe_w, fe_c]
            elif seg_fuse == 'concat':
                DQN_in = tf.concat([FE_tensor, DQN_in], axis=-1, name='Fuse_IN')    # [?, fe_h, fe_w, 2 * fe_c]
            elif seg_fuse == 'diff':
                DQN_in = tf.subtract(FE_tensor, DQN_in, name='Fuse_IN')     # [?, fe_h, fe_w, fe_c]
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

            # Build the DQN header according to the "Dueling Network" mode or not.
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


    #### --------------------- Loss-Summary Declaraion Related ----------------------------
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
        # Calculate the reward for current candidates actions.
        DQN_rewards = self.__reward_calculation(name_space=self._name_space)
        # Calculate the DQN loss.
        DQN_loss = self.__reinforcement_loss(DQN_output, DQN_rewards, name_space=self._name_space)

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
        # Package the reward tensors into dictionary.
        net_util.package_tensor(self._losses, DQN_rewards)

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
            Generate (Cross-Entropy or DICE) loss for the "Segmentation" branch.
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
        crop_method = conf_cus.get('size_matcher', 'crop')
        logit_type = conf_cus.get('result_tensor', 'complete')
        clazz_imb = conf_cus.get('class_imbalance', 'bbox')
        clzImb_thres = conf_cus.get('class_imbalance_threshold', 0.9)
        loss_type = conf_cus.get('segmentation_loss', 'DICE')

        # Get the stage prefix for conveniently inferring inputs holders.
        conf_dqn = self._config['DQN']
        dqn_names = conf_dqn.get('double_dqn', None)
        if dqn_names is not None:
            stage_prefix = dqn_names[0]
        else:
            stage_prefix = self._name_space
        # Get the "Focus Bounding-box" used to crop "Focus" region in label.
        focus_bbox = self.__mediates[stage_prefix + '/Scale_FoBbox']
        # Get the "Focus Map".
        focus_map = self.__mediates[stage_prefix + '/Focus_Map']

        # ---------------------- Definition of UN loss (cross-entropy or dice) ------------------------
        LOSS_name = name_space + '/SegLoss'
        with tf.variable_scope(LOSS_name):
            # The clazz weights used to deal with the "Class imbalance".
            clazz_weights = net_util.placeholder_wrapper(self._losses, tf.float32, [None, classification_dim],
                                                         name='clazz_weights')  # [?, cls]

            # Placeholder of ground truth segmentation.
            GT_label = net_util.placeholder_wrapper(self._losses, tf.int32, input_shape[:-1],
                                                    name='GT_label')  # [?, h, w]
            # Also need to crop label if size do not match. Meanwhile re-assign
            #   the coordinates of "Focus Bounding-box".
            if suit_h != input_shape[1] or suit_w != input_shape[2]:
                # Temporarily expand a dimension for conveniently processing.
                GT_label = tf.expand_dims(GT_label, axis=-1, name='expa01_label')  # [?, h, w, 1]
                # Match the size.
                if crop_method == 'crop':
                    # Crop to target size.
                    GT_label = tf.image.resize_image_with_crop_or_pad(GT_label, suit_h, suit_w)
                elif crop_method == 'bilinear':
                    # Nearest-neighbour resize to target size. (Coz it's segmentation result, integer type)
                    GT_label = tf.image.resize_nearest_neighbor(GT_label, [suit_h, suit_w], name='bi_labels')
                else:
                    raise ValueError('Unknown size match method !!!')
                # Reduce the dimension of labels.
                GT_label = GT_label[:, :, :, 0]     # [?, h, w]

            # Different process to the GT labels according to the "Segmentation Logits Type".
            if logit_type in ['bbox-bi', 'bbox-crop']:
                # Temporarily expand a dimension for conveniently processing.
                GT_label = tf.expand_dims(GT_label, axis=-1, name='expa02_label')  # [?, h, w, 1]
                # Crop and resize (to suit size) the corresponding region in label with respects to image (patch).
                if logit_type == 'bbox-bi':
                    resize_shape = [suit_h, suit_w]
                else:
                    resize_shape = SEG_logits.get_shape().as_list()[1:3]
                bbox_ids = tf.range(tf.reduce_sum(tf.ones_like(focus_bbox, dtype=tf.int32)[:, 0]))
                GT_label = tf.image.crop_and_resize(GT_label, focus_bbox, bbox_ids, resize_shape,
                                                    method='nearest',
                                                    name='focus_label')  # [?, h, w, 1]
                # Reduce the dimension of labels.
                GT_label = GT_label[:, :, :, 0]  # [?, h, w]
            elif logit_type == 'complete':
                # Do not need do anything.
                pass
            else:
                raise ValueError('Unknown segmentation logits type !!!')

            # Cast the data type of labels.
            GT_label = tf.cast(GT_label, 'int32', name='Corr_label')    # [?, h, w]
            # Rectify the clazz weights.
            cw_mask = tf.one_hot(GT_label, depth=classification_dim,
                                 name='one_hot_label')  # [?, h, w, cls]
            clazz_weights = tf.expand_dims(tf.expand_dims(clazz_weights, axis=1), axis=1,
                                           name='expa_CW')  # [?, 1, 1, cls]
            clazz_weights = tf.multiply(clazz_weights, cw_mask, name='rect_weights')    # [?, h, w, cls]
            clazz_weights = tf.reduce_sum(clazz_weights, axis=-1, name='sin_weights')   # [?, h, w]

            # Operations for "Class Weights" to deal with the "Class Imbalance" problem.
            if clazz_imb == 'bbox':
                # Filter the region outside the "Focus Map".
                clazz_weights = tf.multiply(clazz_weights, tf.to_float(focus_map[:, :, :, 0]),
                                            name='filter_focus')    # [?, h, w]
            elif clazz_imb == 'threshold':
                # Filter the gradients of whose probabilities greater than threshold.
                filt_prob = tf.reduce_sum(tf.multiply(SEG_logits, cw_mask), axis=-1,
                                          name='filter_prob')   # [?, h, w]
                thres_map = tf.less(filt_prob, clzImb_thres, name='clzImb_thres_map')   # [?, h, w]
                clazz_weights = tf.multiply(clazz_weights, tf.to_float(thres_map),
                                            name='filter_thres')    # [?, h, w]
            elif clazz_imb == 'W/O':
                # Do nothing.
                pass
            else:
                raise ValueError('Unknown class imbalance solution !!!')

            # Determine the loss function we used to calculate loss for segmentation.
            if loss_type == 'CE':
                loss_func = tf.losses.sparse_softmax_cross_entropy
            elif loss_type == 'DICE':
                loss_func = cus_loss.dice_loss
            else:
                raise ValueError('Unknown segmentation loss type !!!')

            # The single segmentation loss.
            fix_loss = loss_func(
                labels=GT_label, logits=SEG_logits, weights=clazz_weights, scope='FIX_loss')
            # Recursively calculate the loss.
            additional_loss = 0.
            for idx, logits in enumerate(CEloss_tensors):
                additional_loss += loss_func(
                    labels=GT_label, logits=logits, weights=clazz_weights, scope='addition_loss0'+str(idx+1))

            # Add the two parts as the final classification loss.
            SEG_loss = tf.add(fix_loss, score_factor * additional_loss, name='SEG_loss')

            # Print some information.
            print('### Finish the definition of SEG loss, Shape: {}'.format(SEG_loss.shape))

            # Return the segmentation loss.
            return SEG_loss


    def __reward_calculation(self, name_space):
        r'''
            Calculate the rewards for all actions for conveniently use. It will be
                used in both "Inference" and "Training" operation.
        '''

        # Get configuration.
        conf_base = self._config['Base']
        conf_dqn = self._config['DQN']

        # Get detailed parameters.
        input_shape = conf_base.get('input_shape')[1:3]
        clazz_dim = conf_base.get('classification_dimension')
        reward_form = conf_dqn.get('reward_form', 'SDR-DICE')
        err_punish = conf_dqn.get('err_punish', -3.0)
        terminal_dice = conf_dqn.get('terminal_dice_threshold', 0.85)
        terminal_recall = conf_dqn.get('terminal_recall_threshold', 0.85)
        terminal_value = conf_dqn.get('terminal_reward', 3.0)

        # Get the stage prefix for conveniently inferring inputs holders.
        dqn_names = conf_dqn.get('double_dqn', None)
        if dqn_names is not None:
            stage_prefix = dqn_names[0]
        else:
            stage_prefix = self._name_space

        # Get the holders.
        focus_bbox = self._inputs[stage_prefix + '/Focus_Bbox']     # focus bbox
        clazz_weights = self._losses[self._name_space + '/clazz_weights']   # [?, cls]
        GT_label = self._losses[self._name_space + '/GT_label']     # [?, h, w]
        SEG_result = self._outputs[self._name_space + '/SEG_output']    # [?, h, w]

        # ---------------------- Definition of UN cross-entropy loss ------------------------
        LOSS_name = name_space + '/RewardCal'
        with tf.variable_scope(LOSS_name):
            # Translate to "one hot" form.
            GT_label = tf.one_hot(GT_label, depth=clazz_dim, name='one_hot_label')  # [?, h, w, cls]
            SEG_result = tf.one_hot(SEG_result, depth=clazz_dim, name='one_hot_pred')  # [?, h, w, cls]
            # Filter class weights.
            clazz_weights = tf.expand_dims(tf.expand_dims(clazz_weights, axis=1), axis=1,
                                           name='expa_CW')  # [?, 1, 1, cls]
            clazz_weights = tf.multiply(clazz_weights, GT_label, name='rect_weights')  # [?, h, w, cls]
            clazz_weights = tf.reduce_sum(clazz_weights, axis=-1, name='sin_weights')  # [?, h, w]

            # Calculate value for current "Focus Bbox".
            focus_map = net_util.gen_focus_maps_4bbox(focus_bbox, input_shape,
                                                      name='focus_map_4rew')    # [?, h, w, 1]
            cur_clzW = tf.multiply(clazz_weights, tf.to_float(focus_map[:, :, :, 0]),
                                   name='clazz_weights_4cur')   # [?, h, w]
            current_value = cus_metric.DICE(labels=GT_label, predictions=SEG_result, weights=cur_clzW,
                                            scope='Cur_Value')  # [?]

            # ---------------------- Start compute each candidates ------------------------
            # The flag vector indicates whether the input candidate bounding-box is "Bbox-Error".
            BBOX_err = net_util.placeholder_wrapper(self._losses, tf.bool, [None, self._action_dim - 1],
                                                    name='BBOX_err')    # [?, acts-1]
            # The candidates bounding-box (anchors) for next time-step.
            candidates_bbox = net_util.placeholder_wrapper(self._losses, tf.float32, [None, self._action_dim - 1, 4],
                                                           name='Candidates_Bbox')  # [?, acts-1, 4]

            # Expand to add one dimension.
            indt_5d = tf.ones([1, self._action_dim - 1, 1, 1, 1])
            cand_label = tf.expand_dims(GT_label, axis=1) * indt_5d     # [?, acts-1, h, w, cls]
            cand_result = tf.expand_dims(SEG_result, axis=1) * indt_5d  # [?, acts-1, h, w, cls]
            cand_weights = tf.expand_dims(clazz_weights, axis=1) * indt_5d[:, :, :, :, 0]   # [?, acts-1, h, w]
            # Translate to 4-D tensors.
            oh, ow, oc = GT_label.get_shape().as_list()[1:]     # [h, w, cls]
            cand_label = tf.reshape(cand_label, [-1, oh, ow, oc])   # [?*a, h, w, cls]
            cand_result = tf.reshape(cand_result, [-1, oh, ow, oc])     # [?*a, h, w, cls]
            cand_weights = tf.reshape(cand_weights, [-1, oh, ow])       # [?*a, h, w]
            candidates_bbox = tf.reshape(candidates_bbox, [-1, 4])  # [?*a, 4]

            # Calculate value for next candidates.
            cand_maps = net_util.gen_focus_maps_4bbox(candidates_bbox, input_shape,
                                                      name='cand_maps_4rew')    # [?*a, h, w, 1]
            cand_clzWs = tf.multiply(cand_weights, tf.to_float(cand_maps[:, :, :, 0]),
                                     name='clazz_weights_4cands')   # [?*a, h, w]
            candidates_value = cus_metric.DICE(labels=cand_label, predictions=cand_result, weights=cand_clzWs,
                                               scope='Cand_Value')  # [?*a]
            candidates_value = tf.reshape(candidates_value, [-1, self._action_dim - 1],
                                          name='Candidates_Value')  # [?, acts-1]

            # Use the difference of remain value between "Candidates" and "Current Bbox".
            if reward_form == 'SDR-DICE':
                candidates_reward = tf.subtract(candidates_value, tf.expand_dims(current_value, axis=-1),
                                                name='Candidates_diff')     # [?, acts-1]
                candidates_reward = tf.sign(candidates_reward, name='Candidates_Raw')    # [?, acts-1]
            elif reward_form == 'DR-DICE':
                candidates_reward = tf.subtract(candidates_value, tf.expand_dims(current_value, axis=-1),
                                                name='Candidates_Raw')  # [?, acts-1]
            else:
                raise ValueError('Unknown reward form !!!')

            # Filter the error punishment.
            candidates_reward = tf.where(BBOX_err, err_punish * tf.ones_like(candidates_reward),
                                         candidates_reward,
                                         name='Candidates_Reward')  # [?, acts-1]

            # ---------------------- Compute reward for terminal situation ------------------------
            # Check whether "Dice" reaches the threshold.
            cur_dice = cus_metric.DICE(GT_label, SEG_result, clazz_weights, scope='Current_Dice')   # [?]
            dice_reach = tf.greater_equal(cur_dice, terminal_dice, name='Reach_Dice')   # [?]
            # Check whether "Recall" reaches the threshold.
            cur_recall = cus_metric.recall(GT_label, SEG_result, clazz_weights, scope='Current_Recall')  # [?]
            recall_reach = tf.greater_equal(cur_recall, terminal_recall, name='Reach_Recall')   # [?]
            # Determine the reward for "Terminal" action.
            reach_thres = tf.logical_and(dice_reach, recall_reach, name='Reach_Thres')  # [?]
            terminal_reward = tf.where(reach_thres,
                                       terminal_value * tf.ones_like(reach_thres, dtype=tf.float32),
                                       -terminal_value * tf.ones_like(reach_thres, dtype=tf.float32),
                                       name='Terminal_Reward')  # [?]

            # Generate final reward.
            DQN_rewards = tf.concat([candidates_reward, tf.expand_dims(terminal_reward, axis=-1)],
                                    axis=-1, name='final_reward')  # [?, acts]
            DQN_rewards = tf.stop_gradient(DQN_rewards, name='DQN_Rewards')

            # Print some information.
            print('### Finish the rewards calculation of DQN candidates bbox, Shape: {}'.format(DQN_rewards.shape))

            # Finish calculating reward for DQN.
            return DQN_rewards


    def __reinforcement_loss(self, DQN_output, DQN_rewards, name_space):
        r'''
            Generate (L2-regression) loss for "DQN (Region Selection)" branch.
        '''

        # Get configuration.
        conf_dqn = self._config['DQN']
        # Get detailed parameters.
        prioritized_replay = conf_dqn.get('prioritized_replay', True)

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
                                        name='prediction_Q_values')     # [?]
            # Only use the selected rewards, too.
            pred_rewards = tf.reduce_sum(tf.multiply(DQN_rewards, pred_action), axis=-1,
                                         name='prediction_rewards')     # [?]

            # The difference between prediction and target Q values.
            q_diff = tf.subtract(tf.add(target_q_vals, pred_rewards), pred_q_vals, name='Q_diff')   # [?]

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
            tf.summary.scalar('Training_Reward', self._losses[name_space+'/DQN_Rewards'])
            merge_list.append(tf.get_collection(tf.GraphKeys.SUMMARIES, 'Training_Reward'))

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
            summaries = tf.summary.merge(merge_list, name='Summaries')
            net_util.package_tensor(self._summary, summaries)

            # Print some information.
            print('### The summary dict is {}'.format(summaries))

            # Plain return.
            return

