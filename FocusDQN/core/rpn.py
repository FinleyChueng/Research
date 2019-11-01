import tensorflow as tf
import numpy as np



# The Reference Bounding-Boxes Generator
class BoxesGenerator:
    r'''
        Generate reference bounding-boxes for RPN according to the given shape and size.
    '''

    def gen(self, feature_stride, feature_maps_shape, num, dim):
        r'''
            Generation.

        :param feature_stride
        :param feature_maps_shape:
        :param num:
        :param dim:
        :return: A Tensor of reference bounding-boxes respected to the feature maps.
        '''

        # Check the type of feature_stride.
        if not isinstance(feature_stride, int):
            raise TypeError('The feature_stride_shape must be a integer !!!')

        # Check the type of feature_maps_shape.
        if not isinstance(feature_maps_shape, tuple):
            raise TypeError('The feature_maps_shape must be a tuple !!!')

        if len(feature_maps_shape) != 2:
            raise Exception('The feature_maps_shape should be 2-D !!!') # [width, height]

        # Get the scale-ratio list for basic anchors.
        scale_ratio_list = self.__scales_ratios_4anchors(feature_stride, feature_maps_shape, num, dim)

        # Generate reference boxes.
        bboxes = self.__bboxes(feature_stride, feature_maps_shape, scale_ratio_list, num, dim)

        return bboxes


    def __scales_ratios_4anchors(self, feature_stride, shape, num, dim):
        r'''
            Generate the ratios according to the feature map shape.

        :param shape:
        :param num:
        :param dim:
        :return:
        '''

        # Get the minimal edge of the feature maps and computes its order.
        min_edge = min(shape)
        order = np.log2(min_edge)
        # The largest scale.
        floor_order = np.floor(order) - 1   # Decrease one quantity.
        largest_scale = np.exp2(floor_order)
        # Get ratio numbers.
        ratio_num, ratios_list = self.__get_wh_ratios_and_num(num, dim)
        # Generate the scale-ratio list.
        scale_ratio_list = []
        cur_scale = largest_scale
        scale_num = int(num / ratio_num)
        for _ in range(scale_num):
            # The smallest scale of Faster R-CNN is 8. But the feature map size is [15, 15]
            if cur_scale * feature_stride <= 1:
                raise Exception('The shape of feature maps is too small !!! shape: {}'.format(shape))
            # Iteratively compute the width and height of different ratio.
            for ratio in ratios_list:
                width = cur_scale * ratio[0]
                height = cur_scale * ratio[1]
                # Add into scale-ratio list.
                scale_ratio_list.append((width, height))
            # Decrease the scale.
            cur_scale /= 2

        print('The scale-ratio list is {}'.format(scale_ratio_list))

        return scale_ratio_list


    def __get_wh_ratios_and_num(self, num, dim):
        r'''
            Compute the number of ratios of circumcircle and incircle.

        :param num:
        :param dim:
        :return:
        '''

        if num == 9 and dim == 4:
            # The ratio list, the value is the ratio of width and height.
            #   That is, [(1:2), (1:1), (2:1)] this three ratios.
            ratio_list = [(np.sqrt(2.) / 2., np.sqrt(2.)), (1., 1.), (np.sqrt(2.), np.sqrt(2.) / 2.)]
            return 3, ratio_list
        else:
            raise Exception('Invalid num and dim situation !!!')


    def __bboxes(self, stride, fm_shape, scale_ratio_list, num, dim):
        r'''
            Generate reference bounding-boxes according to different situation.

        :param stride
        :param fm_shape:
        :param scale_ratio_list:
        :param num:
        :param dim:
        :return:
        '''

        if num == 9 and dim == 4:
            if len(scale_ratio_list) != 9:
                raise Exception('The length of scale-ratio list is not 9')
            # Generate reference bounding-boxes.
            bboxes = self.__9num_4dim(stride, fm_shape, scale_ratio_list)
        else:
            raise Exception('Invalid given number and dimension of anchors')

        return bboxes


    def __basic_9x4anchors(self, stride, scale_ratio_list):
        r'''
            Generate the basic anchors for convenient computation.

        :param stride:
        :param scale_ratio_list:
        :return:
        '''

        # The basic anchors holder.
        basic_anchors = np.zeros((9, 4), dtype=np.float32)
        # Iteratively assign the anchor value.
        for n in range(9):
            # The radius of circumcircle.
            basic_anchors[n][2] = scale_ratio_list[n][0] * stride
            # The radius of incircle.
            basic_anchors[n][3] = scale_ratio_list[n][1] * stride
        # Convert the type of basic anchor to tensor.
        tensor_basic_anchors = tf.convert_to_tensor(basic_anchors, name='Anchor_Generator/convert_basic_anchor')
        # Reshape the basic anchors.
        t_basic_anchors = tf.reshape(tensor_basic_anchors, (1, 1, 9, 4), name='Anchor_Generator/reshape_basic_anchor')

        return t_basic_anchors


    def __9num_4dim(self, stride, shape, scale_ratio_list):
        r'''
            Generate width * height * 9 4-D reference bounding-boxes.

        :param stride
        :param shape:
        :param scale_ratio_list:
        :return:
        '''

        # Get the width and height of feature maps.
        width, height = shape
        # Get the basic anchors.
        t_basic_anchors = self.__basic_9x4anchors(stride, scale_ratio_list)   # [1, 1, 9, 4]

        # Generate the raw reference bounding-boxes respect to feature maps.
        shift_x = tf.range(width) * stride  # width, [width]
        shift_y = tf.range(height) * stride  # height, [height]
        shift_y, shift_x = tf.meshgrid(shift_y, shift_x)    # x: [width, height], y: [width, height]
        zeros = tf.zeros((width, height), dtype=tf.int32)   # zeros: [width, height]
        shifts = tf.stack([shift_x, shift_y, zeros, zeros], axis=2)   # shifts: [width, height, 4]
        # Expand one dim for raw reference bounding-boxes for convenient fusion.
        raw = tf.expand_dims(shifts, axis=2)    # [width, height, 1, 4]

        # Convert the dtype raw to the same as basic anchors.
        raw = tf.cast(raw, t_basic_anchors.dtype)
        # Add the raw boxes and the basic anchors directly (Using the broadcast)
        #   to get the real reference bounding-boxes for each position of feature map.
        bboxes = tf.add(raw, t_basic_anchors)  # [width, height, 9, 4]

        # bboxes = tf.reshape(bboxes, (width, height, 36))  # [width, height, 36]

        return bboxes



# The RPN Label Translator
class RPNTranslator:

    def __init__(self, feature_stride, bboxes, preds, ground_truth):
        # Check the type of feature_stride.
        if not isinstance(feature_stride, int):
            raise TypeError('The feature_stride_shape must be a integer !!!')

        # Check the type of anchors.
        if not isinstance(bboxes, tf.Tensor):
            raise TypeError('The bboxes must be a Tensorflow.Tensor !!!')
        # Check the dimension of anchors.
        if len(bboxes.get_shape()) != 4:
            raise Exception('The bboxes must be a 4-dim tensor !!!')  # [width, height, anchor_num, anchor_dim]

        # Check the type of preds.
        if not isinstance(preds, tf.Tensor):
            raise TypeError('The preds must be a Tensorflow.Tensor !!!')
        # Check the dimension of preds.
        if len(preds.get_shape()) != 5:
            raise Exception('The preds must be a 5-dim tensor !!!')  # [None, width, height, anchor_num, anchor_dim]

        # Check the type of ground_truth.
        if not isinstance(ground_truth, tf.Tensor):
            raise TypeError('The ground_truth must be a Tensorflow.Tensor !!!')
        # Check the dimension of the ground_truth.
        if len(ground_truth.get_shape()) != 2:
            raise Exception('The ground_truth must be a 2-dim tensor !!!')  # [None, anchor_dim]

        # Get the shape to check shape consistency.
        bboxes_shape = bboxes.get_shape().as_list() # [width, height, anchor_num, anchor_dim]
        pred_shape = np.array(preds.shape[1:].as_list())  # [width, height, anchor_num, anchor_dim]
        bboxes_a_shape = np.array([bboxes_shape[-1]])  # [anchor_dim]
        gt_shape = np.array(ground_truth.shape[1:].as_list())  # [anchor_dim]
        # Check the consistency of three parameters.
        shape_consistency = (bboxes_shape == pred_shape).all() and (bboxes_a_shape == gt_shape).all()
        if not shape_consistency:
            raise Exception('The shape consistency can not satisfied !!!'
                            'The bboxes shape is {}'
                            'The pred shape is {}'
                            'The ground truth shape is {}'.format(bboxes_shape, pred_shape, gt_shape))

        # Record the feature stride for later usage.
        self._feature_stride = feature_stride

        # Then expand the input tensors to same dimension.
        multi_bboxes, expa_deltas, expa_GTs = self.__expand_2same_dimension(bboxes, preds, ground_truth)

        # Initialize the holders to keep the tensors or operations.
        self._multi_bboxes = multi_bboxes
        self._expa_deltas = expa_deltas
        self._expa_GTs = expa_GTs
        # Remember the anchor dimension.
        self._bbox_dim = gt_shape[0]

        # The IOU threshold for positive and negative label, respectively.
        #   Which we will use to filter the anchor map to get the positive
        #   and negative labels.
        self._POS_THRES = 0.2
        # self._POS_THRES = 0.7
        # self._POS_THRES = 0.3
        # self._NEG_THRES = 0.2
        # self._NEG_MIN_THRES = 0.1
        self._NEG_MIN_THRES = 0.0
        self._NEG_THRES = 0.2
        # self._NEG_THRES = 0.3


    def gen_label(self, cls_dim, need_selective):
        # Generate the ground truth deltas, and weights map if needed.
        #   Judge by different anchor dimension.
        if self._bbox_dim == 4 and cls_dim == 2:
            label_deltas = self.__label_4d_bboxes(self._multi_bboxes, self._expa_GTs)
            anchors_map = self.__selective_4d(self._multi_bboxes, self._expa_GTs, need_selective)
            label_cls = self.__label_2d_cls(anchors_map)
            bbox_weights_map, cls_weights_map = self.__weights_map_4d_bbox_2d_cls(anchors_map)
        else:
            raise Exception('No function for the anchor dim: {}, class dim: {}'.format(
                self._bbox_dim, cls_dim))

        # Return the all things.
        return label_deltas, label_cls, bbox_weights_map, cls_weights_map, anchors_map


    def convert2_realbox(self):
        # Generate the predicted bboxes, ground truth deltas, and weights map if needed.
        #   Judge by different anchor dimension.
        if self._bbox_dim == 4:
            predicted_bboxes = self.__pred_4d_bboxes(self._multi_bboxes, self._expa_deltas)
        else:
            raise Exception('No function for the anchor dim: {}'.format(self._bbox_dim))

        return predicted_bboxes


    def __expand_2same_dimension(self, bboxes, deltas, GTs):
        r'''
            Expand the reference bboxes, pred deltas and ground truth to the same dimension.
                For convenient successive processing.

        Parameters:
            bboxes:  Anchors related reference bounding-box. [width, height, anchor_num, anchor_dim]
            deltas:  RPN predicted deltas.   [None, width, height, anchor_num*anchor_dim]
            GTs: Ground truth.   [None, anchor_dim]

        Return:
            bboxes: [1, width, height, anchor_num, anchor_dim]
            deltas: [None, width, height, anchor_num, anchor_dim]
            GTs: [None, 1, 1, 1, anchor_dim]
        '''

        # Expand dimension of anchors for convenient calculation. The additional dimension
        #   is batch size.
        multi_bboxes = tf.expand_dims(bboxes, axis=0)   # [1, width, height, anchor_num, anchor_dim]

        # We do not need to expand the pred_deltas. But we can not directly use the raw tensor. Copy one.
        expa_deltas = tf.multiply(deltas, 1.)   # [None, width, height, anchor_num, anchor_dim]

        # Expand ground truth dimension.
        expa_GTs = tf.expand_dims(GTs, axis=1)  # [None, 1, anchor_dim]
        expa_GTs = tf.expand_dims(expa_GTs, axis=1)  # [None, 1, 1, anchor_dim]
        expa_GTs = tf.expand_dims(expa_GTs, axis=1)  # [None, 1, 1, 1, anchor_dim]

        # Cast the data type.
        multi_bboxes = tf.cast(multi_bboxes, deltas.dtype)  # [1, width, height, anchor_num, anchor_dim]
        expa_GTs = tf.cast(expa_GTs, deltas.dtype)  # [None, 1, 1, 1, anchor_dim]

        # bboxes: [1, width, height, anchor_num, anchor_dim]
        # deltas: [None, width, height, anchor_num, anchor_dim]
        # GTs: [None, 1, 1, 1, anchor_dim]
        return multi_bboxes, expa_deltas, expa_GTs


    def __pred_4d_bboxes(self, multi_bboxes, expa_deltas):

        # Get the x, y, w, h of reference bounding-boxes.
        ref_x = multi_bboxes[:, :, :, :, 0] # [None, width, height, anchor_num]
        ref_y = multi_bboxes[:, :, :, :, 1] # [None, width, height, anchor_num]
        ref_w = multi_bboxes[:, :, :, :, 2] # [None, width, height, anchor_num]
        ref_h = multi_bboxes[:, :, :, :, 3] # [None, width, height, anchor_num]

        # Get the delta of four dimension.
        dx = expa_deltas[:, :, :, :, 0] # 这里的dx指的是相对移动距离，它的值是外接圆半径的倍数
        dy = expa_deltas[:, :, :, :, 1] # 同dx
        dw = expa_deltas[:, :, :, :, 2] # dw指的是放大的倍数，而该倍数为 log(times)，是为了让其值与dx、dy尽量统一
        dh = expa_deltas[:, :, :, :, 3] # 同dw

        # Limit the value of @Variable{dw} and @Variable{dh} in range (-inf, 1].
        #   This is critical due to the reason that it can help avoid the "inf"
        #   situation during the "Inference" computation phrase.
        # By the way, this case should not happened when "Training" is finished.
        limit_dw = tf.minimum(1., dw)   # (-inf, 1]
        limit_dh = tf.minimum(1., dh)   # (-inf, 1], 2.7 times...

        # Compute the predicted bounding-boxes.
        pred_x = tf.add(tf.multiply(dx, ref_w), ref_x)      # x0 + w * dx
        pred_y = tf.add(tf.multiply(dy, ref_h), ref_y)      # y0 + h * dy
        pred_w = tf.multiply(tf.exp(limit_dw), ref_w)       # w * exp(dw)
        pred_h = tf.multiply(tf.exp(limit_dh), ref_h)       # h * exp(dh)

        # Use that to compute the four edges of reference bounding-boxes.
        #   Translate that to the "Bounding Edges" form.
        pred_el = tf.subtract(pred_x, tf.divide(pred_w, 2.))    # left-edge:    pred_x - pred_w / 2.
        pred_er = tf.add(pred_x, tf.divide(pred_w, 2.))         # right-edge:   pred_x + pred_w / 2.
        pred_eu = tf.subtract(pred_y, tf.divide(pred_h, 2.))    # up-edge:      pred_y - pred_h / 2.
        pred_eb = tf.add(pred_y, tf.divide(pred_h, 2.))         # bottom-edge:  pred_y + pred_h / 2.

        # We have to stack the four dimension. Coz we unpack it just now.
        return tf.stack([pred_el, pred_er, pred_eu, pred_eb], axis=4)   # [None, width, height, anchor_num, anchor_dim]


    def __label_4d_bboxes(self, multi_bboxes, expa_GTs):

        # Get reference x, y, w, h, respectively.
        ref_x = multi_bboxes[:, :, :, :, 0] # [1, width, height, anchor_num]
        ref_y = multi_bboxes[:, :, :, :, 1] # [1, width, height, anchor_num]
        ref_w = multi_bboxes[:, :, :, :, 2] # [1, width, height, anchor_num]
        ref_h = multi_bboxes[:, :, :, :, 3] # [1, width, height, anchor_num]

        # Get the GT x, y, w, h, respectively.
        GT_x = expa_GTs[:, :, :, :, 0]  # [None, 1, 1, 1]
        GT_y = expa_GTs[:, :, :, :, 1]  # [None, 1, 1, 1]
        GT_w = expa_GTs[:, :, :, :, 2]  # [None, 1, 1, 1]
        GT_h = expa_GTs[:, :, :, :, 3]  # [None, 1, 1, 1]

        # Translate each dimension of anchor to get the GT_deltas, respectively. Using the broadcast.
        #   The shape is [None, width, height, anchor_num]
        GT_dx = tf.divide(tf.subtract(GT_x, ref_x), ref_w)    # (gt_x - x0) / w
        GT_dy = tf.divide(tf.subtract(GT_y, ref_y), ref_h)    # (gt_y - y0) / h
        GT_dccr = tf.log(tf.divide(GT_w, ref_w))    # log(gt_w / ref_w)
        GT_dicr = tf.log(tf.divide(GT_h, ref_h))    # log(gt_h / ref_h)

        # We have to stack the four dimension. Coz we unpack it just now.
        return tf.stack([GT_dx, GT_dy, GT_dccr, GT_dicr], axis=4)  # [None, width, height, anchor_num, anchor_dim]


    def __selective_4d(self, multi_bboxes, expa_GTs, need_selective):

        # Generate a anchors map with all ones if do not need selective.
        if not need_selective:
            # A anchors map with all ones.
            all_ones = tf.add(tf.multiply(tf.subtract(expa_GTs, multi_bboxes), 0.), 1.)  # anchors = 1
            # Origin shape is [None, width, height, anchor_num, anchor_dim], reduce one dimension.
            all_ones = tf.reduce_mean(all_ones, axis=4)     # [None, width, height, anchor_num]
            return all_ones

        # Generate weight maps for bboxes of current feature maps.
        anchors_map = tf.subtract(tf.multiply(tf.subtract(expa_GTs, multi_bboxes), 0.), 1.)   # anchors = -1
        # Origin shape is [None, width, height, anchor_num, anchor_dim], reduce one dimension.
        anchors_map = tf.reduce_mean(anchors_map, axis=4)  # [None, width, height, anchor_num]

        # Compute the IOU score for each position.
        IOU_score = self.__batch_IOU_score(multi_bboxes, expa_GTs)  # [None, width, height, anchor_num]

        # Generate the outlines map.
        outlines_map = self.__outlines_map(multi_bboxes)    # [1, width, height, anchor_num]

        # Begin to construct the conditions.
        #   The major premise: Can not outlines.
        #   The position will not contribute to gradient if not satisfying the major premise.
        major_premise = tf.logical_not(outlines_map)    # [1, width, height, anchor_num]

        # # We will take outlines into consider when training. (For the "Classification" branch)
        # major_premise = tf.convert_to_tensor([[[[True]]]], dtype=tf.bool)   # [1, 1, 1, 1]

        # The positive label IOU conditions 1st: IOU score > 0.7 .
        cond_IOU_greater_pthres = tf.greater_equal(IOU_score, self._POS_THRES)  # [None, width, height, anchor_num]
        # The positive label IOU conditions 2nd: largest IOU score .
        #   Compute the maximal IOU value of each image.
        image_wise = tf.reduce_max(IOU_score, axis=(1, 2, 3))  # [None]
        #   Expand the dimension to original dimension.
        image_wise_2d = tf.expand_dims(image_wise, axis=-1) # [None, 1]
        image_wise_3d = tf.expand_dims(image_wise_2d, axis=-1)  # [None, 1, 1]
        largest_IOU_score = tf.expand_dims(image_wise_3d, axis=-1)  # [None, 1, 1, 1]
        #   Define the condition.
        cond_largest_IOU = tf.equal(IOU_score, largest_IOU_score)   # [None, width, height, anchor_num]

        # Concat the condition of positive label. Shape: [None, width, height, anchor_num]
        cond_IOU_positive = tf.logical_or(cond_IOU_greater_pthres, cond_largest_IOU)
        cond_positive = tf.logical_and(major_premise, cond_IOU_positive)
        # Mark the positive label in the anchors map. Positive mark: 1
        anchors_map = tf.where(cond_positive, tf.ones_like(anchors_map),
                               anchors_map) # [None, width, height, anchor_num]

        # The negative label IOU conditions: IOU score < 0.3
        cond_IOU_less_nthres = tf.less_equal(IOU_score, self._NEG_THRES)    # [None, width, height, anchor_num]
        cond_IOU_greater_min_nthres = tf.greater_equal(IOU_score, self._NEG_MIN_THRES)    # [None, width, height, anchor_num]
        cond_IOU_within_nthres = tf.logical_and(cond_IOU_less_nthres, cond_IOU_greater_min_nthres)  # [None, width, height, anchor_num]

        # Concat the condition of negative label. Shape: [None, width, height, anchor_num]
        cond_not_largest = tf.logical_not(cond_largest_IOU)
        # Avoid the case that the largest IOU less than negative threshold.
        cond_IOU_negative = tf.logical_and(cond_IOU_within_nthres, cond_not_largest)
        cond_negative = tf.logical_and(major_premise, cond_IOU_negative)
        # Mark the negative label in the anchors map. Negative mark: 0
        anchors_map = tf.where(cond_negative, tf.zeros_like(anchors_map),
                               anchors_map) # [None, width, height, anchor_num]

        # The shape is [None, width, height, anchor_num]
        return anchors_map


    def __batch_IOU_score(self, multi_bboxes, expa_GTs):

        # Get reference x, y, w, h, respectively.
        ref_x = multi_bboxes[:, :, :, :, 0]  # [1, width, height, anchor_num]
        ref_y = multi_bboxes[:, :, :, :, 1]  # [1, width, height, anchor_num]
        ref_w = multi_bboxes[:, :, :, :, 2]  # [1, width, height, anchor_num]
        ref_h = multi_bboxes[:, :, :, :, 3]  # [1, width, height, anchor_num]
        # Use that to compute the four edges of reference bounding-boxes.
        ref_el = tf.subtract(ref_x, tf.divide(ref_w, 2.))   # left-edge:    ref_x - ref_w / 2.
        ref_er = tf.add(ref_x, tf.divide(ref_w, 2.))        # right-edge:   ref_x + ref_w / 2.
        ref_eu = tf.subtract(ref_y, tf.divide(ref_h, 2.))   # up-edge:      ref_y - ref_h / 2.
        ref_eb = tf.add(ref_y, tf.divide(ref_h, 2.))        # bottom-edge:  ref_y + ref_h / 2.

        # Get the GT x, y, w, h, respectively.
        GT_x = expa_GTs[:, :, :, :, 0]  # [None, 1, 1, 1]
        GT_y = expa_GTs[:, :, :, :, 1]  # [None, 1, 1, 1]
        GT_w = expa_GTs[:, :, :, :, 2]  # [None, 1, 1, 1]
        GT_h = expa_GTs[:, :, :, :, 3]  # [None, 1, 1, 1]
        # Use that to compute the four edges of ground truth bounding-boxes.
        GT_el = tf.subtract(GT_x, tf.divide(GT_w, 2.))  # left-edge:    GT_x - GT_w / 2.
        GT_er = tf.add(GT_x, tf.divide(GT_w, 2.))  # right-edge:   GT_x + GT_w / 2.
        GT_eu = tf.subtract(GT_y, tf.divide(GT_h, 2.))  # up-edge:      GT_y - GT_h / 2.
        GT_eb = tf.add(GT_y, tf.divide(GT_h, 2.))  # bottom-edge:  GT_y + GT_h / 2.

        # Begin to calculation. First to compute the width of IOU-area.
        edge_min_right = tf.minimum(ref_er, GT_er)  # min(ref_er, GT_er):   [None, width, height, anchor_num]
        edge_max_left = tf.maximum(ref_el, GT_el)   # max(ref_el, GT_el):   [None, width, height, anchor_num]
        iou_width = tf.subtract(edge_min_right, edge_max_left)  # min_right - max_left: [None, width, height, anchor_num]
        # Filter the value less equal than zeros (<= 0).
        cond_w_less0 = tf.less_equal(iou_width, 0.)
        iou_width = tf.where(cond_w_less0, tf.zeros_like(iou_width), iou_width)

        # Then compute the height of IOU-area.
        edge_min_bottom = tf.minimum(ref_eb, GT_eb)  # max(ref_eb, GT_eb):   [None, width, height, anchor_num]
        edge_max_up = tf.maximum(ref_eu, GT_eu)     # min(ref_eu, GT_eu):   [None, width, height, anchor_num]
        iou_height = tf.subtract(edge_min_bottom, edge_max_up)  # min_up - max_bottom:  [None, width, height, anchor_num]
        # Filter the value less equal than zeros (<= 0).
        cond_h_less0 = tf.less_equal(iou_height, 0.)
        iou_height = tf.where(cond_h_less0, tf.zeros_like(iou_height), iou_height)

        # Compute the area of IOU (Intersection-over-Union).
        IOU_area = tf.multiply(iou_width, iou_height)   # [None, width, height, anchor_num]

        # Compute the area of Reference and Ground Truth.
        ref_area = tf.multiply(ref_w, ref_h)    # ref_w * ref_h:    [1, width, height, anchor_num]
        GT_area = tf.multiply(GT_w, GT_h)  # GT_w * GT_h:  [None, 1, 1, 1]

        # Compute the area of Union-All. define as: ref_area + GT_area - iou_area
        union_area = tf.subtract(tf.add(ref_area, GT_area), IOU_area) # [None, width, height, anchor_num]

        # So we finally can compute the IOU score of each position.
        IOU_score = tf.divide(IOU_area, union_area) # iou_area / union_area:    [None, width, height, anchor_num]

        # Return the IOU score of each position.
        return IOU_score    # [None, width, height, anchor_num]


    def __outlines_map(self, multi_bboxes):

        # The shape of input bboxes is: [1, width, height, anchor_num, anchor_dim].
        #   So we can directly get the width and height of feature maps.
        fea_size = multi_bboxes.get_shape()[1:3].as_list()
        fea_width, fea_height = fea_size
        # Restore the width and height boundary from feature maps size to the
        #   original image size.
        boundary_width = fea_width * self._feature_stride
        boundary_height = fea_height * self._feature_stride

        # Get reference x, y, w, h, respectively.
        ref_x = multi_bboxes[:, :, :, :, 0]  # [1, width, height, anchor_num]
        ref_y = multi_bboxes[:, :, :, :, 1]  # [1, width, height, anchor_num]
        ref_w = multi_bboxes[:, :, :, :, 2]  # [1, width, height, anchor_num]
        ref_h = multi_bboxes[:, :, :, :, 3]  # [1, width, height, anchor_num]
        # Use that to compute the four edges of reference bounding-boxes.
        ref_el = tf.subtract(ref_x, tf.divide(ref_w, 2.))  # left-edge:    ref_x - ref_w / 2.
        ref_er = tf.add(ref_x, tf.divide(ref_w, 2.))  # right-edge:   ref_x + ref_w / 2.
        ref_eu = tf.subtract(ref_y, tf.divide(ref_h, 2.))  # up-edge:      ref_y - ref_h / 2.
        ref_eb = tf.add(ref_y, tf.divide(ref_h, 2.))  # bottom-edge:  ref_y + ref_h / 2.

        # The outline conditions. Shape: [1, width, height, anchor_num]
        cond_out_lb = tf.less(ref_el, 0.)   # left_edge < 0
        cond_out_rb = tf.greater(ref_er, boundary_width)    # right_edge > width
        cond_out_ub = tf.less(ref_eu, 0.)   # up_edge < 0
        cond_out_bb = tf.greater(ref_eb, boundary_height)   # bottom_edge > height
        # Concatenate the conditions. Shape: [1, width, height, anchor_num]
        cond_not_in_width = tf.logical_or(cond_out_lb, cond_out_rb)     # out of width range
        cond_not_in_height = tf.logical_or(cond_out_ub, cond_out_bb)    # out of height range
        cond_outlines = tf.logical_or(cond_not_in_width, cond_not_in_height)    # outlines

        # The outlines map. Shape: [1, width, height, anchor_num]
        outlines_map = cond_outlines

        # Return the outlines map.
        return outlines_map     # [1, width, height, anchor_num]


    def __label_2d_cls(self, anchors_map):

        # We can directly we the bounding-box anchors map to generate the
        #   class map for classification branch.
        #   The shape of anchors map: [None, width, height, anchor_num]

        # Cast the dtype of class map to int. For successive one-hot.
        gt_cls = tf.cast(anchors_map, dtype=tf.int32)

        # Convert to one-hot tensor.
        return tf.one_hot(gt_cls, 2)


    def __weights_map_4d_bbox_2d_cls(self, anchors_map):

        # ----------------- Bounding-Box Weight Map --------------------------

        # Stack the anchors map to get the weights map for Bounding-Box Regression.
        #   The origin shape is [None, width, height, anchor_num]
        bbox_weights_map = tf.stack([anchors_map, anchors_map, anchors_map, anchors_map], axis=4)  # [1, width, height, anchor_num, anchor_dim]

        # Coz we only use the positive anchor to training, so set the anchors value less than 1 to zeros.
        # The condition for anchors in bbox regression training: the position of "bbox_weights_map >= 1"
        cond_bbox_2train = tf.greater_equal(bbox_weights_map, 1.)
        # bbox_weights_map = tf.where(cond_bbox_2train, 1., 0.)  # [None, width, height, anchor_num, anchor_dim]
        # cond_bbox_2train = tf.greater_equal(bbox_weights_map, tf.ones_like(bbox_weights_map))
        bbox_weights_map = tf.where(cond_bbox_2train, tf.ones_like(bbox_weights_map),
                                    tf.zeros_like(bbox_weights_map))  # [None, width, height, anchor_num, anchor_dim]

        # ----------------- Class Weight Map --------------------------

        # Stack the anchors map to get the weights map for Bounding-Box Regression.
        #   The origin shape is [None, width, height, anchor_num]
        cls_weights_map = tf.stack([anchors_map, anchors_map],
                                   axis=4)  # [1, width, height, anchor_num, cls_dim]

        # Coz we will use the positive and negative anchor to training, so set the anchors value equal to -1 to zeros.
        # The condition for anchors in classification training: the position of "cls_weights_map >= 0"
        cond_cls_2train = tf.greater_equal(cls_weights_map, -1.)
        # cond_cls_2train = tf.greater_equal(cls_weights_map, 0.)
        # cls_weights_map = tf.where(cond_cls_2train, 1., 0.)  # [None, width, height, anchor_num, anchor_dim]
        # cond_cls_2train = tf.greater_equal(cls_weights_map, tf.zeros_like(cls_weights_map))
        cls_weights_map = tf.where(cond_cls_2train, tf.ones_like(cls_weights_map),
                                   tf.zeros_like(cls_weights_map))  # [None, width, height, anchor_num, anchor_dim]

        return bbox_weights_map, cls_weights_map



# The prediction bounding-box converter
class PredConverter:

    def __init__(self, max_output_size=10, NMS_THRESH=0.7):

        # The max NMS output number.
        self._NMS_max_output_size = max_output_size
        # The NMS threshold.
        self._NMS_THRESH = NMS_THRESH


    def convert2_practice(self, cls_tf, bbox_tf, score_tf=None):

        # Check the validity of cls_tf.
        if not isinstance(cls_tf, tf.Tensor):
            raise TypeError('The cls_tf must be of @Type{tf.Tensor} !!!')
        if len(cls_tf.shape) != 4:
            raise Exception('The cls_tf must be a 4-D tensor !!!')

        # Check the validity of bbox_tf.
        if not isinstance(bbox_tf, tf.Tensor):
            raise TypeError('The bbox_tf, must be of @Type{tf.Tensor} !!!')
        if len(bbox_tf.shape) != 5:
            raise Exception('The bbox_tf must be a 5-D tensor !!!')

        # Enable the NMS algorithm if the score_tf is not None.
        if score_tf is not None:
            # Set the NMS flag to true.
            NMS_enable = True
            # Check the validity of score_tf.
            if not isinstance(score_tf, tf.Tensor):
                raise TypeError('The score_tf, must be of @Type{tf.Tensor} !!!')
            if len(score_tf.shape) != 5:
                raise Exception('The score_tf must be a 5-D tensor !!!')
        else:
            # Disable the NMS, simply filter the background bounding-box.
            NMS_enable = False

        # Get the practice bounding-box according to the NMS flag.
        if NMS_enable:

            print('Enable the #"NMS"# algorithm !!!')

            # Firstly we filter the background bounding-box and its score. Coz we only
            #   concern about the foreground object.
            fore_bbox, fore_score = self._filter_4fore_nms(cls_tf=cls_tf, score_tf=score_tf, bbox_tf=bbox_tf)

            # We only use the non-background score. So we only use the 2nd dimension of scores.
            fore_score_nonbg = fore_score[:, 1]     # [batch * width * height * anchor_num]

            print(fore_score_nonbg.shape) # el, er, eu, eb

            # We should firstly transpose the 2nd dimension. The bbox is in the form
            #   of (el, er, eu, eb).
            # In detail: transpose the (x1, x2, y1, y2) to (x1, y1, x2, y2)
            x1 = fore_bbox[:, 0]    # [-1]
            x2 = fore_bbox[:, 1]    # [-1]
            y1 = fore_bbox[:, 2]    # [-1]
            y2 = fore_bbox[:, 3]    # [-1]
            swap_fore_bbox = tf.stack((x1, y1, x2, y2), axis=1)     # [-1, 4]

            # Apply the NMS algorithm.
            nms_indices = tf.image.non_max_suppression(boxes=swap_fore_bbox,
                                                       scores=fore_score_nonbg,
                                                       max_output_size=self._NMS_max_output_size,
                                                       iou_threshold=self._NMS_THRESH)  # [-1]

            # Get the bounding-box after NMS algorithm.
            nms_bbox = tf.gather(fore_bbox, indices=nms_indices, axis=0)    # [-1, 4]

            # The final bounding-box is just that.
            practice_bbox = nms_bbox

        else:
            # Just filter the background bounding-box.
            practice_bbox = self._filter_4fore(cls_tf=cls_tf, bbox_tf=bbox_tf)

        # Finish the processing for practice bounding-box.
        return practice_bbox


    def _filter_4fore(self, cls_tf, bbox_tf):
        r'''
            Filter the bounding-box according to the class map.

        :param cls_tf:
        :param bbox_tf:
        :return:
        '''

        # Firstly flatten the class map. Only 1-D
        flat_cls_map = tf.reshape(cls_tf, shape=(-1,))  # [batch * width * height * anchor_num]

        # Then flatten the 5-D bbox map to 2-D map.
        flat_bbox_tf = tf.reshape(bbox_tf, shape=(-1, bbox_tf.shape[-1]))  # [b*w*h*a, 4]

        # The non-zero (zeros means background) position of the class map
        #   is the foreground bounding-box.
        cond_foreground = tf.not_equal(flat_cls_map, 0)  # [b*w*h*a]
        fore_index = tf.where(cond_foreground)  # [-1, 1]
        fore_index = tf.squeeze(fore_index, axis=-1)  # [-1]

        # Finally we can get the foreground bounding-box.
        fore_bbox = tf.gather(flat_bbox_tf, indices=fore_index, axis=0)  # [-1, 4]

        # Finally return the filtered foreground bounding-box and score map.
        return fore_bbox


    def _filter_4fore_nms(self, cls_tf, score_tf, bbox_tf):
        r'''
            Filter the bounding-box and related score map according to the class map,
                so that we can get the foreground bounding-box and its scores.
                They will lately used in the NMS algorithm.

        :param cls_tf:
        :param score_tf:
        :param bbox_tf:
        :return:
        '''

        # Firstly flatten the class map. Only 1-D
        flat_cls_map = tf.reshape(cls_tf, shape=(-1,))   # [batch * width * height * anchor_num]

        # Then flatten the 5-D bbox map to 2-D map.
        flat_bbox_tf = tf.reshape(bbox_tf, shape=(-1, bbox_tf.shape[-1]))       # [b*w*h*a, 4]
        # So as the score map.
        flat_score_tf = tf.reshape(score_tf, shape=(-1, score_tf.shape[-1]))    # [b*w*h*a, 2]

        # The non-zero (zeros means background) position of the class map
        #   is the foreground bounding-box.
        cond_foreground = tf.not_equal(flat_cls_map, 0)    # [b*w*h*a]
        fore_index = tf.where(cond_foreground)   # [-1, 1]
        fore_index = tf.squeeze(fore_index, axis=-1)    # [-1]

        # Finally we can get the foreground bounding-box.
        fore_bbox = tf.gather(flat_bbox_tf, indices=fore_index, axis=0)     # [-1, 4]
        # So as the score map. The foreground scores.
        fore_score = tf.gather(flat_score_tf, indices=fore_index, axis=0)   # [-1, 2]

        # Finally return the filtered foreground bounding-box and score map.
        return fore_bbox, fore_score






