import tensorflow as tf
import numpy as np
import cv2


def placeholder_wrapper(dic, dtype, shape, name):
    r'''
        The wrapper of tensorflow.placeholder, for conveniently
            package the inputs holder.
    '''
    if not isinstance(dic, dict):
        raise TypeError('The dic must be a dictionary !!!')
    y = tf.placeholder(dtype, shape, name)
    # key = y.name.split(':')[0]
    full_name = y.name.split(':')[0]
    fns = full_name.split('/')
    if len(fns) >= 2:
        key = fns[0] + '/' + fns[-1]
    else:
        key = fns[0]
    dic[key] = y
    return y


def package_tensor(dic, tensor):
    r'''
        Package tensors into dictionary. No need to specify the key.
    '''
    if not isinstance(dic, dict):
        raise TypeError('The dic must be a dictionary !!!')
    full_name = tensor.name.split(':')[0]
    fns = full_name.split('/')
    if len(fns) >= 2:
        key = fns[0] + '/' + fns[-1]
    else:
        key = fns[0]
    dic[key] = tensor
    return


def scale_bbox(bbox, src_height, src_width, dst_height, dst_width, name, restrict_boundary=True):
    r'''
        Scale the coordinates of bounding-box according to the source/destination
            size of the corresponding image, it will re-assign (calculate) the
            coordinates.

        ** Note that, this scale operation assumes that: the size change from
            source to destination of corresponding image is "Center Crop".
            (Crop from the center position)
    '''
    # Check validity.
    if not isinstance(bbox, tf.Tensor):
        raise TypeError('The bbox must be a Tensor !!!')
    if len(bbox.shape) != 2 or bbox.shape[1] != 4:
        raise ValueError('The bbox must be of shape [None, 4]')
    if not isinstance(src_height, int) or \
            not isinstance(src_width, int) or \
            not isinstance(dst_height, int) or \
            not isinstance(dst_width, int):
        raise TypeError('Source/Destination height/width must be int !!!')
    # Coordinates.
    y1 = bbox[:, 0]
    x1 = bbox[:, 1]
    y2 = bbox[:, 2]
    x2 = bbox[:, 3]
    # Calculate offset from (0, 0).
    offset_y = (src_height - dst_height) // 2
    offset_x = (src_width - dst_width) // 2
    # Re-assign the coordinates' value.
    y1 = (y1 * src_height - offset_y) / dst_height
    x1 = (x1 * src_width - offset_x) / dst_width
    y2 = (y2 * src_height - offset_y) / dst_height
    x2 = (x2 * src_width - offset_x) / dst_width
    # Restrict the boundary if needed. That is, restrict the normalized
    #   value in range (0, 1).
    if restrict_boundary:
        y1 = tf.maximum(0.0, tf.minimum(1.0, y1))
        x1 = tf.maximum(0.0, tf.minimum(1.0, x1))
        y2 = tf.maximum(0.0, tf.minimum(1.0, y2))
        x2 = tf.maximum(0.0, tf.minimum(1.0, x2))
    # Avoid the "Zero-Scale" situation. Use left-up point as fake bounding-box.
    fy2_1px = 1 / dst_height
    fx2_1px = 1 / dst_width
    fy1 = 0.0 * tf.ones_like(y1)
    fx1 = 0.0 * tf.ones_like(x1)
    fy2 = fy2_1px * tf.ones_like(y2)
    fx2 = fx2_1px * tf.ones_like(x2)
    y1 = tf.where(tf.equal(y2 - y1, 0.0), fy1, y1)
    x1 = tf.where(tf.equal(x2 - x1, 0.0), fx1, x1)
    y2 = tf.where(tf.equal(y2 - y1, 0.0), fy2, y2)
    x2 = tf.where(tf.equal(x2 - x1, 0.0), fx2, x2)
    # Stack.
    re_bbox = tf.stack([y1, x1, y2, x2], axis=-1, name=name)
    return re_bbox


def batch_resize_to_bbox_for_op(x, bbox, cor_size, resize_method, op_func, output_shape, name):
    r'''
        Batch process the input tensor. In detail, it will get the each sub-tensor from
            the batch, then resize the sub-tensor to the corresponding bbox size, and
            finally apply the custom function to the bbox tensor. The shape of output
            tensor depends on the output_shape.

    Parameters:
        x: The tensor waits for batch process.
        bbox: The bounding-box tensor, indicates the bbox size.
        cor_size: The corresponding size of the bounding-box.
        resize_method: The resize method. Default is "bilinear", optional is "nearest"
            and "crop".
        op_func: The custom function used to apply to the bbox-tensor (the tensor
            after re-size).
        output_shape: The shape of output tensor, it must match the output
            of the op_func.
        name: The operation (output tensor) name.

    Return:
        The tensor after bbox-resize and custom operation, whose shape is output_shape.
    '''

    # Check validity of parameters.
    if not isinstance(x, list) or len(x) == 0:
        raise TypeError('The x must be a list consists of at least one tensor !!!')
    x_invar = None
    for e_x in x:
        if not isinstance(e_x, tf.Tensor) or len(e_x.shape) != 4:
            raise TypeError('The elements of x must be 4-D tensor !!!')
        if x_invar is None:
            x_invar = e_x.shape[1:-1]
        if x_invar != e_x.shape[1:-1]:
            raise TypeError('The all input tensors must have same height and width !!!')
    if not isinstance(bbox, tf.Tensor) or len(bbox.shape) != 2 or bbox.shape[1] != 4:
        raise TypeError('The bbox must be [?, 4] Tensor !!!')
    if not isinstance(cor_size, list) or len(cor_size) != 2:
        raise TypeError('The cor_size must be 2-element list !!!')
    if not callable(op_func):
        raise TypeError('The op_func must be a custom function used to receive and process'
                        ' the bbox-shape (re-sized) tensor !!!')
    if output_shape is None:
        pass
    else:
        if not isinstance(output_shape, list):
            raise TypeError('The output_shape must be either None or list !!!')
    if not isinstance(resize_method, list):
        raise TypeError('The resize_method must be a list consists of method names !!!')
    for m in resize_method:
        if m not in ['bilinear', 'nearest', 'crop']:
            raise ValueError('The method must be one of "bilinear", "nearest", "crop" !!!')
    if len(x) != len(resize_method):
        raise ValueError('The size of x and resize_method must matched !!!')

    # Get the height and width of the bounding-box.
    box_height = tf.round(tf.multiply(tf.to_float(cor_size[0]), bbox[:, 2] - bbox[:, 0]))
    box_width = tf.round(tf.multiply(tf.to_float(cor_size[1]), bbox[:, 3] - bbox[:, 1]))
    # Avoid "y1(x1) > y2(x2)".
    box_height = tf.cast(tf.abs(box_height), 'int32')   # [?]
    box_width = tf.cast(tf.abs(box_width), 'int32')     # [?]
    # Avoid "Zero Scale".
    box_height = tf.maximum(1, box_height)
    box_width = tf.maximum(1, box_width)

    # The loop body function.
    def body_func(idx, y):
        # Resize all input tensors to bbox-shape.
        h, w = box_height[idx], box_width[idx]
        candidates = []
        for sub_x, m in zip(x, resize_method):
            cand = tf.slice(sub_x, [idx, 0, 0, 0], [1, -1, -1, -1])
            if m == 'bilinear':
                cand = tf.image.resize_bilinear(cand, [h, w])   # [1, h, w, c]
            elif m == 'nearest':
                cand = tf.image.resize_nearest_neighbor(cand, [h, w])   # [1, h, w, c]
            elif m == 'crop':
                oy = tf.minimum(bbox[idx, 0], bbox[idx, 2])
                ox = tf.minimum(bbox[idx, 1], bbox[idx, 3])
                off_y = tf.cast(tf.round(oy * tf.to_float(h)), 'int32')
                off_x = tf.cast(tf.round(ox * tf.to_float(w)), 'int32')
                cand = tf.image.crop_to_bounding_box(cand, off_y, off_x, h, w)  # [1, h, w, c]
            else:
                raise ValueError('Unknown resize method !!!')
            candidates.append(cand)
        # Custom operation.
        sub_y = op_func(candidates, bbox[idx], cor_size)
        # Check validity.
        if len(y.shape) != len(sub_y.shape) or y.shape[1:] != sub_y.shape[1:]:
            raise ValueError('Invalid sub_y shape ({}), must be same as the y ({}) !!!'
                             ' Please check the custom function. '.format(sub_y.shape, y.shape))
        # Accumulate the sub-y value.
        if len(y.shape) != 0:
            y = tf.concat([y[0: idx], sub_y, y[idx+1:]], axis=0)  # [?, h, w, c]
        else:
            y = tf.add(y, sub_y)
        idx += 1
        return idx, y

    # Declare the output tensor (which is actually pass through the while-loop for
    #   gradually processing).
    if output_shape is None:
        y = 0.
    else:
        bt_ind = tf.reduce_mean(tf.zeros_like(x[0]), axis=(1, 2, 3), keepdims=True)  # [?, 1, 1, 1]
        if len(output_shape) == 0:
            os_ind = tf.zeros([1,])     # [1,]
        else:
            os_ind = tf.expand_dims(tf.zeros(output_shape), axis=0)     # [1, h, w, c]
        dim_diff = len(output_shape) - (len(bt_ind.shape) - 1)
        if dim_diff > 0:  # match
            for _ in range(dim_diff):
                bt_ind = tf.expand_dims(bt_ind, axis=-1)
        elif dim_diff < 0:     # index
            for _ in range(-dim_diff):
                bt_ind = tf.reduce_mean(bt_ind, axis=-1)
        else:
            pass
        y = os_ind * bt_ind

    # Calculate the batch size.
    batch_size = tf.reduce_sum(tf.reduce_mean(tf.ones_like(x[0], dtype=tf.int32), axis=(1, 2, 3)))     # scalar
    # Start to loop.
    index = 0
    _1, y = tf.while_loop(
        cond=lambda idx, _2: tf.less(idx, batch_size),
        body=body_func,
        loop_vars=[index, y],
        name=name)
    # Get the final batch processed tensor.
    return y


def pad_2up(x, bbox, cor_size, name, padding_values=0):
    r'''
        Padding the tensor of "Bounding-box" region size (with '0') to the given up-sample shape.

    Parameters:
        x: The tensor waits for batch process.
        bbox: The bounding-box tensor, indicates the bbox size.
        cor_size: The corresponding size of the bounding-box.
        padding_values: The padding values for fix holes.
        name: The operation (output tensor) name.

    Return:
        The tensor after bbox-resize and custom operation, whose shape is output_shape.
    '''
    # Check validity.
    if not isinstance(x, tf.Tensor) or len(x.shape) != 4:
        raise TypeError('The x must be 4-D tensor !!!')
    if not isinstance(bbox, tf.Tensor) or len(bbox.shape) != 1 or bbox.shape[0] != 4:
        raise TypeError('The bbox must be [4,] Tensor !!!')
    if not isinstance(cor_size, list) or len(cor_size) != 2:
        raise TypeError('The cor_size must be 2-element list !!!')
    # Get the up-sample height and width.
    up_h, up_w = cor_size
    # Get four boundary coord in "Normalized" form.
    oy1 = bbox[0]
    ox1 = bbox[1]
    oy2 = bbox[2]
    ox2 = bbox[3]
    # Compute pad size.
    py_up = tf.minimum(oy1, oy2) - 0.0
    py_up = tf.cast(tf.round(tf.to_float(up_h) * py_up), 'int32')   # Up
    py_bot = 1.0 - tf.maximum(oy1, oy2)
    py_bot = tf.cast(tf.round(tf.to_float(up_h) * py_bot), 'int32')     # Bottom
    px_left = tf.minimum(ox1, ox2) - 0.0
    px_left = tf.cast(tf.round(tf.to_float(up_w) * px_left), 'int32')   # Left
    px_right = 1.0 - tf.maximum(ox1, ox2)
    px_right = tf.cast(tf.round(tf.to_float(up_w) * px_right), 'int32')     # Right
    # Add rectify value to the "left/right" and "up/bottom" coz there's
    #   deviation in the round operation.
    iy_h = tf.reduce_sum(tf.reduce_mean(tf.ones_like(x, dtype=tf.int32), axis=(0, 2, 3)))   # height of y
    iy_w = tf.reduce_sum(tf.reduce_mean(tf.ones_like(x, dtype=tf.int32), axis=(0, 1, 3)))   # width of y
    py_diff = up_h - iy_h - py_up - py_bot
    px_diff = up_w - iy_w - px_left - px_right
    rect_bot = py_bot + py_diff
    rect_right = px_right + px_diff
    py_up = tf.where(tf.greater_equal(rect_bot, 0), py_up, py_up + py_diff)
    py_bot = tf.where(tf.greater_equal(rect_bot, 0), rect_bot, py_bot)
    px_left = tf.where(tf.greater_equal(rect_right, 0), px_left, px_left + px_diff)
    px_right = tf.where(tf.greater_equal(rect_right, 0), rect_right, px_right)
    # Generate pad vector.
    pads = [[0, 0],
            [py_up, py_bot],
            [px_left, px_right],
            [0, 0]]
    # Pad the tensor.
    y = tf.pad(x, pads, constant_values=padding_values)
    y = tf.reshape(y, [y.shape[0], up_h, up_w, y.shape[-1]], name=name)
    return y


def gen_focus_maps_4bbox(bbox, cor_size, name):
    r'''
        Generate the "Focus Maps" for the corresponding "Focus Bounding-box".

    Parameters:
        bbox: The bounding-box tensor, indicates the bbox coordinates.
        cor_size: The corresponding size of the bounding-box.
        name: The operation (output tensor) name.

    Return:
        The "Focus Maps" for corresponding "Focus Bounding-box". its shape likes below:
            [batch, height, width, 1]. (last dimension is always 1) with data type bool.
    '''
    # Check validity.
    if not isinstance(bbox, tf.Tensor) or len(bbox.shape) != 2 or bbox.shape[1] != 4:
        raise TypeError('The bbox must be [?, 4] Tensor !!!')
    if not isinstance(cor_size, list) or len(cor_size) != 2:
        raise TypeError('The cor_size must be 2-element list !!!')
    # Generate the maps.
    h, w = cor_size
    def _gen_map(_inp1):
        _y = []
        for _x1 in _inp1:
            _cy1, _cx1, _cy2, _cx2 = _x1
            _cy1 = int(round(_cy1 * h))
            _cx1 = int(round(_cx1 * w))
            _cy2 = int(round(_cy2 * h))
            _cx2 = int(round(_cx2 * w))
            _m = np.zeros((h, w, 1), dtype=np.bool)
            _m[_cy1: _cy2, _cx1: _cx2, :] = True  # [h, w, 1]
            _y.append(_m)
        _y = np.asarray(_y)
        return _y
    y, = tf.py_func(_gen_map, inp=[bbox], Tout=[tf.bool])
    y = tf.reshape(y, [-1, h, w, 1], name=name)  # [?, h, w, 1]
    return y


def fuzzy_4bbox(x, bbox, name):
    r'''
        Fuzzy the specific region of the input tensor according to the given "Bounding-box".

    Parameters:
        x: The input tensor, must be 4-D.
        bbox: The bounding-box tensor, indicates the bbox coordinates.
        name: The operation (output tensor) name.

    Return:
        The tensor, whose bbox-region clear and outside region fuzzy.
    '''
    # Check validity.
    if not isinstance(x, tf.Tensor) or len(x.shape) != 4:
        raise TypeError('The x must be 4-D tensor !!!')
    if not isinstance(bbox, tf.Tensor) or len(bbox.shape) != 2 or bbox.shape[1] != 4:
        raise TypeError('The bbox must be [?, 4] Tensor !!!')
    # Record the input tensor shape and calculate the slices.
    rs = [-1,]
    rs.extend(x.get_shape().as_list()[1:])
    slice = int(np.ceil(x.get_shape().as_list()[-1] / 512))
    # Declare the sub-function used to fuzzy the region outside the "Focus Bbox".
    def _gause_fuzzy(_inp1, _inp2):
        _y = []
        for _x1, _x2 in zip(_inp1, _inp2):
            _yi = None
            for s in range(slice):
                begin = s * 512
                end = min((s+1) * 512, x.get_shape().as_list()[-1])
                # _y1 = cv2.GaussianBlur(_x1[:, :, begin: end], (9, 9), 0)  # [h, w, c]
                _y1 = cv2.GaussianBlur(_x1[:, :, begin: end], (3, 3), 0.8)  # [h, w, c]
                _map = np.zeros_like(_x1[:, :, begin: end], dtype=np.bool)
                _mh, _mw = _map.shape[:-1]
                _my1 = max(0, int(_mh * _x2[0]))
                _mx1 = max(0, int(_mw * _x2[1]))
                _my2 = min(_mh, int(_mh * _x2[2]))
                _mx2 = min(_mw, int(_mw * _x2[3]))
                _map[_my1: _my2, _mx1: _mx2, :] = True
                _yii = np.where(_map, _x1[:, :, begin: end], _y1)   # [h, w, c]
                if _yi is None:
                    _yi = _yii
                else:
                    _yi = np.concatenate((_yi, _yii), axis=-1)
            _y.append(_yi)
        _y = np.asarray(_y)
        return _y
    # Restore the tensor shape.
    y, = tf.py_func(_gause_fuzzy, inp=[x, bbox], Tout=[tf.float32])
    y = tf.reshape(y, rs, name=name)    # [?, h, w, c]
    return y


def copy_model_parameters(from_scope, to_scope):
    """
    Copies the model's parameters of `from_model` to `to_model`.

    Args:
        from_model: model to copy the parameters from
        to_model:   model to copy the parameters to
    """
    from_model_paras = [
        v for v in tf.trainable_variables() if v.name.startswith(from_scope)
    ]
    from_model_paras = sorted(from_model_paras, key=lambda v: v.name)
    to_model_paras = [
        v for v in tf.trainable_variables() if v.name.startswith(to_scope)
    ]
    to_model_paras = sorted(to_model_paras, key=lambda v: v.name)
    update_ops = []
    for from_model_para, to_model_para in zip(from_model_paras,
                                              to_model_paras):
        op = to_model_para.assign(from_model_para)
        update_ops.append(op)
    return update_ops


def show_all_variables():
    r'''
        Show the quantity of all network variables.
    '''
    total_count = 0
    for idx, op in enumerate(tf.trainable_variables()):
        shape = op.get_shape()
        count = np.prod(shape)
        print("[%2d]\t%s\t%s\t=\t%s" % (idx, op.name, shape, count))
        total_count += int(count)
    print("[Total] variable size: %s" % "{:,}".format(total_count))


def count_flops():
    graph = tf.get_default_graph()
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))


