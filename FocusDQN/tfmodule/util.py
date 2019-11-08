import tensorflow as tf
import numpy as np


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


def scale_bbox(bbox, src_height, src_width, dst_height, dst_width, name):
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
    # Stack.
    re_bbox = tf.stack([y1, x1, y2, x2], axis=-1, name=name)
    return re_bbox


def batch_resize_to_bbox(x, bbox, bbox_size):
# def batch_resize_and_op(x, dst_height, dst_width, op_func):
    r'''
    '''


    pass

    # # Resize SEG (segmentation) tensor into the shape of "Focus Bounding-box" region.
    # FBR_height = tf.multiply(input_shape[1], focus_bbox[:, 2] - focus_bbox[:, 0])

    def body(id, tensor, l):
        y = tf.expand_dims(tensor[id], axis=0) * tf.to_float(id)
        print(y)
        l = tf.concat([l[0: id], y, l[id + 1:]], axis=0)
        print(l)
        id += 1
        # return id, tensor,
        return id, tensor, l

    batch_size = tf.reduce_sum(tf.reduce_mean(tf.ones_like(x, dtype=tf.int32), axis=(1, 2)))    # scalar
    idx = 0
    st = tf.zeros_like(x)
    _1, _2, out = tf.while_loop(
        cond=lambda id, _2, _3: tf.less(id, bs),
        body=body,
        loop_vars=[idx, a, st]
    )
    # out = tf.stack(st, axis=-1)

    # --------------------------------
    x1 = np.random.randint(0, 10, (4, 2, 3))
    sess = tf.Session()
    v1 = sess.run(out, feed_dict={
        a: x1
    })



    pass




    return


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
        print ("[%2d]\t%s\t%s\t=\t%s" % (idx, op.name, shape, count))
        total_count += int(count)
    print("[Total] variable size: %s" % "{:,}".format(total_count))


def count_flops():
    graph = tf.get_default_graph()
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    print('FLOPs: {}'.format(flops.total_float_ops))

