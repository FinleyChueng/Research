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

