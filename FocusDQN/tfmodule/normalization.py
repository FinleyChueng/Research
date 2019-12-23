import tensorflow.contrib.layers as tf_layers


# Define the function used to select feature normalization method.
def feature_normalize(x, normalization, name_space, reuse, bn_training, bn_decay):
    if normalization == 'W/O' or normalization is None:
        y = x
    elif normalization == 'batch':
        if bn_training is None or bn_decay is None:
            raise ValueError('The bn_training and bn_decay must have value'
                             ' when enable Batch Normalization !!!')
        y = tf_layers.batch_norm(x,
                                 reuse=reuse,
                                 is_training=bn_training,
                                 decay=bn_decay,
                                 scope=name_space + '/batch_norm')
    elif normalization == 'instance':
        y = tf_layers.instance_norm(x,
                                    reuse=reuse,
                                    scope=name_space + '/instance_norm')
    elif normalization == 'group':
        y = tf_layers.group_norm(x,
                                 reuse=reuse,
                                 scope=name_space + '/group_norm')
    else:
        raise ValueError('Unknown feature normalization method !!!')
    return y


