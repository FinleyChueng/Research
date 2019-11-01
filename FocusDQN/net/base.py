import tensorflow as tf

# The feature extractor related abstractions.
class FeatureExtractionNetwork:

    def __init__(self):
        self._need_AFPP = None

    def _proc_afpool(self, tensor, proc_func, feats_stride, arg=None):
        r'''
            This function should be called after max pooling layer.
                To do some additional process.

        :param tensor:
        :param proc_func:
        :param feats_stride:
        :param arg:
        :return:
        '''

        # Check the type of input tensor.
        if not isinstance(tensor, tf.Tensor):
            raise TypeError('The tensor must be a Tensorflow.Tensor!!!')

        # If not specify the process function, return the original tensor.
        if proc_func is None:
            return tensor

        # Check whether the proc_func is function or not.
        if not callable(proc_func):
            raise TypeError('proc_func must be callable.')

        # Pass through the additional process function.
        if self._need_AFPP:
            proc_tensor = proc_func(tensor, feats_stride, arg)
        else:
            proc_tensor = tensor

        # Check whether the return type of proc_func is Tensorflow.Tensor or not.
        if not isinstance(proc_tensor, tf.Tensor):
            raise TypeError('The return of the proc_func must be a Tensorflow.Tensor!!!')

        # Finish additional process.
        return proc_tensor


    def build(self, input_layer, weights_regularizer, name_scope, need_AFPP=False, feats_proc_func=None, arg=None):
        r'''
            Build the real architecture.

        :param input_layer:
        :param weights_regularizer:
        :param name_scope:
        :param feats_proc_func:
        :param arg:
        :return:
        '''
        raise NotImplementedError
