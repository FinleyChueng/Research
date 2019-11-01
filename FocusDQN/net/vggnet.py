import inspect
import os
import numpy as np
import tensorflow as tf
import time

from net.base import *

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg16(FeatureExtractionNetwork):

    def __init__(self, vgg16_npy_path=None):
        r'''
            Initialization method of VGG16

        :param vgg16_npy_path:
        '''

        # Super initialization
        FeatureExtractionNetwork.__init__(self)

        # Get the pre-trained parameters dictionary.
        if vgg16_npy_path is None:
            path = inspect.getfile(Vgg16)
            path = os.path.abspath(os.path.join(path, os.pardir))
            path = os.path.join(path, "params/vgg16.npy")
            # path = '.\\params\\vgg16.npy'
            vgg16_npy_path = path
            print(path)

        self.__data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("npy file loaded")

    def build(self, input_layer, weights_regularizer, name_scope, need_AFPP=False, feats_proc_func=None, arg=None):
        r"""
        load variable from npy to build the VGG
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        """

        start_time = time.time()
        print("build model VGG16 started")
        # rgb_scaled = rgb * 255.0
        #
        # # Convert RGB to BGR
        # red, green, blue = tf.split(3, 3, rgb_scaled)
        # assert red.get_shape().as_list()[1:] == [224, 224, 1]
        # assert green.get_shape().as_list()[1:] == [224, 224, 1]
        # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        # bgr = tf.concat(3, [
        #     blue - VGG_MEAN[0],
        #     green - VGG_MEAN[1],
        #     red - VGG_MEAN[2],
        # ])
        # assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # self.input = tf.placeholder(tf.float32,
        #                        [None, 224, 224, 3])

        # input_shape = [None, image_shape[0], image_shape[1], 3]
        # input_shape = [None, image_shape[0], image_shape[1], 4]

        # Reset the "Need AFPP" flag.
        self._need_AFPP = need_AFPP

        # The feature stride for VGG16.
        feature_stride = 1  # The feature map stride

        # Start to build the architecture.
        with tf.variable_scope(name_scope):
            # A channel reduce layer. Coz we use the pre-trained VGG model.
            convert_layer = tf.layers.conv2d(input_layer, 3, 3, 1, 'same', name='VGG_convert')

            # The real VGG architecture.
            conv1_1 = self.__conv_layer(convert_layer, weights_regularizer, "conv1_1")
            conv1_2 = self.__conv_layer(conv1_1, weights_regularizer, "conv1_2")
            pool1 = self.__max_pool(conv1_2, name_scope+'_VGG_pool1')   # 120, 120, ?
            feature_stride *= 2     # Increase the stride
            # Pass through the additional process function.
            pool1 = self._proc_afpool(pool1,
                                      proc_func=feats_proc_func,
                                      feats_stride=feature_stride,
                                      arg=arg)  # 120,120,64

            conv2_1 = self.__conv_layer(pool1, weights_regularizer, "conv2_1")
            conv2_2 = self.__conv_layer(conv2_1, weights_regularizer, "conv2_2")
            pool2 = self.__max_pool(conv2_2, name_scope+'_VGG_pool2')   # 60, 60, ?
            feature_stride *= 2     # Increase the stride
            # Pass through the additional process function.
            pool2 = self._proc_afpool(pool2,
                                      proc_func=feats_proc_func,
                                      feats_stride=feature_stride,
                                      arg=arg)  # 60,60,128

            conv3_1 = self.__conv_layer(pool2, weights_regularizer, "conv3_1")
            conv3_2 = self.__conv_layer(conv3_1, weights_regularizer, "conv3_2")
            conv3_3 = self.__conv_layer(conv3_2, weights_regularizer, "conv3_3")
            pool3 = self.__max_pool(conv3_3, name_scope+'_VGG_pool3')   # 30, 30, ?
            feature_stride *= 2  # Increase the stride
            # Pass through the additional process function.
            pool3 = self._proc_afpool(pool3,
                                      proc_func=feats_proc_func,
                                      feats_stride=feature_stride,
                                      arg=arg)  # 30,30,256

            conv4_1 = self.__conv_layer(pool3, weights_regularizer, "conv4_1")
            conv4_2 = self.__conv_layer(conv4_1, weights_regularizer, "conv4_2")
            conv4_3 = self.__conv_layer(conv4_2, weights_regularizer, "conv4_3")
            pool4 = self.__max_pool(conv4_3, name_scope+'_VGG_pool4')   # 15, 15, ?
            feature_stride *= 2  # Increase the stride
            # Pass through the additional process function.
            pool4 = self._proc_afpool(pool4,
                                      proc_func=feats_proc_func,
                                      feats_stride=feature_stride,
                                      arg=arg)  # 15,15,512

        # dim = 1
        # for d in pool4.shape[1:]:
        #     dim *= int(d)
        # vgg_output = tf.reshape(pool4, [-1, dim])

        # conv5_1 = self.__conv_layer(pool4, weights_regularizer, "conv5_1")
        # conv5_2 = self.__conv_layer(conv5_1, weights_regularizer, "conv5_2")
        # conv5_3 = self.__conv_layer(conv5_2, weights_regularizer, "conv5_3")
        # pool5 = self.__max_pool(conv5_3, 'pool5')
        #
        # print(pool5.shape)
        # dim = 1
        # for d in pool5.shape[1:]:
        #     dim *= int(d)
        # self._output = tf.reshape(pool5, [-1, dim])

        # self.fc6 = self.fc_layer(self.pool5, "fc6")
        # assert self.fc6.get_shape().as_list()[1:] == [4096]
        # self.relu6 = tf.nn.relu(self.fc6)
        #
        # self.fc7 = self.fc_layer(self.relu6, "fc7")
        # self.relu7 = tf.nn.relu(self.fc7)
        #
        # self.fc8 = self.fc_layer(self.relu7, "fc8")
        #
        # self.prob = tf.nn.softmax(self.fc8, name="prob")
        #
        # self.data_dict = None

        print("build model finished: %ds" % (time.time() - start_time))

        return self._need_AFPP, pool4, feature_stride

    def __avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def __conv_layer(self, bottom, regularizer, name):
        with tf.variable_scope(name):
            filt = self.__get_conv_filter(regularizer, name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.__get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def __fc_layer(self, bottom, regularizer, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.__get_fc_weight(regularizer, name)
            biases = self.__get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def __get_conv_filter(self, regularizer, var_name):
        return tf.get_variable('cov_kernel', initializer=self.__data_dict[var_name][0], regularizer=regularizer)
        # return tf.Variable(self.data_dict[name][0], name="filter")
        # return tf.constant(self.data_dict[name][0], name="filter")

    def __get_bias(self, var_name):
        return tf.get_variable('conv_biases', initializer=self.__data_dict[var_name][1])
        # return tf.Variable(self.data_dict[name][1], name="biases")
        # return tf.constant(self.data_dict[name][1], name="biases")

    def __get_fc_weight(self, regularizer, var_name):
        return tf.get_variable('fc_weights', initializer=self.__data_dict[var_name][0], regularizer=regularizer)
        # return tf.Variable(self.data_dict[name][0], name="weights")
        # return tf.constant(self.data_dict[name][0], name="weights")