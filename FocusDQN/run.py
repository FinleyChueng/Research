import os

from dataset.adapter.bratsAdapter import *
# from task.priorityDQNmodel import *
from util.visualization import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# Just Test -------

# import util.evaluation as eva
#
# p = np.zeros((240, 240), dtype=np.int64)
# p[50: 80, 60: 100] = 1
# l = np.zeros((240, 240), dtype=np.int64)
# l[30: 60, 90: 120] = 1
#
# p = p[50: 60, 40: 60]
# l = l[30: 40, 50: 70]
#
# v1 = eva.prop_DICE_metric(p, l, 2, True)
# print(v1)
# v2 = eva.mean_DICE_metric(p, l, 2, True)
# print(v2)
#
#
#
# import matplotlib.pyplot as plt
# plt.subplot(131)
# plt.imshow(p)
# plt.subplot(132)
# plt.imshow(l)
# plt.subplot(133)
# plt.imshow(p*l)
# plt.show()



# import matplotlib.pyplot as plt
# print('Test')
# im = Image.fromarray(np.zeros((240, 240)))
# font = ImageFont.truetype("consola.ttf", 20, encoding="unic")
# draw = ImageDraw.Draw(im)
# # t = 'Hello World'
# # tw, th = font.getsize(t)
# # draw.text((120-tw//2, 120-th//2), t, font=font)
# t = 'Hello World\nGO: 520'
# tw, th = font.getsize_multiline(t)
# draw.multiline_text((120-tw//2, 120-th//2), t, font=font, align='center')
# ar = np.asarray(im)
# print(ar.ndim)
# plt.imshow(ar, cmap='gray')
# plt.show()



# print('Test')
# import tensorflow as tf
# import tfmodule.util as net_util
#
# x1 = [tf.placeholder(tf.float32, [None, 10, 10, 2]),
#       tf.placeholder(tf.float32, [None, 10, 10, 5])]
# # x1 = [tf.placeholder(tf.float32, [None, 10, 10, 2])]
# temp = [[0.2, 0.4, 0.4, 0.6],
#         [0.6, 0.2, 0.8, 0.6],
#         [0.4, 0.4, 0.6, 0.6]]
# x2 = tf.constant(temp)
# x3 = [10, 10]
# x4 = ['bilinear', 'crop']
# # x4 = ['bilinear']
# def x5(cands, bbox):
#     # sub_y = tf.image.resize_nearest_neighbor(sub_y, x3)
#     # sub_y = tf.reduce_sum(cands[0])
#     sub_y = tf.reduce_sum(tf.reduce_mean(cands[0], axis=-1) + 3 * tf.reduce_mean(cands[1], axis=-1))
#     return sub_y
# x6 = None
#
# y = net_util.batch_resize_to_bbox_for_op(x1, x2, x3, x4, x5, x6, None)
#
# # a = np.ones([3, 10, 10, 2], dtype=np.float32)
# # a = np.random.randint(-2, 6, [3, 10, 10, 2])
# a = np.random.randint(-2, 6, [3, 10, 10, 2])
# b = np.random.randint(-2, 6, [3, 10, 10, 5])
# sess = tf.Session()
# v = sess.run(y, feed_dict={x1[0]: a, x1[1]: b})
#
# print('-- orgin --')
# print(a)
# print('== process ==')
# print(v)



# print('Test')
#
# import tensorflow as tf
#
# a = tf.random.uniform([1, 10, 10, 1], maxval=1.0)
# b = tf.random.uniform([1, 10, 10, 1], maxval=1.0)
#
# # a = tf.random.uniform([1, 240, 240, 1], maxval=1.0)
# # b = tf.random.uniform([1, 240, 240, 1], maxval=1.0)
#
# # temp = [[0.3, 0.3, 0.9, 0.9]]
# temp = [[0.2, 0.2, 0.8, 0.8]]
# # temp = [[0.4, 0.4, 0.8, 0.8]]
# bbox = tf.constant(temp)
# bid = tf.constant([0])
#
# src_h, src_w = a.get_shape().as_list()[1:3]
# dst_h = int((temp[0][2]-temp[0][0]) * src_h)
# dst_w = int((temp[0][3]-temp[0][1]) * src_w)
#
# of_y = int(src_h * temp[0][0])
# of_x = int(src_w * temp[0][1])
# y1 = tf.image.crop_and_resize(a, bbox, bid, [dst_h, dst_w])
# lab1 = tf.image.crop_to_bounding_box(b, of_y, of_x, dst_h, dst_w)
# loss1 = tf.reduce_mean(lab1 * tf.log(y1))
# # loss1 = tf.reduce_mean(tf.square(lab1 - y1))
# print('dst_h: {}, dst_w: {}'.format(dst_h, dst_w))
# print('of_y: {}, of_x: {}'.format(of_y, of_x))
#
# y2 = a
# lab2 = tf.image.crop_and_resize(b, bbox, bid, [src_h, src_w])
# loss2 = tf.reduce_mean(lab2 * tf.log(y2))
# # loss2 = tf.reduce_mean(tf.square(lab2 - y2))
#
# by_ratio = bbox[0, 2]-bbox[0, 0]
# bx_ratio = bbox[0, 3]-bbox[0, 1]
#
# sess = tf.Session()
# v1, v2, v3, v4 = sess.run([loss1, loss2, by_ratio, bx_ratio])
# print('loss1: {}, loss2: {}'.format(v1, v2))
# print('l2/l1: {}, bbox --> y-ratio: {}, x-ratio: {}'.format(
#     v2/v1, v3, v4))



# print('Test')
#
# import tensorflow as tf
#
# def body(id, tensor, l):
#     y = tf.expand_dims(tensor[id], axis=0) * tf.to_float(id)
#     print(y)
#     l = tf.concat([l[0: id], y, l[id+1:]], axis=0)
#     print(l)
#     id += 1
#     # return id, tensor,
#     return id, tensor, l
# a = tf.placeholder(tf.float32, [None, 2, 3])
# bs = tf.ones_like(a, dtype=tf.int32)
# bs = tf.reduce_sum(tf.reduce_mean(bs, axis=(1, 2)))     # scalar
# idx = 0
# # st = 0.
# st = tf.zeros_like(a)
# _1, _2, out = tf.while_loop(
#     cond=lambda id, _2, _3: tf.less(id, bs),
#     body=body,
#     loop_vars=[idx, a, st]
# )
#
# # --------------------------------
# x1 = np.random.randint(0, 10, (4, 2, 3))
# sess = tf.Session()
# v1 = sess.run(out, feed_dict={
#     a: x1
# })
#
# print('Origin: ', x1.shape)
# print(x1)
# print('While-loop:, ', v1.shape)
# print(v1)
#
# print('-- end --')

# ----------------------------------------------------------


# Data Visualization Test. ---------------------------------

# adapter = BratsAdapter(enable_data_enhance=False)
# for _ in range(15):
# # for _ in range(35):
#     adapter.next_image_pair('Train', batch_size=155)
# adapter.next_image_pair('Train', batch_size=70)
# img, label, finish = adapter.next_image_pair('Train', batch_size=1)
# plt.subplot(231)
# plt.imshow(img[:, :, 0], cmap='gray')
# plt.subplot(232)
# plt.imshow(img[:, :, 1], cmap='gray')
# plt.subplot(233)
# plt.imshow(img[:, :, 2], cmap='gray')
# plt.subplot(234)
# plt.imshow(img[:, :, 3], cmap='gray')
# plt.subplot(235)
# plt.imshow(np.argmax(label, axis=-1), cmap='gray')
# # plt.imshow(label)
# plt.show()

# --------------------------------------------

# DqnAgent Model Test. -------------------------------------------------------

# * ---> Transfer some holders ...
# | ---> Finish holders transferring !
# |   ===> Inputs holder: dict_keys(['ORG/image', 'ORG/prev_result', 'ORG/Focus_Bbox', 'ORG/position_info', 'ORG/Segment_Stage', 'TEST/Complete_Result'])
# |   ===> Outputs holder: dict_keys(['ORG/DQN_output', 'TEST/SEG_output', 'TEST/FUSE_result'])
# |   ===> Losses holder: dict_keys(['TEST/GT_label', 'TEST/clazz_weights', 'TEST/prediction_actions', 'TEST/target_Q_values', 'TEST/EXP_priority', 'TEST/IS_weights', 'TEST/SEG_loss', 'TEST/DQN_loss', 'TEST/NET_loss', 'TAR/image', 'TAR/prev_result', 'TAR/Focus_Bbox', 'TAR/position_info', 'TAR/Segment_Stage', 'TAR/DQN_output'])
# |   ===> Summary holder: dict_keys(['TEST/Reward', 'TEST/DICE', 'TEST/BRATS_metric', 'TEST/MergeSummary'])
# |   ===> Visual holder: dict_keys([])

# # print('Begin')
# from task.model import *
# import util.config as conf_util
# import tfmodule.util as netutil
# import os
# config_file = '/FocusDQN/config.ini'
# config_file = os.path.abspath(os.path.dirname(os.getcwd())) + config_file
# config = conf_util.parse_config(config_file)
# dqn = DqnAgent(name_space='TEST', config=config)
# inputs, outputs, losses, summary, visual = dqn.definition()
# # netutil.show_all_variables()
# # netutil.count_flops()
# # print('End')
#
# import matplotlib.pyplot as plt
# def show(label, img=None):
#     if img is not None:
#         plt.subplot(231)
#         plt.imshow(img[:, :, 0], cmap='gray')
#         plt.subplot(232)
#         plt.imshow(img[:, :, 1], cmap='gray')
#         plt.subplot(233)
#         plt.imshow(img[:, :, 2], cmap='gray')
#         plt.subplot(234)
#         plt.imshow(img[:, :, 3], cmap='gray')
#         plt.subplot(235)
#         plt.imshow(label, cmap='gray')
#     else:
#         plt.imshow(label, cmap='gray')
#     plt.show()
#     return
#
# # # get image and label.
# # adapter = BratsAdapter(enable_data_enhance=False)
# # for _ in range(15):
# # # for _ in range(35):
# #     adapter.next_image_pair('Train', batch_size=155)
# # adapter.next_image_pair('Train', batch_size=70)
# # img, lab, _1, _2, finish = adapter.next_image_pair('Train', batch_size=4)
# #
# # # show
# # show(lab[0], img[0])
#
# # init.
# sess = tf.Session()
# # infer. ---
# # fake input.
# # img = img[:, :, :, :-1]
# img = np.ones([4, 240, 240, 3])
# img[:, :, :, 1] = 2
# img[:, :, :, 2] = 3
# pr = np.ones([4, 240, 240])
# pi = np.ones([4, 240, 240])
# ss = [True, True, False, False]
# # ss = [False, False, False, False]
# # ss = [True, True, False, False]
# # fb = [[0.5, 0.7, 0.4, 0.6],
# #       [0.5, 0.7, 0.4, 0.6],
# #       [0.5, 0.7, 0.4, 0.6],
# #       [0.5, 0.7, 0.4, 0.6]]
# # fb = [[0.5, 0.4, 0.7, 0.6],
# #       [0.5, 0.4, 0.7, 0.6],
# #       [0.5, 0.4, 0.7, 0.6],
# #       [0.5, 0.4, 0.7, 0.6]]
# fb = [[0.5, 0.7, 0.7, 0.6],
#       [0.5, 0.4, 0.7, 0.8],
#       [0.5, 0.6, 0.9, 0.7],
#       [0.3, 0.4, 0.8, 0.6]]
# # fb = [[0.3, 0.5, 0.7, 0.6],
# #       [0.3, 0.5, 0.7, 0.6],
# #       [0.3, 0.5, 0.7, 0.6],
# #       [0.3, 0.5, 0.7, 0.6]]
# # cs = np.ones([4, 224, 224, 5])
# # cs = np.random.normal(0, 1, [4, 224, 224, 5])
# cs = np.random.uniform(0, 1, [4, 224, 224, 5])
# # cs = np.random.randint(0, 5, [4, 224, 224])
# # input holders.
# x1 = inputs['ORG/image']
# x2 = inputs['ORG/prev_result']
# x3 = inputs['ORG/position_info']
# x4 = inputs['ORG/Segment_Stage']
# x5 = inputs['ORG/Focus_Bbox']
# x6 = inputs['TEST/Complete_Result']
# # output holders.
# y1 = outputs['ORG/DQN_output']
# y2 = outputs['TEST/SEG_output']
# # feed.
# sess.run(tf.global_variables_initializer())
# v1, v2 = sess.run([y1, y2], feed_dict={
#     x1: img,
#     x2: pr,
#     x3: pi,
#     x4: ss,
#     x5: fb,
#     x6: cs
# })
# print('--- infer ---')
# print('DQN_output: ', v1)
# print('SEG_output: ', v2)
# print('all 0: ', np.all(v2 == 0))
# print('any non-zero: ', np.any(v2 != 0))
# print('min cls: ', np.min(v2))
# print('max cls: ', np.max(v2))
# # show(v2[0])
#
# # train. ---
# # fake input.
# lab = np.ones([4, 240, 240])
# cw = [[1, 5, 3, 4, 10],
#       [2, 6, 1, 3, 7],
#       [5, 2, 7, 1, 9],
#       [3, 5, 2, 7, 1]]
# pa = [2, 3, 1, 5]
# tqa = [0.5, 0.6, 0.9, 0.15]
# isw = [0.1, 0.4, 0.5, 0.2]
# # input holders.
# l1 = losses['TEST/GT_label']
# l2 = losses['TEST/clazz_weights']
# l3 = losses['TEST/prediction_actions']
# l4 = losses['TEST/target_Q_values']
# l5 = losses['TEST/IS_weights']
# # output holders.
# y1 = losses['TEST/EXP_priority']
# y2 = losses['TEST/SEG_loss']
# y3 = losses['TEST/DQN_loss']
# y4 = losses['TEST/NET_loss']
# # feed.
# opt = tf.train.AdamOptimizer()
# train = opt.minimize(y4)
# sess.run(tf.global_variables_initializer())
# _, v1, v2, v3, v4 = sess.run([train, y1, y2, y3, y4], feed_dict={
#     x1: img,
#     x2: pr,
#     x3: pi,
#     x4: ss,
#     x5: fb,
#     #------
#     l1: lab,
#     l2: cw,
#     l3: pa,
#     l4: tqa,
#     l5: isw
# })
# print('--- train ---')
# print('EXP_priority: ', v1)
# print('SEG_loss: %10f' % v2)
# print('DQN_loss: %10f' % v3)
# print('NET_loss: %10f' % v4)

# ----------------------------------------------------------------------


# # Environment Test. --------------------------------------------------

# import util.config as conf_util
# import os
# config_file = '/FocusDQN/config.ini'
# config_file = os.path.abspath(os.path.dirname(os.getcwd())) + config_file
# config = conf_util.parse_config(config_file)
#
#
# # from task.model import *
# # dqn = DqnAgent(name_space='TEST', config=config)
# # inputs, outputs, losses, summary, visual = dqn.definition()
# # # input holders.
# # x1 = inputs['ORG/image']
# # x2 = inputs['ORG/prev_result']
# # x3 = inputs['ORG/position_info']
# # x4 = inputs['ORG/Segment_Stage']
# # x5 = inputs['ORG/Focus_Bbox']
# # x6 = inputs['TEST/Complete_Result']
# # # output holders.
# # y1 = outputs['ORG/DQN_output']
# # y2 = outputs['TEST/SEG_output']
#
#
# from dataset.adapter.bratsAdapter import BratsAdapter
# adapter = BratsAdapter(enable_data_enhance=False)
# adapter.reset_position(35*155+70)
#
#
# from task.env import FocusEnv
# env = FocusEnv(config, adapter)
#
# class T:
#
#     def __init__(self):
#         # self._l = [0, 0, 0, 2, 0, 4, 0, 4]
#         self._l = []
#         l = [0, 2]
#         # l = [0, 2, 5, 5, 7, 7, 4, 6, 5, 7]
#         # l = [0, 2, 5, 6, 7, 4, 6, 5, 7]
#         for le in l:
#             self._l.append(0)
#             self._l.append(le)
#         self._idx = 0
#
#     def train_func(self, x):
#         img, SEG_prev, position_info, SEG_stage, focus_bbox, COMP_result = x
#
#         segmentation = SEG_prev.copy()
#         y_min = max(0, min(239, int(round(240 * focus_bbox[0]))))
#         x_min = max(0, min(239, int(round(240 * focus_bbox[1]))))
#         y_max = max(1, min(240, int(round(240 * focus_bbox[2]))))
#         x_max = max(1, min(240, int(round(240 * focus_bbox[3]))))
#         print(y_min, y_max, x_min, x_max)
#         y1 = np.random.randint(y_min, y_max)
#         x1 = np.random.randint(x_min, x_max)
#         y2 = np.random.randint(y_min, y_max)
#         x2 = np.random.randint(x_min, x_max)
#         y1, y2 = min(y1, y2), max(y1, y2)
#         x1, x2 = min(x1, x2), max(x1, x2)
#         c = np.random.randint(0, 5)
#         segmentation[y1: y2, x1: x2] = c
#
#         COMP_res = np.zeros_like(COMP_result)
#
#         # action = np.random.randint(8)
#         # action = np.random.randint(9)
#         if len(self._l) > self._idx:
#             action = self._l[self._idx]
#             self._idx += 1
#         else:
#             # action = np.random.randint(17)
#             action = np.random.randint(9)
#
#         return segmentation, COMP_res, action
#
# t = T()
# for _ in range(3):
#     stage = True
#     env.reset()
#     for _0 in range(25):
#         _1, _2, _3, over, _4, _5 = env.step(t.train_func, stage)
#         if not stage and over:
#             break
#         stage = not stage
#     env.render('gif')


# # ---------------------------------------------------------------------



# Whole Test. ---------------------------------------------------------

from task.framework import DeepQNetwork
import util.config as conf_util
import os

config_file = '/FocusDQN/config.ini'
config_file = os.path.abspath(os.path.dirname(os.getcwd())) + config_file
config = conf_util.parse_config(config_file)

data_adapter = BratsAdapter(enable_data_enhance=False)

dqn = DeepQNetwork(config=config,
                   name_space='ME',
                   data_adapter=data_adapter)

# Train.
dqn.train(epochs=1, max_iteration=260*155)
# Test.
dqn.test(10, is_validate=False)

