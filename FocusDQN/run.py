import os

from dataset.adapter.bratsAdapter import *
# from task.priorityDQNmodel import *
from util.visualization import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# Just Test -------

import util.evaluation as eva

p = np.zeros((240, 240), dtype=np.int64)
p[50: 80, 60: 100] = 1
l = np.zeros((240, 240), dtype=np.int64)
l[30: 60, 90: 120] = 1

p = p[50: 60, 40: 60]
l = l[30: 40, 50: 70]

v1 = eva.prop_DICE_metric(p, l, 2, True)
print(v1)
v2 = eva.mean_DICE_metric(p, l, 2, True)
print(v2)



import matplotlib.pyplot as plt
plt.subplot(131)
plt.imshow(p)
plt.subplot(132)
plt.imshow(l)
plt.subplot(133)
plt.imshow(p*l)
plt.show()



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

# print('Begin')
from task.model import *
import util.config as conf_util
import tfmodule.util as netutil
import os
config_file = '/FocusDQN/config.ini'
config_file = os.path.abspath(os.path.dirname(os.getcwd())) + config_file
config = conf_util.parse_config(config_file)
dqn = DqnAgent(name_space='TEST', config=config)
inputs, outputs, losses, summary, visual = dqn.definition()
# netutil.show_all_variables()
# netutil.count_flops()
# print('End')

import matplotlib.pyplot as plt
def show(label, img=None):
    if img is not None:
        plt.subplot(231)
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.subplot(232)
        plt.imshow(img[:, :, 1], cmap='gray')
        plt.subplot(233)
        plt.imshow(img[:, :, 2], cmap='gray')
        plt.subplot(234)
        plt.imshow(img[:, :, 3], cmap='gray')
        plt.subplot(235)
        plt.imshow(label, cmap='gray')
    else:
        plt.imshow(label, cmap='gray')
    plt.show()
    return

# # get image and label.
# adapter = BratsAdapter(enable_data_enhance=False)
# for _ in range(15):
# # for _ in range(35):
#     adapter.next_image_pair('Train', batch_size=155)
# adapter.next_image_pair('Train', batch_size=70)
# img, lab, _1, _2, finish = adapter.next_image_pair('Train', batch_size=4)
#
# # show
# show(lab[0], img[0])

# init.
sess = tf.Session()
# infer. ---
# fake input.
# img = img[:, :, :, :-1]
img = np.ones([4, 240, 240, 3])
img[:, :, :, 1] = 2
img[:, :, :, 2] = 3
pr = np.ones([4, 240, 240])
pi = np.ones([4, 240, 240])
ss = [True, True, False, False]
# ss = [False, False, False, False]
# ss = [True, True, False, False]
# fb = [[0.5, 0.7, 0.4, 0.6],
#       [0.5, 0.7, 0.4, 0.6],
#       [0.5, 0.7, 0.4, 0.6],
#       [0.5, 0.7, 0.4, 0.6]]
# fb = [[0.5, 0.4, 0.7, 0.6],
#       [0.5, 0.4, 0.7, 0.6],
#       [0.5, 0.4, 0.7, 0.6],
#       [0.5, 0.4, 0.7, 0.6]]
fb = [[0.5, 0.7, 0.7, 0.6],
      [0.5, 0.4, 0.7, 0.8],
      [0.5, 0.6, 0.9, 0.7],
      [0.3, 0.4, 0.8, 0.6]]
# fb = [[0.3, 0.5, 0.7, 0.6],
#       [0.3, 0.5, 0.7, 0.6],
#       [0.3, 0.5, 0.7, 0.6],
#       [0.3, 0.5, 0.7, 0.6]]
# cs = np.ones([4, 224, 224, 5])
# cs = np.random.normal(0, 1, [4, 224, 224, 5])
cs = np.random.uniform(0, 1, [4, 224, 224, 5])
# cs = np.random.randint(0, 5, [4, 224, 224])
# input holders.
x1 = inputs['ORG/image']
x2 = inputs['ORG/prev_result']
x3 = inputs['ORG/position_info']
x4 = inputs['ORG/Segment_Stage']
x5 = inputs['ORG/Focus_Bbox']
x6 = inputs['TEST/Complete_Result']
# output holders.
y1 = outputs['ORG/DQN_output']
y2 = outputs['TEST/SEG_output']
# feed.
sess.run(tf.global_variables_initializer())
v1, v2 = sess.run([y1, y2], feed_dict={
    x1: img,
    x2: pr,
    x3: pi,
    x4: ss,
    x5: fb,
    x6: cs
})
print('--- infer ---')
print('DQN_output: ', v1)
print('SEG_output: ', v2)
print('all 0: ', np.all(v2 == 0))
print('any non-zero: ', np.any(v2 != 0))
print('min cls: ', np.min(v2))
print('max cls: ', np.max(v2))
# show(v2[0])

# train. ---
# fake input.
lab = np.ones([4, 240, 240])
cw = [[1, 5, 3, 4, 10],
      [2, 6, 1, 3, 7],
      [5, 2, 7, 1, 9],
      [3, 5, 2, 7, 1]]
pa = [2, 3, 1, 5]
tqa = [0.5, 0.6, 0.9, 0.15]
isw = [0.1, 0.4, 0.5, 0.2]
# input holders.
l1 = losses['TEST/GT_label']
l2 = losses['TEST/clazz_weights']
l3 = losses['TEST/prediction_actions']
l4 = losses['TEST/target_Q_values']
l5 = losses['TEST/IS_weights']
# output holders.
y1 = losses['TEST/EXP_priority']
y2 = losses['TEST/SEG_loss']
y3 = losses['TEST/DQN_loss']
y4 = losses['TEST/NET_loss']
# feed.
opt = tf.train.AdamOptimizer()
train = opt.minimize(y4)
sess.run(tf.global_variables_initializer())
_, v1, v2, v3, v4 = sess.run([train, y1, y2, y3, y4], feed_dict={
    x1: img,
    x2: pr,
    x3: pi,
    x4: ss,
    x5: fb,
    #------
    l1: lab,
    l2: cw,
    l3: pa,
    l4: tqa,
    l5: isw
})
print('--- train ---')
print('EXP_priority: ', v1)
print('SEG_loss: %10f' % v2)
print('DQN_loss: %10f' % v3)
print('NET_loss: %10f' % v4)

# ----------------------------------------------------------------------


# # Environment Test. --------------------------------------------------

# 17243252
# 17241196

# recorder = MaskVisualVMPY(240, 240, fps=4,
#                           # vision_filename_mask='G:/Finley/Result/dqn-anim/',
#                           vision_filename_mask='D:/Finley-Experiment/Result/dqn-anim/',
#                           # gif_enable=False
#                           gif_enable=True
#                           )
# adapter = BratsAdapter()
# for _ in range(45):
#     _2, label = adapter.next_image_pair('Train', 1)
#
# env = MazeDQNEnv(data_adapter=adapter,
#                  proc_imgsize=(240, 240),
#                  feats_stride=16,
#                  respective_size=5,
#                  anim_recorder=recorder)
#
#
# env.switch(without_process=True)
# env.switch(without_process=True)
# img = env.switch()
#
# st = time.time()
# # reset the upper environment.
# env.reset((np.ones((20, 20, 4)),))
# # env.reset()
# pred_list = [2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 3]
# terminal = False
# while not terminal:
#     if len(pred_list) == 0:
#         act = np.random.randint(4)
#         # act = np.random.randint(5)
#     else:
#         act = pred_list.pop()
#     # act = np.random.randint(5)
#     _1, reward, terminal, _3 = env.step(act)
#     # _1, reward, terminal, _3 = env.step(act, arg=q)
#     print(reward)
#
# print(time.time() - st)
#
# # env.switch()
# env.render()
#
# print(img.shape)
# print(img)

# # ---------------------------------------------------------------------


# Reward Test. --------------------------------------------------

# import collections
# a = collections.deque()
#
# a.append(1)
# a.append(2)
# a.append(3)
#
# print(a)
#
# a.popleft()
# print(a)
# a.append(4)
# print(a)
#
# a.extend([0, 0, 0, 0])
# a.extend(np.zeros(4))
#
# print(a)

# # ---------------------------------------------------------------------



# Whole Test. ---------------------------------------------------------

from net.resnet import *

recorder = MaskVisualVMPY(240, 240, fps=4,
                          # vision_filename_mask='E:/Finley/Result/caldqn-333/',
                          vision_filename_mask='E:/Finley/Result/caldqn-anim/',
                          # vision_filename_mask='G:/Finley/Result/dqn-anim/',
                          # vision_filename_mask='D:/Finley-Experiment/Result/dqn-anim/',
                          )
data_adapter = BratsAdapter(enable_data_enhance=False)
# For Debug
# for _ in range(858):
# for _ in range(35):
# for _ in range(3750):
#     data_adapter.next_image_pair('Train', 1)
# for _ in range(51):
#     data_adapter.next_image_pair('Test', 1)

# epsilon_dict = [(0, 1.0),
#                 (3000, 0.9),
#                 (6000, 0.8),
#                 (9000, 0.7),
#                 (12000, 0.6),
#                 (15000, 0.5),
#                 (18000, 0.4),
#                 (21000, 0.3),
#                 (24000, 0.2),
#                 (27000, 0.1),
#                 (30000, 0.0),
#                 ]

# epsilon_dict = [(0, 0.8),
#                 (3750, 0.7),
#                 (7500, 0.6),
#                 (11250, 0.5),
#                 (15000, 0.4),
#                 (18750, 0.3),
#                 (22500, 0.2),
#                 (26250, 0.1),
#                 (30000, 0.0)
#                 ]

# epsilon_dict = [(25, 0.7),
#                 (50, 0.6),
#                 (75, 0.5),
#                 (100, 0.4),
#                 (125, 0.3),
#                 (150, 0.2),
#                 (175, 0.1),
#                 (200, 0.0)
#                 ]

dqn = DeepQNetwork(data_adapter=data_adapter,
                   input_image_size=(240, 240),
                   clazz_dim=5,
                   feature_extraction_network=ResNet(),
                   anim_recorder=recorder,
                   # Custom
                   # epsilon_dict=epsilon_dict,
                   # batch_size=2,
                   # track_len=5,
                   batch_size=2,
                   track_len=8,
                   # batch_size=4,
                   # batch_size=8,
                   learning_rate=1e-4,
                   # learning_rate=4e-5,
                   # learning_rate=10e-6,
                   # replay_memory_size=1000,
                   # replay_memory_size=2000,
                   replay_memory_size=20,
                   # replay_memory_size=30,
                   # replay_memory_size=50,
                   save_per_step=1000,
                   # save_per_step=100,
                   # save_per_step=10000,
                   # replay_period=100,
                   replay_period=2,
                   # replay_period=50,
                   # replay_period=20,
                   breakpoint_dir='./tmp/breakpoint',
                   params_dir='./tmp/ImageSegmentation/',
                   summary_dir='./tmp/Summary/',
                   log_dir='./tmp/Logs/whole-log.txt',
                   use_all_masks=False,
                   # use_all_masks=True,
                   # prioritized_replay=False,
                   prioritized_replay=True,
                   double_q=True)



# dqn.train(5, max_turns=260, restore_from_bp=False)
# dqn.train(1, max_turns=5*260, restore_from_bp=False)

dqn.train(1, max_turns=6*260, restore_from_bp=False)

# dqn.train(1, max_turns=5*260, restore_from_bp=False)


dqn.test(10)

dqn.test(110, real_test=True)



#
#   1. 把 学习率 改为了 10000 次下降 0.5
#   2. 把 正则化系数 改成了 1e-7 ， 并包含了 kernel 和 bias
#   3. 把 ResNet 换成了 slim 里的结构， 把 transition layer 改成了 二层的结构
#   4. 把 ResNet 里的 BN 全都换成 true
#   5. 把 BN 改成了 0.9 并去掉了 renorm
#   6. (未添加--就是还原BN的顺序这样，没加)

