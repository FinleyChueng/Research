import os

from dataset.adapter.bratsAdapter import *
# from task.priorityDQNmodel import *
from util.visualization import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



# Just Test -------
print('Begin')
from task.model import *
import tfmodule.util as netutil
import os
config_file = '/FocusDQN/config.ini'
config_file = os.path.abspath(os.path.dirname(os.getcwd())) + config_file
config = conf_util.parse_config(config_file)
dqn = DqnAgent(name_space='TEST', config=config)
dqn.definition()
netutil.show_all_variables()
# netutil.count_flops()
print('End')


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

# from net.vggnet import *
# from net.resnet import *
# from net.densenet import *
#
#
# agent = DqnAgent(
#     input_image_size=(240, 240),
#     action_dim=8,
#     clazz_dim=5,
#     train_batch_size=32,
#     train_track_len=8,
#     use_all_masks=False,
#     # feature_extraction_network=Vgg16()
#     feature_extraction_network=ResNet()
# )
#
# # infer_dict, loss_dict, summary_dict = agent.definition(dqn_name_scope_pair=None,
# infer_dict, loss_dict, summary_dict = agent.definition(dqn_name_scope_pair=('ORG', 'TAR'),
#                                                        prioritized_replay=True,
#                                                        dueling_network=True,
#                                                        fuse_RF_feats=False
#                                                        )
#
# vnum = 0
# train_list = tf.trainable_variables()
# for var in train_list:
#     print(var)
#     dim = 1
#     for d in var.shape:
#         dim *= int(d)
#     vnum += dim
# print('The total number of variables: {}'.format(vnum))

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

