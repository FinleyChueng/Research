import collections
import random
import sys
import time

import tensorflow as tf
from keras.utils import to_categorical
from task.segmodel import DqnAgent

from core.dqn import *
from task.env import CalDQNEnv


# --------- Version Info  ---------
# Date 18/04/02: 1) Add "Prioritized Replay" into DQN, including logic and relative data structure.
#                       The data structure is actually a prioritized deque (implemented by the small
#                       top heap from "heapq" module).
# Date 18/04/05: 1) Rectify the "Prioritized Replay" logic, the calculation of priority should use
#                       the "Loss" of each epoch other than the "Reward".
#                2) Completing the whole "Prioritized Replay" logic, including the adjust to the
#                       gradient function.
# Date 18/04/22: 1) Rectify the logic of "Update Priority" of "Prioritized Pool".
#                2) Restore the training procedure of "Prioritized Replay" to previous version,
#                       coz the result of gradients computation of "weights(n) * inputs(n)" is far
#                       from the result of gradients accumulation of n "weight * gradient" due to
#                       the "Accuracy Error" of float operation.
# Date 18/04/24: 1) Rectify the logic of gradients (loss) computation in "Prioritized Replay".
# Date 18/04/26: 1) Rectify the whole implementation of "Prioritized Replay".
# Date 18/04/27: 1) Add the logic for translating the origin observation of "Image Segmentation"
#                       into DQN network input, especially the size translation.
# ---------------------------------

# just used for trying.

class DeepQNetwork(DQN):

    def __init__(self,
                 data_adapter,
                 input_image_size,
                 clazz_dim,
                 feature_extraction_network,
                 anim_recorder,
                 regularized_coef=0.3,
                 learning_rate=10e-6,
                 gamma=0.9,
                 replay_memory_size=10000,
                 replay_period=1,
                 batch_size=32,
                 track_len=8,
                 epsilon_dict=None,
                 breakpoint_dir=None,
                 params_dir=None,
                 summary_dir=None,
                 save_per_step=1000,
                 test_per_epoch=100,
                 num_testes=10,
                 use_all_masks=True,
                 double_q=False,
                 prioritized_replay=False,
                 dueling_network=False,
                 log_level=logging.INFO,
                 log_dir=None,
                 ):
        '''Q-Learning algorithm
        Args:
            model:                q funtion
            env:                  environment
            optimizer:            Tensorflow optimizer
            learning_rate:        learning rate
            gamma:                decay factor of future reward
            replay_memory_size:   replay memory size (Experience Replay)
            batch_size:           batch size for every train step
            initial_epsilon:      ε-greedy exploration's initial ε
            final_epsilon:        ε-greedy exploration's final ε
            decay_factor:         ε-greedy exploration's decay factor
            explore_policy:       explore policy, default is `lambda epsilon: random.randint(0, self.num_actions - 1)`
            logdir：              dir to save model
            save_per_step:        save per step
            test_per_epoch:       test per epoch
        '''

        DQN.__init__(self, log_level, log_dir)

        # Hold the data adapter.
        self._DA_holder = data_adapter
        self._Ins_holder = input_image_size
        self._UAM_flag = use_all_masks

        # The valid class dimension.
        self._clazz_dim = clazz_dim

        # The action and direction dimension of model.
        model_acts_dim = 8

        # Define the model. (Mainly initialize the variable here.)
        self._model = DqnAgent(input_image_size=input_image_size,
                               action_dim=model_acts_dim,
                               clazz_dim=clazz_dim,
                               train_batch_size=batch_size,
                               train_track_len=track_len,
                               feature_extraction_network=feature_extraction_network,
                               use_all_masks=use_all_masks,
                               # enable_regularization=False
                               enable_regularization=True
                               )

        self._act_num = model_acts_dim
        self._mask_num = self._model.mask_dim

        self._gamma = gamma
        self._epsilon = None
        self._epsilon_dict = epsilon_dict

        self._regularized_coef = regularized_coef   # 网络正则化系数

        self._bpt_dir = breakpoint_dir
        self._params_dir = params_dir
        self._summary_dir = summary_dir
        self._should_save_model = params_dir is not None
        self._test_per_epoch = test_per_epoch
        self._num_testes = num_testes
        self._save_per_step = save_per_step

        # Define a unified optimizer.
        self._global_step = tf.Variable(0, trainable=False, name='global_step')
        share_learning_rate = tf.train.exponential_decay(learning_rate, self._global_step, 10000, 0.5, staircase=True)
        # share_learning_rate = tf.train.exponential_decay(learning_rate, self._global_step, 14000, 0.5, staircase=True)
        self._optimizer = tf.train.AdamOptimizer(share_learning_rate)

        # Specify the replay memory.
        if prioritized_replay is False:
            replay_memory = collections.deque()
        else:
            replay_memory = PrioritizedPool(replay_memory_size)
        self._replay_memory = replay_memory

        # Double DQN
        if double_q:
            self._DDQN_name_pair = ('ORG', 'TAR')
        else:
            self._DDQN_name_pair = None
        # Prioritized Replay
        self._prioritized_replay = prioritized_replay
        # Dueling Network
        self._dueling_network = dueling_network

        self._replay_period = replay_period
        self._replay_memory_size = replay_memory_size
        self._batch_size = batch_size
        self._track_len = track_len

        # # reward of every epoch
        # self._rewards = []

        # # session
        # config = tf.ConfigProto()
        # # 最多占gpu资源的70%
        # config.gpu_options.per_process_gpu_memory_fraction=0.7
        # # 开始不会给tensorflow全部gpu资源 而是按需增加
        # config.gpu_options.allow_growth = True
        # self._sess = tf.InteractiveSession(config=config)

        # session
        self._sess = tf.InteractiveSession()
        # # Use the @Util{tfdebug}...
        # from tensorflow.python import debug as tf_debug
        # # 使用tf_debug的wrapper来包裹原来的session，使得启动后能进入
        # #   CLI调试界面.
        # self._sess = tf_debug.LocalCLIDebugWrapperSession(self._sess)
        # # 添加过滤器，用于监测 NAN 和 INF
        # self._sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Define the whole model.
        self._define_whole_model()

        # The feature extract holders. It's actually the input and
        #   output holder of "Feature Extract" Network.
        FE_holders = (self._sess,
                      self._tp_flag,
                      self._img,
                      self._fe_out,
                      list(self._feats_dict.values()))

        # The segmentation holders which will pass to environment.
        #   It's actually the "Up-sample" Network. Including session.
        segmentation_holders = (self._sess,
                                self._tp_flag,
                                list(self._feats_dict.values()),
                                self._fe_out,
                                self._un_sm,
                                self._up_fet,
                                self._un_out)

        # Define the environment.
        self._env = CalDQNEnv(data_adapter=data_adapter,
                              feats_ext_util=FE_holders,
                              segment_util=segmentation_holders,
                              cls_dim=clazz_dim,
                              proc_imgsize=input_image_size,
                              use_all_masks=use_all_masks,
                              anim_recorder=anim_recorder
                              )

        # Check validity.
        env_acts_dim = self._env.acts_dim
        assert model_acts_dim == env_acts_dim
        assert self._mask_num == self._env.mask_dim

        # Simply assignment.
        self._test_env = self._env
        # # The test environment.
        # self._test_env = copy.deepcopy(env)

        # check saved model
        self._check_model()

        # # finalize the graph
        # self._sess.graph.finalize()

    def _check_model(self):
        r'''
            Check whether to load model or not.

        Parameters:
            save_per_step: Indicates the frequence of save operation.
        '''

        if self._params_dir is not None:
            if not self._params_dir.endswith('/'): self._params_dir += '/'
            self._saver = tf.train.Saver()
            checkpoint_state = tf.train.get_checkpoint_state(self._params_dir)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                path = checkpoint_state.model_checkpoint_path
                self._saver.restore(self._sess, path)
                print('Restore from {} successfully.'.format(path))
            else:
                print('No checkpoint.')
                self._sess.run(tf.global_variables_initializer())
            # self._summaries = tf.summary.merge_all()
            # self._summary_writer = tf.summary.FileWriter(
            #     self._summary_dir, self._sess.graph)
            sys.stdout.flush()
        else:
            self._sess.run(tf.global_variables_initializer())
            pass
        # Finish checking ...
        return

    def _define_whole_model(self):
        r'''
            The definition of whole model.

        ----------------------------------------------------------------------------
        The Reference of dictionary elements.

        model_ios_dict = {
            # public holders.
            'Training_phrase': self._train_phrase,
            'FEATURE_Stride': self._feature_stride,
            'image': self._image,
            'FEN/output': self._fe_output,
            'FEN/feats_dict': feats_dict,
            # DRQN holders.
            'DRQN/input': self._drqn_in,
            'DRQN/GRU_sin': self._drqn_gru_Sin,
            'DRQN/GRU_sout': self._drqn_gru_state,
            'DRQN/output': self._drqn_output,
            # UN holders.
            'UN/select_mask': self._selective_mask,
            'UN/opt_conv': self._opt_conv,
            'UN/up_fet': upfet_trans,
            'UN/output': self._un_output,
            # CC holders.
            'CC/ohact_vec': self._cc_ohact_vec,
            'CC/RNN_sin': self._cc_rnn_Sin,
            'CC/RNN_sout': self._cc_rnn_state,
            'CC/output': self._cc_output
        }

        model_loss_dict = {
            # UN-related.
            'UN/GT_segmentation': self._GT_segmentation,
            'UN/loss': self._un_loss,
            # CC-related.
            'CC/label': self._CC_label
            'CC/loss': self._cc_loss
            # DRQN-related.
            'DRQN/target_input': self._target_drqn_in,
            'DRQN/target_GRU_sin': self._target_drqn_gru_Sin,
            'DRQN/target_output': self._target_drqn_output,
            'DRQN/input_act': self._drqn_input_act,
            'DRQN/target_q_val': self._drqn_target_q_val,
            'DRQN/exp_priority': self._exp_priority,
            'DRQN/ISWeights': self._ISWeights,
            'DRQN/loss': self._drqn_loss,
            # Whole-related.
            'NET/Whole_loss': self._whole_loss,
        }

        model_summaries_dict = {
            'DQN/statis_rewards': self._rewards,
            'Net/summaries': self._summaries
        }
        '''

        # # The global step. Used to save model and logging.
        # self._global_step = tf.Variable(0, trainable=False, name='global_step')

        # Define the whole model.
        infer_dict, loss_dict, summary_dict = self._model.definition(
            dqn_name_scope_pair=self._DDQN_name_pair,
            prioritized_replay=self._prioritized_replay,
            dueling_network=self._dueling_network,
            fuse_RF_feats=False
        )

        ######################## Start to get the inference holders ########################
        # Public holders.
        self._tp_flag = infer_dict['Training_phrase']
        self._feature_stride = infer_dict['FEATURE_Stride']
        self._img = infer_dict['image']
        self._fe_out = infer_dict['FEN/output']
        self._feats_dict = infer_dict['FEN/feats_dict']
        # DRQN holders.
        self._drqn_in = infer_dict['DRQN/input']
        self._drqn_gsi = infer_dict['DRQN/GRU_sin']
        self._drqn_gso = infer_dict['DRQN/GRU_sout']
        self._drqn_out = infer_dict['DRQN/output']
        # UN holders.
        self._un_sm = infer_dict['UN/select_mask']
        self._up_fet = infer_dict['UN/up_fet']
        self._un_out = infer_dict['UN/output']

        ######################## Start to get the loss holders. ########################
        # UN-related.
        self._un_cw = loss_dict['UN/clazz_weights']
        self._vf = loss_dict['UN/visit_flag']
        self._un_gt_seg = loss_dict['UN/GT_segmentation']
        un_loss = loss_dict['UN/loss']
        # DRQN-related.
        self._drqn_cur_in_act = loss_dict['DRQN/input_act']
        self._drqn_cur_tar_qval = loss_dict['DRQN/target_q_val']
        drqn_loss = loss_dict['DRQN/loss']
        # Get the target DQN holders of duplicate model if enable the "Double DQN".
        if self._DDQN_name_pair is not None:
            self._tar_img = loss_dict['DRQN/target_image']
            self._tar_drqn_in = loss_dict['DRQN/target_input']
            self._tar_drqn_gsi = loss_dict['DRQN/target_GRU_sin']
            self._tar_drqn_out = loss_dict['DRQN/target_output']
        # Get prioritized replay holders for DQN if enabled.
        if self._prioritized_replay:
            self._exp_pri = loss_dict['DRQN/exp_priority']
            self._ISweights = loss_dict['DRQN/ISWeights']
        # Whole-related.
        self._whole_loss = loss_dict['NET/Whole_loss']

        ######################## Start to get the summary holders. ########################
        # The summary for model.
        self._statis_rewards = summary_dict['DRQN/statis_rewards']
        self._statis_dice = summary_dict['NET/DICE']
        self._statis_brats = summary_dict['NET/BRATS']
        self._summary = summary_dict['NET/summaries']

        ######################## Start to define the training holders. ########################
        # Train the whole network end-to-end.
        #   So directly use the whole loss to train.
        self._e2e_train_op = self._optimizer.minimize(self._whole_loss, global_step=self._global_step)
        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(update_ops)
        # update_ops = [elem for elem in update_ops if self._DDQN_name_pair[0] in elem.name or 'UN' in elem.name]
        # print(update_ops)
        # with tf.control_dependencies(update_ops):
        #     self._e2e_train_op = self._optimizer.minimize(self._whole_loss, global_step=self._global_step)

        ######################## Start to define the summary writer. ########################
        # The summary writter for whole model.
        self._summary_writer = tf.summary.FileWriter(
            self._summary_dir, self._sess.graph)

        # Finish the definition of whole model.
        return


    # num_epochs: train epochs
    def train(self, num_epochs, max_turns=260, restore_from_bp=False):
        r'''
            The "Training" phrase. Train the whole model many times.

        Parameters:
            num_epochs: Indicates the whole epochs that whole model should train.
            max_turns: Specify the max turns for "DQN Agent".
            restore_from_bp: Whether to start from breakpoint.
        '''

        # Whether end-to-end train or not.
        print('End-to-End Training mode !!!')
        # Train the whole model many times. (Depend on epochs)
        for epoch in range(num_epochs):
            # Start training.
            self.__e2e_train(max_turns, restore_from_bp)
            print('Finish the epoch {}'.format(epoch))

        # Finish the whole training phrase.
        return


    def test(self, turn, real_test=False, max_step_per_test=100000):

        # show ------
        self._logger.info('Testing...')
        # -----------
        sys.stdout.flush()

        # Metric used to record total rewards of "Maze" environment.
        total_rewards = []
        # Dice Metric.
        dice_metrics = []
        # BRATS Metric.
        brats_metrics = []

        # Start to Testing ...
        for t in range(turn):
            # Metric used to record rewards of current turn if not None.
            turn_rewards = []
            turn_dice = []
            turn_brats = []

            # Reset the "Maze" environment.
            state, info = self._test_env.reset(arg=None, is_training=False, test_flag=real_test)

            # Reset (Initialize) the recurrent layer's hidden state (GRU).
            #   Note that, we will update it (Iteratively) when in inference
            #   phrase. But always set to 0 when training.
            h_size = self._model.gru_hsize
            gru_S = np.zeros([1, h_size], dtype=np.float32)

            # Debug metric used to record the cost time of current turn.
            start_time = time.time()

            # 9999999999: max step per epoch
            for step in range(max_step_per_test):
                # Use the model (DRQN) to predict the action index for current state.
                action_index, gru_Sout = self._action(state, arg=(gru_S, ), test_flag=(True, step))
                # Push action index into environment, so that we can get the next state and reward.
                next_state, reward, terminal, info = self._test_env.step(action_index, arg=t)

                # ### DEBUG
                # print('equeal - gru: ', (gru_S == gru_Sout).all())
                # print('equeal - state: ', (state[0] == next_state[0]).all())
                # ### DEBUG

                # Change current state to next state. (Iteratively)
                state = next_state
                # What's more, re-assign the GRU hidden state.
                gru_S = gru_Sout

                # ### DEBUG
                # print("Test- gru S : ", gru_S)
                # ### DEBUG

                # Unpack information.
                if not real_test:
                    cls_dice, brats_m = info

                # Calculate rewards of current turn of "Recursive" if it is not None.
                if reward is not None:
                    # turn_rewards.append(np.mean(reward))
                    turn_rewards.append(reward)
                    turn_dice.append(cls_dice)
                    turn_brats.append(brats_m)

                # Terminated. One turn ended (Process of current image is finished yet).
                if terminal:
                    # Show the value. --------------------------------------------
                    self._logger.info("Test-Turn {} --> total_reward: {}, DICE: {}, BRATS: {}, Finished !!!".format(
                        t, np.sum(turn_rewards, axis=0), np.mean(turn_dice, axis=0), np.mean(turn_brats, axis=0)))
                    # ------------------------------------------------------------
                    sys.stdout.flush()
                    # Simply break the loop.
                    break

            # Render. - Record the process of current image in "GIF" form.
            if not real_test:
                self._test_env.render()

            # Record the current turn situation into total level.
            total_rewards.append(turn_rewards)
            dice_metrics.append(turn_dice)
            brats_metrics.append(turn_brats)

            print('Test-Turn {} cost time: {}'.format(t, time.time() - start_time))

        # Calculate the average rewards of "Recursive" environment.
        average_reward = np.mean(total_rewards, axis=(0, 1))
        average_dice = np.mean(dice_metrics, axis=(0, 1))
        average_brats = np.mean(brats_metrics, axis=(0, 1))
        # Show. ---------------------------------------
        self._logger.info("Test-epoch {:5} --> average_reward: {}, avg_DICE: {}, avg_BRATS: {}".format(
            turn, average_reward, average_dice, average_brats))
        # ---------------------------------------------
        sys.stdout.flush()

        # Finish Testing. And return the reward info list.
        return total_rewards, dice_metrics, brats_metrics


    def __restore_from_bp(self):
        if self._bpt_dir is not None:
            with open(self._bpt_dir, 'r') as f:
                start_pos = int(f.read())
        else:
            print('Not specify the breakpoint path, can not restore training ...')
            start_pos = 0

        return start_pos


    def __e2e_train(self, images_per_epoch, restore_from_bp=False):
        r'''
            End-to-end train the whole model.

        :param images_per_epoch:
        :return:
        '''

        print('-----> Start the "End-to-End" training policy !!!')

        # Restore from the last time position if enabled.
        if restore_from_bp:
            start_pos = self.__restore_from_bp()
        else:
            start_pos = 0

        # generate epsilon dictionary according to the training epochs if
        #   no specify the dictionary.
        if self._epsilon_dict is None and images_per_epoch != 0:
            nte = images_per_epoch
            decay_rate = max(1, nte // 10)  # Avoid die-loop.
            e = 0.0
            dict = []
            while not nte < 0:
                dict.append((nte, e))
                nte -= decay_rate
                e += 0.1
            dict.reverse()
            self._epsilon_dict = dict
        print(self._epsilon_dict)

        # Total steps count used to control whether to training or not.
        total_steps = 0
        # Remainder of last difference of total steps.
        ts_diff_remainder = 0

        # Start to training ...
        for turn in range(start_pos, images_per_epoch):
            # Used to record total rewards for current turn.
            turn_rewards = 0

            # Reset the "Maze" environment (by pass through the proposals).
            state, (image, label, select_mask, cls_weights) = self._env.reset(arg=None)
            # The experience trajectory list of current processing image.
            exp_traj = []

            # Reset (Initialize) the recurrent layer's hidden state (GRU).
            #   Note that, we will update it (Iteratively) when in inference
            #   phrase. But always set to 0 when training.
            h_size = self._model.gru_hsize
            gru_S = np.zeros([1, h_size])

            # Debug metric used to record the cost time of current turn.
            start_time = time.time()
            # Current total steps, which is used to control the training time.
            cur_tosteps = total_steps

            # 9999999999: max step per epoch
            for step in range(9999999999):
                # ε-greedy exploration
                action_index, gru_Sout = self._egreedy_action(state, turn=turn, arg=(gru_S, ))
                # Push action index into environment, so that we can get the next state and reward.
                next_state, reward, terminal, (next_image, next_label, next_SM, next_CW) = self._env.step(
                    action_index
                )
                # Generate one-hot vector for action index, which is lately used to calculate loss.
                one_hot_action = to_categorical(action_index, num_classes=self._act_num)    # [cls, act_dim]
                # one_hot_action = np.zeros(self._act_num)
                # one_hot_action[action_index] = 1

                ### DEBUG
                # print(reward)
                ### DEBUG

                # Experience.
                exp = (state,
                       one_hot_action,
                       reward,
                       next_state,
                       terminal,
                       image,
                       label,
                       select_mask,
                       cls_weights,
                       [0]   # visited count
                       )

                # Add current experience into trajectory.
                exp_traj.append(exp)

                # What's more, re-assign the GRU hidden state.
                gru_S = gru_Sout

                # Change current state to next state. (Iteratively)
                state = next_state
                image = next_image  # change to next processing image.
                label = next_label  # change to next label.
                select_mask = next_SM   # change to next select mask.
                cls_weights = next_CW   # change to next clazz weights.

                # Calculate rewards of current turn of "Recursive" environment.
                turn_rewards += np.mean(reward)     # turn_rewards += reward
                # Auto-increase the total steps count.
                total_steps += 1

                # Terminated. One turn ended (Process of current image is finished yet).
                if terminal:
                    # Only store experience trajectory when its size greater than
                    #   pre-defined track length.
                    if len(exp_traj) > self._track_len:
                        # Package the experience trajectory with
                        #   1) FEN (deep) feature maps.
                        #   2) raw image
                        #   3) its label
                        experience = (exp_traj, )
                        # Store the transition (experience) in "Replay Memory".
                        if self._prioritized_replay is False:
                            self._replay_memory.append(experience)
                            # Remove element (experience) if exceeds max size.
                            if len(self._replay_memory) > self._replay_memory_size:
                                self._replay_memory.popleft()
                        else:
                            # Using the "Prioritized Replay".
                            self._replay_memory.store(experience)

                    # Show the value. --------------------------------------------
                    self._logger.debug("Turn {} --> total_rewards: {}, epsilon: {}, ".format(
                        turn, turn_rewards, self._epsilon))
                    # ------------------------------------------------------------
                    sys.stdout.flush()

                    # Simply break the loop.
                    break

            # Render. - Record the process of current image in "GIF" form.
            self._env.render()

            # Check whether to training the DQN agent now or not.
            exec_train = len(self._replay_memory) > 0       # self._batch_size
            # Start training the DQN agent.
            if exec_train:
                # Calculate the training times.
                tc_denominator = total_steps - cur_tosteps + ts_diff_remainder
                train_count = tc_denominator // self._replay_period
                ts_diff_remainder = tc_denominator % self._replay_period
                # Train the model many times.
                for tc in range(train_count):
                    # Metric used to see the training time.
                    train_start_time = time.time()
                    # Really training the DQN agent.
                    v_cost = self._do_train_net(turn=turn,
                                                max_turns=images_per_epoch)
                    # Debug. ------------------------------------------------------
                    self._logger.info("Turn {} - Train Count {}, Loss: {}, Training time: {}".format(
                        turn, tc, v_cost, time.time() - train_start_time)
                    )
                    # -------------------------------------------------------------

            self._logger.debug('Image {} cost time: {}'.format(turn, time.time() - start_time))

        # Finish the end-to-end component training.
        return


    # The real training code.
    def _do_train_net(self, turn, max_turns):
        r'''
            The method used to really execute the one-turn DQN training.

            The single exp-cell consists of five part elements:
                (state, one_hot_action, reward, next_state, terminal, direction, select_mask)
            The experience consists of: (exp_cell_list, )
            That is, the experience just likes below:
                ([(s, a, r, ns, t, i, l, sm, cw, vf), ..., (s, a, r, ns, t, i, l, sm, cw, vf)], )

            state: The current state of "Maze" environment, which consists of:
                (features_maps, )

        Parameters:
            turn: The current turn of training phrase.
            max_turns: The max turn of whole training phrase.
        '''

        sample_time = time.time()     # Debug

        # The tree idx batch and ISweights batch used to update the priority if
        #   enable the "Prioritized Replay".
        if self._prioritized_replay:
            tree_idx = []
            ISWeights = []

        # The train-related batch.
        # -----------------------------------------------
        # Split into different batches for state.
        FEN_feats_batch = []
        # Split into different batches for next state.
        next_FEN_feats_batch = []
        # Action batch holder.
        act_batch = []
        # Reward batch holder.
        reward_batch = []
        # Terminal batch holder.
        terminal_batch = []
        # Raw image batch holder.
        image_batch = []
        # Label batch holder.
        label_batch = []
        # Select mask batch holder.
        selectM_batch = []
        # Class weights batch holder.
        clsWeights_batch = []
        # Visited flag.
        visitFlag_batch = []

        # Sample experiences from the replay memory of specified batch size.
        for _ in range(self._batch_size):
            if self._prioritized_replay:
                # Prioritized replay. So the we will get three batches: tree_idx, mini_batches, ISWeights.
                tid, experience, isweight = self._replay_memory.sample(1)
                tid, experience, isweight = tid[0], experience[0], isweight[0]
                # Add the tree_idx and ISweights into batch, respectively.
                tree_idx.append(tid)
                ISWeights.append(isweight)
            else:
                # Randomly select a experience (uniform replay)
                experience = random.sample(self._replay_memory, 1)[0]

            # Divide experience into different element.
            exp_traj, = experience

            # Random select a time-step as the begin time.
            rb_tstep = np.random.randint(len(exp_traj) - self._track_len)   # scalar
            # Get the selected fixed-len trajectory.
            rselect_exps = exp_traj[rb_tstep: rb_tstep + self._track_len]   # list. len: track_len

            # Package to time-related list.
            feats00 = []
            nfeats30 = []
            act1 = []
            rew2 = []
            img5 = []
            lab6 = []
            sm7 = []
            cw8 = []
            ter4 = []
            vf9 = []
            for rs_exp in rselect_exps:
                feats00.append(rs_exp[0][0])
                nfeats30.append(rs_exp[3][0])
                act1.append(rs_exp[1])
                rew2.append(rs_exp[2])
                img5.append(rs_exp[5])
                lab6.append(rs_exp[6])
                sm7.append(rs_exp[7])
                cw8.append(rs_exp[8])
                ter4.append(rs_exp[4])
                # Meanwhile reset the flag.
                rs_exp[9][0] += 1
                vf9.append(rs_exp[9].copy()[0])

            # Pack into the State batch.
            FEN_feats_batch.extend(feats00)
            # Pack into the next neighbours and modalities (State) batch.
            next_FEN_feats_batch.extend(nfeats30)
            # Pack into the action batch.
            act_batch.append(act1)
            # Pack into the reward batch.
            reward_batch.append(rew2)
            # Pack into the terminal batch.
            terminal_batch.append(ter4)
            # Pack into the visit flag batch.
            visitFlag_batch.append(vf9)

            # Pack into the raw image batch.
            image_batch.extend(img5)
            # Pack into the segmentation label batch.
            label_batch.append(lab6)
            # Pack into the select mask batch.
            selectM_batch.extend(sm7)
            # Pack into the class weights batch.
            clsWeights_batch.append(cw8)
        # ------------------------- Finish Split Batch -------------------

        # Union the State batch and Next-State batch.
        state_batch = (
            FEN_feats_batch,
        )
        next_state_batch = (
            next_FEN_feats_batch,
        )

        # The holder used to store the target Q values.
        target_q_values = []

        # Calculate the next Q values by passing the next states batch into DQN agent.
        next_q_values = self._next_Qvalues_function(next_state_batch, image_batch=image_batch)  # [b, t]

        # Iteratively calculate the target Q values.
        for t_ter, t_rew, t_nqv in zip(terminal_batch, reward_batch, next_q_values):
            track_tar_qvals = []
            for terminal, reward, next_qval in zip(t_ter, t_rew, t_nqv):
                # The target Q values is depend on whether terminated or not.
                if terminal:
                    # Directly add the reward value if it's terminated.
                    track_tar_qvals.append(reward)
                else:
                    # Discounted future reward (plus reward) if not terminated.
                    track_tar_qvals.append(reward + self._gamma * next_qval)
            # Add into batch level.
            target_q_values.append(track_tar_qvals)

        # Debug. --------------------------------------------------------
        self._logger.debug("The sample and package time cost: {}".format(time.time() - sample_time))  # DEBUG
        # ---------------------------------------------------------------

        bp_time = time.time()   # DEBUG

        # Always set LSTM initial state to 0 when training.
        h_size = self._model.gru_hsize
        gru_Si = np.zeros([self._batch_size, h_size])

        # Generate the basic feed dictionary lately used for training the DQN agent.
        feed_dict = {
            # The public.
            self._tp_flag: True,
            self._img: image_batch,
            # The initial hidden state for GRU in DRQN.
            self._drqn_gsi: gru_Si,
            # The input action and target Q values. For DRQN.
            self._drqn_cur_in_act: act_batch,
            self._drqn_cur_tar_qval: target_q_values,
            # # The select mask and GT segmentation. For UN
            self._un_sm: selectM_batch,
            self._un_gt_seg: label_batch,
            self._un_cw: clsWeights_batch,
            self._vf: visitFlag_batch
        }

        # Really training the DQN agent by passing the state, action and target Q values into
        #   train operation of DQN agent according to the "Experience Replay" policy.
        if self._prioritized_replay:
            # Used to update priority. Coz the "Prioritized Replay" need to update the
            #   priority for experience.
            feed_dict[self._ISweights] = ISWeights
            # Execute the training operation.
            v_cost, _, exp_priority = self._sess.run([self._whole_loss, self._e2e_train_op, self._exp_pri],
                                                     feed_dict=feed_dict)
            # Merge the ISweights of same tree_idx.
            ul_dict = {}
            for ul_tid, ul_ep in zip(tree_idx, exp_priority):
                if ul_tid not in ul_dict.keys():
                    ul_dict[ul_tid] = [ul_ep]
                else:
                    ul_dict[ul_tid].append(ul_ep)
            tree_idx = []
            exp_priority = []
            for ul_key in ul_dict.keys():
                tree_idx.append(ul_key)
                exp_priority.append(np.mean(ul_dict[ul_key]))
            # Convert to Numpy.ndarray.
            tree_idx = np.asarray(tree_idx)
            exp_priority = np.asarray(exp_priority)
            # update probability for leafs.
            self._replay_memory.batch_update(tree_idx, exp_priority)  # update priority
        else:
            # Simply feed the dictionary to the selected operation is OK.
            v_cost, _ = self._sess.run([self._whole_loss, self._e2e_train_op],
                                       feed_dict=feed_dict)

        self._logger.debug("The BP time cost: {}".format(time.time() - bp_time))     # DEBUG

        # Get the global step lately used to determine whether to save model or not.
        step = self._global_step.eval()
        # Calculate the summary to get the statistic graph.
        if step > 0 and step % self._save_per_step == 0:
        # if step > 0 and step % 100 == 0:
        # if step > 0 and step % 10 == 0:
            # Execute test environment to get the reward situation of current DQN model.
            # cur_reward_list = self.test(10)
            cur_reward_list, cur_DICE_list, cur_BRATS_list = self.test(2)
            # Compute the summary value and add into statistic graph.
            feed_dict[self._statis_rewards] = cur_reward_list
            feed_dict[self._statis_dice] = cur_DICE_list
            feed_dict[self._statis_brats] = cur_BRATS_list
            summary = self._sess.run(self._summary, feed_dict=feed_dict)
            self._summary_writer.add_summary(summary, step)
            self._summary_writer.flush()
        # Save the model (parameters) within the fix period.
        if self._should_save_model and step > 0 and step % self._save_per_step == 0:
            # Save the model.
            self._saver.save(self._sess, self._params_dir, 233)
            # Record the breakpoint in order to support the "Continue Training".
            with open(self._bpt_dir, 'w') as f:
                if turn == max_turns - 1:
                    # Reset the start position to 0.
                    f.write('0')
                else:
                    f.write(str(turn))
        # Regularly copy the parameters to "Target" DQN network.
        if self._DDQN_name_pair is not None and step % self._save_per_step == 0:
            # Notify the network to copy the parameters from "Origin" to "Target".
            self._model.notify_copy2_DDQN(self._sess, only_head=False)

        # Finish one turn train. Return the cost value.
        return v_cost

    def _explore_policy(self, distribution='uniform'):
        r'''
            Randomly select an action.

        :param distribution:
        :return:
        '''

        return np.random.randint(self._act_num, size=self._mask_num, dtype=np.int64)
        # return np.random.randint(self._act_num)

    def _egreedy_action(self, state, turn, arg=None):
        r'''
            The ε-greedy exploration for DQN agent.

        Parameters:
            state: The current state of environment.
            turn: The turn index, which used to get the value of epsilon.
            arg: Additional argument. Here includes:
                    1) the deep feature maps produced by FEN.
                    2) the GRU hidden state.

        Return:
            The action index produced by "Random" or DRQN agent. Meanwhile return
                the LSTM hidden state of next time-step.
        '''

        # Get current epsilon.
        for idx in range(len(self._epsilon_dict) - 1):
            lower = self._epsilon_dict[idx][0]
            upper = self._epsilon_dict[idx+1][0]
            if turn in range(lower, upper):
                self._epsilon = self._epsilon_dict[idx][1]
                break

        # Exploration or Exploitation according to the epsilon.
        if random.random() <= self._epsilon:
            # Exploration.
            action_index = self._explore_policy()
            # Coz we do not execute the DRQN, the hidden state is
            #   not changed. Return the raw GRU hidden state is OK.
            gru_Sout, = arg
        else:
            # Exploitation. Use network (DRQN) to predict action index.
            action_index, gru_Sout = self._action(state, arg)

        # Return the action index. Meanwhile return the
        #   GRU hidden state of next time-step.
        return action_index, gru_Sout

    def _action(self, state, arg, test_flag=(False, -1)):
        r'''
            Let DQN select action for current state, that is, select the action with max Q value.

            Moreover, this method is only called in "Reference" phrase.

        Parameters:
            state: The current state of "Recursive" environment.
                The state of "Recursive" consists of:
                    (feature_maps, )
                    The deep feature maps produced by FEN, which the DRQN interacts with.
            arg: Additional arguments. Here includes:
                    1) gru_hstate: The GRU hidden state.

        Return:
            The action index with max Q value. Meanwhile return the LSTM hidden state of
                next time-step.
        '''

        # Destruct the detail state elements from state.
        feature_maps, = state

        # Get the GRU hidden state.
        gru_Si, = arg

        # Calculate the Q values of each action within current state.
        drqn_q_val, drqn_lstm_sout = self._sess.run(
            [self._drqn_out, self._drqn_gso],
            feed_dict={
                self._tp_flag: False,
                self._fe_out: [feature_maps],
                self._drqn_gsi: gru_Si
            })

        # Unpack the wrapper.
        drqn_q_val = drqn_q_val[0, 0]
        lstm_Sout = drqn_lstm_sout

        # Select action according to the output of "Deep Recurrent Q Network".
        action = np.argmax(drqn_q_val, axis=-1)      # [mask]

        # # Debug
        # print('PN probability:', pn_probs)
        # print('PN fake Qvals:', pn_fqvals)
        # print('FTN Qvals:', ftn_qvals)
        # ri, ci = self._sess.run([self._model.debug_raw_input, self._model.debug_crop_input],
        #                         feed_dict={
        #                             self._img: [cur_image],
        #                             self._rois: [neighbour_rois],
        #                             self._pi: [position_indicator],
        #                             self._th: [tackle_history],
        #                             self._pq: [parent_Q_vals],
        #                             self._gs: [grid_shape]
        #                         })
        # import matplotlib.pyplot as plt
        # plt.subplot(231)
        # plt.imshow(ri[0, :, :, 0])
        # plt.subplot(232)
        # plt.imshow(ri[0, :, :, 4])
        # plt.subplot(233)
        # plt.imshow(ri[0, :, :, 5])
        # plt.subplot(234)
        # plt.imshow(ci[0, :, :, 0])
        # plt.subplot(235)
        # plt.imshow(ci[0, :, :, 4])
        # plt.subplot(236)
        # plt.imshow(ci[0, :, :, 5])
        # plt.show()
        # # Debug

        # if test_flag[0] and test_flag[1] % 30 == 0:
        #
        #     print('--- Cls Q ---')
        #     print(drqn_q_val)
        #
        #     raw_q, ca_prob = self._sess.run(
        #         [self._model.DEBUG_drqn_raw_out, self._model.DEBUG_t_CA_prob],
        #         feed_dict={
        #             self._tp_flag: False,
        #             self._fe_out: [feature_maps],
        #             self._drqn_gsi: gru_Si
        #         })
        #     raw_q = raw_q[0, 0]
        #     ca_prob = ca_prob[0, 0]
        #
        #     print('--- Raw Q ---')
        #     print(raw_q)
        #     print('--- CA prob ---')
        #     print(ca_prob)


        # Return the action select by DRQN. Meanwhile return the
        #   LSTM hidden state of next time-step.
        return action, lstm_Sout

    def _next_Qvalues_function(self, state_batch, image_batch):
        r'''
            The enhanced version of @method{self._action()}, it's used to calculate
                next Q values for batch input. Which is used to calculate the
                "Target Q Values".

            Note that, this method is only used in "Training" phrase.

        Parameters:
            state_batch: The state batch. Which is consisted as below:
                state_batch = (
                    FEN_feats_batch,
                )
            feats_batch: The FEN feature maps batch. Which is produced by FEN.

        Return:
            The next Q values batch. Which is used to calculate the "Target Q Values".
        '''

        # Get the different element batch, respectively.
        FEN_feats_batch, = state_batch

        # Computes the Q values of states batch. And use the "Double Q" algorithm
        #   according to the flag.
        if self._DDQN_name_pair is not None:
            # Both run the "Target Network" and "Origin Network" to get the target Q values
            #   and origin Q values. And use the target Q values to predict action, but
            #   use origin Q values to evaluate true values.
            tar_q_vals, org_q_vals = self._sess.run(
                [self._tar_drqn_out, self._drqn_out],
                feed_dict={
                    # # Public.
                    # self._tp_flag: True,
                    # self._fe_out: FEN_feats_batch,
                    # # Target Network.
                    # self._tar_drqn_in: FEN_feats_batch,
                    # # Origin Network.
                    # self._drqn_in: FEN_feats_batch

                    # Public.
                    self._tp_flag: True,
                    # Target Network.
                    self._tar_img: image_batch,
                    # Origin Network.
                    self._drqn_in: FEN_feats_batch
                })  # [b, t, cls, act_dim]

            # Imitate selecting action. Compute the action with max Q value.
            tar_acts = np.argmax(tar_q_vals, axis=-1)   # [b, t, mask]

            # Recover to one-hot form for convenient selecting the target Q value.
            tar_ohacts = to_categorical(tar_acts, num_classes=self._act_num)    # [b, t, mask, act_dim]
            # Multiply and reduce-sum to get the real target Q value.
            next_q_values = np.sum(org_q_vals * tar_ohacts, axis=-1)    # [b, t, mask]

        else:
            # Only run the "Target Network" to get the target Q values. And use it to predict action.
            tar_q_vals = self._sess.run(self._tar_drqn_out, feed_dict={
                # # Public.
                # self._tp_flag: True,
                # self._fe_out: FEN_feats_batch,
                # # Target Network.
                # self._tar_drqn_in: FEN_feats_batch

                # Public.
                self._tp_flag: True,
                # Target Network.
                self._tar_drqn_in: image_batch
            })  # [b, t, mask, act_dim]

            # Use the max Q value as the target Q values.
            next_q_values = np.max(tar_q_vals, axis=-1)     # [b, t, mask]

        # print('Next Q: \n {}'.format(next_q_values))
        # print("Shape: {}".format(next_q_values.shape))

        # note here we do not truncate the q values, coz this function
        #   is only used in training state to compute the target q values.
        return next_q_values


