import collections
import random
import sys
import time

import tensorflow as tf
from keras.utils import to_categorical


from core.dqn import *
from task.model import DqnAgent
from task.env import FocusEnv



class DeepQNetwork(DQN):

    def __init__(self,
                 config,
                 name_space,
                 data_adapter,
                 # input_image_size,
                 # clazz_dim,
                 # feature_extraction_network,
                 # anim_recorder,
                 # regularized_coef=0.3,
                 # learning_rate=10e-6,
                 # gamma=0.9,
                 # replay_memory_size=10000,
                 # replay_period=1,
                 # batch_size=32,
                 # track_len=8,
                 # epsilon_dict=None,
                 # breakpoint_dir=None,
                 # params_dir=None,
                 # summary_dir=None,
                 # save_per_step=1000,
                 # test_per_epoch=100,
                 # num_testes=10,
                 # use_all_masks=True,
                 # double_q=False,
                 # prioritized_replay=False,
                 # dueling_network=False,
                 log_level=logging.INFO,
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


        # Initialize.
        self._config = config
        self._name_space = name_space

        conf_others = self._config['Others']
        log_dir = conf_others.get('log_path')

        DQN.__init__(self, log_level, log_dir)



        # Data adapter.
        self._data_adapter = data_adapter

        # Define the model. (Mainly initialize the variable here.)
        self._model = DqnAgent(self._config, name_space=self._name_space)

        # Define the environment.
        self._env = FocusEnv(self._config, data_adapter=data_adapter)


        # Get config.
        conf_train = self._config['Training']
        learning_rate = conf_train.get('learning_rate')
        decay_iter = conf_train.get('learning_decay_iter')
        decay_rate = conf_train.get('learning_decay_rate')
        conf_dqn = self._config['DQN']
        prioritized_replay = conf_dqn.get('prioritized_replay', True)

        replay_memories = conf_others.get('replay_memories')






        # self._epsilon_book = None



        # self._bpt_dir = breakpoint_dir
        # self._params_dir = params_dir
        # self._summary_dir = summary_dir
        # self._should_save_model = params_dir is not None
        # self._test_per_epoch = test_per_epoch
        # self._num_testes = num_testes
        # self._save_per_step = save_per_step



        # Get the valid action quantity.
        self._acts_dim = self._env.acts_dim

        # Specify the replay memory.
        if prioritized_replay is False:
            replay_memory = collections.deque()
        else:
            replay_memory = PrioritizedPool(replay_memories)
        self._replay_memory = replay_memory

        # Declare the normal storage for segmentation sample.
        self._sample_storage = collections.deque()


        # # Prioritized Replay
        # self._prioritized_replay = prioritized_replay
        # # Dueling Network
        # self._dueling_network = dueling_network

        # self._replay_period = replay_period
        # self._replay_memory_size = replay_memory_size
        # self._batch_size = batch_size

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


        r'''
        The definition of whole model.
        ----------------------------------------------------------------------------
        Inputs holder: dict_keys(['ORG/image', 'ORG/prev_result', 'ORG/Focus_Bbox', 'ORG/position_info', 'ORG/Segment_Stage', 'TEST/Complete_Result'])
        Outputs holder: dict_keys(['ORG/DQN_output', 'TEST/SEG_output', 'TEST/FUSE_result'])
        Losses holder: dict_keys(['TEST/GT_label', 'TEST/clazz_weights', 'TEST/prediction_actions', 'TEST/target_Q_values', 'TEST/EXP_priority', 'TEST/IS_weights', 'TEST/SEG_loss', 'TEST/DQN_loss', 'TEST/NET_loss', 'TAR/image', 'TAR/prev_result', 'TAR/Focus_Bbox', 'TAR/position_info', 'TAR/Segment_Stage', 'TAR/DQN_output'])
        Summary holder: dict_keys(['TEST/Reward', 'TEST/DICE', 'TEST/BRATS_metric', 'TEST/MergeSummary'])
        Visual holder: dict_keys([])
        '''

        # Define the whole model.
        inputs, outputs, losses, summary, visual = self._model.definition()
        self._inputs = inputs
        self._outputs = outputs
        self._losses = losses
        self._summary = summary
        self._visual = visual

        # Define a unified optimizer.
        self._global_step = tf.Variable(0, trainable=False, name='global_step')
        share_learning_rate = tf.train.exponential_decay(learning_rate, self._global_step,
                                                         decay_iter, decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(share_learning_rate)
        # Define the training operator.
        net_loss = self._losses[self._name_space + '/NET_loss']
        self._train_op = optimizer.minimize(net_loss, global_step=self._global_step)

        # The summary writer for whole model.
        summary_path = conf_others.get('summary_path')
        self._summary_writer = tf.summary.FileWriter(summary_path, self._sess.graph)


        # check saved model
        self._check_model()


        # # Reset the start position of single iteration for data adapter
        # #   if enable "restore from breakpoint".
        # conf_others = self._config['Others']
        # restore_from_bp = conf_others.get('restore_breakpoint', True)
        # if restore_from_bp:
        #     # Compute last iter.
        #     glo_step = self._global_step.eval(self._sess)
        #     replay_iter = conf_dqn.get('replay_turn', 1)
        #     last_iter = glo_step * replay_iter
        #     # Compute the start position of single iteration.
        #     iter_offset = last_iter % max_iteration









        # # finalize the graph
        # self._sess.graph.finalize()

    def _check_model(self):
        r'''
            Check whether to load model from previous parameters or initialize it.
        '''
        conf_others = self._config['Others']
        params_dir = conf_others.get('net_params')
        if params_dir is not None:
            if not params_dir.endswith('/'): params_dir += '/'
            var_list = tf.trainable_variables()
            g_list = tf.global_variables()
            for v in g_list:
                if 'global_step' in v.name:
                    var_list.append(v)
                    break
            self._saver = tf.train.Saver(var_list=var_list)
            # self._saver = tf.train.Saver()
            checkpoint_state = tf.train.get_checkpoint_state(params_dir)
            if checkpoint_state and checkpoint_state.model_checkpoint_path:
                path = checkpoint_state.model_checkpoint_path
                self._saver.restore(self._sess, path)
                print('Restore from {} successfully.'.format(path))
            else:
                print('No checkpoint.')
                self._sess.run(tf.global_variables_initializer())
            sys.stdout.flush()
        else:
            self._sess.run(tf.global_variables_initializer())
        return



    # num_epochs: train epochs
    def train(self, num_epochs, max_iteration):
        r'''
            The "Training" phrase. Train the whole model many times.

        Parameters:
            num_epochs: Indicates the whole epochs that whole model should train.
            max_turns: Specify the max turns for "DQN Agent".
            restore_from_bp: Whether to start from breakpoint.
        '''

        # Check validity.
        if num_epochs <= 0 or max_iteration <= 0:
            raise ValueError('The epochs and iteration should be positive !!!')

        # Get config.
        conf_train = self._config['Training']
        epsilon_dict = conf_train.get('epsilon_dict', None)
        replay_iter = conf_train.get('replay_iter', 1)

        # Get or compute the epsilon dict here.
        if epsilon_dict is None:
            epsilon_dict = {}
            for interval in range(11):
                iv = interval * 0.1
                epsilon_dict[str(iv)] = 1.0 - iv

        # Translate to "iteration -> epsilon" form.
        total_turns = num_epochs * max_iteration
        epsilon_book = []
        for k in epsilon_dict.keys():
            iter = int(float(k) * total_turns)
            epsilon_book.append((iter, epsilon_dict[k]))
        print('### --> Epsilon book: {}'.format(epsilon_book))
        # self._epsilon_book = epsilon_book

        # Determine the start position if enable restore from last position.
        conf_others = self._config['Others']
        restore_from_bp = conf_others.get('restore_breakpoint', True)
        if restore_from_bp:
            # Compute last iter.
            glo_step = self._global_step.eval(self._sess)
            last_iter = glo_step * replay_iter
            # Compute the start epoch and iteration.
            start_epoch = last_iter // max_iteration
            start_iter = last_iter % max_iteration
        else:
            start_epoch = start_iter = 0

        # Train the whole model many times. (Depend on epochs)
        print('\n\nEnd-to-End Training !!!')
        for epoch in range(start_epoch, num_epochs):
            # Reset the start position of iteration.
            self._data_adapter.reset_position(start_iter)
            # Start training.
            self._train(max_iteration, start_iter, epsilon_book)
            # Re-assign the start position iteration to zero.
            #   Note that, only the first epoch is specified,
            #   others start from zero.
            start_iter = 0
            # Print some info.
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


    # def __restore_from_bp(self, breakpoint_path):
    #     if breakpoint_path is not None:
    #         with open(breakpoint_path, 'r') as f:
    #             start_pos = int(f.read())
    #     else:
    #         print('Not specify the breakpoint path, can not restore training ...')
    #         start_pos = 0
    #     return start_pos


    # def _train(self, max_iteration, start_pos, epsilon_book):
    #     r'''
    #         End-to-end train the whole model.
    #
    #     :param images_per_epoch:
    #     :return:
    #     '''
    #
    #     print('-----> Start the "End-to-End" training policy !!!')
    #
    #
    #
    #     # # Total steps count used to control whether to training or not.
    #     # total_steps = 0
    #     # # Remainder of last difference of total steps.
    #     # ts_diff_remainder = 0
    #
    #     # Start to training ...
    #     for turn in range(start_pos, max_iteration):
    #         # Used to record total rewards for current turn.
    #         turn_rewards = 0
    #
    #         # Reset the "Maze" environment (by pass through the proposals).
    #         state, (image, label, select_mask, cls_weights) = self._env.reset(arg=None)
    #         # The experience trajectory list of current processing image.
    #         exp_traj = []
    #
    #         # Reset (Initialize) the recurrent layer's hidden state (GRU).
    #         #   Note that, we will update it (Iteratively) when in inference
    #         #   phrase. But always set to 0 when training.
    #         h_size = self._model.gru_hsize
    #         gru_S = np.zeros([1, h_size])
    #
    #         # Debug metric used to record the cost time of current turn.
    #         start_time = time.time()
    #         # Current total steps, which is used to control the training time.
    #         cur_tosteps = total_steps
    #
    #         # 9999999999: max step per epoch
    #         for step in range(9999999999):
    #             # ε-greedy exploration
    #             action_index, gru_Sout = self._egreedy_action(state, turn=turn, arg=(gru_S, ))
    #             # Push action index into environment, so that we can get the next state and reward.
    #             next_state, reward, terminal, (next_image, next_label, next_SM, next_CW) = self._env.step(
    #                 action_index
    #             )
    #             # Generate one-hot vector for action index, which is lately used to calculate loss.
    #             one_hot_action = to_categorical(action_index, num_classes=self._act_num)    # [cls, act_dim]
    #             # one_hot_action = np.zeros(self._act_num)
    #             # one_hot_action[action_index] = 1
    #
    #             ### DEBUG
    #             # print(reward)
    #             ### DEBUG
    #
    #             # Experience.
    #             exp = (state,
    #                    one_hot_action,
    #                    reward,
    #                    next_state,
    #                    terminal,
    #                    image,
    #                    label,
    #                    select_mask,
    #                    cls_weights,
    #                    [0]   # visited count
    #                    )
    #
    #             # Add current experience into trajectory.
    #             exp_traj.append(exp)
    #
    #             # What's more, re-assign the GRU hidden state.
    #             gru_S = gru_Sout
    #
    #             # Change current state to next state. (Iteratively)
    #             state = next_state
    #             image = next_image  # change to next processing image.
    #             label = next_label  # change to next label.
    #             select_mask = next_SM   # change to next select mask.
    #             cls_weights = next_CW   # change to next clazz weights.
    #
    #             # Calculate rewards of current turn of "Recursive" environment.
    #             turn_rewards += np.mean(reward)     # turn_rewards += reward
    #             # Auto-increase the total steps count.
    #             total_steps += 1
    #
    #             # Terminated. One turn ended (Process of current image is finished yet).
    #             if terminal:
    #                 # Only store experience trajectory when its size greater than
    #                 #   pre-defined track length.
    #                 if len(exp_traj) > self._track_len:
    #                     # Package the experience trajectory with
    #                     #   1) FEN (deep) feature maps.
    #                     #   2) raw image
    #                     #   3) its label
    #                     experience = (exp_traj, )
    #                     # Store the transition (experience) in "Replay Memory".
    #                     if self._prioritized_replay is False:
    #                         self._replay_memory.append(experience)
    #                         # Remove element (experience) if exceeds max size.
    #                         if len(self._replay_memory) > self._replay_memory_size:
    #                             self._replay_memory.popleft()
    #                     else:
    #                         # Using the "Prioritized Replay".
    #                         self._replay_memory.store(experience)
    #
    #                 # Show the value. --------------------------------------------
    #                 self._logger.debug("Turn {} --> total_rewards: {}, epsilon: {}, ".format(
    #                     turn, turn_rewards, self._epsilon))
    #                 # ------------------------------------------------------------
    #                 sys.stdout.flush()
    #
    #                 # Simply break the loop.
    #                 break
    #
    #         # Render. - Record the process of current image in "GIF" form.
    #         self._env.render()
    #
    #         # Check whether to training the DQN agent now or not.
    #         exec_train = len(self._replay_memory) > 0       # self._batch_size
    #         # Start training the DQN agent.
    #         if exec_train:
    #             # Calculate the training times.
    #             tc_denominator = total_steps - cur_tosteps + ts_diff_remainder
    #             train_count = tc_denominator // self._replay_period
    #             ts_diff_remainder = tc_denominator % self._replay_period
    #             # Train the model many times.
    #             for tc in range(train_count):
    #                 # Metric used to see the training time.
    #                 train_start_time = time.time()
    #                 # Really training the DQN agent.
    #                 v_cost = self._do_train_net(turn=turn,
    #                                             max_turns=images_per_epoch)
    #                 # Debug. ------------------------------------------------------
    #                 self._logger.info("Turn {} - Train Count {}, Loss: {}, Training time: {}".format(
    #                     turn, tc, v_cost, time.time() - train_start_time)
    #                 )
    #                 # -------------------------------------------------------------
    #
    #         self._logger.debug('Image {} cost time: {}'.format(turn, time.time() - start_time))
    #
    #     # Finish the end-to-end component training.
    #     return


    def _train(self, max_iteration, start_pos, epsilon_book):
        r'''
            End-to-end train the whole model.

        :param images_per_epoch:
        :return:
        '''

        print('-----> Start the "End-to-End" training policy !!!')

        # Get config.
        conf_dqn = self._config['DQN']
        prioritized_replay = conf_dqn.get('prioritized_replay', True)
        conf_others = self._config['Others']
        memory_size = conf_others.get('replay_memories')
        store_type = conf_others.get('sample_type', 'light')

        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)
        replay_iter = conf_train.get('replay_iter', 1)

        # Declare the store function for "Sample/Experience Store".
        def store_2mem(exp, SEG_stage):
            r'''
                experience likes below:
                (sample_meta, SEG_stage, SEG_prev, cur_bbox, position_info,
                    action, reward, terminal, SEG_cur, focus_bbox, next_posinfo)
            '''
            # Add to sample storage.
            if SEG_stage:
                self._sample_storage.append(exp)
                if len(self._sample_storage) > memory_size:
                    self._sample_storage.popleft()
            else:
                # store the experience in "Replay Memory".
                if prioritized_replay:
                    # using the "Prioritized Replay".
                    self._replay_memory.store(exp)
                else:
                    self._replay_memory.append(exp)
                    # remove element (experience) if exceeds max size.
                    if len(self._replay_memory) > memory_size:
                        self._replay_memory.popleft()
            # Finish.
            return

        # ------------------------ Train Loop Part ------------------------
        # Start to training ...
        for turn in range(start_pos, max_iteration):
            # Some info holder.
            turn_rewards = 0
            start_time = time.time()

            # Get current epsilon.
            for idx in range(len(epsilon_book) - 1):
                lower = epsilon_book[idx][0]
                upper = epsilon_book[idx + 1][0]
                if turn in range(lower, upper):
                    self._epsilon = epsilon_book[idx][1]
                    break

            # Ensure environment works in "Train" mode.
            self._env.switch_phrase('Train')

            # ------------------- Start single iteration. -------------------
            # Reset the environment.
            image, label, clazz_weights, (mha_idx, inst_idx) = self._env.reset()

            # Determine the store type.
            if store_type == 'light':
                sample_meta = (mha_idx, inst_idx, clazz_weights)
            elif store_type == 'heave':
                sample_meta = (image, label, clazz_weights)
            else:
                raise ValueError('Unknown sample store type !!!')

            # ---------------------------- Core Part ----------------------------
            # Pre-execute one step.
            (S_SEG_prev, S_cur_bbox, S_position_info), \
                S_action, S_reward, S_terminal, \
                (S_SEG_cur, S_focus_bbox, S_next_posinfo), \
                S_info = self._env.step(self._func_4train, SEG_stage=True)
            store_2mem((sample_meta, True, S_SEG_prev, S_cur_bbox, S_position_info,
                        S_action, S_reward, S_terminal, S_SEG_cur, S_focus_bbox, S_next_posinfo),
                       SEG_stage=True)
            # The "Focus".
            (F_SEG_prev, F_cur_bbox, F_position_info), \
                F_action, F_reward, F_terminal, \
                (F_SEG_cur, F_focus_bbox, F_next_posinfo), \
                F_info = self._env.step(self._func_4train, SEG_stage=False)
            #  --> update visual.
            turn_rewards += F_reward
            # The terminal flag only works when in "Focus" phrase.
            if F_terminal:
                # Store all the elements if terminate in one step.
                store_2mem((sample_meta, False, F_SEG_prev, F_cur_bbox, F_position_info,
                            F_action, F_reward, F_terminal, F_SEG_cur, F_focus_bbox, F_next_posinfo),
                           SEG_stage=False)
                # Show some info. --------------------------------------------
                self._logger.debug("Iter {} --> total_steps: {} total_rewards: {}, epsilon: {}, "
                                   "cost time: {}".format(
                    turn, 1, turn_rewards, self._epsilon, time.time() - start_time))
                # ------------------------------------------------------------

            # Start iteration if not terminal in the very beginning.
            else:
                # Record the "Previous State" for "Focus" (DQN)
                F_SEG_PREV = F_SEG_prev.copy()
                F_CUR_BBOX = F_cur_bbox.copy()
                F_POS_INFO = F_position_info.copy()

                # 9999999999: max step per iteration.
                for step in range(9999999999):
                    # Execute the "Segment" phrase.
                    (S_SEG_prev, S_cur_bbox, S_position_info), \
                        S_action, S_reward, S_terminal, \
                        (S_SEG_cur, S_focus_bbox, S_next_posinfo), \
                        S_info = self._env.step(self._func_4train, SEG_stage=True)
                    # Store the sample into "Segment" storage.
                    store_2mem((sample_meta, True, S_SEG_prev, S_cur_bbox, S_position_info,
                                S_action, S_reward, S_terminal, S_SEG_cur, S_focus_bbox, S_next_posinfo),
                               SEG_stage=True)

                    # Execute the "Focus" phrase.
                    (F_SEG_prev, F_cur_bbox, F_position_info), \
                        F_action, F_reward, F_terminal, \
                        (F_SEG_cur, F_focus_bbox, F_next_posinfo), \
                        F_info = self._env.step(self._func_4train, SEG_stage=False)
                    # Store the experience into "Focus" replay memory.
                    store_2mem((sample_meta, False, F_SEG_PREV, F_CUR_BBOX, F_POS_INFO,
                                F_action, F_reward, F_terminal, F_SEG_prev, F_cur_bbox, F_position_info),
                               SEG_stage=False)

                    # The terminal flag only works when in "Focus" phrase.
                    if F_terminal:
                        # Store the all elements (Cur - Next) if it's terminate now.
                        #   Note that, it's the last experience, so we have to add it.
                        store_2mem((sample_meta, False, F_SEG_prev, F_cur_bbox, F_position_info,
                                    F_action, F_reward, F_terminal, F_SEG_cur, F_focus_bbox, F_next_posinfo),
                                   SEG_stage=False)
                        # Show some info. --------------------------------------------
                        self._logger.debug("Iter {} --> total_steps: {} total_rewards: {}, epsilon: {}, "
                                           "cost time: {}".format(
                            turn, step, turn_rewards, self._epsilon, time.time() - start_time))
                        # ------------------------------------------------------------
                        break

                    # Switch to next state (SEG_prev, cur_bbox, pos_info) for "Focus" phrase.
                    F_SEG_PREV = F_SEG_prev.copy()
                    F_CUR_BBOX = F_cur_bbox.copy()
                    F_POS_INFO = F_position_info.copy()

                    # Update some info for visual.
                    turn_rewards += F_reward
            # ---------------------------- End of core part ----------------------------

            # Finish the process of current image. Render.
            self._env.render('video')

            # Check whether to training or not.
            exec_train = (len(self._sample_storage) >= (batch_size // 2)) and \
                         (len(self._replay_memory) >= (batch_size - batch_size // 2)) and \
                         (turn % replay_iter == 0)
            # Start training the DQN agent.
            if exec_train:
                # Metric used to see the training time.
                train_time = time.time()
                # Really training the DQN agent.
                v1_cost, v2_cost, v3_cost = self.__do_train()
                # Debug. ------------------------------------------------------
                self._logger.info("Train iter {} - Net Loss: {}, SEG Loss: {}, DQN Loss: {}, Training time: {}".format(
                    turn // replay_iter, v1_cost, v2_cost, v3_cost, time.time() - train_time)
                )
                # -------------------------------------------------------------

            # Print some info.
            self._logger.debug('Iter {} finish !!! Cost time: {}'.format(turn, time.time() - start_time))

        # Finish one epoch.
        return


    # The real training code.
    def __do_train(self):
        r'''
            The function is used to really execute the one-iteration training for whole model.

            ** Note that, The experience (the element of memory storage) consists of:
                (sample_meta, SEG_stage, SEG_prev, cur_bbox, position_info, action, reward, terminal,
                    SEG_cur, focus_bbox, next_posinfo)
        '''

        # Get config.
        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)

        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        prioritized_replay = conf_dqn.get('prioritized_replay', True)
        gamma = conf_dqn.get('discount_factor', 0.9)

        conf_others = self._config['Others']
        # memory_size = conf_others.get('replay_memories')
        store_type = conf_others.get('sample_type', 'light')
        save_steps = conf_others.get('save_steps', 1000)
        params_dir = conf_others.get('net_params')

        # ------------------------------- Data Preparation ------------------------
        # The input batch holders.
        images = []
        labels = []
        clazz_weights = []
        SEG_stages = []
        SEG_prevs = []
        cur_bboxes = []
        position_infos = []
        actions = []
        rewards = []
        terminals = []
        SEG_curs = []
        focus_bboxes = []
        next_pos_infos = []

        # The function used to allocate each element to its corresponding batch.
        def allocate_2batches(data_batch):
            for sample in data_batch:
                if store_type == 'light':
                    mha_idx, inst_idx, weights = sample[0]
                    img, lab = self._data_adapter.precise_locate(mha_idx, inst_idx)
                elif store_type == 'heave':
                    img, lab, weights = sample[0]
                else:
                    raise ValueError('Unknown sample store type !!!')
                # Add to each batch.
                images.append(img)
                labels.append(lab)
                clazz_weights.append(weights)
                SEG_stages.append(sample[1])
                SEG_prevs.append(sample[2])
                cur_bboxes.append(sample[3])
                position_infos.append(sample[4])
                actions.append(sample[5])
                rewards.append(sample[6])
                terminals.append(sample[7])
                SEG_curs.append(sample[8])
                focus_bboxes.append(sample[9])
                next_pos_infos.append(sample[10])
            return

        # Randomly select a mini batch (uniform replay) for "Segmentation" branch.
        mini_batch = random.sample(self._sample_storage, batch_size//2)
        allocate_2batches(mini_batch)

        # Prioritized or randomly select a mini batch for "Focus" (DQN) branch.
        if prioritized_replay:
            # Prioritized replay. So the we will get three batches: tree_idx, mini_batches, ISWeights.
            tree_idx, exp_batch, ISWeights = self._replay_memory.sample(batch_size - batch_size//2)
        else:
            # Randomly select experience batch (uniform replay)
            exp_batch = random.sample(self._replay_memory, batch_size - batch_size//2)
        allocate_2batches(exp_batch)

        # Calculate the target Q values.
        #   1) get next Q values, so here we pass through (SEG_curs, focus_bboxes, next_pos_infos)
        #   2) iteratively add "reward" or "reward + next q val" as the target Q values.
        target_q_values = []
        next_q_values = self.__next_Qvalues((images, SEG_curs, next_pos_infos, SEG_stages, focus_bboxes))
        for t, r, nqv in zip(terminals, rewards, next_q_values):
            if t:
                target_q_values.append(r)
            else:
                target_q_values.append(r + gamma * nqv)     # Discounted future reward

        # ------------------------------- Train Model ------------------------
        # Get origin input part.
        if double_q is None:
            org_name = self._name_space
        else:
            org_name = double_q[0]
        x1 = self._inputs[org_name + '/image']
        x2 = self._inputs[org_name + '/prev_result']
        x3 = self._inputs[org_name + '/position_info']
        x4 = self._inputs[org_name + '/Segment_Stage']
        x5 = self._inputs[org_name + '/Focus_Bbox']
        # Get loss input part.
        l1 = self._losses[self._name_space + '/GT_label']
        l2 = self._losses[self._name_space + '/clazz_weights']
        l3 = self._losses[self._name_space + '/prediction_actions']
        l4 = self._losses[self._name_space + '/target_Q_values']
        # Get loss value holder.
        sloss = self._losses[self._name_space + '/SEG_loss']
        dloss = self._losses[self._name_space + '/DQN_loss']
        nloss = self._losses[self._name_space + '/NET_loss']

        # Generate the basic feed dictionary lately used for training the DQN agent.
        feed_dict = {
            # Origin input part.
            x1: images,
            x2: SEG_prevs,
            x3: position_infos,
            x4: SEG_stages,
            x5: cur_bboxes,
            # Loss part.
            l1: labels,
            l2: clazz_weights,
            l3: actions,
            l4: target_q_values
        }

        # Do train.
        if prioritized_replay:
            # Used to update priority. Coz the "Prioritized Replay" need to update the
            #   priority for experience.
            ISw = self._losses[self._name_space + '/IS_weights']
            exp_pri = self._losses[self._name_space + '/EXP_priority']
            feed_dict[ISw] = ISWeights
            # Execute the training operation.
            _, exp_priority, v1_cost, v2_cost, v3_cost = self._sess.run(
                [self._train_op, exp_pri, nloss, sloss, dloss], feed_dict=feed_dict)
            # update probability for leafs.
            self._replay_memory.batch_update(tree_idx, exp_priority)  # update priority
        else:
            # Simply feed the dictionary to the selected operation is OK.
            _, v1_cost, v2_cost, v3_cost = self._sess.run(
                [self._train_op, nloss, sloss, dloss], feed_dict=feed_dict)

        # ------------------------------- Save Model Parameters ------------------------
        # Get the global step lately used to determine whether to save model or not.
        step = self._global_step.eval()
        # Calculate the summary to get the statistic graph.
        if step > 0 and step % save_steps == 0:
            # Get summary holders.
            s1 = self._summary[self._name_space + '/Reward']
            s2 = self._summary[self._name_space + '/DICE']
            s3 = self._summary[self._name_space + '/BRATS_metric']
            out_summary = self._summary[self._name_space + '/MergeSummary']
            # Execute test environment to get the reward situation of current DQN model.
            # cur_reward_list = self.test(10)
            reward_list, DICE_list, BRATS_list = self.test(2)
            # Compute the summary value and add into statistic graph.
            feed_dict[s1] = reward_list
            feed_dict[s2] = DICE_list
            feed_dict[s3] = BRATS_list
            summary = self._sess.run(out_summary, feed_dict=feed_dict)
            self._summary_writer.add_summary(summary, step)
            self._summary_writer.flush()
        # Save the model (parameters) within the fix period.
        if step > 0 and step % save_steps == 0:
            self._saver.save(self._sess, params_dir, 233)
        # Regularly copy the parameters to "Target" DQN network.
        if double_q is not None and step % save_steps == 0:
            self._model.notify_copy2_DDQN(self._sess, only_head=False)

        # Finish one turn train. Return the cost value.
        return v1_cost, v2_cost, v3_cost


    def _func_4train(self, x):
        r'''
            Use model to generate the segmentation result for current region.
            What's more, it will use the Q values generated by DQN to select action
                or random select actions according to the epsilon value (which
                indicates it's "Exploitation" or "Exploration" phrase).

            ** Note that, this function should be called in "Validate" or "Test" phrase.

        ----------------------------------------------------------------
        Parameters:
            x: The current observation of environment. It consists of:
                (image, SEG_prev, position_info, SEG_stage, focus_bbox, COMP_res)
        ----------------------------------------------------------------
        Return:
            The (whole-view) segmentation result, and the complete result value
                (for next segmentation of iteration).
            What's more, return the action.
        '''

        # Get each input data.
        image, SEG_prev, position_info, SEG_stage, focus_bbox, COMP_result = x

        # Determine the stage prefix.
        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        if double_q is None:
            stage_prefix = self._name_space
        else:
            stage_prefix = double_q[0]
        # Get each input holder of model.
        x1 = self._inputs[stage_prefix + '/image']
        x2 = self._inputs[stage_prefix + '/prev_result']
        x3 = self._inputs[stage_prefix + '/position_info']
        x4 = self._inputs[stage_prefix + '/Segment_Stage']
        x5 = self._inputs[stage_prefix + '/Focus_Bbox']
        x6 = self._inputs[self._name_space + '/Complete_Result']
        # Get segmentation output holder of model.
        y1 = self._outputs[self._name_space + '/SEG_output']
        y2 = self._outputs[self._name_space + '/FUSE_result']
        y3 = self._outputs[stage_prefix + '/DQN_output']

        # Generate the segmentation result for current region (focus bbox).
        segmentation, COMP_res, q_val = self._sess.run([y1, y2, y3], feed_dict={
            x1: [image],
            x2: [SEG_prev],
            x3: [position_info],
            x4: [SEG_stage],
            x5: [focus_bbox],
            x6: [COMP_result]
        })

        # Get the segmentation and complete value result.
        segmentation = segmentation[0]
        COMP_res = COMP_res[0]
        q_val = q_val[0]
        # Select action according to the output of "Deep Q Network".
        action = np.argmax(q_val, axis=-1)  # scalar
        # e-greedy action policy.
        action = self.__egreedy_action(action, self._epsilon)

        # Return both segmentation result and DQN result.
        return segmentation, COMP_res, action


    def _segment(self, x):
        r'''
            Use model to generate the segmentation result for current region.

            ** Note that, this function should be called in "Validate" or "Test" phrase.

        ----------------------------------------------------------------
        Parameters:
            x: The current observation of environment. It consists of:
                (image, SEG_prev, position_info, SEG_stage, focus_bbox, COMP_res)
        ----------------------------------------------------------------
        Return:
            The (whole-view) segmentation result, and the complete result value
                (for next segmentation of iteration).
        '''

        # Get each input data.
        image, SEG_prev, position_info, SEG_stage, focus_bbox, COMP_result = x

        # Determine the stage prefix.
        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        if double_q is None:
            stage_prefix = self._name_space
        else:
            stage_prefix = double_q[0]
        # Get each input holder of model.
        x1 = self._inputs[stage_prefix + '/image']
        x2 = self._inputs[stage_prefix + '/prev_result']
        x3 = self._inputs[stage_prefix + '/position_info']
        x4 = self._inputs[stage_prefix + '/Segment_Stage']
        x5 = self._inputs[stage_prefix + '/Focus_Bbox']
        x6 = self._inputs[self._name_space + '/Complete_Result']
        # Get segmentation output holder of model.
        y1 = self._outputs[self._name_space + '/SEG_output']
        y2 = self._outputs[self._name_space + '/FUSE_result']

        # Generate the segmentation result for current region (focus bbox).
        segmentation, COMP_res = self._sess.run([y1, y2], feed_dict={
            x1: [image],
            x2: [SEG_prev],
            x3: [position_info],
            x4: [SEG_stage],
            x5: [focus_bbox],
            x6: [COMP_result]
        })
        # Get the segmentation and complete value result.
        segmentation = segmentation[0]
        COMP_res = COMP_res[0]
        return segmentation, COMP_res


    def _action(self, x):
        r'''
            Let DQN select action for current state, that is,
                select the action with max Q value.

            ** Note that, this function should be called in "Validate" or "Test" phrase.

        ----------------------------------------------------------------
        Parameters:
            x: The current observation of environment. It consists of:
                (image, SEG_prev, position_info, SEG_stage, focus_bbox)
        ----------------------------------------------------------------
        Return:
            The action index with max Q value.
        '''

        # Get each input data.
        image, SEG_prev, position_info, SEG_stage, focus_bbox = x

        # Determine the stage prefix.
        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        if double_q is None:
            stage_prefix = self._name_space
        else:
            stage_prefix = double_q[0]
        # Get each input holder of model.
        x1 = self._inputs[stage_prefix + '/image']
        x2 = self._inputs[stage_prefix + '/prev_result']
        x3 = self._inputs[stage_prefix + '/position_info']
        x4 = self._inputs[stage_prefix + '/Segment_Stage']
        x5 = self._inputs[stage_prefix + '/Focus_Bbox']
        # Get DQN output holder of model.
        y = self._outputs[stage_prefix + '/DQN_output']

        # Calculate the Q values of each action within current state.
        q_val = self._sess.run(y, feed_dict={
            x1: [image],
            x2: [SEG_prev],
            x3: [position_info],
            x4: [SEG_stage],
            x5: [focus_bbox]
        })[0]

        # Select action according to the output of "Deep Q Network".
        action = np.argmax(q_val, axis=-1)  # scalar
        return action





    # # The real training code.
    # def _do_train_net(self, turn, max_turns):
    #     r'''
    #         The method used to really execute the one-turn DQN training.
    #
    #         The single exp-cell consists of five part elements:
    #             (state, one_hot_action, reward, next_state, terminal, direction, select_mask)
    #         The experience consists of: (exp_cell_list, )
    #         That is, the experience just likes below:
    #             ([(s, a, r, ns, t, i, l, sm, cw, vf), ..., (s, a, r, ns, t, i, l, sm, cw, vf)], )
    #
    #         state: The current state of "Maze" environment, which consists of:
    #             (features_maps, )
    #
    #     Parameters:
    #         turn: The current turn of training phrase.
    #         max_turns: The max turn of whole training phrase.
    #     '''
    #
    #     sample_time = time.time()     # Debug
    #
    #     # The tree idx batch and ISweights batch used to update the priority if
    #     #   enable the "Prioritized Replay".
    #     if self._prioritized_replay:
    #         tree_idx = []
    #         ISWeights = []
    #
    #     # The train-related batch.
    #     # -----------------------------------------------
    #     # Split into different batches for state.
    #     FEN_feats_batch = []
    #     # Split into different batches for next state.
    #     next_FEN_feats_batch = []
    #     # Action batch holder.
    #     act_batch = []
    #     # Reward batch holder.
    #     reward_batch = []
    #     # Terminal batch holder.
    #     terminal_batch = []
    #     # Raw image batch holder.
    #     image_batch = []
    #     # Label batch holder.
    #     label_batch = []
    #     # Select mask batch holder.
    #     selectM_batch = []
    #     # Class weights batch holder.
    #     clsWeights_batch = []
    #     # Visited flag.
    #     visitFlag_batch = []
    #
    #     # Sample experiences from the replay memory of specified batch size.
    #     for _ in range(self._batch_size):
    #         if self._prioritized_replay:
    #             # Prioritized replay. So the we will get three batches: tree_idx, mini_batches, ISWeights.
    #             tid, experience, isweight = self._replay_memory.sample(1)
    #             tid, experience, isweight = tid[0], experience[0], isweight[0]
    #             # Add the tree_idx and ISweights into batch, respectively.
    #             tree_idx.append(tid)
    #             ISWeights.append(isweight)
    #         else:
    #             # Randomly select a experience (uniform replay)
    #             experience = random.sample(self._replay_memory, 1)[0]
    #
    #         # Divide experience into different element.
    #         exp_traj, = experience
    #
    #         # Random select a time-step as the begin time.
    #         rb_tstep = np.random.randint(len(exp_traj) - self._track_len)   # scalar
    #         # Get the selected fixed-len trajectory.
    #         rselect_exps = exp_traj[rb_tstep: rb_tstep + self._track_len]   # list. len: track_len
    #
    #         # Package to time-related list.
    #         feats00 = []
    #         nfeats30 = []
    #         act1 = []
    #         rew2 = []
    #         img5 = []
    #         lab6 = []
    #         sm7 = []
    #         cw8 = []
    #         ter4 = []
    #         vf9 = []
    #         for rs_exp in rselect_exps:
    #             feats00.append(rs_exp[0][0])
    #             nfeats30.append(rs_exp[3][0])
    #             act1.append(rs_exp[1])
    #             rew2.append(rs_exp[2])
    #             img5.append(rs_exp[5])
    #             lab6.append(rs_exp[6])
    #             sm7.append(rs_exp[7])
    #             cw8.append(rs_exp[8])
    #             ter4.append(rs_exp[4])
    #             # Meanwhile reset the flag.
    #             rs_exp[9][0] += 1
    #             vf9.append(rs_exp[9].copy()[0])
    #
    #         # Pack into the State batch.
    #         FEN_feats_batch.extend(feats00)
    #         # Pack into the next neighbours and modalities (State) batch.
    #         next_FEN_feats_batch.extend(nfeats30)
    #         # Pack into the action batch.
    #         act_batch.append(act1)
    #         # Pack into the reward batch.
    #         reward_batch.append(rew2)
    #         # Pack into the terminal batch.
    #         terminal_batch.append(ter4)
    #         # Pack into the visit flag batch.
    #         visitFlag_batch.append(vf9)
    #
    #         # Pack into the raw image batch.
    #         image_batch.extend(img5)
    #         # Pack into the segmentation label batch.
    #         label_batch.append(lab6)
    #         # Pack into the select mask batch.
    #         selectM_batch.extend(sm7)
    #         # Pack into the class weights batch.
    #         clsWeights_batch.append(cw8)
    #     # ------------------------- Finish Split Batch -------------------
    #
    #     # Union the State batch and Next-State batch.
    #     state_batch = (
    #         FEN_feats_batch,
    #     )
    #     next_state_batch = (
    #         next_FEN_feats_batch,
    #     )
    #
    #     # The holder used to store the target Q values.
    #     target_q_values = []
    #
    #     # Calculate the next Q values by passing the next states batch into DQN agent.
    #     next_q_values = self._next_Qvalues_function(next_state_batch, image_batch=image_batch)  # [b, t]
    #
    #     # Iteratively calculate the target Q values.
    #     for t_ter, t_rew, t_nqv in zip(terminal_batch, reward_batch, next_q_values):
    #         track_tar_qvals = []
    #         for terminal, reward, next_qval in zip(t_ter, t_rew, t_nqv):
    #             # The target Q values is depend on whether terminated or not.
    #             if terminal:
    #                 # Directly add the reward value if it's terminated.
    #                 track_tar_qvals.append(reward)
    #             else:
    #                 # Discounted future reward (plus reward) if not terminated.
    #                 track_tar_qvals.append(reward + self._gamma * next_qval)
    #         # Add into batch level.
    #         target_q_values.append(track_tar_qvals)
    #
    #     # Debug. --------------------------------------------------------
    #     self._logger.debug("The sample and package time cost: {}".format(time.time() - sample_time))  # DEBUG
    #     # ---------------------------------------------------------------
    #
    #     bp_time = time.time()   # DEBUG
    #
    #     # Always set LSTM initial state to 0 when training.
    #     h_size = self._model.gru_hsize
    #     gru_Si = np.zeros([self._batch_size, h_size])
    #
    #     # Generate the basic feed dictionary lately used for training the DQN agent.
    #     feed_dict = {
    #         # The public.
    #         self._tp_flag: True,
    #         self._img: image_batch,
    #         # The initial hidden state for GRU in DRQN.
    #         self._drqn_gsi: gru_Si,
    #         # The input action and target Q values. For DRQN.
    #         self._drqn_cur_in_act: act_batch,
    #         self._drqn_cur_tar_qval: target_q_values,
    #         # # The select mask and GT segmentation. For UN
    #         self._un_sm: selectM_batch,
    #         self._un_gt_seg: label_batch,
    #         self._un_cw: clsWeights_batch,
    #         self._vf: visitFlag_batch
    #     }
    #
    #     # Really training the DQN agent by passing the state, action and target Q values into
    #     #   train operation of DQN agent according to the "Experience Replay" policy.
    #     if self._prioritized_replay:
    #         # Used to update priority. Coz the "Prioritized Replay" need to update the
    #         #   priority for experience.
    #         feed_dict[self._ISweights] = ISWeights
    #         # Execute the training operation.
    #         v_cost, _, exp_priority = self._sess.run([self._whole_loss, self._e2e_train_op, self._exp_pri],
    #                                                  feed_dict=feed_dict)
    #         # Merge the ISweights of same tree_idx.
    #         ul_dict = {}
    #         for ul_tid, ul_ep in zip(tree_idx, exp_priority):
    #             if ul_tid not in ul_dict.keys():
    #                 ul_dict[ul_tid] = [ul_ep]
    #             else:
    #                 ul_dict[ul_tid].append(ul_ep)
    #         tree_idx = []
    #         exp_priority = []
    #         for ul_key in ul_dict.keys():
    #             tree_idx.append(ul_key)
    #             exp_priority.append(np.mean(ul_dict[ul_key]))
    #         # Convert to Numpy.ndarray.
    #         tree_idx = np.asarray(tree_idx)
    #         exp_priority = np.asarray(exp_priority)
    #         # update probability for leafs.
    #         self._replay_memory.batch_update(tree_idx, exp_priority)  # update priority
    #     else:
    #         # Simply feed the dictionary to the selected operation is OK.
    #         v_cost, _ = self._sess.run([self._whole_loss, self._e2e_train_op],
    #                                    feed_dict=feed_dict)
    #
    #     self._logger.debug("The BP time cost: {}".format(time.time() - bp_time))     # DEBUG
    #
    #     # Get the global step lately used to determine whether to save model or not.
    #     step = self._global_step.eval()
    #     # Calculate the summary to get the statistic graph.
    #     if step > 0 and step % self._save_per_step == 0:
    #     # if step > 0 and step % 100 == 0:
    #     # if step > 0 and step % 10 == 0:
    #         # Execute test environment to get the reward situation of current DQN model.
    #         # cur_reward_list = self.test(10)
    #         cur_reward_list, cur_DICE_list, cur_BRATS_list = self.test(2)
    #         # Compute the summary value and add into statistic graph.
    #         feed_dict[self._statis_rewards] = cur_reward_list
    #         feed_dict[self._statis_dice] = cur_DICE_list
    #         feed_dict[self._statis_brats] = cur_BRATS_list
    #         summary = self._sess.run(self._summary, feed_dict=feed_dict)
    #         self._summary_writer.add_summary(summary, step)
    #         self._summary_writer.flush()
    #     # Save the model (parameters) within the fix period.
    #     if self._should_save_model and step > 0 and step % self._save_per_step == 0:
    #         # Save the model.
    #         self._saver.save(self._sess, self._params_dir, 233)
    #         # Record the breakpoint in order to support the "Continue Training".
    #         with open(self._bpt_dir, 'w') as f:
    #             if turn == max_turns - 1:
    #                 # Reset the start position to 0.
    #                 f.write('0')
    #             else:
    #                 f.write(str(turn))
    #     # Regularly copy the parameters to "Target" DQN network.
    #     if self._DDQN_name_pair is not None and step % self._save_per_step == 0:
    #         # Notify the network to copy the parameters from "Origin" to "Target".
    #         self._model.notify_copy2_DDQN(self._sess, only_head=False)
    #
    #     # Finish one turn train. Return the cost value.
    #     return v_cost

    def __explore_policy(self, distribution='uniform'):
        r'''
            Randomly select an action.
        '''
        return np.random.randint(self._acts_dim)

    def __egreedy_action(self, action, epsilon):
        r'''
            The ε-greedy exploration for DQN agent.
        '''
        # Exploration or Exploitation according to the epsilon.
        if random.random() <= epsilon:
            # Exploration.
            action_index = self.__explore_policy()
        else:
            # Exploitation. Use network (DRQN) to predict action index.
            action_index = action
        return action_index

    # def _action(self, state, arg, test_flag=(False, -1)):
    #     r'''
    #         Let DQN select action for current state, that is, select the action with max Q value.
    #
    #         Moreover, this method is only called in "Reference" phrase.
    #
    #     Parameters:
    #         state: The current state of "Recursive" environment.
    #             The state of "Recursive" consists of:
    #                 (feature_maps, )
    #                 The deep feature maps produced by FEN, which the DRQN interacts with.
    #         arg: Additional arguments. Here includes:
    #                 1) gru_hstate: The GRU hidden state.
    #
    #     Return:
    #         The action index with max Q value. Meanwhile return the LSTM hidden state of
    #             next time-step.
    #     '''
    #
    #     # Destruct the detail state elements from state.
    #     feature_maps, = state
    #
    #     # Get the GRU hidden state.
    #     gru_Si, = arg
    #
    #     # Calculate the Q values of each action within current state.
    #     drqn_q_val, drqn_lstm_sout = self._sess.run(
    #         [self._drqn_out, self._drqn_gso],
    #         feed_dict={
    #             self._tp_flag: False,
    #             self._fe_out: [feature_maps],
    #             self._drqn_gsi: gru_Si
    #         })
    #
    #     # Unpack the wrapper.
    #     drqn_q_val = drqn_q_val[0, 0]
    #     lstm_Sout = drqn_lstm_sout
    #
    #     # Select action according to the output of "Deep Recurrent Q Network".
    #     action = np.argmax(drqn_q_val, axis=-1)      # [mask]
    #
    #     # # Debug
    #     # print('PN probability:', pn_probs)
    #     # print('PN fake Qvals:', pn_fqvals)
    #     # print('FTN Qvals:', ftn_qvals)
    #     # ri, ci = self._sess.run([self._model.debug_raw_input, self._model.debug_crop_input],
    #     #                         feed_dict={
    #     #                             self._img: [cur_image],
    #     #                             self._rois: [neighbour_rois],
    #     #                             self._pi: [position_indicator],
    #     #                             self._th: [tackle_history],
    #     #                             self._pq: [parent_Q_vals],
    #     #                             self._gs: [grid_shape]
    #     #                         })
    #     # import matplotlib.pyplot as plt
    #     # plt.subplot(231)
    #     # plt.imshow(ri[0, :, :, 0])
    #     # plt.subplot(232)
    #     # plt.imshow(ri[0, :, :, 4])
    #     # plt.subplot(233)
    #     # plt.imshow(ri[0, :, :, 5])
    #     # plt.subplot(234)
    #     # plt.imshow(ci[0, :, :, 0])
    #     # plt.subplot(235)
    #     # plt.imshow(ci[0, :, :, 4])
    #     # plt.subplot(236)
    #     # plt.imshow(ci[0, :, :, 5])
    #     # plt.show()
    #     # # Debug
    #
    #     # if test_flag[0] and test_flag[1] % 30 == 0:
    #     #
    #     #     print('--- Cls Q ---')
    #     #     print(drqn_q_val)
    #     #
    #     #     raw_q, ca_prob = self._sess.run(
    #     #         [self._model.DEBUG_drqn_raw_out, self._model.DEBUG_t_CA_prob],
    #     #         feed_dict={
    #     #             self._tp_flag: False,
    #     #             self._fe_out: [feature_maps],
    #     #             self._drqn_gsi: gru_Si
    #     #         })
    #     #     raw_q = raw_q[0, 0]
    #     #     ca_prob = ca_prob[0, 0]
    #     #
    #     #     print('--- Raw Q ---')
    #     #     print(raw_q)
    #     #     print('--- CA prob ---')
    #     #     print(ca_prob)
    #
    #
    #     # Return the action select by DRQN. Meanwhile return the
    #     #   LSTM hidden state of next time-step.
    #     return action, lstm_Sout

    # def _next_Qvalues_function(self, state_batch, image_batch):
    #     r'''
    #         The enhanced version of @method{self._action()}, it's used to calculate
    #             next Q values for batch input. Which is used to calculate the
    #             "Target Q Values".
    #
    #         Note that, this method is only used in "Training" phrase.
    #
    #     Parameters:
    #         state_batch: The state batch. Which is consisted as below:
    #             state_batch = (
    #                 FEN_feats_batch,
    #             )
    #         feats_batch: The FEN feature maps batch. Which is produced by FEN.
    #
    #     Return:
    #         The next Q values batch. Which is used to calculate the "Target Q Values".
    #     '''
    #
    #     # Get the different element batch, respectively.
    #     FEN_feats_batch, = state_batch
    #
    #     # Computes the Q values of states batch. And use the "Double Q" algorithm
    #     #   according to the flag.
    #     if self._DDQN_name_pair is not None:
    #         # Both run the "Target Network" and "Origin Network" to get the target Q values
    #         #   and origin Q values. And use the target Q values to predict action, but
    #         #   use origin Q values to evaluate true values.
    #         tar_q_vals, org_q_vals = self._sess.run(
    #             [self._tar_drqn_out, self._drqn_out],
    #             feed_dict={
    #                 # # Public.
    #                 # self._tp_flag: True,
    #                 # self._fe_out: FEN_feats_batch,
    #                 # # Target Network.
    #                 # self._tar_drqn_in: FEN_feats_batch,
    #                 # # Origin Network.
    #                 # self._drqn_in: FEN_feats_batch
    #
    #                 # Public.
    #                 self._tp_flag: True,
    #                 # Target Network.
    #                 self._tar_img: image_batch,
    #                 # Origin Network.
    #                 self._drqn_in: FEN_feats_batch
    #             })  # [b, t, cls, act_dim]
    #
    #         # Imitate selecting action. Compute the action with max Q value.
    #         tar_acts = np.argmax(tar_q_vals, axis=-1)   # [b, t, mask]
    #
    #         # Recover to one-hot form for convenient selecting the target Q value.
    #         tar_ohacts = to_categorical(tar_acts, num_classes=self._act_num)    # [b, t, mask, act_dim]
    #         # Multiply and reduce-sum to get the real target Q value.
    #         next_q_values = np.sum(org_q_vals * tar_ohacts, axis=-1)    # [b, t, mask]
    #
    #     else:
    #         # Only run the "Target Network" to get the target Q values. And use it to predict action.
    #         tar_q_vals = self._sess.run(self._tar_drqn_out, feed_dict={
    #             # # Public.
    #             # self._tp_flag: True,
    #             # self._fe_out: FEN_feats_batch,
    #             # # Target Network.
    #             # self._tar_drqn_in: FEN_feats_batch
    #
    #             # Public.
    #             self._tp_flag: True,
    #             # Target Network.
    #             self._tar_drqn_in: image_batch
    #         })  # [b, t, mask, act_dim]
    #
    #         # Use the max Q value as the target Q values.
    #         next_q_values = np.max(tar_q_vals, axis=-1)     # [b, t, mask]
    #
    #     # print('Next Q: \n {}'.format(next_q_values))
    #     # print("Shape: {}".format(next_q_values.shape))
    #
    #     # note here we do not truncate the q values, coz this function
    #     #   is only used in training state to compute the target q values.
    #     return next_q_values


    def __next_Qvalues(self, x):
        r'''
            The enhanced version of @method{self._action()}, it's used to calculate
                next Q values for batch input. Which is used to calculate the
                "Target Q Values".

            ** Note that, this function is only used in "Training" phrase.
        ------------------------------------------------------------------------------
        Parameters:
            x: The tuple of all input batches.
        ------------------------------------------------------------------------------
        Return:
            The next Q values batch. Which is used to calculate the "Target Q Values".
        '''

        # Get each input data.
        images, SEG_prevs, position_infos, SEG_stages, focus_bboxes = x

        # Determine the stage prefix.
        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        if double_q is None:
            org_name = tar_name = self._name_space
        else:
            org_name, tar_name = double_q

        # Get each input holder of model.
        x1 = self._inputs[org_name + '/image']
        x2 = self._inputs[org_name + '/prev_result']
        x3 = self._inputs[org_name + '/position_info']
        x4 = self._inputs[org_name + '/Segment_Stage']
        x5 = self._inputs[org_name + '/Focus_Bbox']
        # Get DQN output holder of model.
        y_org = self._outputs[org_name + '/DQN_output']

        # Calculate the Q values according to the "Double DQN" mode.
        if double_q is None:
            # Pure mode, use origin model to compute the Q values.
            q_vals = self._sess.run(y_org, feed_dict={
                x1: images,
                x2: SEG_prevs,
                x3: position_infos,
                x4: SEG_stages,
                x5: focus_bboxes
            })  # [batch, act_dim]
            # Use the max values of actions as next Q value.
            next_q_values = np.max(q_vals, axis=-1)     # [batch]
        else:
            # Get loss holders of model.
            l1 = self._losses[tar_name + '/image']
            l2 = self._losses[tar_name + '/prev_result']
            l3 = self._losses[tar_name + '/position_info']
            l4 = self._losses[tar_name + '/Segment_Stage']
            l5 = self._losses[tar_name + '/Focus_Bbox']
            # Get DQN output holder of model.
            y_tar = self._losses[tar_name + '/DQN_output']
            # "Double DQN" mode, use the actions selected by "target" model
            #   to filter (get) the next Q values from "origin" model.
            org_qvals, tar_qvals = self._sess.run([y_org, y_tar], feed_dict={
                # origin part
                x1: images,
                x2: SEG_prevs,
                x3: position_infos,
                x4: SEG_stages,
                x5: focus_bboxes,
                # target part
                l1: images,
                l2: SEG_prevs,
                l3: position_infos,
                l4: SEG_stages,
                l5: focus_bboxes
            })
            # Generate the action from "target" and get Q values from "origin".
            tar_acts = np.argmax(tar_qvals, axis=-1)    # [batch]
            tar_acts = np.eye(self._acts_dim)[tar_acts]     # [batch, act_dim]
            next_q_values = np.sum(org_qvals * tar_acts, axis=-1)   # [batch]

        # Return the next Q values.
        return next_q_values


