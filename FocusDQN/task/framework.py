import collections
import random
import sys
import time
import tensorflow as tf

from core.dqn import *
from task.model import DqnAgent
from task.env import FocusEnvWrapper
import tfmodule.util as tf_util
import util.evaluation as eva
from util.visualization import PictureVisual



class DeepQNetwork(DQN):
    '''
        Deep Q-Learning algorithm. This is a framework that user can easily use.
            If user wants to change the config, only need to modify the config
            file (just like './config.ini')
    '''

    def __init__(self, config, name_space, data_adapter, log_level=logging.INFO):
        '''
            Initialization method. Mainly to declare the model and corresponding
                environment according to the given configuration.
        '''

        # Normal Assign.
        self._config = config
        self._name_space = name_space

        # Get config.
        conf_train = self._config['Training']
        learning_rate = conf_train.get('learning_rate')
        decay_iter = conf_train.get('learning_decay_iter')
        decay_rate = conf_train.get('learning_decay_rate')
        learning_policy = conf_train.get('learning_rate_policy', 'continuous')
        conf_dqn = self._config['DQN']
        prioritized_replay = conf_dqn.get('prioritized_replay', True)
        conf_others = self._config['Others']
        log_dir = conf_others.get('log_path')
        replay_memories = conf_others.get('replay_memories')

        # Initialize the parent class.
        DQN.__init__(self, log_level, log_dir)

        # Data adapter.
        self._data_adapter = data_adapter

        # Define the model. (Mainly initialize the variable here.)
        self._model = DqnAgent(self._config, name_space=self._name_space)

        # Define the environment.
        self._env = FocusEnvWrapper.get_instance(self._config, data_adapter=data_adapter)

        # Get the valid action quantity.
        self._acts_dim = self._env.acts_dim

        # Specify the replay memory.
        if prioritized_replay is False:
            replay_memory = collections.deque()
        else:
            replay_memory = PrioritizedPool(replay_memories)
        self._replay_memory = replay_memory

        # session
        self._sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # # Use the @Util{tfdebug}...
        # from tensorflow.python import debug as tf_debug
        # # 使用tf_debug的wrapper来包裹原来的session，使得启动后能进入
        # #   CLI调试界面.
        # self._sess = tf_debug.LocalCLIDebugWrapperSession(self._sess)
        # # 添加过滤器，用于监测 NAN 和 INF
        # self._sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        # Define the whole model.
        inputs, outputs, losses, summary, visual = self._model.definition()
        self._inputs = inputs
        self._outputs = outputs
        self._losses = losses
        self._summary = summary
        self._visual = visual
        # Show the quantity of parameters.
        tf_util.show_all_variables()

        # Determine the e-decay learning rate.
        self._global_step = tf.Variable(0, trainable=False, name='global_step')
        share_learning_rate = tf.train.exponential_decay(learning_rate, self._global_step,
                                                         decay_iter, decay_rate, staircase=True)

        # Define a unified ("End-to-end") optimizer.
        optimizer = tf.train.AdamOptimizer(share_learning_rate)
        # Define the training operator for "End-to-End".
        net_loss = self._losses[self._name_space + '/NET_loss']
        self._train_op = optimizer.minimize(net_loss, global_step=self._global_step)

        # Define the training operator for single "Segmentation".
        if learning_policy == 'fixed':
            pretrain_optimizer = tf.train.AdamOptimizer(learning_rate)
            self._pre_step = tf.Variable(0, trainable=False, name='pretrain_step')
        elif learning_policy == 'continuous':
            pretrain_optimizer = tf.train.AdamOptimizer(share_learning_rate)
            self._pre_step = self._global_step
        else:
            raise ValueError('Unknown learning rate policy !!!')
        sloss = self._losses[self._name_space + '/SEG_loss']
        self._pre_op = pretrain_optimizer.minimize(sloss, global_step=self._pre_step)
        # Visualization util for pre-train.
        self._pre_visutil = None
        anim_path = conf_others.get('animation_path', None)
        if anim_path is not None:
            conf_base = self._config['Base']
            input_shape = conf_base.get('input_shape')[1:3]
            suit_h = conf_base.get('suit_height')
            suit_w = conf_base.get('suit_width')
            clazz_dim = conf_base.get('classification_dimension')
            self._pre_visutil = PictureVisual(image_height=input_shape[0],
                                              image_width=input_shape[1],
                                              result_categories=clazz_dim,
                                              file_path=anim_path,
                                              name_scope='SinSEG',
                                              suit_height=suit_h,
                                              suit_width=suit_w)

        # The summary writer for whole model.
        summary_path = conf_others.get('summary_path')
        self._summary_writer = tf.summary.FileWriter(summary_path, self._sess.graph)

        # check saved model
        self._check_model()

        # finalize the graph
        self._sess.graph.finalize()

        # Finish initialization.
        return



    # num_epochs: train epochs
    def train(self, epochs, max_instances):
        r'''
            The "Training" phrase. Train the whole model many times.

        Parameters:
            epochs: Indicates the total epochs that whole model should train.
            max_instances: Specify the max instances of one epoch.
        '''

        # Check validity.
        if not isinstance(epochs, (int, np.int, np.int32, np.int64)) or \
                not isinstance(max_instances, (int, np.int, np.int32, np.int64)):
            raise TypeError('The epochs and max_instances must be integer !!!')
        if epochs <= 0 or max_instances <= 0:
            raise ValueError('The epochs and max_instances should be positive !!!')

        # Compute the max iterations from max instances.
        max_iteration = max_instances * self._data_adapter.slices_3d

        # Get config.
        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)
        epsilon_dict = conf_train.get('epsilon_dict', None)
        replay_iter = conf_train.get('replay_iter', 1)
        pre_epochs = conf_train.get('customize_pretrain_epochs', 0)
        learning_policy = conf_train.get('learning_rate_policy', 'continuous')
        conf_others = self._config['Others']
        restore_from_bp = conf_others.get('restore_breakpoint', True)

        # Check the train epochs validity. Coz we use the "Pre-train".
        if epochs < pre_epochs:
            raise Exception('The total epochs must no less than pre-train epochs !!! '
                            'Please change the total epochs or pre-train epochs.')

        # Get or compute the epsilon dict here.
        if epsilon_dict is None:
            epsilon_dict = {}
            for interval in range(11):
                iv = interval * 0.1
                epsilon_dict[str(iv)] = 1.0 - iv

        # Translate to "iteration -> epsilon" form.
        total_turns = (epochs - pre_epochs) * max_iteration
        epsilon_book = []
        for k in epsilon_dict.keys():
            ep_iter = int(float(k) * total_turns)
            epsilon_book.append((ep_iter, epsilon_dict[k]))
        print('### --> Epsilon book: {}'.format(epsilon_book))

        # Compute the steps of single instance when in "Pre-training" phrase.
        pre_single_instance_steps = int(np.ceil(self._data_adapter.slices_3d / batch_size))

        # ---------------- Pre-train the "Segmentation" branch (model). ----------------
        print('\n\nPre-training the SEGMENTATION model !!!')
        # Determine the start position.
        if restore_from_bp:
            # Compute last iter.
            pre_step = int(self._pre_step.eval(self._sess))
            pre_complete_instance_iters = (pre_step // pre_single_instance_steps) * self._data_adapter.slices_3d
            pre_remain_iters = (pre_step % pre_single_instance_steps) * batch_size
            last_iter = pre_complete_instance_iters + pre_remain_iters
            # Compute the start epoch and iteration.
            if last_iter < pre_epochs * max_iteration:
                start_epoch = last_iter // max_iteration
                start_iter = last_iter % max_iteration
            else:
                start_epoch = pre_epochs
                start_iter = -1
        else:
            start_epoch = start_iter = 0
        # Start train the "Segmentation" model.
        for epoch in range(start_epoch, pre_epochs):
            # Reset the start position of iteration.
            self._data_adapter.reset_position(start_iter)
            # Compute the start loop and max loops using the {start_pos, start_epoch} and {max_epoch}.
            start_instance = start_iter // self._data_adapter.slices_3d
            start_ins_index = start_iter % self._data_adapter.slices_3d
            start_loop = start_instance * pre_single_instance_steps + start_ins_index // batch_size
            max_loop = max_instances * pre_single_instance_steps
            # Start training.
            self._customize_pretrain(cur_epoch=epoch, max_loop=max_loop, start_loop=start_loop)
            # Re-assign the start position iteration to zero.
            #   Note that, only the first epoch is specified,
            #   others start from zero.
            start_iter = 0
            # Print some info.
            print('Finish the epoch {} for SEGMENTATION'.format(epoch))

        # Save the immediate model here for conveniently conduct experiments.
        if pre_epochs != 0:
            params_dir = conf_others.get('net_params')
            if not params_dir.endswith('/'): params_dir += '/'
            self._saver.save(self._sess, params_dir, 233)   # training folder.
            params_dir += '/warm-up/'
            self._saver.save(self._sess, params_dir, 233)   # immediate folder.

        # Determine the total pre-train steps according to the learning policy.
        if learning_policy == 'fixed':
            total_presteps = 0
        elif learning_policy == 'continuous':
            total_presteps = pre_epochs * max_instances * pre_single_instance_steps
        else:
            raise ValueError('Unknown learning rate policy !!!')

        # ---------------- Train the whole model (End-to-end) many times. (Depend on epochs) ----------------
        print('\n\nEnd-to-End Training !!!')
        # Determine the start position.
        if restore_from_bp:
            # Compute last iter.
            glo_step = int(self._global_step.eval(self._sess))
            # Compute the start epoch and iteration.
            e2e_step = glo_step - total_presteps
            last_iter = e2e_step * replay_iter
            start_epoch = last_iter // max_iteration
            start_iter = last_iter % max_iteration
        else:
            start_epoch = start_iter = 0
        # Start train the whole model.
        for epoch in range(start_epoch, epochs - pre_epochs):
            # Reset the start position of iteration.
            self._data_adapter.reset_position(start_iter)
            # Start training.
            self._train(max_iteration, epoch, start_iter, epsilon_book, total_presteps)
            # Re-assign the start position iteration to zero.
            #   Note that, only the first epoch is specified,
            #   others start from zero.
            start_iter = 0
            # Print some info.
            print('Finish the epoch {} for WHOLE'.format(epoch))

        # Finish the whole training phrase.
        return


    def test(self, instances_num, is_validate, is_e2e=True):
        r'''
            Test or validate the model of given iterations.

            ** Note that, it's evaluation metric is mainly designed for 3-D data.
        '''

        # Check validity.
        if not isinstance(instances_num, int):
            raise TypeError('The instances_num must be an integer !!!')
        if instances_num <= 0:
            raise ValueError('The instances_num must be positive !!!')
        if not isinstance(is_validate, bool):
            raise TypeError('The is_validate must be a boolean !!!')
        if not isinstance(is_e2e, bool):
            raise TypeError('The is_e2e must be a boolean !!!')

        # Indicating -------------------
        if is_validate:
            self._logger.info('Validate...')
        else:
            self._logger.info('Testing...')
        self._logger.info('\t Instance Number: {}'.format(instances_num))
        sys.stdout.flush()
        # ------------------------------

        # The cost time.
        start_time = time.time()  # time.

        # Different test method according to the flag.
        if is_e2e:
            mean_dice, mean_brats, mean_reward = self._end2end_test(instances_num, is_validate)
        else:
            mean_dice, mean_brats = self._customize_test(instances_num, is_validate)
            mean_reward = None

        # Show some information. ---------------------------------------
        self._logger.info('Finish testing --> average_reward: {}, avg_DICE: {}, '
                          'avg_BRATS: {}, cost time: {}'.format(
            mean_reward, mean_dice, mean_brats, time.time() - start_time))
        sys.stdout.flush()
        # ---------------------------------------------

        # Finish Testing. And return the reward info list.
        return mean_reward, mean_dice, mean_brats



    def _check_model(self):
        r'''
            Check whether to load model from previous parameters or initialize it.
        '''
        conf_others = self._config['Others']
        params_dir = conf_others.get('net_params')
        restore_from_bp = conf_others.get('restore_breakpoint', True)
        if params_dir is not None:
            if not params_dir.endswith('/'): params_dir += '/'
            var_list = None if restore_from_bp else tf.trainable_variables()
            self._saver = tf.train.Saver(var_list=var_list)
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


    def _customize_pretrain(self, cur_epoch, max_loop, start_loop):
        r'''
            Pre-train the "Segmentation" network for better performance.
        '''

        # Get config.
        conf_base = self._config['Base']
        input_shape = conf_base.get('input_shape')[1:3]
        clazz_dim = conf_base.get('classification_dimension')
        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)
        clazz_weights = conf_train.get('clazz_weights', None)
        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        step_thres = conf_dqn.get('step_threshold', 10)
        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')
        conf_others = self._config['Others']
        save_steps = conf_others.get('pre_save_steps', 400)
        validate_steps = conf_others.get('pre_validate_steps', 500)
        valnum_per = conf_others.get('instances_per_validate', 2)
        params_dir = conf_others.get('net_params')

        # Get origin input part.
        if double_q is None:
            org_name = self._name_space
        else:
            org_name = double_q[0]
        x1 = self._inputs[org_name + '/image']
        x2 = self._inputs[org_name + '/prev_result']
        x3 = self._inputs[org_name + '/position_info']
        x4 = self._inputs[org_name + '/Focus_Bbox']
        x5 = self._inputs[org_name + '/Bbox_History']
        x6 = self._inputs[org_name + '/History_Length']
        # Get loss input part.
        l1 = self._losses[self._name_space + '/GT_label']
        l2 = self._losses[self._name_space + '/clazz_weights']
        # Get loss value holder.
        sloss = self._losses[self._name_space + '/SEG_loss']

        # Fake data. (Focus bbox, SEG )
        focus_bboxes = np.asarray([[0.0, 0.0, 1.0, 1.0]])   # [[y1, x1, y2, x2]]
        focus_bboxes = focus_bboxes * np.ones((batch_size, 4))
        SEG_prevs = np.zeros((batch_size, input_shape[0], input_shape[1]))
        if pos_method == 'map':
            position_infos = np.ones((batch_size, input_shape[0], input_shape[1]))
        elif pos_method == 'coord':
            position_infos = focus_bboxes.copy()
        elif pos_method == 'w/o':
            position_infos = None
        else:
            raise ValueError('Unknown position information fusion method !!!')
        bboxes_his = np.asarray([[self._env.Fake_Bbox]])
        bboxes_his = bboxes_his * np.ones((batch_size, step_thres, 4))
        his_len = np.zeros(batch_size, dtype=np.int32)

        # Start to train.
        for ite in range(start_loop, max_loop):
            # Visual info.
            st_time = time.time()

            # Prepare the input batch.
            images, labels, weights, _4 = self._data_adapter.next_image_pair('Train', batch_size=batch_size)
            if clazz_weights is not None:
                weights = np.asarray([clazz_weights])
            else:
                weights = np.asarray([weights])
            weights = weights * np.ones((batch_size, clazz_dim))
            # The length of input batches, mainly for last slice.
            last_len = images.shape[0]
            # Generate the basic feed dictionary lately used for training the "Segmentation" network.
            feed_dict = {
                # Origin input part.
                x1: images[:last_len],
                x2: SEG_prevs[:last_len],
                x3: position_infos[:last_len],
                x4: focus_bboxes[:last_len],
                x5: bboxes_his[:last_len],
                x6: his_len[:last_len],
                # Loss part.
                l1: labels[:last_len],
                l2: weights[:last_len]
            }

            # Execute the training operator for "Segmentation".
            _, v1_cost = self._sess.run([self._pre_op, sloss], feed_dict=feed_dict)

            # Print some info. --------------------------------------------
            self._logger.info("Epoch: {}, Iter: [{}/{}] --> SEG loss: {}, cost time: {} ".format(
                cur_epoch, ite+1, max_loop, v1_cost, time.time() - st_time))
            # ------------------------------------------------------------

            # Get the pretrain step lately used to determine whether to save model or not.
            step = self._pre_step.eval(self._sess)
            # Calculate the summary to get the statistic graph.
            if step > 0 and step % validate_steps == 0:
                # Get summary holders.
                s1 = self._summary[self._name_space + '/DICE']
                s2 = self._summary[self._name_space + '/BRATS_metric']
                out_summary = self._summary[self._name_space + '/SEG_Summaries']
                # Execute "Validate".
                _1, DICE_list, BRATS_list = self.test(valnum_per, is_validate=True, is_e2e=False)
                # Print some info. --------------------------------------------
                self._logger.info("Epoch: {}, Iter: [{}/{}] --> DICE: {}, BRATS: {} ".format(
                    cur_epoch, ite+1, max_loop, DICE_list, BRATS_list))
                # ------------------------------------------------------------
                # Compute the summary value and add into statistic graph.
                feed_dict[s1] = DICE_list
                feed_dict[s2] = BRATS_list
                summary = self._sess.run(out_summary, feed_dict=feed_dict)
                self._summary_writer.add_summary(summary, step)
                self._summary_writer.flush()
            # Save the model (parameters) within the fix period.
            if step > 0 and step % save_steps == 0:
                self._saver.save(self._sess, params_dir, 233)
        # Finish.
        return


    def _train(self, max_iteration, cur_epoch, start_pos, epsilon_book, total_presteps):
        r'''
            End-to-end train the whole model.
        '''

        # Get config.
        conf_dqn = self._config['DQN']
        prioritized_replay = conf_dqn.get('prioritized_replay', True)
        conf_others = self._config['Others']
        memory_size = conf_others.get('replay_memories')
        eins_num = conf_others.get('environment_instances', 16)
        validate_steps = conf_others.get('e2e_validate_steps', 500)
        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)
        replay_iter = conf_train.get('replay_iter', 1)
        conf_others = self._config['Others']
        visualize_interval = conf_others.get('visualize_interval_per_2dimage', 25)

        # Declare the store function for "Experience Store".
        def store_2mem(experiences):
            r'''
                experience likes below:
                (sample_meta, (SEG_prev, cur_bbox, position_info, acts_prev), \
                    action, terminal, anchors.copy(), BBOX_errs.copy(), \
                    (SEG_cur, focus_bbox, next_posinfo, acts_cur))
            '''
            # Store all the experiences in "Replay Memory".
            for exp in experiences:
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

        # Calculate the max loops for current epoch.
        max_turn = validate_steps * replay_iter
        # Calculate the decrease extent of global step.
        #   Coz there's remainder util the end of each epoch.
        decrease_extent = cur_epoch * ((max_iteration - (max_iteration // max_turn) * max_turn) // replay_iter)

        # Calculate the instance that "Focus Model" already finished.
        #   It's also the start instance id for very beginning loop.
        model_finished = cur_epoch * max_iteration + start_pos
        # Calculate the loop threshold for very beginning loop.
        loop_thres = max_turn - (start_pos % max_turn)
        # Calculate the finished loops from start position.
        finished_loops = int(np.floor(start_pos / max_turn))
        # Calculate the remain loops.
        remain_loops = int(np.ceil(max_iteration / max_turn)) - finished_loops

        # ------------------------ Train Loop Part ------------------------
        for loop_id in range(remain_loops):
            # Reset the start instance id at the very beginning of each epoch.
            self._env.reset_instance_id(p='Train', ins_id=model_finished)
            # Calculate the max iter for current loop.
            tp_maxIter = self._env.instance_id(p='Train') + loop_thres

            # The previous training position.
            prev_tpos = model_finished
            # The previous animation position.
            prev_aniPos = model_finished

            # Some (metric) info holder.
            turn_rewards = [0] * eins_num
            turn_sts = [time.time()] * eins_num
            turn_steps = [0] * eins_num
            # Loop related.
            loop_time = time.time()

            # -------------------------- Start to training ... -----------------------------
            while True:
                # Get current epsilon.
                for idx in range(len(epsilon_book) - 1):
                    lower = epsilon_book[idx][0]
                    upper = epsilon_book[idx + 1][0]
                    if model_finished in range(lower, upper):
                        self._epsilon = epsilon_book[idx][1]
                        break

                # Ensure environment works in "Train" mode, what's more, reset the max iteration of environment.
                self._env.switch_phrase(p='Train', max_iter=tp_maxIter)

                # Determine whether should record picture or not.
                apos_remain = prev_aniPos // visualize_interval * visualize_interval
                apos_count = (model_finished - apos_remain) // visualize_interval
                if apos_count > 0:
                    anim_type = 'pic'
                    prev_aniPos = model_finished
                else:
                    anim_type = None

                # Single roll out.
                experiences, terminals, rewards, infos, reach_max, sample_ids, env_ids = self._env.roll_out(
                    segment_func=self._segment_func,
                    op_func=self._core_func,
                    anim_type=anim_type
                )   # unknown batch size.

                # Store the experiences to the replay buffer.
                store_2mem(experiences)

                # Traverse to check the terminal situation so that we can increase
                #   the total model finish number.
                # What's more, update and print some information.
                for eid, sid, t, r, i in zip(env_ids, sample_ids, terminals, rewards, infos):
                    # Update some information for visual.
                    turn_rewards[eid] += r
                    turn_steps[eid] += 1
                    if t:
                        # Show some info. --------------------------------------------
                        self._logger.debug(
                            "Epoch: {}, Loop: [{}/{}], Iter: [{}/{}, id - {}] "
                            "--> total_steps: {} total_rewards: {}, epsilon: {}, cost time: {}".format(
                                cur_epoch, loop_id+1, remain_loops,
                                model_finished + 1 - cur_epoch * max_iteration, max_iteration,
                                sid - cur_epoch * max_iteration,
                                turn_steps[eid], turn_rewards[eid],
                                self._epsilon, time.time() - turn_sts[eid]))
                        # ------------------------------------------------------------
                        turn_steps[eid] = 0
                        turn_rewards[eid] = 0
                        turn_sts[eid] = time.time()
                        # Increase the total model finish number.
                        model_finished += 1

                # Calculate the training count.
                train_count = ((prev_tpos % replay_iter) + (model_finished - prev_tpos)) // replay_iter
                # Only update previous train position when train count greater than 0. Reserve remain value.
                if train_count > 0:
                    prev_tpos = model_finished
                # Check whether to training or not.
                exec_train = (len(self._replay_memory) >= batch_size) and train_count > 0
                # Start training the DQN agent.
                if exec_train:
                    for tc in range(train_count):
                        # Metric used to see the training time.
                        train_time = time.time()
                        # Really training the DQN agent.
                        v1_cost, v2_cost, v3_cost, bias_rew = self.__do_train(total_presteps, decrease_extent)
                        # Debug. ------------------------------------------------------
                        self._logger.info("Epoch: {}, Loop: [{}/{}], Iter: [{}/{}], "
                                          "TrainCount: {} - Net Loss: {}, SEG Loss: {}, DQN Loss: {}, "
                                          "Reward Bias: {}, Training time: {}".format(
                            cur_epoch, loop_id+1, remain_loops,
                            model_finished - cur_epoch * max_iteration - (finished_loops + loop_id) * max_turn, max_turn,
                            tc, v1_cost, v2_cost, v3_cost, bias_rew, time.time() - train_time)
                        )
                        # -------------------------------------------------------------

                # Break the loop when reach the max iteration.
                if reach_max:
                    break
            # -------------------------- End of training core part -----------------------------

            # Check the validity of total finished number.
            MF_valid = min(cur_epoch * max_iteration + (finished_loops + loop_id + 1) * max_turn,
                           (cur_epoch + 1) * max_iteration)
            if model_finished != MF_valid:
                raise Exception('Error coding !!! model_finished: {}, MF_valid: {}'.format(model_finished, MF_valid))
            # Reset the loop threshold.
            loop_thres = min((cur_epoch + 1) * max_iteration - model_finished, max_turn)

            # Print some information.
            self._logger.debug("Epoch: {}, Loop: [{}/{}], finished !!!, cost time: {}".format(
                cur_epoch, loop_id, remain_loops, time.time() - loop_time))

        # Finish one epoch.
        return


    # The real training code.
    def __do_train(self, total_presteps, decrease_extent):
        r'''
            The function is used to really execute the one-iteration training for whole model.

            ** Note that, The experience (the element of memory storage) consists of:
                (sample_meta, (SEG_prev, cur_bbox, position_info, acts_prev, bboxes_prev, his_plen, comp_prev),
                    action, terminal, anchors, BBOX_errs,
                    (SEG_cur, focus_bbox, next_posinfo, acts_cur, bboxes_cur, his_clen, comp_cur), reward)
        '''

        # Get config.
        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)

        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        prioritized_replay = conf_dqn.get('prioritized_replay', True)
        gamma = conf_dqn.get('discount_factor', 0.9)

        conf_others = self._config['Others']
        store_type = conf_others.get('sample_type', 'heave')
        save_steps = conf_others.get('e2e_save_steps', 100)
        validate_steps = conf_others.get('e2e_validate_steps', 500)
        valnum_per = conf_others.get('instances_per_validate', 2)
        params_dir = conf_others.get('net_params')

        # ------------------------------- Data Preparation ------------------------
        # The input batch holders. -- sample meta
        images = []
        labels = []
        clazz_weights = []
        # state
        SEG_prevs = []
        cur_bboxes = []
        position_infos = []
        acts_prevs = []
        bboxes_prevs = []
        his_plens = []
        comp_prevs = []
        # dqn
        actions = []
        terminals = []
        anchors = []
        BBOX_errs = []
        # next state
        SEG_curs = []
        focus_bboxes = []
        next_pos_infos = []
        acts_curs = []
        bboxes_curs = []
        his_clens = []
        comp_curs = []
        # The previous reward.
        VIS_reward = []

        # The function used to allocate each element to its corresponding batch.
        def allocate_2batches(data_batch):
            for sample in data_batch:
                if store_type == 'light':
                    mha_idx, inst_idx, weights = sample[0]
                    img, lab = self._data_adapter.precise_locate((mha_idx, inst_idx))
                elif store_type == 'heave':
                    img, lab, weights = sample[0]
                else:
                    raise ValueError('Unknown sample store type !!!')
                # Data meta.
                images.append(img)
                labels.append(lab)
                clazz_weights.append(weights)
                # state.
                s11, s12, s13, s14, s15, s16, s17 = sample[1]
                SEG_prevs.append(s11)
                cur_bboxes.append(s12)
                position_infos.append(s13)
                acts_prevs.append(s14)
                bboxes_prevs.append(s15)
                his_plens.append(s16)
                comp_prevs.append(s17)
                # dqn.
                actions.append(sample[2])
                terminals.append(sample[3])
                anchors.append(sample[4])
                BBOX_errs.append(sample[5])
                # next state.
                s61, s62, s63, s64, s65, s66, s67 = sample[6]
                SEG_curs.append(s61)
                focus_bboxes.append(s62)
                next_pos_infos.append(s63)
                acts_curs.append(s64)
                bboxes_curs.append(s65)
                his_clens.append(s66)
                comp_curs.append(s67)
                # previous reward.
                VIS_reward.append(sample[7])
            return

        # Prioritized or randomly select a mini batch for "Focus" (DQN) branch.
        if prioritized_replay:
            # Prioritized replay. So the we will get three batches: tree_idx, mini_batches, ISWeights.
            tree_idx, exp_batch, ISWeights = self._replay_memory.sample(batch_size)
        else:
            # Randomly select experience batch (uniform replay)
            exp_batch = random.sample(self._replay_memory, batch_size)
        allocate_2batches(exp_batch)

        # Calculate the target Q values.
        #   1) get next Q values, so here we pass through
        #       (SEG_curs, focus_bboxes, next_pos_infos, acts_curs, bboxes_curs, bboxes_clens, comp_curs)
        #   2) iteratively add "0." or "next-q-val" as the target Q values.
        target_q_values = []
        next_q_values = self.__next_Qvalues((images, SEG_curs, next_pos_infos, focus_bboxes, acts_curs,
                                             bboxes_curs, his_clens, comp_curs))
        for t, nqv in zip(terminals, next_q_values):
            if t:
                target_q_values.append(0.)
            else:
                target_q_values.append(gamma * nqv)     # Discounted future reward

        # ------------------------------- Train Model ------------------------
        # Get origin input part.
        if double_q is None:
            org_name = self._name_space
        else:
            org_name = double_q[0]
        x1 = self._inputs[org_name + '/image']
        x2 = self._inputs[org_name + '/prev_result']
        x3 = self._inputs[org_name + '/position_info']
        x4 = self._inputs[org_name + '/Focus_Bbox']
        x5 = self._inputs[org_name + '/actions_history']
        x6 = self._inputs[org_name + '/Bbox_History']
        x7 = self._inputs[org_name + '/History_Length']
        x8 = self._inputs[self._name_space + '/Complete_Result']
        # Get loss input part.
        l1 = self._losses[self._name_space + '/GT_label']
        l2 = self._losses[self._name_space + '/clazz_weights']
        l3 = self._losses[self._name_space + '/prediction_actions']
        l4 = self._losses[self._name_space + '/target_Q_values']
        l5 = self._losses[self._name_space + '/Candidates_Bbox']
        l6 = self._losses[self._name_space + '/BBOX_err']
        # Get loss value holder.
        sloss = self._losses[self._name_space + '/SEG_loss']
        dloss = self._losses[self._name_space + '/DQN_loss']
        nloss = self._losses[self._name_space + '/NET_loss']
        dreward = self._losses[self._name_space + '/DQN_Rewards']

        # Generate the basic feed dictionary lately used for training the whole model.
        feed_dict = {
            # Segmentation part.
            x8: comp_prevs,
            # Origin input part.
            x1: images,
            x2: SEG_prevs,
            x3: position_infos,
            x4: cur_bboxes,
            x5: acts_prevs,
            x6: bboxes_prevs,
            x7: his_plens,
            # Loss part.
            l1: labels,
            l2: clazz_weights,
            l3: actions,
            l4: target_q_values,
            l5: anchors,
            l6: BBOX_errs
        }

        # Do train.
        if prioritized_replay:
            # Used to update priority. Coz the "Prioritized Replay" need to update the
            #   priority for experience.
            ISw = self._losses[self._name_space + '/IS_weights']
            exp_pri = self._losses[self._name_space + '/EXP_priority']
            # Add to feed dictionary.
            feed_dict[ISw] = ISWeights
            # Execute the training operation.
            _, exp_priority, v1_cost, v2_cost, v3_cost, v4_rew = self._sess.run(
                [self._train_op, exp_pri, nloss, sloss, dloss, dreward], feed_dict=feed_dict)
            # Update probability for leafs.
            self._replay_memory.batch_update(tree_idx, exp_priority)  # update priority
        else:
            # Simply feed the dictionary to the selected operation is OK.
            _, v1_cost, v2_cost, v3_cost, v4_rew = self._sess.run(
                [self._train_op, nloss, sloss, dloss, dreward], feed_dict=feed_dict)

        # Compute the bias of "Training-Reward" and "Previous-Reward".
        bias_reward = np.mean(v4_rew) - np.mean(VIS_reward)

        # ------------------------------- Save Model Parameters ------------------------
        # Get the global step lately used to determine whether to save model or not.
        step = self._global_step.eval(self._sess)
        # Rectify the global step.
        step -= total_presteps
        step -= decrease_extent
        # Calculate the summary to get the statistic graph.
        if step > 0 and step % validate_steps == 0:
            # Get summary holders.
            s1 = self._summary[self._name_space + '/Reward']
            s2 = self._summary[self._name_space + '/DICE']
            s3 = self._summary[self._name_space + '/BRATS_metric']
            out_summary = self._summary[self._name_space + '/WHOLE_Summaries']
            # Reset the sample id for "Validate" mode.
            val_sampleId = (step//validate_steps-1) * valnum_per * self._data_adapter.slices_3d
            self._env.reset_instance_id(p='Validate', ins_id=val_sampleId)
            # Execute "Validate".
            reward_list, DICE_list, BRATS_list = self.test(valnum_per, is_validate=True)
            # Compute the summary value and add into statistic graph.
            feed_dict[s1] = reward_list
            feed_dict[s2] = DICE_list
            feed_dict[s3] = BRATS_list
            summary = self._sess.run(out_summary, feed_dict=feed_dict)
            self._summary_writer.add_summary(summary, step + total_presteps + decrease_extent)
            self._summary_writer.flush()
        # Save the model (parameters) within the fix period.
        if step > 0 and step % save_steps == 0:
            self._saver.save(self._sess, params_dir, 233)
        # Regularly copy the parameters to "Target" DQN network.
        if double_q is not None and step % save_steps == 0:
            copy_ops = self._losses[self._name_space + '/copy_params']
            self._sess.run(copy_ops)
        # Finish one turn train. Return the cost value.
        return v1_cost, v2_cost, v3_cost, bias_reward


    def _core_func(self, x, with_explore, explore_valids, with_reward):
        r'''
            Use model to generate the segmentation result for current region.
            What's more, it will use the Q values generated by DQN to select action
                or random select actions according to the epsilon value (which
                indicates it's "Exploitation" or "Exploration" phrase).

            ** Note that, this function will be called in all phrase. It's the
                core function. The return elements depends on input.

        ----------------------------------------------------------------
        Parameters:
            x: The current observations of environment.
                When with rewards, each of it consists of:
                    (image, SEG_prev, position_info, focus_bbox, acts_prev, bboxes_prev, his_plen, COMP_res,
                    anchors, BBOX_errs, label)
                Otherwise:
                    (image, SEG_prev, position_info, focus_bbox, acts_prev, bboxes_prev, his_plen, COMP_res)
            with_explore: The flag indicates whether enable the "E-greedy Exploration" or not.
            explore_valids: The valid actions list of current time-step of environment that will be used
                as the indicator for "Exploration" .
            with_reward: The flag that indicating whether compute rewards or not.
        ----------------------------------------------------------------
        Return:
            The (whole-view) segmentation result, and the complete result value
                (for next segmentation of iteration).
            What's more, return the action. (And rewards if given)
        '''

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
        x4 = self._inputs[stage_prefix + '/Focus_Bbox']
        x5 = self._inputs[stage_prefix + '/actions_history']
        x6 = self._inputs[stage_prefix + '/Bbox_History']
        x7 = self._inputs[stage_prefix + '/History_Length']
        x8 = self._inputs[self._name_space + '/Complete_Result']
        # Get reward calculation part.
        l1 = self._losses[self._name_space + '/Candidates_Bbox']
        l2 = self._losses[self._name_space + '/BBOX_err']
        l3 = self._losses[self._name_space + '/GT_label']
        l4 = self._losses[self._name_space + '/clazz_weights']
        # Get segmentation output holder of model.
        y1 = self._outputs[self._name_space + '/SEG_output']
        y2 = self._outputs[self._name_space + '/Region_Result']
        y3 = self._outputs[stage_prefix + '/DQN_output']
        y4 = self._losses[self._name_space + '/DQN_Rewards']

        # The function used to package input data.
        def package_xbatch(data):
            i1 = []
            i2 = []
            i3 = []
            i4 = []
            i5 = []
            i6 = []
            i7 = []
            i8 = []
            if with_reward:
                i9 = []
                i10 = []
                i11 = []
                i12 = []
            for d in data:
                i1.append(d[0])
                i2.append(d[1])
                i3.append(d[2])
                i4.append(d[3])
                i5.append(d[4])
                i6.append(d[5])
                i7.append(d[6])
                i8.append(d[7])
                if with_reward:
                    i9.append(d[8])
                    i10.append(d[9])
                    i11.append(d[10])
                    i12.append(d[11])
            if with_reward:
                return i1, i2, i3, i4, i5, i6, i7, i8, \
                       i9, i10, i11, i12
            else:
                return i1, i2, i3, i4, i5, i6, i7, i8
        # -----------------------------------------------------

        # Different execution logic according to the flag.
        if with_reward:
            # Get each input elements.
            images, SEG_prevs, position_infos, focus_bboxs, acts_prevs, bboxes_prevs, his_plens, \
                COMP_results, anchors, BBOX_errs, labels, clazz_weights = package_xbatch(x)
            # Generate the segmentation result for current region (focus bbox).
            segmentations, COMP_ress, q_vals, rewards = self._sess.run([y1, y2, y3, y4], feed_dict={
                # Input part.
                x1: images,
                x2: SEG_prevs,
                x3: position_infos,
                x4: focus_bboxs,
                x5: acts_prevs,
                x6: bboxes_prevs,
                x7: his_plens,
                x8: COMP_results,
                # Reward part.
                l1: anchors,
                l2: BBOX_errs,
                l3: labels,
                l4: clazz_weights
            })
        else:
            # Get input elements.
            images, SEG_prevs, position_infos, focus_bboxs, acts_prevs, \
                bboxes_prevs, his_plens, COMP_results = package_xbatch(x)
            # Generate the segmentation result for current region (focus bbox).
            segmentations, COMP_ress, q_vals = self._sess.run([y1, y2, y3], feed_dict={
                x1: images,
                x2: SEG_prevs,
                x3: position_infos,
                x4: focus_bboxs,
                x5: acts_prevs,
                x6: bboxes_prevs,
                x7: his_plens,
                x8: COMP_results,
            })

        # Select action according to the output of "Deep Q Network".
        actions = np.argmax(q_vals, axis=-1)    # [?]
        # e-greedy action policy.
        if with_explore:
            actions = self.__egreedy_action(actions, explore_valids=explore_valids, epsilon=self._epsilon)  # [?]

        # Return both segmentation result and DQN result. What's more, return reward if given.
        if with_reward:
            return segmentations, COMP_ress, actions, rewards
        else:
            return segmentations, COMP_ress, actions


    def _segment_func(self, x):
        r'''
            Use model to generate the segmentation result for current region.

            ** Note that, this function should be called in "Validate" or "Test" phrase.

        ----------------------------------------------------------------
        Parameters:
            x: The current observation of environment. It consists of:
                (image, SEG_prev, position_info, focus_bbox, bboxes_prev, his_plen, COMP_res)
        ----------------------------------------------------------------
        Return:
            The (whole-view) segmentation result, and the complete result value
                (for next segmentation of iteration).
        '''

        # Get each input data.
        image, SEG_prev, position_info, focus_bbox, bboxes_prev, his_plen, COMP_result = x

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
        x4 = self._inputs[stage_prefix + '/Focus_Bbox']
        x5 = self._inputs[stage_prefix + '/Bbox_History']
        x6 = self._inputs[stage_prefix + '/History_Length']
        x7 = self._inputs[self._name_space + '/Complete_Result']
        # Get segmentation output holder of model.
        y1 = self._outputs[self._name_space + '/SEG_output']
        y2 = self._outputs[self._name_space + '/Region_Result']

        # Generate the segmentation result for current region (focus bbox).
        segmentation, COMP_res = self._sess.run([y1, y2], feed_dict={
            x1: [image],
            x2: [SEG_prev],
            x3: [position_info],
            x4: [focus_bbox],
            x5: [bboxes_prev],
            x6: [his_plen],
            x7: [COMP_result]
        })
        # Get the segmentation and complete value result.
        segmentation = segmentation[0]
        COMP_res = COMP_res[0]
        return segmentation, COMP_res


    def __explore_policy(self, explore_valids, distribution='uniform'):
        r'''
            Randomly select an action.
        '''
        # Randomly select actions (only in valid actions).
        rand_actions = []
        for ev in explore_valids:
            valid_acts = np.where(ev)[0]
            ract_idx = np.random.randint(len(valid_acts))
            rand_actions.append(valid_acts[ract_idx])
        return rand_actions
        # return np.random.randint(self._acts_dim, size=vector_len)


    def __egreedy_action(self, actions, explore_valids, epsilon):
        r'''
            The ε-greedy exploration for DQN agent.
        '''
        # Check the validity.
        if len(actions) != len(explore_valids):
            raise Exception('The length of actions and explore_valids must be same !!!')
        # Exploration or Exploitation according to the epsilon.
        exploration_flag = np.random.rand(len(actions)) <= epsilon  # [?,]
        exploration_actions = self.__explore_policy(explore_valids)    # [?,]
        action_indexs = np.where(exploration_flag, exploration_actions, actions)    # [?,]
        return action_indexs


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
        images, SEG_prevs, position_infos, focus_bboxes, acts_prevs, bboxes_prevs, his_plens, comp_prevs = x

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
        x4 = self._inputs[org_name + '/Focus_Bbox']
        x5 = self._inputs[org_name + '/actions_history']
        x6 = self._inputs[org_name + '/Bbox_History']
        x7 = self._inputs[org_name + '/History_Length']
        x8 = self._inputs[self._name_space + '/Complete_Result']
        # Get DQN output holder of model.
        y_org = self._outputs[org_name + '/DQN_output']

        # Calculate the Q values according to the "Double DQN" mode.
        if double_q is None:
            # Pure mode, use origin model to compute the Q values.
            q_vals = self._sess.run(y_org, feed_dict={
                x1: images,
                x2: SEG_prevs,
                x3: position_infos,
                x4: focus_bboxes,
                x5: acts_prevs,
                x6: bboxes_prevs,
                x7: his_plens,
                x8: comp_prevs
            })  # [batch, act_dim]
            # Use the max values of actions as next Q value.
            next_q_values = np.max(q_vals, axis=-1)     # [batch]
        else:
            # Get loss holders of model.
            l1 = self._losses[tar_name + '/image']
            l2 = self._losses[tar_name + '/prev_result']
            l3 = self._losses[tar_name + '/position_info']
            l4 = self._losses[tar_name + '/Focus_Bbox']
            l5 = self._losses[tar_name + '/actions_history']
            l6 = self._losses[tar_name + '/Bbox_History']
            l7 = self._losses[tar_name + '/History_Length']
            # Get DQN output holder of model.
            y_tar = self._losses[tar_name + '/DQN_output']
            # "Double DQN" mode, use the actions selected by "target" model
            #   to filter (get) the next Q values from "origin" model.
            org_qvals, tar_qvals = self._sess.run([y_org, y_tar], feed_dict={
                # segmentation part.
                x8: comp_prevs,
                # origin part
                x1: images,
                x2: SEG_prevs,
                x3: position_infos,
                x4: focus_bboxes,
                x5: acts_prevs,
                x6: bboxes_prevs,
                x7: his_plens,
                # target part
                l1: images,
                l2: SEG_prevs,
                l3: position_infos,
                l4: focus_bboxes,
                l5: acts_prevs,
                l6: bboxes_prevs,
                l7: his_plens,
            })
            # Generate the action from "target" and get Q values from "origin".
            tar_acts = np.argmax(tar_qvals, axis=-1)    # [batch]
            tar_acts = np.eye(self._acts_dim)[tar_acts]     # [batch, act_dim]
            next_q_values = np.sum(org_qvals * tar_acts, axis=-1)   # [batch]

        # Return the next Q values.
        return next_q_values


    def _end2end_test(self, instances_num, is_validate):
        r'''
            Test or validate the whole model (including segmentation and DQN branch)
                for given iterations.

            ** Note that, it's evaluation metric is mainly designed for 3-D data.

        Parameters:
            instances_num: The number of instances used to validate or test.
            is_validate: Flag that indicates whether it's validate or test phrase.

        Return:
            The Brats-metric and Dice-metric of each category.
        '''

        # Get config.
        conf_base = self._config['Base']
        clazz_dim = conf_base.get('classification_dimension')
        img_h, img_w = conf_base.get('input_shape')[1:3]
        conf_others = self._config['Others']
        visualize_interval = conf_others.get('visualize_interval_per_2dimage', 25)

        # Check whether need additional iteration to deal with 3-D data.
        if self._data_adapter.slices_3d <= 0:
            iter_3d = 1
        else:
            iter_3d = self._data_adapter.slices_3d

        # Metrics.
        total_rewards = np.zeros((instances_num, iter_3d))  # Rewards
        dice_metrics = np.zeros((instances_num, clazz_dim))  # Dice Metric.
        brats_metrics = np.zeros((instances_num, 3))  # BRATS Metric.
        # Temporary holders.
        preds_holder = np.zeros((instances_num, iter_3d, img_h, img_w))  # Predictions
        labels_holder = np.zeros((instances_num, iter_3d, img_h, img_w))  # Labels
        MHAs_holder = np.zeros((instances_num, iter_3d))  # MHA ids

        # Ensure the right phrase of "environment".
        if is_validate:
            # The oldest (from beginning).
            oldest_insId = self._env.instance_id(p='Validate')
            max_iter = oldest_insId + instances_num * iter_3d
            self._env.switch_phrase(p='Validate', max_iter=max_iter)
        else:
            # The oldest (from beginning).
            oldest_insId = self._env.instance_id(p='Test')
            max_iter = oldest_insId + instances_num * iter_3d
            self._env.switch_phrase(p='Test', max_iter=max_iter)

        # The total model finished number and the previous training position.
        model_finished = oldest_insId
        prev_aniPos = oldest_insId

        # --------------------------------------- Core body ---------------------------------------
        # Start to Testing ...
        while True:
            # Determine whether should record animation or not.
            apos_remain = prev_aniPos // visualize_interval * visualize_interval
            apos_count = (model_finished - apos_remain) // visualize_interval
            if apos_count > 0:
                anim_type = 'video'
                prev_aniPos = model_finished
            else:
                anim_type = None

            # Single roll out.
            segmentations, labels, terminals, rewards, infos, reach_max, \
            sample_ids, env_ids, data_identities = self._env.roll_out(
                segment_func=self._segment_func,
                op_func=self._core_func,
                anim_type=anim_type
            )  # unknown batch size.

            # Record predictions and labels. What's more, update the total model finished number.
            for eid, sid, di, s, t, l, r, i in zip(
                    env_ids, sample_ids, data_identities, segmentations, terminals, labels, rewards, infos):
                # Calculate the batch_id and the iter_id.
                batch_id = (sid - oldest_insId) // iter_3d
                iter_id = (sid - oldest_insId) % iter_3d
                # Update metric info.
                if is_validate:
                    total_rewards[batch_id, iter_id] += r
                # Record prediction and its label.
                if t:
                    if is_validate:
                        labels_holder[batch_id, iter_id] = l
                    preds_holder[batch_id, iter_id] = s
                    MHAs_holder[batch_id, iter_id] = di[0]
                    # Update the total model finished number.
                    model_finished += 1

            # Break the loop if finish max iterations.
            if reach_max:
                break
        # ---------------------------------- End of loop ----------------------------------

        # Check the validity of whole operation. The mha_id of same instance must be same.
        for mha in MHAs_holder:
            if ((mha - np.mean(mha)) != 0).any():
                raise Exception('Error coding !!! The The mha_id of same instance must be same !!!')

        # Calculate total metric or write the result according to the phrase.
        if is_validate:
            for ins_id, elem in enumerate(zip(preds_holder, labels_holder)):
                pred_3d, label_3d = elem
                brats_1 = eva.BRATS_Complete(pred=pred_3d, label=label_3d)
                brats_2 = eva.BRATS_Core(pred=pred_3d, label=label_3d)
                brats_3 = eva.BRATS_Enhance(pred=pred_3d, label=label_3d)
                brats_vector = np.asarray([brats_1, brats_2, brats_3])  # [3]
                brats_metrics[ins_id] = brats_vector  # [instance, 3]
                cate_dice = []
                for c in range(clazz_dim):
                    cate_pred = pred_3d == c
                    cate_lab = label_3d == c
                    cdice = eva.DICE_Bi(pred=cate_pred, label=cate_lab)
                    cate_dice.append(cdice)
                cate_dice = np.asarray(cate_dice)  # [category]
                dice_metrics[ins_id] = cate_dice  # [instance, category]
        else:
            for ins_id, elem in enumerate(zip(MHAs_holder, preds_holder)):
                mha_id, pred_3d = elem
                self._data_adapter.write_result(MHA_id=mha_id[0], result=pred_3d, name=str(ins_id))

        # Mean the metrics.
        DICE = np.mean(dice_metrics, axis=0)    # [category]
        BRATS = np.mean(brats_metrics, axis=0)  # [3]
        REWARD = np.mean(total_rewards) # scalar

        # Return the metrics.
        return DICE, BRATS, REWARD


    def _customize_test(self, instances_num, is_validate):
        r'''
            Test or validate the pure segmentation branch of model for given iterations.

            ** Note that, it's evaluation metric is mainly designed for 3-D data.

        Parameters:
            instances_num: The number of instances used to validate or test.
            is_validate: Flag that indicates whether it's validate or test phrase.

        Return:
            The Brats-metric and Dice-metric of each category.
        '''

        # Get config.
        conf_base = self._config['Base']
        input_shape = conf_base.get('input_shape')[1:3]
        suit_h = conf_base.get('suit_height')
        suit_w = conf_base.get('suit_width')
        clazz_dim = conf_base.get('classification_dimension')
        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)
        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        step_thres = conf_dqn.get('step_threshold', 10)
        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')
        CR_method = conf_cus.get('result_fusion', 'prob')
        conf_others = self._config['Others']
        validate_steps = conf_others.get('pre_validate_steps', 500)

        # Get origin input part.
        if double_q is None:
            org_name = self._name_space
        else:
            org_name = double_q[0]
        x1 = self._inputs[org_name + '/image']
        x2 = self._inputs[org_name + '/prev_result']
        x3 = self._inputs[org_name + '/position_info']
        x4 = self._inputs[org_name + '/Focus_Bbox']
        x5 = self._inputs[org_name + '/Bbox_History']
        x6 = self._inputs[org_name + '/History_Length']

        # Fake data. (Focus bbox, SEG )
        focus_bboxes = np.asarray([[0.0, 0.0, 1.0, 1.0]])  # [[y1, x1, y2, x2]]
        focus_bboxes = focus_bboxes * np.ones((batch_size, 4))
        SEG_prevs = np.zeros((batch_size, input_shape[0], input_shape[1]))
        if pos_method == 'map':
            position_infos = np.ones((batch_size, input_shape[0], input_shape[1]))
        elif pos_method == 'coord':
            position_infos = focus_bboxes.copy()
        elif pos_method == 'w/o':
            position_infos = None
        else:
            raise ValueError('Unknown position information fusion method !!!')
        bboxes_his = np.asarray([[self._env.Fake_Bbox]])
        bboxes_his = bboxes_his * np.ones((batch_size, step_thres, 4))
        his_len = np.zeros(batch_size, dtype=np.int32)

        # The sub-function used to "Validate". Meanwhile define some holders and data.
        x7 = self._inputs[self._name_space + '/Complete_Result']
        o1 = self._outputs[self._name_space + '/SEG_output']
        if CR_method == 'logit' or CR_method == 'prob':
            COMP_results = np.zeros((batch_size, step_thres, suit_h, suit_w, clazz_dim), dtype=np.float32)
        elif CR_method in ['mask-lap', 'mask-vote']:
            COMP_results = np.zeros((batch_size, step_thres, suit_h, suit_w), dtype=np.int64)
        else:
            raise ValueError('Unknown result fusion method !!!')

        # Calculate the instance id.
        step = self._pre_step.eval(self._sess)
        last_insId = (step//validate_steps-1) * instances_num * int(np.ceil(self._data_adapter.slices_3d/batch_size))

        # Start validate/test loop ...
        BRATSs = []
        DICEs = []
        for i in range(instances_num):
            mha_id = -1
            pred_3d = []
            lab_3d = []
            for j in range(self._data_adapter.slices_3d // batch_size + 1):
                if is_validate:
                    imgs, labs, _3, data_args = self._data_adapter.next_image_pair('Validate', batch_size=batch_size)
                else:
                    imgs, labs, data_args = self._data_adapter.next_image_pair('Test', batch_size=batch_size)
                if mha_id < 0: mha_id = data_args[0]
                la_l = imgs.shape[0]
                feed_dict = {
                    # Origin input part.
                    x1: imgs[:la_l],
                    x2: SEG_prevs[:la_l],
                    x3: position_infos[:la_l],
                    x4: focus_bboxes[:la_l],
                    x5: bboxes_his[:la_l],
                    x6: his_len[:la_l],
                    x7: COMP_results[:la_l]
                }
                preds = self._sess.run(o1, feed_dict=feed_dict)
                pred_3d.extend(preds)
                lab_3d.extend(labs)
                # Visualize.
                if self._pre_visutil is not None:
                    ins_id = last_insId + i * (self._data_adapter.slices_3d // batch_size + 1) + j
                    save_dir = 'Train' if is_validate else 'Test'
                    self._pre_visutil.visualize(ins_id, (imgs[0], labs[0], preds[0]), mode=save_dir)
            # Calculate metrics if validate, write results if test.
            if is_validate:
                pred_3d = np.asarray(pred_3d)
                lab_3d = np.asarray(lab_3d)
                brats_1 = eva.BRATS_Complete(pred=pred_3d, label=lab_3d)
                brats_2 = eva.BRATS_Core(pred=pred_3d, label=lab_3d)
                brats_3 = eva.BRATS_Enhance(pred=pred_3d, label=lab_3d)
                BRATSs.append([brats_1, brats_2, brats_3])
                dice = []
                for c in range(clazz_dim):
                    cate_pred = pred_3d == c
                    cate_lab = lab_3d == c
                    dice.append(eva.DICE_Bi(pred=cate_pred, label=cate_lab))
                DICEs.append(dice)
            else:
                pred_3d = np.asarray(pred_3d)
                self._data_adapter.write_result(MHA_id=mha_id, result=pred_3d, name=str(i))

        # Mean the metrics.
        DICE = np.mean(np.asarray(DICEs), axis=0)   # [category]
        BRATS = np.mean(np.asarray(BRATSs), axis=0)     # [3]

        # Return the metrics.
        return DICE, BRATS


