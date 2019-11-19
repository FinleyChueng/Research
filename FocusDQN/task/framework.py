import collections
import random
import sys
import time
import tensorflow as tf

from core.dqn import *
from task.model import DqnAgent
from task.env import FocusEnv
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
        self._env = FocusEnv(self._config, data_adapter=data_adapter)

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

        # Define a unified optimizer.
        self._global_step = tf.Variable(0, trainable=False, name='global_step')
        share_learning_rate = tf.train.exponential_decay(learning_rate, self._global_step,
                                                         decay_iter, decay_rate, staircase=True)
        optimizer = tf.train.AdamOptimizer(share_learning_rate)
        # Define the training operator for "End-to-End".
        net_loss = self._losses[self._name_space + '/NET_loss']
        self._train_op = optimizer.minimize(net_loss, global_step=self._global_step)

        # Define the training operator for single "Segmentation".
        self._pre_step = tf.Variable(0, trainable=False, name='pretrain_step')
        pretrain_learning_rate = tf.train.exponential_decay(learning_rate, self._pre_step,
                                                            decay_iter, decay_rate, staircase=True)
        pretrain_optimizer = tf.train.AdamOptimizer(pretrain_learning_rate)
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

        # # finalize the graph
        # self._sess.graph.finalize()

        # Finish initialization.
        return



    # num_epochs: train epochs
    def train(self, epochs, max_iteration):
        r'''
            The "Training" phrase. Train the whole model many times.

        Parameters:
            epochs: Indicates the total epochs that whole model should train.
            max_iteration: Specify the max iteration of one epoch.
        '''

        # Check validity.
        if not isinstance(epochs, int) or not isinstance(max_iteration, int):
            raise TypeError('The epochs and max_iteration must be integer !!!')
        if epochs <= 0 or max_iteration <= 0:
            raise ValueError('The epochs and iteration should be positive !!!')

        # Get config.
        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)
        epsilon_dict = conf_train.get('epsilon_dict', None)
        replay_iter = conf_train.get('replay_iter', 1)
        pre_epochs = conf_train.get('customize_pretrain_epochs', 1)
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
            iter = int(float(k) * total_turns)
            epsilon_book.append((iter, epsilon_dict[k]))
        print('### --> Epsilon book: {}'.format(epsilon_book))

        # ---------------- Pre-train the "Segmentation" branch (model). ----------------
        print('\n\nPre-training the SEGMENTATION model !!!')
        # Determine the start position.
        if restore_from_bp:
            glo_step = int(self._global_step.eval(self._sess))
            if glo_step != 0:
                # Means it was in "End-to-end" phrase last time.
                start_epoch = pre_epochs
                start_iter = -1
            else:
                # Compute last iter.
                pre_step = int(self._pre_step.eval(self._sess))
                last_iter = pre_step * batch_size
                # Compute the start epoch and iteration.
                start_epoch = last_iter // max_iteration
                start_iter = last_iter % max_iteration
        else:
            start_epoch = start_iter = 0
        # Start train the "Segmentation" model.
        for epoch in range(start_epoch, pre_epochs):
            # Reset the start position of iteration.
            self._data_adapter.reset_position(start_iter)
            # Start training.
            self._customize_pretrain(max_iteration, start_iter)
            # Re-assign the start position iteration to zero.
            #   Note that, only the first epoch is specified,
            #   others start from zero.
            start_iter = 0
            # Print some info.
            print('Finish the epoch {} for SEGMENTATION'.format(epoch))

        # ---------------- Train the whole model many times. (Depend on epochs) ----------------
        print('\n\nEnd-to-End Training !!!')
        # Determine the start position.
        if restore_from_bp:
            # Compute last iter.
            glo_step = int(self._global_step.eval(self._sess))
            last_iter = glo_step * replay_iter
            # Compute the start epoch and iteration.
            start_epoch = last_iter // max_iteration
            start_iter = last_iter % max_iteration
        else:
            start_epoch = start_iter = 0
        # Start train the whole model.
        for epoch in range(start_epoch, epochs - pre_epochs):
            # Reset the start position of iteration.
            self._data_adapter.reset_position(start_iter)
            # Start training.
            self._train(max_iteration, start_iter, epsilon_book)
            # Re-assign the start position iteration to zero.
            #   Note that, only the first epoch is specified,
            #   others start from zero.
            start_iter = 0
            # Print some info.
            print('Finish the epoch {} for WHOLE'.format(epoch))

        # Finish the whole training phrase.
        return


    def test(self, instances_num, is_validate):
        r'''
            Test or validate the model of given iterations.

            ** Note that, it's evaluation metric is mainly designed for 3-D data.
        '''

        # Check validity.
        if not isinstance(instances_num, int):
            raise TypeError('The instances_num must be an integer !!!')
        if not isinstance(is_validate, bool):
            raise TypeError('The is_validate must be a boolean !!!')
        if instances_num <= 0:
            raise ValueError('The instances_num must be positive !!!')

        # Indicating -------------------
        self._logger.info('Testing...')
        sys.stdout.flush()
        # ------------------------------

        # Get config.
        conf_base = self._config['Base']
        clazz_dim = conf_base.get('classification_dimension')

        # Ensure the right phrase of "environment".
        if is_validate:
            self._env.switch_phrase('Validate')
        else:
            self._env.switch_phrase('Test')

        # Check whether need additional iteration to deal with 3-D data.
        if self._data_adapter.slices_3d <= 0:
            iter_3d = 1
        else:
            iter_3d = self._data_adapter.slices_3d

        # Metrics.
        total_rewards = []  # rewards
        dice_metrics = []   # Dice Metric.
        brats_metrics = []  # BRATS Metric.
        start_time = time.time()  # time.

        # Start to Testing ...
        for t in range(instances_num):
            # Metric used to record rewards of current turn if not None.
            turn_reward = []
            # 3D data holder. Used to compute the metric.
            label_3d = []
            pred_3d = []

            # 3D data
            for it in range(iter_3d):
                # Metric.
                it_reward = []

                # Reset the environment.
                label = self._env.reset()
                # Iteratively process.
                for step in range(9999999999):  # 9999999999: max step per epoch
                    # "Segment" phrase.
                    _1, (segmentation, _3, _4) = self._env.step(self._segment, SEG_stage=True)
                    # "Focus" phrase.
                    terminal, (_2, reward, info) = self._env.step(self._action, SEG_stage=False)

                    # Update metric info.
                    if is_validate:
                        it_reward.append(reward)

                    # Current slice is finish.
                    if terminal:
                        if is_validate:
                            label_3d.append(label)
                        pred_3d.append(segmentation)
                        break
                # Render environment. No need visualize each one.
                if it % (iter_3d // 6) == 0:
                    self._env.render(anim_type='video')

                # Update turn metric.
                turn_reward.append(np.sum(it_reward))

            # Update total metric or write the result according to the phrase.
            if is_validate:
                total_rewards.append(turn_reward)   # [turn, iter_3d]
                label_3d = np.asarray(label_3d)
                pred_3d = np.asarray(pred_3d)
                brats_1 = eva.BRATS_Complete(pred=pred_3d, label=label_3d)
                brats_2 = eva.BRATS_Core(pred=pred_3d, label=label_3d)
                brats_3 = eva.BRATS_Enhance(pred=pred_3d, label=label_3d)
                brats_metrics.append([brats_1, brats_2, brats_3])   # [turn, 3]
                cate_dice = []
                for c in range(clazz_dim):
                    cate_pred = pred_3d == c
                    cate_lab = label_3d == c
                    cdice = eva.DICE_Bi(pred=cate_pred, label=cate_lab)
                    cate_dice.append(cdice)
                dice_metrics.append(cate_dice)  # [turn, category]
            else:
                self._data_adapter.write_result(result=pred_3d, name=str(t))

        # ----------------------------- End of loop. -------------------------

        # Mean the metrics.
        mean_reward = np.mean(total_rewards)
        mean_dice = np.mean(dice_metrics, axis=0)
        mean_brats = np.mean(brats_metrics, axis=0)

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
        if params_dir is not None:
            if not params_dir.endswith('/'): params_dir += '/'
            # var_list = tf.trainable_variables()
            # g_list = tf.global_variables()
            # for v in g_list:
            #     if 'global_step' in v.name:
            #         var_list.append(v)
            #         break
            # self._saver = tf.train.Saver(var_list=var_list)
            self._saver = tf.train.Saver()
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


    def _customize_pretrain(self, max_iteration, start_pos):
        r'''
            Pre-train the "Segmentation" network for better performance.
        '''

        # Get config.
        conf_base = self._config['Base']
        input_shape = conf_base.get('input_shape')[1:3]
        suit_h = conf_base.get('suit_height')
        suit_w = conf_base.get('suit_width')
        clazz_dim = conf_base.get('classification_dimension')
        conf_train = self._config['Training']
        batch_size = conf_train.get('batch_size', 32)
        clazz_weights = conf_train.get('clazz_weights', None)
        conf_dqn = self._config['DQN']
        double_q = conf_dqn.get('double_dqn', None)
        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')
        CR_method = conf_cus.get('result_fusion', 'prob')
        conf_others = self._config['Others']
        save_steps = conf_others.get('save_steps', 100)
        save_steps *= 4
        validate_steps = conf_others.get('validate_steps', 500)
        validate_steps //= 4
        params_dir = conf_others.get('net_params')

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
        # Get loss value holder.
        sloss = self._losses[self._name_space + '/SEG_loss']

        # Fake data. (Focus bbox, SEG stage)
        SEG_stages = [True,] * batch_size
        focus_bboxes = np.asarray([[0.0, 0.0, 1.0, 1.0]])   # [[y1, x1, y2, x2]]
        focus_bboxes = focus_bboxes * np.ones((batch_size, 4))
        SEG_prevs = np.zeros((batch_size, input_shape[0], input_shape[1]))
        if pos_method == 'map':
            position_infos = np.ones((batch_size, input_shape[0], input_shape[1]))
        elif pos_method == 'coord':
            position_infos = focus_bboxes.copy()
        elif pos_method == 'sight':
            position_infos = focus_bboxes.copy()
        else:
            raise ValueError('Unknown position information fusion method !!!')

        # The sub-function used to "Validate". Meanwhile define some holders and data.
        x6 = self._inputs[self._name_space + '/Complete_Result']
        o1 = self._outputs[self._name_space + '/SEG_output']
        if CR_method == 'logit' or CR_method == 'prob':
            COMP_results = np.zeros((batch_size, suit_h, suit_w, clazz_dim), dtype=np.float32)
        elif CR_method == 'mask':
            COMP_results = np.zeros((batch_size, suit_h, suit_w), dtype=np.float32)
        else:
            raise ValueError('Unknown result fusion method !!!')
        # Validate function.
        def seg_validate(instance_num):
            BRATSs = []
            DICEs = []
            for _1 in range(instance_num):
                pred_3d = []
                lab_3d = []
                for _2 in range(self._data_adapter.slices_3d // batch_size + 1):
                    val_imgs, val_labs = self._data_adapter.next_image_pair('Validate', batch_size=batch_size)
                    la_l = val_imgs.shape[0]
                    feed_dict = {
                        # Origin input part.
                        x1: val_imgs[:la_l],
                        x2: SEG_prevs[:la_l],
                        x3: position_infos[:la_l],
                        x4: SEG_stages[:la_l],
                        x5: focus_bboxes[:la_l],
                        x6: COMP_results[:la_l]
                    }
                    preds = self._sess.run(o1, feed_dict=feed_dict)
                    pred_3d.extend(preds)
                    lab_3d.extend(val_labs)
                    # visual.
                    if self._pre_visutil is not None:
                        self._pre_visutil.visualize((val_imgs[0], val_labs[0], preds[0]), mode='Train')
                # metric.
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
            DICE = np.mean(np.asarray(DICEs), axis=0)      # category
            BRATS = np.mean(np.asarray(BRATSs), axis=0)    # 3
            return DICE, BRATS
        # ------------------------- end of sub-func ------------------------

        # Start to train.
        for ite in range(start_pos // batch_size, max_iteration // batch_size + 1):
            # Prepare the input batch.
            images, labels, weights, _4 = self._data_adapter.next_image_pair('Train', batch_size=batch_size)
            if clazz_weights is not None:
                weights = np.asarray([clazz_weights])
                weights = weights * np.ones((batch_size, clazz_dim))
            # The length of input batches, mainly for last slice.
            last_len = images.shape[0]
            # Generate the basic feed dictionary lately used for training the "Segmentation" network.
            feed_dict = {
                # Origin input part.
                x1: images[:last_len],
                x2: SEG_prevs[:last_len],
                x3: position_infos[:last_len],
                x4: SEG_stages[:last_len],
                x5: focus_bboxes[:last_len],
                # Loss part.
                l1: labels[:last_len],
                l2: weights[:last_len]
            }

            # Execute the training operator for "Segmentation".
            _, v1_cost = self._sess.run([self._pre_op, sloss], feed_dict=feed_dict)

            # Print some info. --------------------------------------------
            self._logger.info("Iter {} --> SEG loss: {} ".format(ite, v1_cost))
            # ------------------------------------------------------------

            # Get the pretrain step lately used to determine whether to save model or not.
            step = self._pre_step.eval(self._sess)
            # Calculate the summary to get the statistic graph.
            if step > 0 and step % validate_steps == 0:
                # Get summary holders.
                s1 = self._summary[self._name_space + '/Reward']
                s2 = self._summary[self._name_space + '/DICE']
                s3 = self._summary[self._name_space + '/BRATS_metric']
                out_summary = self._summary[self._name_space + '/MergeSummary']
                # Execute "Validate".
                reward_list = 0
                DICE_list, BRATS_list = seg_validate(2)
                # Print some info. --------------------------------------------
                self._logger.info("Iter {} --> DICE: {}, BRATS: {} ".format(ite, DICE_list, BRATS_list))
                # ------------------------------------------------------------
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
        # Finish.
        return


    def _train(self, max_iteration, start_pos, epsilon_book):
        r'''
            End-to-end train the whole model.
        '''

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
            self._env.render(None)

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
        fake_weight = conf_dqn.get('fake_exp_priority', 0.25)

        conf_others = self._config['Others']
        # memory_size = conf_others.get('replay_memories')
        store_type = conf_others.get('sample_type', 'light')
        save_steps = conf_others.get('save_steps', 100)
        validate_steps = conf_others.get('validate_steps', 500)
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
                    img, lab = self._data_adapter.precise_locate((mha_idx, inst_idx))
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

        # Generate the basic feed dictionary lately used for training the whole model.
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
            # Extend the weights with fake sample weights.
            FAKE_W = [fake_weight] * len(mini_batch)
            FAKE_W.extend(ISWeights)
            ISWeights = FAKE_W
            # Add to feed dictionary.
            feed_dict[ISw] = ISWeights
            # Execute the training operation.
            _, exp_priority, v1_cost, v2_cost, v3_cost = self._sess.run(
                [self._train_op, exp_pri, nloss, sloss, dloss], feed_dict=feed_dict)
            # Truncate the values to get the real experience priority.
            exp_priority = exp_priority[len(mini_batch):]
            # Update probability for leafs.
            self._replay_memory.batch_update(tree_idx, exp_priority)  # update priority
        else:
            # Simply feed the dictionary to the selected operation is OK.
            _, v1_cost, v2_cost, v3_cost = self._sess.run(
                [self._train_op, nloss, sloss, dloss], feed_dict=feed_dict)

        # ------------------------------- Save Model Parameters ------------------------
        # Get the global step lately used to determine whether to save model or not.
        step = self._global_step.eval(self._sess)
        # Calculate the summary to get the statistic graph.
        if step > 0 and step % validate_steps == 0:
            # Get summary holders.
            s1 = self._summary[self._name_space + '/Reward']
            s2 = self._summary[self._name_space + '/DICE']
            s3 = self._summary[self._name_space + '/BRATS_metric']
            out_summary = self._summary[self._name_space + '/MergeSummary']
            # Execute "Validate".
            reward_list, DICE_list, BRATS_list = self.test(2, is_validate=True)
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

