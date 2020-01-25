import gc
from util.visualization import *
from core.data_structure import *
from dataset.adapter.base import *



# Env-related abstractions
class Env:

    def step(self, action_index, arg):
        r'''
            Execute the action_index specific action, and return a tuple of
            (state, reward, terminal, info).

        --------------------------------------------------------------------
        Parameters:
            action_index: The index of action chosen by DQN agent.
            arg: Additional parameters used in the implementation.

        --------------------------------------------------------------------
        Return:
            The tuple of (state, reward, terminal, info).
                state: The current state after executed the specific action.
                reward: The reward of specific action.
                terminal: A flag implies whether the task is in terminal state.
                info: Some other information, which is optional.
        '''
        raise NotImplementedError

    def reset(self, arg):
        r'''
            Reset the environment. (That is states)

        --------------------------------------------------------------------
        Parameters:
            arg: Additional parameters used in the implementation.

        --------------------------------------------------------------------
        Return:
            The initial state of the environment.
        '''
        raise NotImplementedError

    def render(self):
        r'''
            The method that visualize the environment (specially its states).
        '''
        raise NotImplementedError



# ---------------------------------------------------------------------------
# The "Focus Environment". Which is "End-to-end" type.
#   Directly use it is not recommended. Better to use the wrapper.
# ---------------------------------------------------------------------------

class FocusEnvCore:
    r'''
        The task-specific environment. This one is an "End-to-end" type.
            It will generate the next "Focus Bounding-box" for each step.
    '''

    def __init__(self, config, data_adapter, is_separate):
        r'''
            Initialization method for environment. Mainly to declare some
                holder and initialize some config.
        '''

        # Configuration.
        self._config = config

        # Get basic parameter.
        conf_base = self._config['Base']
        clazz_dim = conf_base.get('classification_dimension')
        suit_height = conf_base.get('suit_height')
        suit_width = conf_base.get('suit_width')

        # Data adapter.
        self._adapter = data_adapter

        # Action quantity.
        conf_dqn = self._config['DQN']
        restrict_mode = conf_dqn.get('restriction_action')
        aban_par = conf_dqn.get('abandon_parents')
        if restrict_mode:
            self._act_dim = 8 if aban_par else 9
        else:
            self._act_dim = 13 if aban_par else 17

        # Clazz weights (for segmentation).
        self._clazz_weights = None

        # Animation recorder.
        self._anim_recorder = None
        conf_other = self._config['Others']
        anim_path = conf_other.get('animation_path', None)
        anim_fps = conf_other.get('animation_fps', 4)
        if anim_path is not None:
            self._anim_recorder = MaskVisualVMPY(240, 240,
                                                 fps=anim_fps,
                                                 result_categories=clazz_dim,
                                                 suit_height=suit_height,
                                                 suit_width=suit_width,
                                                 vision_path=anim_path,
                                                 name_scope='FocusEnv')

        # The flag indicating whether is in "Train", "Validate" or "Test" phrase.
        self._phrase = 'Train'  # Default in "Train"

        # --------------------- Declare some input data placeholder. ---------------------
        self._image = None  # Current processing image.
        self._label = None  # Label for current image.

        # The "Action History".
        self._ACTION_his = None

        # The "Bbox History".
        self._BBOX_his = None

        # The focus bounding-box. (y1, x1, y2, x2)
        self._focus_bbox = None
        # --> fake bbox, only (1, 1)-pixel shape.
        input_shape = conf_base.get('input_shape')
        image_height, image_width = input_shape[1:3]
        fy2_1px = 1 / image_height
        fx2_1px = 1 / image_width
        self._FAKE_BBOX = np.asarray([0.0, 0.0, fy2_1px, fx2_1px])  # [y1, x1, y2, x2]

        # The "Focus Boundary" used to restrict the range of target.
        self._FOCUS_boundary = None

        # Get "Position Information" fusion method.
        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')
        # Declare the position information generation function, for convenient usage.
        def gen_position_info(focus_bbox):
            # The position information. Declaration according to config.
            if pos_method == 'map':
                pos_info = np.zeros([image_height, image_width], dtype=np.int64)
                fy1, fx1, fy2, fx2 = focus_bbox.copy()
                fy1 = int(round(image_height * fy1))
                fx1 = int(round(image_width * fx1))
                fy2 = int(round(image_height * fy2))
                fx2 = int(round(image_width * fx2))
                pos_info[fy1: fy2, fx1: fx2] = 1
            elif pos_method == 'coord':
                pos_info = focus_bbox.copy()
            elif pos_method == 'w/o':
                pos_info = None
            else:
                raise ValueError('Unknown position information fusion method !!!')
            return pos_info
        # Assign the holder for lately usage.
        self._position_func = gen_position_info

        # The previous "Segmentation" result (real form).
        self._SEG_prev = None

        # The complete "Segmentation" result (logit/prob/mask), which will be used
        #   as part of input for "Result Fusion". So it's not real "Result"-form.
        self._COMP_result = None

        # The previous "Relative Directions". (used in "restrict" mode)
        self._RelDir_prev = None
        self._RELATIVE_DIRECTION = ('left-up', 'right-up', 'left-bottom', 'right-bottom')

        # The time-step.
        self._time_step = None

        # The process flag.
        self._finished = True   # finish current image
        self._FB_opted = False  # whether optimized the "Focus Bbox" or not.

        # "Separate Execution" related variables.
        self._separate_execution = is_separate
        self._exec_1st = False

        # "Instance Id" for current processing image.
        self._sample_id = -1

        # Finish initialization.
        return


    @property
    def acts_dim(self):
        r'''
            The action quantity (supported) of environment.
        '''
        return self._act_dim


    @property
    def Fake_Bbox(self):
        r'''
            The fake "Bounding-Box" used in environment.
        '''
        return self._FAKE_BBOX.copy()


    @property
    def finished(self):
        r'''
            Whether the process of current image is finished or not.
        '''
        return self._finished


    @property
    def sample_id(self):
        r'''
            Get the sample id for current processing image.
        '''
        return self._sample_id


    def switch_phrase(self, p):
        r'''
            Switch to target phrase.
        '''
        if p not in ['Train', 'Validate', 'Test']:
            raise ValueError('Unknown phrase value !!!')
        self._phrase = p
        return


    def reset(self, sample_id, segment_func=None):
        r'''
            Reset the environment. Mainly to reset the related parameters,
                and switch to next image-label pair.

            ** Note that, it will be initialized according to whether
                enable the "Optimize Initial Bbox" or not.
        '''

        # Check the validity of execution logic.
        if not self._finished:
            raise Exception('The processing of current image is not yet finished, '
                            'please keep calling @Method{step} !!!')

        # Check validity of parameters.
        if not isinstance(sample_id, (int, np.int, np.int32, np.int64)) or sample_id < 0:
            raise TypeError('The sample_id must be non-negative integer !!!')
        if segment_func is not None and not callable(segment_func):
            raise TypeError('The segment_func must be None or function !!!')

        # --------------------------- Reset some holders ---------------------------
        # Get detailed config parameters.
        conf_base = self._config['Base']
        input_shape = conf_base.get('input_shape')
        image_height, image_width = input_shape[1:3]
        suit_h = conf_base.get('suit_height')
        suit_w = conf_base.get('suit_width')
        clazz_dim = conf_base.get('classification_dimension')
        conf_cus = self._config['Custom']
        CR_method = conf_cus.get('result_fusion', 'prob')
        pos_method = conf_cus.get('position_info', 'map')
        conf_dqn = self._config['DQN']
        his_len = conf_dqn.get('actions_history', 10)
        step_thres = conf_dqn.get('step_threshold', 10)
        initFB_optimize = conf_dqn.get('initial_bbox_optimize', True)
        initFB_pad = conf_dqn.get('initial_bbox_padding', 20)

        # Reset the "Previous Segmentation".
        self._SEG_prev = np.zeros([image_height, image_width], dtype=np.int64)

        # Reset the "Complete Result" (segmentation).
        if CR_method == 'logit' or CR_method == 'prob':
            self._COMP_result = collections.deque([np.zeros([suit_h, suit_w, clazz_dim], dtype=np.float32)] * step_thres)
        elif CR_method in ['mask-lap', 'mask-vote']:
            self._COMP_result = collections.deque([np.zeros([suit_h, suit_w], dtype=np.int64)] * step_thres)
        else:
            raise ValueError('Unknown result fusion method !!!')

        # Reset the "Focus Bounding-box".
        self._focus_bbox = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)   # (y1, x1, y2, x2)

        # Reset the "Focus Boundary".
        self._FOCUS_boundary = (0.0, 0.0, 1.0, 1.0)     # (y1, x1, y2, x2)

        # Reset the "Action History".
        self._ACTION_his = collections.deque([-1,] * his_len)

        # Reset the "Bbox History".
        self._BBOX_his = collections.deque([self._FAKE_BBOX] * step_thres)

        # Reset the "Previous Relative Directions".
        self._RelDir_prev = []     # clear.

        # Reset the time-step.
        self._time_step = 0

        # Reset the process flag.
        self._finished = False

        # Reset the "Step" state.
        if self._separate_execution:
            self._exec_1st = False

        # Reset the "Sample Id" for current processing image of this environment.
        self._sample_id = sample_id

        # -------------------------- Switch to next image-label pair. ---------------------------
        # Get the next image-label pair according to the phrase.
        if self._phrase == 'Train':
            # Get the next train sample pair.
            img, label, clazz_weights, data_arg = self._adapter.next_image_pair(mode='Train', batch_size=1)
        elif self._phrase == 'Validate':
            # Get the next validate image-label pair, and additional arguments.
            img, label, clazz_weights, data_arg = self._adapter.next_image_pair(mode='Validate', batch_size=1)
        elif self._phrase == 'Test':
            # Get the next test image-label pair, and additional arguments.
            img, label, data_arg = self._adapter.next_image_pair(mode='Test', batch_size=1)
        else:
            raise ValueError('Unknown phrase !!!')

        # Assign the image and label holder.
        self._image = img
        self._label = label
        # Get clazz weights from configuration here.
        conf_train = self._config['Training']
        self._clazz_weights = conf_train.get('clazz_weights', None)
        # Reset clazz weights if needed.
        if self._clazz_weights is None:     # do not config
            if self._label is not None:     # train or validate
                self._clazz_weights = clazz_weights

        # Record the process.
        if self._anim_recorder is not None:
            self._anim_recorder.reload((sample_id, img.copy(), label.copy() if label is not None else None))

        # -------------------------- Optimize Part. ---------------------------
        # Optimize the initial "Focus Bbox" if specified.
        if initFB_optimize and segment_func is not None:
            # The position information. Declaration according to config.
            if pos_method == 'map':
                pos_info = np.ones([image_height, image_width], dtype=np.int64)
            elif pos_method == 'coord':
                pos_info = self._focus_bbox.copy()
            elif pos_method == 'sight':
                pos_info = self._focus_bbox.copy()
            else:
                raise ValueError('Unknown position information fusion method !!!')
            # Get the initial segmentation.
            segmentation, COMP_res = segment_func((self._image.copy(),
                                                   self._SEG_prev.copy(),
                                                   pos_info,
                                                   self._focus_bbox.copy(),
                                                   self._BBOX_his.copy(),
                                                   self._time_step,
                                                   self._COMP_result.copy()))
            # Only operate when the segmentation is not pure "Background".
            if not (segmentation == 0).all():
                # Record the very beginning frame.
                if self._anim_recorder is not None:
                    # "Segment" phrase.
                    self._anim_recorder.record(
                        (self._focus_bbox.copy(), None, None, self._SEG_prev.copy(), None))  # Need copy !!!
                # Assign the "Focus Bbox".
                init_region = np.where(segmentation != 0)
                init_y1 = max(0, np.min(init_region[0]) - initFB_pad) / image_height
                init_x1 = max(0, np.min(init_region[1]) - initFB_pad) / image_width
                init_y2 = min(image_height, np.max(init_region[0]) + initFB_pad) / image_height
                init_x2 = min(image_width, np.max(init_region[1]) + initFB_pad) / image_width
                self._focus_bbox = np.asarray([init_y1, init_x1, init_y2, init_x2])
                # Re-assign the "Focus Boundary".
                self._FOCUS_boundary = (init_y1, init_x1, init_y2, init_x2)
                # Assign the "Current Segmentation" holders.
                self._SEG_prev = segmentation
                # Append region result into "Complete Results".
                self._COMP_result.append(COMP_res)
                self._COMP_result.popleft()
                # Record the current bbox to the history.
                self._BBOX_his.append(self._focus_bbox.copy())
                self._BBOX_his.popleft()
                # Increase the time-step
                self._time_step += 1

        # Whether return the train samples meta according to phrase.
        if self._phrase == 'Train':
            # Return train samples.
            return img.copy(), label.copy(), self._clazz_weights.copy(), data_arg
        else:
            # Return the label used to compute metric when in "Validate" phrase. (And mha_id used to save results)
            return label.copy() if label is not None else None, data_arg


    def step_comp(self, op_func):
        r'''
            Step (interact with) the environment. Here prepare inputs for "Focus Model",
                and then get results from it, finally re-assign some holders for
                "Step-by-step" logic. What's more, it will return the training elements.

        Parameters:
            op_func: The operation function. Containing the both two of "Segment" and "Focus"
                function. Which will be applied to generate the fusion result of previous
                complete result, current region result and select the next region
                to precisely segment.

        Return:
            The tuple of (state, action, terminal, anchors, err, next_state, info) when in "Train" phrase.
                state: (SEG_prev, cur_bbox, position_info, acts_prev, bboxes_prev, his_plen, comp_prev)
                next_state: (SEG_cur, focus_bbox, next_posinfo, acts_cur, bboxes_cur, his_clen, comp_cur)
            The tuple of (terminal, SEG_cur, reward, info) when in "Validate" or "Test" phrase.
                ** Note that, the reward is None when in "Test" phrase, and the
                SEG_cur will be used to write result in "Test" phrase.
            ---------------------------------------------------------------
            SEG_prev: The previous segmentation result.
            cur_bbox: The current bbox of image.
            position_info: The position information for current time-step.
            acts_prev: The current actions history.
            bboxes_prev: The current bboxes history.
            his_plen: The valid length of current history.
            comp_prev: The current "Complete Result".
            ---------------------------------------------------------------
            action: The action executed of current time-step.
            terminal: Flag that indicates whether current image is finished.
            anchors: The anchors for current bbox.
            BBOX_errs: The "BBOX erros" vector for current anchors.
            ---------------------------------------------------------------
            SEG_cur: Current segmentation result.
            focus_bbox: Next focus bbox of image.
            next_posinfo: The fake next position information.
            acts_cur: The next actions history.
            bboxes_cur: The next bboxes history.
            his_clen: The valid length of next history.
            comp_cur: The next "Complete Result".
            ---------------------------------------------------------------
            info: Extra information.(Optional, default is type.None)
        '''

        # Check whether should use this "Step" version.
        if self._separate_execution:
            raise Exception('One must call the @Method{step_1st} and @Method{step_2nd} instead'
                            ' of this step version !!!')

        # Check the validity of execution logic.
        if self._finished:
            raise Exception('The process of current image-label pair is finished, it '
                            'can not be @Method{step} any more !!! '
                            'You should call the @Method{reset} to reset the '
                            'environment to process the next image.')

        # Check validity.
        if not callable(op_func):
            raise TypeError('The op_func must be a function !!!')

        # Get the detailed config.
        conf_dqn = self._config['DQN']
        step_threshold = conf_dqn.get('step_threshold', 10)

        # The current state.
        cur_bbox = self._focus_bbox.copy()
        SEG_prev = self._SEG_prev.copy()
        position_info = self._position_func(self._focus_bbox)
        acts_prev = np.asarray(self._ACTION_his.copy())
        bboxes_prev = np.asarray(self._BBOX_his.copy())
        his_plen = self._time_step
        comp_prev = self._COMP_result.copy()

        # -------------------------------- Core Part -----------------------------
        # Generate anchors.
        conf_dqn = self._config['DQN']
        anchor_scale = conf_dqn.get('anchors_scale', 2)
        if self._act_dim == 8 or self._act_dim == 9:
            # "Restrict" mode.
            anchors = self._anchors(self._focus_bbox.copy(), anchor_scale, self._act_dim,
                                    arg=self._RelDir_prev[-1])   # the newest relative direction.
        elif self._act_dim == 13 or self._act_dim == 17:
            # "Whole Candidates" mode.
            anchors = self._anchors(self._focus_bbox.copy(), anchor_scale, self._act_dim == 17)
        else:
            raise Exception('Unknown action dimension, there is no logic with respect to '
                            'current act_dim, please check your code !!!')

        # Get the "Bbox Error" vector for all the candidates (anchors).
        BBOX_errs, infos = self._check_candidates(anchors)

        # Execute the operation function to conduct "Segment" and "Focus".
        if self._label is not None:
            segmentation, COMP_res, action, reward = op_func([(self._image.copy(),
                                                              SEG_prev.copy(),
                                                              position_info.copy(),
                                                              cur_bbox.copy(),
                                                              acts_prev.copy(),
                                                              bboxes_prev.copy(),
                                                              his_plen,
                                                              comp_prev.copy(),
                                                              anchors.copy(),
                                                              BBOX_errs.copy(),
                                                              self._label.copy(),
                                                              self._clazz_weights.copy())],
                                                             with_explore=self._phrase == 'Train',
                                                             with_reward=True)
            segmentation = segmentation[0]
            COMP_res = COMP_res[0]
            action = action[0]
            reward = reward[0][action]
        else:
            segmentation, COMP_res, action = op_func([(self._image.copy(),
                                                      SEG_prev.copy(),
                                                      position_info.copy(),
                                                      cur_bbox.copy(),
                                                      acts_prev.copy(),
                                                      bboxes_prev.copy(),
                                                      his_plen,
                                                      comp_prev.copy())],
                                                     with_explore=False,
                                                     with_reward=False)
            segmentation= segmentation[0]
            COMP_res = COMP_res[0]
            action = action[0]
            reward = None

        # Push forward the environment.
        dst_bbox, terminal, info = self._push_forward(action,
                                                      candidates=anchors,
                                                      BBOX_errs=BBOX_errs,
                                                      step_thres=step_threshold,
                                                      infos=infos)
        # Append the current action into "Action History".
        self._ACTION_his.append(action)
        self._ACTION_his.popleft()
        # Append the current bbox into "Bbox History".
        self._BBOX_his.append(self._focus_bbox.copy())
        self._BBOX_his.popleft()
        # Append the current region result into "Complete Result".
        self._COMP_result.append(COMP_res)
        self._COMP_result.popleft()
        # Iteratively re-assign the "Segmentation" result and "Focus Bbox".
        self._SEG_prev = segmentation
        self._focus_bbox = np.asarray(dst_bbox)
        # -------------------------------- Core Part -----------------------------

        # Record the process.
        if self._anim_recorder is not None:
            # "Segment" phrase.
            self._anim_recorder.record(
                (cur_bbox.copy(), None, None, self._SEG_prev.copy(), None))  # Need copy !!!
            # "Focus" phrase.
            self._anim_recorder.record(
                (cur_bbox.copy(), self._focus_bbox.copy(), reward, self._SEG_prev.copy(), info))  # Need copy !!!

        # Reset the process flag when current process terminate.
        if terminal:
            self._finished = True

        # Generate next state.
        SEG_cur = self._SEG_prev.copy()
        focus_bbox = self._focus_bbox.copy()
        next_posinfo = self._position_func(self._focus_bbox)
        acts_cur = np.asarray(self._ACTION_his.copy())
        bboxes_cur = np.asarray(self._BBOX_his.copy())
        his_clen = self._time_step
        comp_cur = self._COMP_result.copy()

        # Return sample when in "Train" phrase.
        if self._phrase == 'Train':
            return (SEG_prev, cur_bbox, position_info, acts_prev, bboxes_prev, his_plen, comp_prev), \
                   action, terminal, anchors.copy(), BBOX_errs.copy(), \
                   (SEG_cur, focus_bbox, next_posinfo, acts_cur, bboxes_cur, his_clen, comp_cur), \
                   reward, info
        else:
            # Return terminal flag (most important), and other information.
            return terminal, (SEG_cur, reward, info)


    def step_1st(self):
        r'''
            Step (interact with) the environment, this is only part one.
            Here we prepare the inputs (state) data for the "Focus Model", including:
                inherent input elements, and the candidates bounding-box.

        Return:
            state_elems: The elements fed for "Focus Model". Detailed as belows:
                The tuple of (image, SEG_prev , position_info, cur_bbox, acts_prev, bboxes_prev, his_plen, comp_prev)
                    when in "Test" phrase. (Only state elements)
                When in "Train/Validate" phrase, the additional information (evaluation elements):
                    (anchors, BBOX_errs, label, clazz_weights)
                ---------------------------------------------------------------
                image: The current processing image.
                SEG_prev: The previous segmentation result.
                position_info: The position information for current time-step.
                cur_bbox: The current bbox of image.
                acts_prev: The current actions history.
                bboxes_prev: The current bboxes history.
                his_plen: The valid length of current history.
                comp_prev: The current "Complete Result".
                ---------------------------------------------------------------
                anchors: The anchors for current bbox.
                BBOX_errs: The "BBOX erros" vector for current anchors.
                label: The corresponding label for current image.
                clazz_weights: The class weights for current image.
            push_elems: The elements used to "Push Forward Environment". Detail as belows:
                (anchors, BBOX_errs, infos)
        '''

        # Check whether should use this "Step" version.
        if not self._separate_execution:
            raise Exception('One must call the @Method{step_comp} instead'
                            ' of this step version !!!')

        # Check the validity of execution logic.
        if self._finished:
            raise Exception('The process of current image-label pair is finished, it '
                            'can not be @Method{step_1st} any more !!! '
                            'You should call the @Method{reset} to reset the '
                            'environment to process the next image.')
        if self._exec_1st:
            raise Exception('Already execute the @Method{step_1st}, one must call the '
                            '@Method{step_2nd} to go on !!!')

        # The current state.
        cur_bbox = self._focus_bbox.copy()
        SEG_prev = self._SEG_prev.copy()
        position_info = self._position_func(self._focus_bbox)
        acts_prev = np.asarray(self._ACTION_his.copy())
        bboxes_prev = np.asarray(self._BBOX_his.copy())
        his_plen = self._time_step
        comp_prev = self._COMP_result.copy()

        # -------------------------------- Core Part -----------------------------
        # Generate anchors.
        conf_dqn = self._config['DQN']
        anchor_scale = conf_dqn.get('anchors_scale', 2)
        if self._act_dim == 8 or self._act_dim == 9:
            # "Restrict" mode.
            anchors = self._anchors(self._focus_bbox.copy(), anchor_scale, self._act_dim,
                                    arg=self._RelDir_prev[-1])   # the newest relative direction.
        elif self._act_dim == 13 or self._act_dim == 17:
            # "Whole Candidates" mode.
            anchors = self._anchors(self._focus_bbox.copy(), anchor_scale, self._act_dim)
        else:
            raise Exception('Unknown action dimension, there is no logic with respect to '
                            'current act_dim, please check your code !!!')

        # Get the "Bbox Error" vector for all the candidates (anchors).
        BBOX_errs, infos = self._check_candidates(anchors)

        # Update the "Step" state.
        self._exec_1st = True

        # Return the "State" (and "Evaluation") elements for lately usage (inferring model...).
        if self._label is not None:
            return (self._image.copy(), SEG_prev.copy(), position_info.copy(),
                    cur_bbox.copy(), acts_prev.copy(), bboxes_prev.copy(), his_plen, comp_prev.copy(),
                    anchors.copy(), BBOX_errs.copy(), self._label.copy(), self._clazz_weights.copy()), \
                   (anchors.copy(), BBOX_errs.copy(), infos)
        else:
            return (self._image.copy(), SEG_prev.copy(), position_info.copy(),
                    cur_bbox.copy(), acts_prev.copy(), bboxes_prev.copy(), his_plen, comp_prev.copy()), \
                   (anchors.copy(), BBOX_errs.copy(), infos)


    def step_2nd(self, results, state_elems, push_elems):
        r'''
            Step (interact with) the environment, this is only part two.
            Here we get the results from "Focus Model", and later we re-assign some
                holder for "Step-by-step" logic. What's more, it will return the
                "State" and "Evaluation" elements for "Train" phrase, and return
                "Terminal" flag when in "Test" phrase.

        Parameters:
            results: The results from "Focus Model", including the "Segmentation",
                "Complete Result" and "Action" for current time-step.
            state_elems: The state elements fed for "Focus Model". It will be used
                to train the tensorflow model.
            push_elems: The evaluation elements used to "Push Forward Environment".

        Return:
            The tuple of (state, action, terminal, anchors, err, next_state, info) when in "Train" phrase.
                state: (SEG_prev, cur_bbox, position_info, acts_prev, bboxes_prev, his_plen, comp_prev)
                next_state: (SEG_cur, focus_bbox, next_posinfo, acts_cur, bboxes_cur, his_clen, comp_cur)
            The tuple of (terminal, SEG_cur, reward, info) when in "Validate" or "Test" phrase.
                ** Note that, the reward is None when in "Test" phrase, and the
                SEG_cur will be used to write result in "Test" phrase.
            ---------------------------------------------------------------
            SEG_prev: The previous segmentation result.
            cur_bbox: The current bbox of image.
            position_info: The position information for current time-step.
            acts_prev: The current actions history.
            bboxes_prev: The current bboxes history.
            his_plen: The valid length of current history.
            comp_prev: The current "Complete Result".
            ---------------------------------------------------------------
            action: The action executed of current time-step.
            terminal: Flag that indicates whether current image is finished.
            anchors: The anchors for current bbox.
            BBOX_errs: The "BBOX erros" vector for current anchors.
            ---------------------------------------------------------------
            SEG_cur: Current segmentation result.
            focus_bbox: Next focus bbox of image.
            next_posinfo: The fake next position information.
            acts_cur: The next actions history.
            bboxes_cur: The next bboxes history.
            his_clen: The valid length of next history.
            comp_cur: The next "Complete Result".
            ---------------------------------------------------------------
            info: Extra information.(Optional, default is type.None)
        '''

        # Check whether should use this "Step" version.
        if not self._separate_execution:
            raise Exception('One must call the @Method{step_comp} instead'
                            ' of this step version !!!')

        # Check the validity of execution logic.
        if self._finished:
            raise Exception('The process of current image-label pair is finished, it '
                            'can not be @Method{step_2nd} any more !!! '
                            'You should call the @Method{reset} to reset the '
                            'environment to process the next image.')
        if not self._exec_1st:
            raise Exception('Not yet execute the @Method{step_1st}, one must call it before calling '
                            '@Method{step_2nd}!!!')

        # Check validity of parameters.
        if not isinstance(results, tuple):
            raise TypeError('The results should be tuple !!!')
        else:
            if self._label is not None and len(results) != 4:
                raise ValueError('The results should contain 4 elements when label is not None !!!')
            elif self._label is None and len(results) != 3:
                raise ValueError('The results should contain 3 elements when label is None !!!')
            else:
                pass
        if not isinstance(state_elems, tuple):
            raise TypeError('The state_elems should be tuple !!!')
        else:
            if self._label is not None and len(state_elems) != 12:
                raise ValueError('The state_elems should contain 12 elements when label is not None !!!')
            elif self._label is None and len(state_elems) != 8:
                raise ValueError('The state_elems should contain 8 elements when label is None !!!')
            else:
                pass
        if not isinstance(push_elems, tuple) or len(push_elems) != 3:
            raise TypeError('The push_elems should be 3-elements tuple !!!')

        # Get the detailed config.
        conf_dqn = self._config['DQN']
        step_threshold = conf_dqn.get('step_threshold', 10)

        # Get the detailed elements from "State Elements".
        if self._label is not None:
            _1, SEG_prev, position_info, cur_bbox, acts_prev, bboxes_prev, his_plen, comp_prev, \
                _9, _10, _11, _12 = state_elems
        else:
            _1, SEG_prev, position_info, cur_bbox, acts_prev, bboxes_prev, his_plen, comp_prev = state_elems

        # Get detailed results for "Segment" and "Focus".
        if self._label is not None:
            segmentation, COMP_res, action, reward = results
            reward = reward[action]
        else:
            segmentation, COMP_res, action = results
            reward = None

        # Get detailed elements from "Push Forward Elements".
        anchors, BBOX_errs, infos = push_elems

        # Push forward the environment.
        dst_bbox, terminal, info = self._push_forward(action,
                                                      candidates=anchors,
                                                      BBOX_errs=BBOX_errs,
                                                      step_thres=step_threshold,
                                                      infos=infos)
        # Append the current action into "Action History".
        self._ACTION_his.append(action)
        self._ACTION_his.popleft()
        # Append the current bbox into "Bbox History".
        self._BBOX_his.append(dst_bbox.copy())
        self._BBOX_his.popleft()
        # Append the current region result into "Complete Result".
        self._COMP_result.append(COMP_res)
        self._COMP_result.popleft()
        # Iteratively re-assign the "Segmentation" result and "Focus Bbox".
        self._SEG_prev = segmentation
        self._focus_bbox = np.asarray(dst_bbox)
        # -------------------------------- Core Part -----------------------------

        # Record the process.
        if self._anim_recorder is not None:
            # "Segment" phrase.
            self._anim_recorder.record(
                (cur_bbox.copy(), None, None, self._SEG_prev.copy(), None))  # Need copy !!!
            # "Focus" phrase.
            self._anim_recorder.record(
                (cur_bbox.copy(), self._focus_bbox.copy(), reward, self._SEG_prev.copy(), info))  # Need copy !!!

        # Reset the process flag when current process terminate.
        if terminal:
            self._finished = True

        # Generate next state.
        SEG_cur = self._SEG_prev.copy()
        focus_bbox = self._focus_bbox.copy()
        next_posinfo = self._position_func(self._focus_bbox)
        acts_cur = np.asarray(self._ACTION_his.copy())
        bboxes_cur = np.asarray(self._BBOX_his.copy())
        his_clen = self._time_step
        comp_cur = self._COMP_result.copy()

        # Update the "Step" state.
        self._exec_1st = False

        # Return sample when in "Train" phrase.
        if self._phrase == 'Train':
            return (SEG_prev, cur_bbox, position_info, acts_prev, bboxes_prev, his_plen, comp_prev), \
                   action, terminal, anchors.copy(), BBOX_errs.copy(), \
                   (SEG_cur, focus_bbox, next_posinfo, acts_cur, bboxes_cur, his_clen, comp_cur), \
                   reward, info
        else:
            # Return terminal flag (most important), and other information.
            return terminal, (SEG_cur, reward, info)


    def render(self, anim_type):
        r'''
            Render. That is, visualize the result (and the process if specified).
        '''
        # Check whether can render or not.
        if self._finished:
            pass
        else:
            raise Exception('The processing of current image is not yet finish, '
                            'can not @Method{render} !!!')
        # Save the process into filesystem.
        if self._anim_recorder is not None:
            self._anim_recorder.show(mode=self._phrase, anim_type=anim_type)
        return



    def _anchors(self, bbox, scale, adim, arg=None):
        r'''
            Generate the anchors for current bounding-box with the given scale.
                What's more, it can restrict some anchors if given additional
                arguments.

            ** Note that, do not change the order of anchors !!! Its special
                order is for the convenience of coding "Restrict Action" mode.
        '''

        # Get the four coordinates (normalized).
        y1, x1, y2, x2 = bbox
        h = y2 - y1
        w = x2 - x1
        # Get the four boundaries (normalized).
        B_y1, B_x1, B_y2, B_x2 = self._FOCUS_boundary

        # -------> Generate the whole situation for convenient selection.
        s_anchors = []
        # ---> Children situation.
        cld_h = h / scale
        cld_w = w / scale
        # S1: Child left-up.
        s1_y1 = y1
        s1_x1 = x1
        s1_y2 = min(B_y2, y1 + cld_h)
        s1_x2 = min(B_x2, x1 + cld_w)
        s_anchors.append([s1_y1, s1_x1, s1_y2, s1_x2])
        # S2: Child right-up.
        s2_y1 = max(B_y1, y2 - cld_h)
        s2_x1 = x1
        s2_y2 = y2
        s2_x2 = min(B_x2, x1 + cld_w)
        s_anchors.append([s2_y1, s2_x1, s2_y2, s2_x2])
        # S3: Child left-bottom.
        s3_y1 = y1
        s3_x1 = max(B_x1, x2 - cld_w)
        s3_y2 = min(B_y2, y1 + cld_h)
        s3_x2 = x2
        s_anchors.append([s3_y1, s3_x1, s3_y2, s3_x2])
        # S4: Child right-bottom.
        s4_y1 = max(B_y1, y2 - cld_h)
        s4_x1 = max(B_x1, x2 - cld_w)
        s4_y2 = y2
        s4_x2 = x2
        s_anchors.append([s4_y1, s4_x1, s4_y2, s4_x2])
        # ---> Parents situation.
        par_h = h * scale
        par_w = w * scale
        # S5: Parent left-up.
        s5_y1 = max(B_y1, y2 - par_h)
        s5_x1 = max(B_x1, x2 - par_w)
        s5_y2 = y2
        s5_x2 = x2
        s_anchors.append([s5_y1, s5_x1, s5_y2, s5_x2])
        # S6: Parent right-up.
        s6_y1 = y1
        s6_x1 = max(B_x1, x2 - par_w)
        s6_y2 = min(B_y2, y1 + par_h)
        s6_x2 = x2
        s_anchors.append([s6_y1, s6_x1, s6_y2, s6_x2])
        # S7: Parent left-bottom.
        s7_y1 = max(B_y1, y2 - par_h)
        s7_x1 = x1
        s7_y2 = y2
        s7_x2 = min(B_x2, x1 + par_w)
        s_anchors.append([s7_y1, s7_x1, s7_y2, s7_x2])
        # S8: Parent right-bottom.
        s8_y1 = y1
        s8_x1 = x1
        s8_y2 = min(B_y2, y1 + par_h)
        s8_x2 = min(B_x2, x1 + par_w)
        s_anchors.append([s8_y1, s8_x1, s8_y2, s8_x2])
        # ---> Peers situation.
        pee_h = h
        pee_w = w
        # S9: Peer left-up.
        s9_y1 = max(B_y1, y2 - par_h)
        s9_x1 = max(B_x1, x2 - par_w)
        s9_y2 = max(B_y1, y2 - par_h + pee_h)
        s9_x2 = max(B_x1, x2 - par_w + pee_w)
        s_anchors.append([s9_y1, s9_x1, s9_y2, s9_x2])
        # S10: Peer right-up.
        s10_y1 = min(B_y2, y1 + par_h - pee_h)
        s10_x1 = max(B_x1, x2 - par_w)
        s10_y2 = min(B_y2, y1 + par_h)
        s10_x2 = max(B_x1, x2 - par_w + pee_w)
        s_anchors.append([s10_y1, s10_x1, s10_y2, s10_x2])
        # S11: Peer left-bottom.
        s11_y1 = max(B_y1, y2 - par_h)
        s11_x1 = min(B_x2, x1 + par_w - pee_w)
        s11_y2 = max(B_y1, y2 - par_h + pee_h)
        s11_x2 = min(B_x2, x1 + par_w)
        s_anchors.append([s11_y1, s11_x1, s11_y2, s11_x2])
        # S12: Peer right-bottom.
        s12_y1 = min(B_y2, y1 + par_h - pee_h)
        s12_x1 = min(B_x2, x1 + par_w - pee_w)
        s12_y2 = min(B_y2, y1 + par_h)
        s12_x2 = min(B_x2, x1 + par_w)
        s_anchors.append([s12_y1, s12_x1, s12_y2, s12_x2])
        # S13: Peer pure-left.
        s13_y1 = max(B_y1, y2 - par_h)
        s13_x1 = x1
        s13_y2 = max(B_y1, y2 - par_h + pee_h)
        s13_x2 = x2
        s_anchors.append([s13_y1, s13_x1, s13_y2, s13_x2])
        # S14: Peer pure-up.
        s14_y1 = y1
        s14_x1 = max(B_x1, x2 - par_w)
        s14_y2 = y2
        s14_x2 = max(B_x1, x2 - par_w + pee_w)
        s_anchors.append([s14_y1, s14_x1, s14_y2, s14_x2])
        # S15: Peer pure-right.
        s15_y1 = min(B_y2, y1 + par_h - pee_h)
        s15_x1 = x1
        s15_y2 = min(B_y2, y1 + par_h)
        s15_x2 = x2
        s_anchors.append([s15_y1, s15_x1, s15_y2, s15_x2])
        # S16: Peer pure-bottom.
        s16_y1 = y1
        s16_x1 = min(B_x2, x1 + par_w - pee_w)
        s16_y2 = y2
        s16_x2 = min(B_x2, x1 + par_w)
        s_anchors.append([s16_y1, s16_x1, s16_y2, s16_x2])

        # Now select anchors to return according to the mode.
        if adim == 8:   # restrict, abandon
            # Check validity.
            if arg is None:
                raise Exception('One must assign the arg with previous action when adim = 8 !!!')
            # Select candidates for relative direction.
            relative = arg
            if relative == 'left-up':
                cands = [
                    0, 1, 2, 3,     # all children
                    14, 15, 11  # peers - right, bottom, right-bottom
                ]
            elif relative == 'right-up':
                cands = [
                    0, 1, 2, 3,  # all children
                    12, 10, 15  # peers - left, left-bottom, bottom
                ]
            elif relative == 'left-bottom':
                cands = [
                    0, 1, 2, 3,  # all children
                    13, 9, 14   # peers - up, right-up, right
                ]
            elif relative == 'right-bottom':
                cands = [
                    0, 1, 2, 3,  # all children
                    8, 13, 12   # peers - left-up, up, left
                ]
            else:
                raise Exception('Invalid relative value !!!')
            # Iteratively get corresponding anchors.
            r_anchors = []
            for c in cands:
                r_anchors.append(s_anchors[c])
            return r_anchors
        # Restrict, not-abandon.
        elif adim == 9:
            # Check validity.
            if arg is None:
                raise Exception('One must assign the arg with previous action when adim = 9 !!!')
            # Select candidates for relative direction.
            relative = arg
            if relative == 'left-up':
                cands = [
                    0, 1, 2, 3,     # all children
                    7,  # parent - right-bottom
                    14, 15, 11  # peers - right, bottom, right-bottom
                ]
            elif relative == 'right-up':
                cands = [
                    0, 1, 2, 3,  # all children
                    6,  # parent - left-bottom
                    12, 10, 15  # peers - left, left-bottom, bottom
                ]
            elif relative == 'left-bottom':
                cands = [
                    0, 1, 2, 3,  # all children
                    5,  # parent - right-up
                    13, 9, 14   # peers - up, right-up, right
                ]
            elif relative == 'right-bottom':
                cands = [
                    0, 1, 2, 3,  # all children
                    4,  # parent - left-up
                    8, 13, 12   # peers - left-up, up, left
                ]
            else:
                raise Exception('Invalid relative value !!!')
            # Iteratively get corresponding anchors.
            r_anchors = []
            for c in cands:
                r_anchors.append(s_anchors[c])
            return r_anchors
        elif adim == 13:
            cands = [
                0, 1, 2, 3,  # all children
                8, 9, 10, 11, 12, 13, 14, 15    # all peers
            ]
            # Iteratively get corresponding anchors.
            r_anchors = []
            for c in cands:
                r_anchors.append(s_anchors[c])
            return r_anchors
        # Not-restrict, not-abandon .
        elif adim == 17:
            # Return all the anchors.
            return s_anchors
        else:
            raise ValueError('Unknown adim, this adim\'s anchors don\'t support now !!!')


    def _check_candidates(self, candidates):
        r'''
            Check the validity of all the candidates bounding-box.
        '''

        # "Bounding-box error" vector.
        BBOX_err = [False, ] * (self._act_dim - 1)
        info = [None, ] * (self._act_dim - 1)
        # Check the validity of each candidate.
        for idx, cand in enumerate(candidates):
            y1, x1, y2, x2 = cand
            if (y2 - y1) == 0.0 or (x2 - x1) == 0.0:
                BBOX_err[idx] = True
                info[idx] = 'Zero-scale Bbox !'
        # Check if it's the outermost level.
        if len(self._RelDir_prev) == 0:
            for _idx in range(4, self._act_dim - 1):
                BBOX_err[_idx] = True
                info[_idx] = 'Out-of-Boundary !'

        # Return the "Bbox Error" vector, and info vector.
        return BBOX_err, info


    def _push_forward(self, action, candidates, BBOX_errs, step_thres, infos):
        r'''
            Push forward the environment according to the given action.
                Mainly to get the destination bounding-box and terminal flag.
        '''

        # Check validity.
        if not isinstance(action, (int, np.int, np.int32, np.int64)):
            raise TypeError('The action must be an integer !!!')
        action = int(action)
        if action not in range(self._act_dim):
            raise ValueError('The action must be in range(0, {}) !!!'.format(self._act_dim))

        # Terminal flag.
        terminal = False
        # The out-most flag.
        outmost = len(self._RelDir_prev) == 0

        # Execute the relative direction procedure for given action.
        #   Different procedure according to action quantity.
        if self._act_dim == 8:  # restrict, abandon
            # --> select children.
            if action <= 3:
                # push the relative direction. -- focus in.
                rel_dirc = self._RELATIVE_DIRECTION[action]  # coz just the same order.
                self._RelDir_prev.append(rel_dirc)
            # --> select peers.
            elif action <= 6:
                # only enable when it isn't out-most.
                if not outmost:
                    # translate the current (newest) relative direction.  -- focus peer.
                    cur_Rdirc = self._RelDir_prev.pop()
                    # metaphysics formulation.
                    CRD_idx = self._RELATIVE_DIRECTION.index(cur_Rdirc)
                    NRD_idx = action - 4
                    if NRD_idx - CRD_idx <= 0:
                        NRD_idx -= 1
                    next_Rdirc = self._RELATIVE_DIRECTION[NRD_idx]
                    # pop out and push new relative direction.
                    self._RelDir_prev.append(next_Rdirc)
            # --> stop, terminal.
            else:
                terminal = True
        # Restrict, not-abandon
        elif self._act_dim == 9:
            # --> select children.
            if action <= 3:
                # push the relative direction. -- focus in.
                rel_dirc = self._RELATIVE_DIRECTION[action]  # coz just the same order.
                self._RelDir_prev.append(rel_dirc)
            # --> select parent.
            elif action == 4:
                # only enable when it isn't out-most.
                if not outmost:
                    # pop out the relative direction. -- focus out.
                    self._RelDir_prev.pop()
            # --> select peers.
            elif action <= 7:
                # only enable when it isn't out-most.
                if not outmost:
                    # translate the current (newest) relative direction.  -- focus peer.
                    cur_Rdirc = self._RelDir_prev.pop()
                    # metaphysics formulation.
                    CRD_idx = self._RELATIVE_DIRECTION.index(cur_Rdirc)
                    NRD_idx = action - 4
                    if NRD_idx - CRD_idx <= 0:
                        NRD_idx -= 1
                    next_Rdirc = self._RELATIVE_DIRECTION[NRD_idx]
                    # pop out and push new relative direction.
                    self._RelDir_prev.append(next_Rdirc)
            # --> stop, terminal.
            else:
                terminal = True
        # Not restrict, abandon.
        elif self._act_dim == 13:
            # Fake relative direction for "Out-of-Boundary" error check.
            fake_RD = self._RELATIVE_DIRECTION[0]
            # --> select children. -- focus in. push
            if action <= 3:
                self._RelDir_prev.append(fake_RD)
            # --> select peers.
            if action <= 11:
                pass
            # --> stop, terminal.
            else:
                terminal = True
        # Not restrict, not-abandon.
        elif self._act_dim == 17:
            # Fake relative direction for "Out-of-Boundary" error check.
            fake_RD = self._RELATIVE_DIRECTION[0]
            # --> select children. -- focus in. push
            if action <= 3:
                self._RelDir_prev.append(fake_RD)
            # --> select parent. -- focus out. pop
            elif action <= 7:
                # only enable when it isn't out-most.
                if not outmost:
                    # check if it's the outermost level.
                    self._RelDir_prev.pop()
            # --> select peers. -- do nothing ...
            elif action <= 15:
                pass
            # --> stop, terminal.
            else:
                terminal = True
        # Else.
        else:
            raise Exception('Unknown action dimension, there is no logic with respect to '
                            'current act_dim, please check your code !!!')

        # Check whether selected the invalid candidate.
        if not terminal:
            terminal = BBOX_errs[action]
            info = infos[action]
        else:
            info = 'Terminal !'

        # Get the selected bounding-box.
        if not terminal:
            dst_bbox = candidates[action]
        else:
            dst_bbox = self._FAKE_BBOX

        # Increase the time-step.
        self._time_step += 1
        # Check whether reaches the time-step threshold.
        if self._time_step >= step_thres:
            terminal = True
            info = 'Reach Threshold !'

        # Return the destination bbox, the terminal flag and info.
        return dst_bbox, terminal, info



# ---------------------------------------------------------------------------
# The wrapper for "Focus Environment", for the conveniently usage purpose.
# ---------------------------------------------------------------------------

class FocusEnvWrapper:
    r'''
        The wrapper for the "Focus Environment". It can be easily used.
    '''

    @staticmethod
    def get_instance(config, data_adapter):
        r'''
            Get the specific wrapper instance for use.
        '''
        is_multi = config['Others'].get('environment_instances', 16) > 1
        if is_multi:
            print('### ~~~ Focus-Environment start in "Multi-instance" mode !!!')
            return FocusEnvWrapper.FocusEnvMulti(config, data_adapter)
        else:
            print('### ~~~ Focus-Environment start in "Single-instance" mode !!!')
            return FocusEnvWrapper.FocusEnvSingle(config, data_adapter)


    class FocusEnv:
        r'''
            Abstract Wrapper.
        '''

        def __init__(self, config, data_adapter):
            r'''
                Initialization.
            '''
            # Normal.
            self._config = config
            # Check validity of data adapter.
            conf_base = self._config['Base']
            clazz_dim = conf_base.get('classification_dimension')
            self._verify_data(data_adapter, clazz_dim)
            self._data_adapter = data_adapter
            # Phrase.
            self._phrase = None
            # Instance id for each mode.
            self._train_Iid = 0
            self._validate_Iid = 0
            self._test_Iid = 0
            # Max iteration.
            self._max_iter = -1

        @property
        def acts_dim(self):
            r'''
                Get the valid actions quantity (dimension) of real environment.
            '''
            raise NotImplementedError

        @property
        def Fake_Bbox(self):
            r'''
                The fake "Bounding-Box" used in real environment.
            '''
            raise NotImplementedError

        def instance_id(self, p):
            r'''
                Get the (newest) instance id of specific phrase of environment.
            '''
            if p == 'Train':
                return self._train_Iid
            elif p == 'Validate':
                return self._validate_Iid
            elif p == 'Test':
                return self._test_Iid
            else:
                raise ValueError('Unknown phrase value !!!')

        def reset_instance_id(self, p, ins_id):
            r'''
                Reset the (instance id) counter for specific phrase. It mainly
                    used with the "restore from breakpoint" policy.
            '''
            if not isinstance(ins_id, (int, np.int, np.int32, np.int64)) or ins_id < 0:
                raise ValueError('The ins_id must be non-negative integer !!!')
            if p == 'Train':
                self._train_Iid = ins_id
            elif p == 'Validate':
                self._validate_Iid = ins_id
            elif p == 'Test':
                self._test_Iid = ins_id
            else:
                raise ValueError('Unknown phrase value !!!')
            return

        def switch_phrase(self, p, max_iter):
            r'''
                Switch to target phrase. And reset the max iteration.
            '''
            if p not in ['Train', 'Validate', 'Test']:
                raise ValueError('Unknown phrase value !!!')
            self._phrase = p
            if not isinstance(max_iter, (int, np.int, np.int32, np.int64)) or max_iter <= 0:
                raise ValueError('The max_iter must be None or positive integer !!!')
            self._max_iter = max_iter
            return

        def roll_out(self, segment_func, op_func, anim_type):
            r'''
                Roll out environment one time-step. What's more, it will return the
                    training elements for "Train" phrase, or just terminal flag for
                    "Validate/Test" phrase.

            Parameters:
                segment_func: The function used to segmentation.
                op_func: The function used to operate for step.
                anim_type: The visualization type.

            Return:
                The tuple of (experiences, terminals, rewards, infos, reach_max, sample_ids, env_ids)
                    when in "Train" phrase.
                The tuple of (segmentations, labels, terminals, rewards, infos, reach_max, sample_ids, env_ids,
                    data_identities) when in "Validate" or "Test" phrase.
            '''
            raise NotImplementedError

        def _verify_data(self, adapter, category):
            r'''
                Verify the data supported by adapter.
            '''

            # Firstly check the type of adapter.
            if not isinstance(adapter, Adapter):
                raise TypeError('The adapter should be @Type{Adapter} !!!')

            # Get a train image-label pair to verify.
            img, label, clazz_weights, data_arg = adapter.next_image_pair(mode='Train', batch_size=1)
            MHA_idx, inst_idx = data_arg

            # Check the type and dimension of the data.
            if not isinstance(img, np.ndarray) or not isinstance(label, np.ndarray):
                raise TypeError('The type of image and label both should be'
                                '@Type{numpy.ndarray} !!!')

            # Now we only support single image processing.
            if img.ndim != 3 or label.ndim != 2:  # [width, height, modalities], [width, height]
                raise Exception('The dimension of the image and label both should '
                                'be 3 and 2 !!! img: {}, label: {}\n'
                                'Now we only support single image processing ...'.format(img.ndim, label.ndim))

            # Check the shape consistency.
            shape_img = img.shape[:-1]
            shape_label = label.shape
            shape_consistency = shape_img == shape_label
            if not shape_consistency:
                raise Exception('The shape of image and label are not satisfy consistency !!! '
                                'img: {}, label: {}'.format(shape_img, shape_label))

            # Check the validity of MHA_idx and inst_idx.
            if not isinstance(MHA_idx, (int, np.int, np.int32, np.int64)):
                raise TypeError('The MHA_idx must be of integer type !!!')
            if not isinstance(inst_idx, (int, np.int, np.int32, np.int64)):
                raise TypeError('The inst_idx must be of integer type !!!')

            # Check the validity of clazz_weights.
            if not isinstance(clazz_weights, np.ndarray) or \
                            clazz_weights.ndim != 1 or \
                            clazz_weights.shape[0] != category:
                raise Exception('The class weights should be of 2-D numpy '
                                'array with shape (?, category) !!!')

            # Finish.
            return


    class FocusEnvSingle(FocusEnv):
        r'''
            Single instance implementation.
        '''

        def __init__(self, config, data_adapter):
            r'''
                Initialization.
            '''
            # Normal.
            super().__init__(config, data_adapter)
            # Declare the environment instance.
            self._env_Ins = FocusEnvCore(config, data_adapter, is_separate=False)
            # Sample Meta.
            self._sample_meta = None
            # Label holder and data identity.
            self._label_holder = None
            self._data_identity = None

        @property
        def acts_dim(self):
            r'''
                Get the valid actions quantity (dimension) of real environment.
            '''
            return self._env_Ins.acts_dim

        @property
        def Fake_Bbox(self):
            r'''
                The fake "Bounding-Box" used in real environment.
            '''
            return self._env_Ins.Fake_Bbox

        def switch_phrase(self, p, max_iter):
            r'''
                Switch to target phrase. And reset the max iteration.
                    Note that, one must switch the real environment at the same time.
            '''
            super().switch_phrase(p, max_iter)
            self._env_Ins.switch_phrase(p=p)
            return

        def roll_out(self, segment_func, op_func, anim_type):
            r'''
                Roll out the environment for one time-step. It will return the experiences for
                    training and some other variables.

            Parameters:
                segment_func: The function used to segmentation.
                op_func: The function used to operate for step.
                anim_type: The visualization type.

            Return:
                The tuple of (experiences, terminals, rewards, infos, reach_max, sample_ids, env_ids)
                    when in "Train" phrase.
                The tuple of (segmentations, labels, terminals, rewards, infos, reach_max, sample_ids, env_ids,
                    data_identities) when in "Validate" or "Test" phrase.
            '''

            # Check the validity of execution logic.
            if self._phrase is None or self._max_iter == -1:
                raise Exception('One must call @Method{swith_phrase} once before calling '
                                '@Method{roll_out} !!!')

            # Check the validity of reach max iteration.
            if self._phrase == 'Train':
                miter_flag = self._train_Iid >= self._max_iter
            elif self._phrase == 'Validate':
                miter_flag = self._validate_Iid >= self._max_iter
            elif self._phrase == 'Test':
                miter_flag = self._test_Iid >= self._max_iter
            else:
                raise ValueError('Unknown phrase value !!!')
            if miter_flag and self._env_Ins.finished:
                raise Exception('Already reach the max iteration, can not be '
                                'rolled out any more !!!')

            # Different process logic according to the phrase.
            if self._phrase == 'Train':
                # Reset the environment.
                if self._env_Ins.finished:
                    image, label, clazz_weights, (mha_idx, inst_idx) = self._env_Ins.reset(sample_id=self._train_Iid,
                                                                                           segment_func=segment_func)
                    # Increase the instance id.
                    self._train_Iid += 1
                    # Determine the store type, and then determine the sample meta.
                    conf_others = self._config['Others']
                    store_type = conf_others.get('sample_type', 'light')
                    if store_type == 'light':
                        self._sample_meta = [mha_idx, inst_idx, clazz_weights]
                    elif store_type == 'heave':
                        self._sample_meta = [image, label, clazz_weights]
                    else:
                        raise ValueError('Unknown sample store type !!!')
                # Step the environment.
                state, action, terminal, anchors, BBOX_errs, \
                    next_state, reward, info = self._env_Ins.step_comp(op_func=op_func)
                # Package the experience.
                if self._sample_meta is None:
                    raise Exception('Invalid coding !!!')
                experience = (tuple(self._sample_meta.copy()),
                              state,
                              action,
                              terminal,
                              anchors,
                              BBOX_errs,
                              next_state,
                              reward)
                # Render the environment.
                if terminal:
                    self._env_Ins.render(anim_type=anim_type)
                # Check whether reach the max iteration or not.
                reach_max = self._train_Iid >= self._max_iter and self._env_Ins.finished
                # Return the experiences, terminals, rewards, infos, reach_max flag, sample_ids, env_ids.
                return [experience], [terminal], [reward], [info], reach_max, [self._env_Ins.sample_id], [0]

            # "Validate" or "Test" phrase.
            else:
                # Reset the environment.
                if self._env_Ins.finished:
                    if self._phrase == 'Validate':
                        self._label_holder, self._data_identity = self._env_Ins.reset(
                            sample_id=self._validate_Iid,
                            segment_func=segment_func
                        )
                        # Increase the instance id.
                        self._validate_Iid += 1
                    else:
                        self._label_holder, self._data_identity = self._env_Ins.reset(
                            sample_id=self._test_Iid,
                            segment_func=segment_func
                        )
                        # Increase the instance id.
                        self._test_Iid += 1
                label = self._label_holder.copy() if self._label_holder is not None else None
                # Step the environment.
                terminal, (segmentation, reward, info) = self._env_Ins.step_comp(op_func=op_func)
                # Render the environment.
                if terminal:
                    self._env_Ins.render(anim_type=anim_type)
                # Check whether reach the max iteration or not.
                if self._phrase == 'Validate':
                    reach_max = self._validate_Iid >= self._max_iter and self._env_Ins.finished
                else:
                    reach_max = self._test_Iid >= self._max_iter and self._env_Ins.finished
                # Return the segmentations, labels, terminals, rewards, infos, reach_max,
                #   sample_ids, env_ids, data_identities.
                return [segmentation], [label], [terminal], [reward], [info], reach_max, \
                       [self._env_Ins.sample_id], [0], [tuple(list(self._data_identity).copy())]


    class FocusEnvMulti(FocusEnv):
        r'''
            Multi-instance implementation.
        '''

        def __init__(self, config, data_adapter):
            r'''
                Initialization.
            '''
            # Normal.
            super().__init__(config, data_adapter)
            # Declare the environment instance list.
            conf_others = self._config['Others']
            instance_num = conf_others.get('environment_instances', 16)
            self._env_cands = []
            # Sample Meta list.
            self._sample_meta_cands = []
            # Label holder list and data identities.
            self._label_holder_cands = []
            self._data_identity_cands = []
            # Iteratively add.
            for _ in range(instance_num):
                self._env_cands.append(FocusEnvCore(config, data_adapter, is_separate=True))
                self._sample_meta_cands.append(None)
                self._label_holder_cands.append(None)
                self._data_identity_cands.append(None)

        @property
        def acts_dim(self):
            r'''
                Get the valid actions quantity (dimension) of real environment.
            '''
            return self._env_cands[0].acts_dim

        @property
        def Fake_Bbox(self):
            r'''
                The fake "Bounding-Box" used in real environment.
            '''
            return self._env_cands[0].Fake_Bbox

        def switch_phrase(self, p, max_iter):
            r'''
                Switch to target phrase. And reset the max iteration.
                    Note that, one must switch the real environment at the same time.
            '''
            super().switch_phrase(p, max_iter)
            for env in self._env_cands:
                env.switch_phrase(p=p)
            return

        def roll_out(self, segment_func, op_func, anim_type):
            r'''
                Roll out the environment for one time-step. It will return the experiences for
                    training and some other variables.

            Parameters:
                segment_func: The function used to segmentation.
                op_func: The function used to operate for step.
                anim_type: The visualization type.

            Return:
                The tuple of (experiences, terminals, rewards, infos, reach_max, sample_ids, env_ids)
                    when in "Train" phrase.
                The tuple of (segmentations, labels, terminals, rewards, infos, reach_max, sample_ids, env_ids,
                    data_identities) when in "Validate" or "Test" phrase.
            '''

            # Check the validity of execution logic.
            if self._phrase is None or self._max_iter == -1:
                raise Exception('One must call @Method{swith_phrase} once before calling '
                                '@Method{roll_out} !!!')

            # Check the validity of reach max iteration.
            if self._phrase == 'Train':
                miter_flag = self._train_Iid >= self._max_iter
            elif self._phrase == 'Validate':
                miter_flag = self._validate_Iid >= self._max_iter
            elif self._phrase == 'Test':
                miter_flag = self._test_Iid >= self._max_iter
            else:
                raise ValueError('Unknown phrase value !!!')
            for e in self._env_cands:
                miter_flag = miter_flag and e.finished
            if miter_flag:
                raise Exception('Already reach the max iteration, can not be '
                                'rolled out any more !!!')

            # Different process logic according to the phrase.
            if self._phrase == 'Train':
                # The return elements.
                experiences = []
                terminals = []
                rewards = []
                infos = []
                sample_ids = []
                # step 01 elements.
                state_elems = []
                push_elems = []
                exec_ids = []
                # --------------------------- Reset and Step_1st ---------------------------
                # Iteratively push forward (step 01) all the environment instance.
                for eid, env in enumerate(self._env_cands):
                    # Check whether fulfill the iteration or not. Do not assign new process if fulfilled.
                    if self._train_Iid >= self._max_iter and env.finished:
                        continue
                    # Reset the environment.
                    if env.finished:
                        image, label, clazz_weights, (mha_idx, inst_idx) = env.reset(
                            sample_id=self._train_Iid,
                            segment_func=segment_func)
                        # Increase the instance id.
                        self._train_Iid += 1
                        # Determine the store type, and then determine the sample meta.
                        conf_others = self._config['Others']
                        store_type = conf_others.get('sample_type', 'light')
                        if store_type == 'light':
                            self._sample_meta_cands[eid] = [mha_idx, inst_idx, clazz_weights]
                        elif store_type == 'heave':
                            self._sample_meta_cands[eid] = [image, label, clazz_weights]
                        else:
                            raise ValueError('Unknown sample store type !!!')
                    # Step 1st the environment.
                    state_e, push_e = env.step_1st()
                    # Append related information.
                    state_elems.append(state_e)
                    push_elems.append(push_e)
                    exec_ids.append(eid)
                # --------------------------- Operation Function ---------------------------
                # Execute the operation function to get the "Results".
                results = op_func(state_elems, with_explore=True, with_reward=True)
                # Re-package the results.
                r_1, r_2, r_3, r_4 = results
                r_all = []
                for ri_1, ri_2, ri_3, ri_4 in zip(r_1, r_2, r_3, r_4):
                    r_all.append((ri_1, ri_2, ri_3, ri_4))
                results = r_all
                # --------------------------- Step_2nd ---------------------------
                # Iteratively get results and re-assign (step 02) for all the environment instance.
                for ex_id, r, se, pe in zip(exec_ids, results, state_elems, push_elems):
                    # Step 2nd the environment.
                    state, action, terminal, anchors, BBOX_errs, \
                        next_state, reward, info = self._env_cands[ex_id].step_2nd(results=r,
                                                                                   state_elems=se,
                                                                                   push_elems=pe)
                    # Package the experience.
                    if self._sample_meta_cands[ex_id] is None:
                        raise Exception('Invalid coding !!!')
                    exp = (tuple(self._sample_meta_cands[ex_id].copy()),
                           state,
                           action,
                           terminal,
                           anchors,
                           BBOX_errs,
                           next_state,
                           reward)
                    # Do some operation and render the environment.
                    if terminal:
                        self._env_cands[ex_id].render(anim_type=anim_type)
                    # Package the each elements into batch.
                    experiences.append(exp)
                    terminals.append(terminal)
                    rewards.append(reward)
                    infos.append(info)
                    sample_ids.append(self._env_cands[ex_id].sample_id)
                # Check whether reach the max iteration or not.
                reach_max = self._train_Iid >= self._max_iter
                for e in self._env_cands:
                    reach_max = reach_max and e.finished
                # Return the experience, terminal, reward, info, reach_max flag, sample_ids, env_ids.
                return experiences, terminals, rewards, infos, reach_max, sample_ids, exec_ids

            # "Validate" or "Test" phrase.
            else:
                # The return elements.
                segmentations = []
                labels = []
                data_identities = []
                terminals = []
                rewards = []
                infos = []
                sample_ids = []
                # step 01 elements.
                state_elems = []
                push_elems = []
                exec_ids = []
                # --------------------------- Reset and Step_1st ---------------------------
                # Iteratively push forward all the environment instance.
                for eid, env in enumerate(self._env_cands):
                    # Check whether fulfill the iteration or not. Do not assign new process if fulfilled.
                    if self._phrase == 'Validate':
                        if self._validate_Iid >= self._max_iter and env.finished:
                            continue
                    else:
                        if self._test_Iid >= self._max_iter and env.finished:
                            continue
                    # Reset the environment.
                    if env.finished:
                        if self._phrase == 'Validate':
                            self._label_holder_cands[eid], self._data_identity_cands[eid] = env.reset(
                                sample_id=self._validate_Iid,
                                segment_func=segment_func
                            )
                            # Increase the instance id.
                            self._validate_Iid += 1
                        else:
                            self._label_holder_cands[eid], self._data_identity_cands[eid] = env.reset(
                                sample_id=self._test_Iid,
                                segment_func=segment_func
                            )
                            # Increase the instance id.
                            self._test_Iid += 1
                    # Package label and data identity into batch.
                    lab = self._label_holder_cands[eid].copy() if self._label_holder_cands[eid] is not None else None
                    labels.append(lab)
                    data_id = tuple(list(self._data_identity_cands[eid]).copy()) \
                        if self._data_identity_cands[eid] is not None else None
                    data_identities.append(data_id)
                    # Step 1st the environment.
                    state_e, push_e = env.step_1st()
                    # Append related information.
                    state_elems.append(state_e)
                    push_elems.append(push_e)
                    exec_ids.append(eid)
                # --------------------------- Operation Function ---------------------------
                # Execute the operation function to get the "Results".
                results = op_func(state_elems, with_explore=False, with_reward=self._phrase=='Validate')
                # Re-package the results.
                if self._phrase == 'Validate':
                    r_1, r_2, r_3, r_4 = results
                    r_all = []
                    for ri_1, ri_2, ri_3, ri_4 in zip(r_1, r_2, r_3, r_4):
                        r_all.append((ri_1, ri_2, ri_3, ri_4))
                    results = r_all
                else:
                    r_1, r_2, r_3 = results
                    r_all = []
                    for ri_1, ri_2, ri_3 in zip(r_1, r_2, r_3):
                        r_all.append((ri_1, ri_2, ri_3))
                    results = r_all
                # --------------------------- Step_2nd ---------------------------
                # Iteratively get results and re-assign (step 02) for all the environment instance.
                for ex_id, r, se, pe in zip(exec_ids, results, state_elems, push_elems):
                    # Step 2nd the environment.
                    terminal, (SEG_cur, reward, info) = self._env_cands[ex_id].step_2nd(results=r,
                                                                                        state_elems=se,
                                                                                        push_elems=pe)
                    # Render the environment.
                    if terminal:
                        self._env_cands[ex_id].render(anim_type=anim_type)
                    # Package the each elements into batch.
                    segmentations.append(SEG_cur)
                    terminals.append(terminal)
                    rewards.append(reward)
                    infos.append(info)
                    sample_ids.append(self._env_cands[ex_id].sample_id)
                # Check whether reach the max iteration or not.
                if self._phrase == 'Validate':
                    reach_max = self._validate_Iid >= self._max_iter
                else:
                    reach_max = self._test_Iid >= self._max_iter
                for e in self._env_cands:
                    reach_max = reach_max and e.finished
                # Return the segmentations, labels, terminals, rewards, infos, reach_max,
                #   sample_ids, env_ids, data_identities.
                return segmentations, labels, terminals, rewards, infos, reach_max, \
                    sample_ids, exec_ids, data_identities


