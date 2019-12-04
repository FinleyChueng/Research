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
# The two-stage segmentation environment. Which consists of "Segment" and
#   "Focus" phrase.
# ---------------------------------------------------------------------------

class FocusEnv:
    r'''
        The task-specific environment. This one is a "Two-stage" environment, which
            consists of "Segment" and "Focus" stage.

        When in "Segment" stage, it will supply the segmentation result for given region.
        When in "Focus" stage, it will calculate the next "Focus Bbox" coordinates for
            precisely process.
    '''

    def __init__(self, config, data_adapter):
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
        self._verify_data(data_adapter, clazz_dim)
        self._adapter = data_adapter

        # Action quantity.
        conf_dqn = self._config['DQN']
        if conf_dqn.get('restriction_action'):
            self._act_dim = 9
        else:
            self._act_dim = 17

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
        self._FB_opted = False     # whether optimized the "Focus Bbox" or not.

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


    def switch_phrase(self, p):
        r'''
            Switch to target phrase.
        '''
        if p not in ['Train', 'Validate', 'Test']:
            raise ValueError('Unknown phrase value !!!')
        self._phrase = p
        return


    def reset(self, segment_func=None):
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
            self._COMP_result = np.zeros([suit_h, suit_w, clazz_dim], dtype=np.float32)
        elif CR_method == 'mask':
            self._COMP_result = np.zeros([suit_h, suit_w], dtype=np.float32)
        else:
            raise ValueError('Unknown result fusion method !!!')

        # Reset the "Focus Bounding-box".
        self._focus_bbox = np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32)   # (y1, x1, y2, x2)

        # Reset the "Action History".
        self._ACTION_his = collections.deque([-1,] * his_len)

        # Reset the "Bbox History".
        self._BBOX_his = collections.deque([self._FAKE_BBOX] * step_thres)

        # Reset the "Previous Relative Directions".
        self._RelDir_prev = ['left-up']     # add default outermost direction.

        # Reset the time-step.
        self._time_step = 0

        # Reset the process flag.
        self._finished = False

        # -------------------------- Switch to next image-label pair. ---------------------------
        # Get the next image-label pair according to the phrase.
        if self._phrase == 'Train':
            # Get the next train sample pair.
            img, label, clazz_weights, data_arg = self._adapter.next_image_pair(mode='Train', batch_size=1)
        elif self._phrase == 'Validate':
            # Get the next validate image-label pair.
            img, label = self._adapter.next_image_pair(mode='Validate', batch_size=1)
        elif self._phrase == 'Test':
            # Get the next test image-label pair.
            img, label = self._adapter.next_image_pair(mode='Test', batch_size=1)
        else:
            raise ValueError('Unknown phrase !!!')

        # Assign the image and label holder.
        self._image = img
        self._label = label
        # Get clazz weights from configuration here.
        conf_train = self._config['Training']
        self._clazz_weights = conf_train.get('clazz_weights', None)

        # Record the process.
        if self._anim_recorder is not None:
            self._anim_recorder.reload((img.copy(), label.copy() if label is not None else None))

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
                init_y1 = max(0, min(init_region[0]) - initFB_pad) / image_height
                init_x1 = max(0, min(init_region[1]) - initFB_pad) / image_width
                init_y2 = min(image_height, max(init_region[0]) + initFB_pad) / image_height
                init_x2 = min(image_width, max(init_region[1]) + initFB_pad) / image_width
                self._focus_bbox = np.asarray([init_y1, init_x1, init_y2, init_x2])
                # Assign the "Current Segmentation" and "Complete info" holders.
                self._SEG_prev = segmentation
                self._COMP_result = COMP_res
                # Append the "Relative Direction" history (for wide range).
                self._RelDir_prev.append('right-bottom')
                # Increase the time-step, and record the current bbox to the history.
                self._time_step += 1
                self._BBOX_his.append(self._focus_bbox.copy())
                self._BBOX_his.popleft()

        # Whether return the train samples meta according to phrase.
        if self._phrase == 'Train':
            # Reset lazz weights if needed.
            if self._clazz_weights is None:  # do not config
                self._clazz_weights = clazz_weights
            # Return train samples.
            return img.copy(), label.copy(), self._clazz_weights.copy(), data_arg
        else:
            # Return the label used to compute metric when in "Validate" phrase.
            return label.copy() if label is not None else None


    def step(self, op_func):
        r'''
            Step (interact with) the environment. Here coz we build
                the two-stage alternative model, so this function will
                execute the "Segment" and "Focus" stage, alternatively.

        Parameters:
            op_func: The operation function. It must be one of the
                type listed likes below:
                -----------------------------
                1) segment_func: The segment function used in "Segment" stage,
                which will be applied to generate the fusion result
                of previous complete result and current region result.
                -----------------------------
                2) focus_func: The focus function used in "Focus" (and "Segment")
                stage, which will be applied to select the next region
                to precisely segment.
                -----------------------------
                3) train_func: Containing the both two of "Segment" and "Focus"
                function. Only used in "Training" phrase.
                -----------------------------

        Return:
            The tuple of (state, action, terminal, anchors, err, next_state, info)
                when in "Train" phrase.
            state: (SEG_prev, cur_bbox, position_info, acts_prev, bboxes_prev, his_plen, comp_prev)
            next_state: (SEG_cur, focus_bbox, next_posinfo, acts_cur, bboxes_cur, his_clen, comp_cur)
            ---------------------------------------------------------------
            SEG_prev: The previous segmentation result.
            cur_bbox: The current bbox of image.
            position_info: The position information for current time-step.
            acts_prev: The current actions history.
            bboxes_prev: The current bboxes history.
            his_plen: The valid length of current history.
            comp_prev: The current "Complete Result".
            action: The action executed of current time-step.
            terminal: Flag that indicates whether current image is finished.
            anchors: The anchors for current bbox.
            BBOX_errs: The "BBOX erros" vector for current anchors.
            SEG_cur: Current segmentation result.
            focus_bbox: Next focus bbox of image.
            next_posinfo: The fake next position information.
            acts_cur: The next actions history.
            bboxes_cur: The next bboxes history.
            his_clen: The valid length of next history.
            comp_cur: The next "Complete Result".
            info: Extra information.(Optional, default is type.None)
            ---------------------------------------------------------------
            The tuple of (terminal, SEG_cur, reward, info) when in "Validate" or "Test" phrase.
                ** Note that, the reward is None when in "Test" phrase, and the
                    SEG_cur will be used to write result in "Test" phrase.
        '''

        # Check validity.
        if not callable(op_func):
            raise TypeError('The op_func must be a function !!!')

        # Check the validity of execution logic.
        if self._finished:
            raise Exception('The process of current image-label pair is finished, it '
                            'can not be @Method{step} any more !!! '
                            'You should call the @Method{reset} to reset the '
                            'environment to process the next image.')

        # Get the detailed config.
        conf_base = self._config['Base']
        input_shape = conf_base.get('input_shape')
        image_height, image_width = input_shape[1:3]
        conf_dqn = self._config['DQN']
        step_threshold = conf_dqn.get('step_threshold', 10)
        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')

        # The position information generation function, for convenient usage.
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

        # The current state.
        cur_bbox = self._focus_bbox.copy()
        SEG_prev = self._SEG_prev.copy()
        position_info = gen_position_info(self._focus_bbox)
        acts_prev = np.asarray(self._ACTION_his.copy())
        bboxes_prev = np.asarray(self._BBOX_his.copy())
        his_plen = self._time_step
        comp_prev = self._COMP_result.copy()

        # -------------------------------- Core Part -----------------------------
        # Generate anchors.
        conf_dqn = self._config['DQN']
        anchor_scale = conf_dqn.get('anchors_scale', 2)
        if self._act_dim == 9:
            # "Restrict" mode.
            anchors = self._anchors(self._focus_bbox.copy(), anchor_scale, 9,
                                    arg=self._RelDir_prev[-1])   # the newest relative direction.
        elif self._act_dim == 17:
            # "Whole Candidates" mode.
            anchors = self._anchors(self._focus_bbox.copy(), anchor_scale, 17)
        else:
            raise Exception('Unknown action dimension, there is no logic with respect to '
                            'current act_dim, please check your code !!!')

        # Get the "Bbox Error" vector for all the candidates (anchors).
        BBOX_errs, infos = self._check_candidates(anchors)

        # Execute the operation fucntion to conduct "Segment" and "Focus".
        if self._label is not None:
            segmentation, COMP_res, action, reward = op_func((self._image.copy(),
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
                                                              self._clazz_weights.copy()),
                                                             with_explore=self._phrase == 'Train',
                                                             with_reward=True)
            reward = reward[action]
        else:
            segmentation, COMP_res, action = op_func((self._image.copy(),
                                                      SEG_prev.copy(),
                                                      position_info.copy(),
                                                      cur_bbox.copy(),
                                                      acts_prev.copy(),
                                                      bboxes_prev.copy(),
                                                      his_plen,
                                                      comp_prev.copy()),
                                                     with_explore=False,
                                                     with_reward=False)
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
        self._BBOX_his.append(dst_bbox.copy())
        self._BBOX_his.popleft()
        # Iteratively re-assign the "Segmentation" result, "Complete info" and "Focus Bbox".
        self._SEG_prev = segmentation
        self._COMP_result = COMP_res
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
        next_posinfo = gen_position_info(self._focus_bbox)
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
        # Release the memory.
        gc.collect()
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

        # -------> Generate the whole situation for convenient selection.
        s_anchors = []
        # ---> Children situation.
        cld_h = h / scale
        cld_w = w / scale
        # S1: Child left-up.
        s1_y1 = y1
        s1_x1 = x1
        s1_y2 = min(1.0, y1 + cld_h)
        s1_x2 = min(1.0, x1 + cld_w)
        s_anchors.append([s1_y1, s1_x1, s1_y2, s1_x2])
        # S2: Child right-up.
        s2_y1 = max(0.0, y2 - cld_h)
        s2_x1 = x1
        s2_y2 = y2
        s2_x2 = min(1.0, x1 + cld_w)
        s_anchors.append([s2_y1, s2_x1, s2_y2, s2_x2])
        # S3: Child left-bottom.
        s3_y1 = y1
        s3_x1 = max(0.0, x2 - cld_w)
        s3_y2 = min(1.0, y1 + cld_h)
        s3_x2 = x2
        s_anchors.append([s3_y1, s3_x1, s3_y2, s3_x2])
        # S4: Child right-bottom.
        s4_y1 = max(0.0, y2 - cld_h)
        s4_x1 = max(0.0, x2 - cld_w)
        s4_y2 = y2
        s4_x2 = x2
        s_anchors.append([s4_y1, s4_x1, s4_y2, s4_x2])
        # ---> Parents situation.
        par_h = h * scale
        par_w = w * scale
        # S5: Parent left-up.
        s5_y1 = max(0.0, y2 - par_h)
        s5_x1 = max(0.0, x2 - par_w)
        s5_y2 = y2
        s5_x2 = x2
        s_anchors.append([s5_y1, s5_x1, s5_y2, s5_x2])
        # S6: Parent right-up.
        s6_y1 = y1
        s6_x1 = max(0.0, x2 - par_w)
        s6_y2 = min(1.0, y1 + par_h)
        s6_x2 = x2
        s_anchors.append([s6_y1, s6_x1, s6_y2, s6_x2])
        # S7: Parent left-bottom.
        s7_y1 = max(0.0, y2 - par_h)
        s7_x1 = x1
        s7_y2 = y2
        s7_x2 = min(1.0, x1 + par_w)
        s_anchors.append([s7_y1, s7_x1, s7_y2, s7_x2])
        # S8: Parent right-bottom.
        s8_y1 = y1
        s8_x1 = x1
        s8_y2 = min(1.0, y1 + par_h)
        s8_x2 = min(1.0, x1 + par_w)
        s_anchors.append([s8_y1, s8_x1, s8_y2, s8_x2])
        # ---> Peers situation.
        pee_h = h
        pee_w = w
        # S9: Peer left-up.
        s9_y1 = max(0.0, y2 - par_h)
        s9_x1 = max(0.0, x2 - par_w)
        s9_y2 = max(0.0, y2 - par_h + pee_h)
        s9_x2 = max(0.0, x2 - par_w + pee_w)
        s_anchors.append([s9_y1, s9_x1, s9_y2, s9_x2])
        # S10: Peer right-up.
        s10_y1 = min(1.0, y1 + par_h - pee_h)
        s10_x1 = max(0.0, x2 - par_w)
        s10_y2 = min(1.0, y1 + par_h)
        s10_x2 = max(0.0, x2 - par_w + pee_w)
        s_anchors.append([s10_y1, s10_x1, s10_y2, s10_x2])
        # S11: Peer left-bottom.
        s11_y1 = max(0.0, y2 - par_h)
        s11_x1 = min(1.0, x1 + par_w - pee_w)
        s11_y2 = max(0.0, y2 - par_h + pee_h)
        s11_x2 = min(1.0, x1 + par_w)
        s_anchors.append([s11_y1, s11_x1, s11_y2, s11_x2])
        # S12: Peer right-bottom.
        s12_y1 = min(1.0, y1 + par_h - pee_h)
        s12_x1 = min(1.0, x1 + par_w - pee_w)
        s12_y2 = min(1.0, y1 + par_h)
        s12_x2 = min(1.0, x1 + par_w)
        s_anchors.append([s12_y1, s12_x1, s12_y2, s12_x2])
        # S13: Peer pure-left.
        s13_y1 = max(0.0, y2 - par_h)
        s13_x1 = x1
        s13_y2 = max(0.0, y2 - par_h + pee_h)
        s13_x2 = x2
        s_anchors.append([s13_y1, s13_x1, s13_y2, s13_x2])
        # S14: Peer pure-up.
        s14_y1 = y1
        s14_x1 = max(0.0, x2 - par_w)
        s14_y2 = y2
        s14_x2 = max(0.0, x2 - par_w + pee_w)
        s_anchors.append([s14_y1, s14_x1, s14_y2, s14_x2])
        # S15: Peer pure-right.
        s15_y1 = min(1.0, y1 + par_h - pee_h)
        s15_x1 = x1
        s15_y2 = min(1.0, y1 + par_h)
        s15_x2 = x2
        s_anchors.append([s15_y1, s15_x1, s15_y2, s15_x2])
        # S16: Peer pure-bottom.
        s16_y1 = y1
        s16_x1 = min(1.0, x1 + par_w - pee_w)
        s16_y2 = y2
        s16_x2 = min(1.0, x1 + par_w)
        s_anchors.append([s16_y1, s16_x1, s16_y2, s16_x2])

        # Now select anchors to return according to the mode.
        if adim == 9:
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
        # No restriction.
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
        RelDir_his = self._RelDir_prev.copy()
        RelDir_his.pop()
        if len(RelDir_his) == 0:
            for _idx in range(self._act_dim - 1):
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

        # Execute the given action. Different procedure according to action quantity.
        if self._act_dim == 9:
            # --> select children.
            if action <= 3:
                # push the relative direction. -- focus in.
                rel_dirc = self._RELATIVE_DIRECTION[action]  # coz just the same order.
                self._RelDir_prev.append(rel_dirc)
            # --> select parent.
            elif action == 4:
                # pop out the relative direction. -- focus out.
                self._RelDir_prev.pop()
            # --> select peers.
            elif action <= 7:
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
        # "Normal" mode.
        elif self._act_dim == 17:
            # Fake relative direction for "Out-of-Boundary" error check.
            fake_RD = self._RELATIVE_DIRECTION[0]
            # --> select children. -- focus in. push
            if action <= 3:
                self._RelDir_prev.append(fake_RD)
            # --> select parent. -- focus out. pop
            elif action <= 7:
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
        if img.ndim != 3 or label.ndim != 2:    # [width, height, modalities], [width, height]
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

