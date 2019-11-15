import gc
import util.evaluation as eva
from util.visualization import *
from core.env import *
from dataset.adapter.base import *



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
                                                 vision_filename_mask=anim_path)

        # The flag indicating whether is in "Train", "Validate" or "Test" phrase.
        self._phrase = 'Train'  # Default in "Train"

        # --------------------- Declare some input data placeholder. ---------------------
        self._image = None  # Current processing image.
        self._label = None  # Label for current image.
        self._SEG_stage = None  # Flag indicating whether is "Segmentation" stage or not.

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


    def switch_phrase(self, p):
        r'''
            Switch to target phrase.
        '''
        if p not in ['Train', 'Validate', 'Test']:
            raise ValueError('Unknown phrase value !!!')
        self._phrase = p
        return


    def reset(self):
        r'''
            Reset the environment. Mainly to reset the related parameters,
                and switch to next image-label pair.
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

        # Reset the "Previous Relative Directions".
        self._RelDir_prev = ['left-up']     # add default outermost direction.

        # Reset the time-step.
        self._time_step = 0

        # Reset the process flag.
        self._finished = False
        self._FB_opted = False  # "Focus Bbox" not optimized.

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

        # Whether return the train samples meta according to phrase.
        if self._phrase == 'Train':
            # Record the process.
            if self._anim_recorder is not None:
                self._anim_recorder.reload((img.copy(), label.copy()))
            # Clazz weights.
            conf_train = self._config['Training']
            weights = conf_train.get('clazz_weights', None)
            if weights is None:  # do not config
                weights = clazz_weights
            # Return train samples.
            return img.copy(), label.copy(), weights.copy(), data_arg
        else:
            # Record the process.
            if self._anim_recorder is not None:
                self._anim_recorder.reload((img.copy(), label.copy() if label is not None else None))
            # Return the label used to compute metric when in "Validate" phrase.
            return label.copy() if label is not None else None


    def step(self, op_func, SEG_stage):
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
            SEG_stage: Indicating whether is "Segment" or "Focus" stage.

        Return:
            The tuple of (SEG_prev, cur_bbox, position_info, action, reward, over,
                SEG_cur, focus_bbox, next_posinfo, info) when in "Train" phrase.
            ---------------------------------------------------------------
            SEG_prev: The previous segmentation result.
            cur_bbox: The current bbox of image.
            position_info: The position information for current time-step.
            action: The action executed of current time-step.
            reward: The reward of current action.
            over: Flag that indicates whether current image is finished.
            SEG_cur: Current segmentation result.
            focus_bbox: Next focus bbox of image.
            next_posinfo: The fake next position information.
            info: Extra information.(Optional, default is type.None)
            ---------------------------------------------------------------
            The tuple of (over, SEG_cur, reward, info) when in "Validate" or "Test" phrase.
                ** Note that, the reward is None when in "Test" phrase, and the
                    SEG_cur will be used to write result in "Test" phrase.
        '''

        # Check validity.
        if not callable(op_func):
            raise TypeError('The op_func must be a function !!!')
        if not isinstance(SEG_stage, bool):
            raise TypeError('The SEG_stage must be a boolean !!!')

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
        initFB_optimize = conf_dqn.get('initial_bbox_optimize', True)
        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')
        conf_train = self._config['Training']
        sample_share = conf_train.get('sample_share', False)

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
            elif pos_method == 'sight':
                pos_info = focus_bbox.copy()
            else:
                raise ValueError('Unknown position information fusion method !!!')
            return pos_info

        # Declare the position information here.
        position_info = gen_position_info(self._focus_bbox)

        # Visualiztion related.
        vis_bbox = self._focus_bbox.copy()

        # Old "Focus Bounding-box", which will be used as part of
        #   experience for "Focus" branch.
        cur_bbox = self._focus_bbox.copy()
        # Old "Previous Segmentation", which will be used as part
        #   of sample for both "Segment" and "Focus" branch.
        SEG_prev = self._SEG_prev.copy()

        # The "Focus Bbox" holder, coz we need even "fake" focus bbox generated
        #   when in "Segment" phrase.
        focus_bbox = np.zeros_like(self._focus_bbox)
        # The "Next Segmentation", which will be used as part
        #   of experience for "Focus" branch.
        SEG_cur = np.zeros_like(self._SEG_prev)
        # The fake "Next position information", which will be used in both
        #   "Segment" and "Focus" branch.
        next_posinfo = np.zeros_like(position_info)

        # -------------------------------- Core Part -----------------------------
        # Different procedure according to phrase. "Train" phrase.
        if self._phrase == 'Train':
            # Execute train function.
            segmentation, COMP_res, action = op_func((self._image.copy(),
                                                      self._SEG_prev.copy(),
                                                      position_info,
                                                      SEG_stage,
                                                      self._focus_bbox.copy(),
                                                      self._COMP_result.copy()))
            # Execution to get the next state (bbox), reward, terminal flag.
            RelDirc_his = self._RelDir_prev.copy() if SEG_stage else self._RelDir_prev
            dst_bbox, reward, over, info = self._exec_4ActRew(action, self._focus_bbox.copy(), RelDirc_his)

            # Bound the initial segmentation region as initial "Focus Bbox".
            #   Just for precise and fast "Focus".
            if initFB_optimize and not self._FB_opted and SEG_stage and not ((segmentation == 0).all()):
                init_region = np.where(segmentation != 0)
                init_y1 = min(init_region[0]) / image_height
                init_x1 = min(init_region[1]) / image_width
                init_y2 = max(init_region[0]) / image_height
                init_x2 = max(init_region[1]) / image_width
                self._focus_bbox = np.asarray([init_y1, init_x1, init_y2, init_x2])
                vis_bbox = self._focus_bbox.copy()  # re-assign for initial situation.
                self._FB_opted = True  # means optimized

            # Real "Segment", fake "Focus".
            if SEG_stage:
                # To generate the next state for "Focus" if enable "Sample Share".
                if sample_share:
                    #   1) the destination bbox in time for "Segment" phrase.
                    #   2) the current segmentation result for "DQN" training.
                    #   3) the fake next position information for both two branch.
                    focus_bbox = dst_bbox.copy()
                    next_posinfo = gen_position_info(focus_bbox)
                    SEG_cur, _2, _3 = op_func((self._image.copy(),
                                               self._SEG_prev.copy(),
                                               next_posinfo,
                                               SEG_stage,
                                               focus_bbox,
                                               self._COMP_result.copy()))
                # Re-assign the "Segmentation" result and Complete info.
                self._SEG_prev = segmentation
                self._COMP_result = COMP_res
                reward = info = None  # fake, for conveniently coding.
                over = False  # "Segment" stage, not over.
            # Fake "Segment", real "Focus".
            else:
                # Re-assign the current "Focus Bbox".
                self._focus_bbox = np.asarray(dst_bbox)
                self._time_step += 1
                # Use the "Real Next Focus Bbox" when it's over. Meanwhile
                #   update the next position information.
                if over:
                    focus_bbox = self._focus_bbox.copy()
                    next_posinfo = gen_position_info(focus_bbox)

        # "Validate" or "Test".
        else:
            # "Segment" stage.
            if SEG_stage:
                # calculate first.
                segmentation, COMP_res = op_func((self._image.copy(),
                                                  self._SEG_prev.copy(),
                                                  position_info,
                                                  SEG_stage,
                                                  self._focus_bbox.copy(),
                                                  self._COMP_result.copy()))
                # bound initial "Focus Bbox", for rapidly and precisely processing.
                if initFB_optimize and not self._FB_opted and not ((segmentation == 0).all()):
                    init_region = np.where(segmentation != 0)
                    init_y1 = min(init_region[0]) / image_height
                    init_x1 = min(init_region[1]) / image_width
                    init_y2 = max(init_region[0]) / image_height
                    init_x2 = max(init_region[1]) / image_width
                    self._focus_bbox = np.asarray([init_y1, init_x1, init_y2, init_x2])
                    vis_bbox = self._focus_bbox.copy()  # re-assign for initial situation.
                    self._FB_opted = True  # means optimized
                # iteratively assign.
                self._SEG_prev = segmentation
                self._COMP_result = COMP_res
                action = reward = info = None   # fake, for conveniently coding.
                SEG_cur = segmentation.copy()   # real, used to compute metric in "Validate" phrase, and write result.
                over = False    # "Segment" stage, not over.
            # "Focus" stage.
            else:
                action = op_func((self._image.copy(),
                                  self._SEG_prev.copy(),
                                  position_info,
                                  SEG_stage,
                                  self._focus_bbox.copy()))
                dst_bbox, reward, over, info = self._exec_4ActRew(action, self._focus_bbox.copy(), self._RelDir_prev)
                self._focus_bbox = np.asarray(dst_bbox)
                self._time_step += 1
        # -------------------------------- Core Part -----------------------------

        # Record the process.
        if self._anim_recorder is not None:
            # fo_bbox = self._focus_bbox.copy() if (cur_bbox != self._focus_bbox).any() else None
            fo_bbox = self._focus_bbox.copy() if not SEG_stage else None
            self._anim_recorder.record((vis_bbox.copy(), fo_bbox, reward, self._SEG_prev.copy(), info))    # Need copy !!!

        # Reset the process flag. Only the over of "Focus" stage means real terminate.
        if not SEG_stage and over:
            self._finished = True

        # Return sample when in "Train" phrase.
        if self._phrase == 'Train':
            return (SEG_prev, cur_bbox, position_info), \
                   action, reward, over, \
                   (SEG_cur, focus_bbox, next_posinfo), \
                   info
        else:
            # Return terminal flag (most important), and other information.
            return over, (SEG_cur, reward, info)


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
        # # Get config.
        # conf_other = self._config['Others']
        # anim_type = conf_other.get('animation_type', 'gif')

        # Save the process into filesystem.
        if self._anim_recorder is not None:
            self._anim_recorder.show(mode=self._phrase, anim_type=anim_type)
        # Release the memory.
        gc.collect()
        return



    def _exec_4ActRew(self, action, bbox, RelDirc_his):
        r'''
            Execute the given action, and compute corresponding reward.

            ** Note that, it will not affect the current "Focus Bounding-box",
                but to return a new bbox-coordinates.
        '''

        # Check validity.
        if not isinstance(action, (int, np.int, np.int32, np.int64)):
            raise TypeError('The action must be an integer !!!')
        action = int(action)
        if action not in range(self._act_dim):
            raise ValueError('The action must be in range(0, {}) !!!'.format(self._act_dim))
        if not isinstance(bbox, np.ndarray) or bbox.ndim != 1 or bbox.shape[0] != 4:
            raise ValueError('The bbox must be a 4-elements numpy array '
                             'just like (y1, x1, y2, x2) !!!')

        # Get detailed config.
        conf_base = self._config['Base']
        clazz_dim = conf_base.get('classification_dimension')
        conf_dqn = self._config['DQN']
        anchor_scale = conf_dqn.get('anchors_scale', 2)
        reward_metric = conf_dqn.get('reward_metric', 'Dice-M')
        terminate_threshold = conf_dqn.get('terminal_threshold', 0.8)
        step_threshold = conf_dqn.get('step_threshold', 10)

        # Out-of-Boundary (focus out too much) error.
        OOB_err = False
        # Terminal flag.
        terminal = False

        # Execute the given action. Different procedure according to action quantity.
        if self._act_dim == 9:
            # "Restrict" mode.
            anchors = self.__anchors(bbox, anchor_scale, 9,
                                     arg=RelDirc_his[-1])   # the newest relative direction.
            # --> select children.
            if action <= 3:
                # push the relative direction. -- focus in.
                rel_dirc = self._RELATIVE_DIRECTION[action]     # coz just the same order.
                RelDirc_his.append(rel_dirc)
            # --> select parent.
            elif action == 4:
                # pop out the relative direction. -- focus out.
                RelDirc_his.pop()
                # check if it's the outermost level.
                if len(RelDirc_his) == 0:
                    OOB_err = True
            # --> select peers.
            elif action <= 7:
                # translate the current (newest) relative direction.  -- focus peer.
                cur_Rdirc = RelDirc_his.pop()
                # metaphysics formulation.
                CRD_idx = self._RELATIVE_DIRECTION.index(cur_Rdirc)
                NRD_idx = action - 4
                if NRD_idx - CRD_idx <= 0:
                    NRD_idx -= 1
                next_Rdirc = self._RELATIVE_DIRECTION[NRD_idx]
                # pop out and push new relative direction.
                RelDirc_his.append(next_Rdirc)
            # --> stop, terminal.
            else:
                terminal = True

        elif self._act_dim == 17:
            # "Whole Candidates" mode.
            anchors = self.__anchors(bbox, anchor_scale, 17)
            # Fake relative direction for "Out-of-Boundary" error check.
            fake_RD = self._RELATIVE_DIRECTION[0]
            # --> select children. -- focus in. push
            if action <= 3:
                RelDirc_his.append(fake_RD)
            # --> select parent. -- focus out. pop
            elif action <= 7:
                # check if it's the outermost level.
                if len(RelDirc_his) == 0:
                    OOB_err = True
                else:
                    RelDirc_his.pop()
            # --> select peers. -- do nothing ...
            elif action <= 15:
                pass
            # --> stop, terminal.
            else:
                terminal = True

        else:
            raise Exception('Unknown action dimension, there is no logic with respect to '
                            'current act_dim, please check your code !!!')

        # Compute the reward for given action. (Only in "Train" phrase)
        if self._phrase == 'Train':
            cal_rewards = self.__rewards(self._SEG_prev.copy(), self._label.copy(),
                                         category=clazz_dim, anchors=anchors,
                                         OOB_err=OOB_err, terminal=terminal,
                                         ter_thres=terminate_threshold,
                                         metric_type=reward_metric)
            if isinstance(cal_rewards, list):
                reward = cal_rewards[action]
            else:
                reward = cal_rewards
        else:
            reward = None

        # Judge whether game over or not.
        over, info = self.__game_over(anchors[action] if action != self._act_dim - 1 else None,
                                      OOB_err, terminal,
                                      time_step=self._time_step,
                                      step_thres=step_threshold)

        # Get the destination bbox.
        if not over:
            dst_bbox = anchors[action]
        else:
            dst_bbox = self._FAKE_BBOX

        # Return the 1)destination bbox, 2)reward, 3)over flag and 4)information.
        return dst_bbox, reward, over, info


    def __anchors(self, bbox, scale, adim, arg=None):
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


    def __rewards(self, pred, label, category, anchors, OOB_err, terminal, ter_thres, metric_type):
        r'''
            Calculate the rewards for each situation. Including:
                1) The remaining value of given anchors.
                2) The terminal moment.
                3) The Out-of-Boundary error.
        '''

        # Reward (punishment) for "Out-of-Boundary" error.
        if OOB_err:
            return 0.0

        # Reward for "Terminal" action.
        if terminal:
            if metric_type == 'Dice-M':
                v = eva.mean_DICE_metric(pred, label, category, ignore_BG=True)
            elif metric_type == 'Dice-P':
                v = eva.prop_DICE_metric(pred, label, category, ignore_BG=True)
            else:
                raise ValueError('Unknown metric type !!!')
            return 3.0 if v >= ter_thres else -3.0

        # Rewards for each "Anchor".
        cor_h, cor_w = label.shape
        rewards = []
        for bbox in anchors:
            # boundary.
            y1, x1, y2, x2 = bbox
            up = int(round(y1 * cor_h))
            left = int(round(x1 * cor_w))
            bottom = int(round(y2 * cor_h))
            right = int(round(x2 * cor_w))
            # calculate.
            region_pred = pred[up: bottom, left: right]
            region_lab = label[up: bottom, left: right]
            if metric_type == 'Dice-M':
                v = eva.mean_DICE_metric(region_pred, region_lab, category, ignore_BG=True)
            elif metric_type == 'Dice-P':
                v = eva.prop_DICE_metric(region_pred, region_lab, category, ignore_BG=True)
            else:
                raise ValueError('Unknown metric type !!!')
            # use the remaining value of "Dice" metric as reward.
            v = 1.0 - v
            # package.
            rewards.append(v)

        # Return all rewards for anchors.
        return rewards


    def __game_over(self, candidate, OOB_err, terminal, time_step, step_thres):
        r'''
            Judge whether current processing is over according to each flag.
        '''
        if OOB_err:
            return True, 'Out-of-Boundary !'
        if terminal:
            return True, 'Terminal !'
        if time_step >= step_thres:
            return True, 'Reach Threshold !'
        if candidate is not None:
            y1, x1, y2, x2 = candidate
            if (y2 - y1) == 0.0 or (x2 - x1) == 0.0:
                return True, 'Zero-scale Bbox !'
        return False, None


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

