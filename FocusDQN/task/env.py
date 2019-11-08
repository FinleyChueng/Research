import gc

import cv2
import tensorflow as tf
from keras.utils import to_categorical
# from skimage import morphology
from tensorflow.python import debug as tf_debug

import util.evaluation as eval
from core.env import *
from dataset.adapter.base import *



class FocusEnv:


    def __init__(self, config, data_adapter, tfnet_holders):


        # Configuration.
        self._config = config

        # Data adapter.
        self._adapter = data_adapter

        # Get tensorflow model inputs and outputs, respectively.
        self._tfnet_inputs, self._tfnet_outputs = tfnet_holders

        # The flag indicating whether is in "Train", "Validate" or "Test" phrase.
        self._phrase = 'Train'  # Default in "Train"


        # self._reset = False


        # | ---> Finish holders transferring !
        # |   ===> Inputs holder: dict_keys(['ORG/image', 'ORG/prev_result', 'ORG/position_info', 'ORG/Segment_Stage', 'ORG/Focus_Bbox'])
        # |   ===> Outputs holder: dict_keys(['ORG/DQN_output', 'TEST/SEG_output'])
        # |   ===> Losses holder: dict_keys(['TEST/GT_label', 'TEST/clazz_weights', 'TEST/prediction_actions', 'TEST/target_Q_values', 'TEST/EXP_priority', 'TEST/IS_weights', 'TEST/SEG_loss', 'TEST/DQN_loss', 'TEST/NET_loss', 'TAR/image', 'TAR/prev_result', 'TAR/position_info', 'TAR/Segment_Stage', 'TAR/Focus_Bbox', 'TAR/DQN_output'])
        # |   ===> Summary holder: dict_keys(['TEST/Reward', 'TEST/DICE', 'TEST/BRATS_metric', 'TEST/MergeSummary'])
        # |   ===> Visual holder: dict_keys([])

        # Get detailed parameters.
        conf_base = self._config['Base']
        input_shape = conf_base.get('input_shape')
        image_height, image_width = input_shape[1:3]

        conf_cus = self._config['Custom']
        pos_method = conf_cus.get('position_info', 'map')

        # Some input data placeholder.
        self._image = None  # Current processing image.
        self._label = None  # Label for current image.
        self._SEG_stage = None  # Flag indicating whether is "Segmentation" stage or not.
        self._focus_bbox = np.zeros([4])    # Focus bounding-box.
        # The result of "Segmentation" stage, will used as part of input.
        self._SEG_prev = np.zeros([image_height, image_width])
        # The position information. Declaration according to config.
        if pos_method == 'map':
            self._position_info = np.zeros([image_height, image_width])
        elif pos_method == 'coord':
            self._position_info = np.zeros([4])
        elif pos_method == 'sight':
            self._position_info = np.zeros([4])
        else:
            raise ValueError('Unknown position information fusion method !!!')


        return


    def switch_phrase(self, p):
        r'''
            Switch to target phrase.
        '''
        if p not in ['Train', 'Validate', 'Test']:
            raise ValueError('Unknown phrase value !!!')
        self._phrase = p
        return


    def reset(self, arg):
        r'''
            Reset the environment. Mainly to reset the related parameters,
                and switch to next image-label pair.
        '''

        # # Check the validity of calling.
        # if self._reset:
        #     raise Exception('Invalid process state: the processing of current image is not yet finished, '
        #                     'can not reset the environment !!!')


        # | ---> Finish holders transferring !
        # |   ===> Inputs holder: dict_keys(['ORG/image', 'ORG/prev_result', 'ORG/position_info', 'ORG/Segment_Stage', 'ORG/Focus_Bbox'])
        # |   ===> Outputs holder: dict_keys(['ORG/DQN_output', 'TEST/SEG_output'])
        # |   ===> Losses holder: dict_keys(['TEST/GT_label', 'TEST/clazz_weights', 'TEST/prediction_actions', 'TEST/target_Q_values', 'TEST/EXP_priority', 'TEST/IS_weights', 'TEST/SEG_loss', 'TEST/DQN_loss', 'TEST/NET_loss', 'TAR/image', 'TAR/prev_result', 'TAR/position_info', 'TAR/Segment_Stage', 'TAR/Focus_Bbox', 'TAR/DQN_output'])
        # |   ===> Summary holder: dict_keys(['TEST/Reward', 'TEST/DICE', 'TEST/BRATS_metric', 'TEST/MergeSummary'])
        # |   ===> Visual holder: dict_keys([])


        # Reset the previous segmentation result. Generate it in first time.
        self._prev_result[:] = 0.

        # Reset the segmentation result.
        self._segmentation[:] = 0.      # [w, h]


        # Reset the previous metric.
        self._prev_lab[:] = 0.

        # Reset the terminal flag.
        self._terminal = False

        # Generate initial states.
        initial_state = self._gen_state()

        # Generate the fake state if it's None.
        if self._fake_state is None:
            self._fake_state = (np.zeros_like(self._feature_maps),)


        import tensorflow as tf
        tf.image.resize_nearest_neighbor




        # Set the reset flag to True. Remember to set False when terminate
        #   current processing image.
        self._reset = True

        # Reset the execution steps of current image.
        self._step = 0

        # Additional info.
        additional_info = self._pack_additional_info('Reset ~!', arg=(None,))

        # Record the process.
        if self._anim_recorder is not None:
            # Reload the animation recorder.
            self._anim_recorder.reload((not self._train_mode,))
            # self._anim_recorder.reload((False,))
            # self._anim_recorder.reload((True,))

        # print('Maze Environment ------ Reset the environment !!!')

        # Finish the reset and get the initial state.
        return initial_state, additional_info


    def __switch2_next_imglabel(self, is_training, test_flag):
        r'''
            Switch to the next processing image.

        :param is_training: Indicating whether it is training mode or not.
        :param test_flag: Indicating whether it's real testing phrase.
        '''

        # Check the validity of parameters.
        if not isinstance(is_training, bool):
            raise TypeError('The is_training must be of @Type{bool} !!!')
        if not isinstance(test_flag, bool):
            raise TypeError('The test_flag must be of @Type{bool} !!!')

        # Fake class weights.
        clazz_weights = None

        # Get the next image-label pair according to the mode.
        if is_training and not test_flag:
            # Get the next train sample pair.
            img, label, finish_CMHA, self._cur_Cweights = self._adapter.next_image_pair(mode='Train', batch_size=1)
        elif not is_training and not test_flag:
            # Get the next validate image-label pair.
            img, label, finish_CMHA = self._adapter.next_image_pair(mode='Validate', batch_size=1)
        elif not is_training and test_flag:
            # Get the next test image-label pair.
            img, label, finish_CMHA = self._adapter.next_image_pair(mode='Test', batch_size=1)
        else:
            raise Exception('The training and test flag can not both be True !!!')

        # Get the original size.
        org_size = img.shape[:-1]     # (width, height)
        # Reshape the original size to target size.
        if org_size != self._proc_size:
            img = cv2.resize(img, self._proc_size)
            # Resize only when label is not None.
            if label is not None:
                label = cv2.resize(label, self._proc_size)
        else:
            pass
        # Record the four modalities and ground truth. The four modalities is
        #   lately used as the input to network.
        self._4mod = img        # [width, height, modalities]
        # Assign the label.
        self._label = label     # [width, height, cls_dim]

        # print('Maze Environment ------ Switch to next image-label pair !!!')

        # Finish switching to next processing image. And return the
        #   "Finish current MHA" flag.
        return finish_CMHA







# ------------------------------------------------------------------------------------------------------
# This environment is used to training the "DQN" agent. Moreover is the real processing environment.
# ------------------------------------------------------------------------------------------------------
class CalDQNEnv:

    def __init__(self,
                 data_adapter,
                 feats_ext_util,
                 segment_util,
                 cls_dim,
                 proc_imgsize,
                 use_all_masks,
                 rev_func='x',
                 anim_recorder=None):

        # Normal initialization.
        self._proc_size = proc_imgsize

        # The valid number of actions of "Maze" Environment.
        self._acts_num = 8

        # The total class number of the "Segmentation" task.
        self._cls_num = cls_dim

        # The dimension of class-based mask waits for generation.
        if use_all_masks:
            self._mask_num = cls_dim
        else:
            self._mask_num = cls_dim - 1

        # Check the tensorflow utility first. Which is actually the Tensorflow.Tensor.
        self.__verify_tfutil(FE_util=feats_ext_util, segment_util=segment_util)
        # The feature maps extract utility holder.
        self._feats_ext_util = feats_ext_util
        # The segmentation utility holder.
        self._segment_util = segment_util

        # Check the adapter, and then assign it if valid.
        self.__verify_data(data_adapter)
        self._adapter = data_adapter
        # Animation recorder.
        self._anim_recorder = anim_recorder

        # Training mode flag.
        self._train_mode = None
        # Testing mode flag.
        self._test_flag = None

        # Finish resetting environment.
        self._reset = False

        # The four modalities image. Shape: [width, height, modalities]
        self._4mod = None
        # The ground truth of the original image. Only one. Shape: [width, height, cls_dim]
        self._label = None

        # The current class weights.
        self._cur_Cweights = np.ones(self._cls_num, dtype=np.float32)
        # Whether finish current MHA flag.
        self._last_ofCMHA = None

        # The "Operation Conv Kernel" instance.
        self._opt_kernel = OptKernel()

        # The previous segmentation result (Binary).
        self._prev_result = np.zeros((self._proc_size[0], self._proc_size[1], self._mask_num), dtype=np.float32)
        # self._prev_result = np.zeros(self._proc_size, dtype=np.float32)
        # self._prev_result = None

        # The holder of segmentation result of "Maze" environment.
        #   Moreover, it is real segmentation result.
        self._segmentation = np.zeros(self._proc_size, dtype=np.float32)

        # The terminal flag.
        self._terminal = False

        # The feature maps of current processing image provided by FEN.
        self._feature_maps = None
        # The detailed feature maps each level of provided by FEN.
        self._FEN_lvl_feats = None

        # The length of the history.
        self._history_len = 8
        # The oldest feature maps of the length-fixed history.
        self._oldest_feats = None
        # The iterative hidden state value of "Critic" RNN cell.
        self._CCrnn_hstates = None
        # The features history. (Record the past 8 elements).
        self._feats_history = collections.deque()
        # The one-hot action history. (Record the past 8 elements).
        self._ohact_history = collections.deque()

        # Fake state used to return when the processing of current image
        #   is terminal. This is just for the convenience of implementation
        #   of training framework.
        # The Fake state is all zeros.
        self._fake_state = None
        # self._fake_state = (np.zeros((self._maze_size[0], self._maze_size[1])),  # sight mask
        #                     np.zeros(4),  # respective-field bbox
        #                     np.zeros(2)  # player position
        #                     )

        # # The previous metric (that is, the past step metric).
        # self._prev_metric = None

        # The previous label (that is, the past step label).
        self._prev_lab = np.zeros((self._proc_size[0], self._proc_size[1], self._mask_num), dtype=np.float32)

        # The total execution steps of current image.
        #   Which is mainly used to control the terminal condition.
        self._step = None


    @property
    def acts_dim(self):
        r'''
            Get the action number of "Cal" environment.
        '''
        return self._acts_num

    @property
    def mask_dim(self):
        r'''
            Get the mask (previous segmentation result) number of "Cal" environment.
        '''
        return self._mask_num



    def reset(self, arg, is_training=True, test_flag=False):
        r'''
            Reset the "Cal" environment and get the initial state.
                Mainly to reset each holder.

        Parameter:
            arg: Additional arguments. Including:
                1) The feature maps provided by DQN agent.

        Return:
            The initial state.
        '''

        # Check the validity of flag.
        if is_training and test_flag:
            raise Exception('The training and testing flag can not both be True !!!')

        # Check the reset flag situation.
        if self._reset:
            raise Exception('The processing of current image is not yet finished, '
                            'can not reset the environment !!!')

        # Initialize the flag.
        self._train_mode = is_training
        self._test_flag = test_flag

        # Reset the previous segmentation result. Generate it in first time.
        self._prev_result[:] = 0.

        # Reset the segmentation result.
        self._segmentation[:] = 0.      # [w, h]

        # Reset (Clear) the FEN feats and one-hot action history.
        self._feats_history.clear()
        self._ohact_history.clear()

        # # Reset the previous metric.
        # self._prev_metric = 0.

        # Reset the previous metric.
        self._prev_lab[:] = 0.

        # Reset the terminal flag.
        self._terminal = False

        # Generate initial states.
        initial_state = self._gen_state()

        # Generate the fake state if it's None.
        if self._fake_state is None:
            self._fake_state = (np.zeros_like(self._feature_maps),)

        # Generate the initial oldest feature maps of FEN.
        self._oldest_feats = np.zeros_like(self._feature_maps)

        # Generate the initial hidden state value for "Critic" RNN cells.
        #   Note that, they're all zeros.
        self._CCrnn_hstates = (np.zeros([1, self._mask_num * self._acts_num], dtype=np.float32),
                               np.zeros([1, 512], dtype=np.float32),
                               [np.zeros_like(self._feature_maps)]
                               )

        # # Reset the previous segmentation result. Generate it in first time.
        # if self._prev_result is not None:
        #     self._prev_result[:] = 0.
        # else:
        #     fw, fh = self._feature_maps.shape[:-1]
        #     self._prev_result = np.zeros((fw, fh), dtype=np.float32)

        # Set the reset flag to True. Remember to set False when terminate
        #   current processing image.
        self._reset = True

        # Reset the execution steps of current image.
        self._step = 0

        # Additional info.
        additional_info = self._pack_additional_info('Reset ~!', arg=(None,))

        # Record the process.
        if self._anim_recorder is not None:
            # Reload the animation recorder.
            self._anim_recorder.reload((not self._train_mode,))
            # self._anim_recorder.reload((False,))
            # self._anim_recorder.reload((True,))

        # print('Maze Environment ------ Reset the environment !!!')

        # Finish the reset and get the initial state.
        return initial_state, additional_info


    def step(self, action, arg=None):
        r'''
            Execute the specific "Maze Game" action, which is predicted (selected)
                by the DQN agent. What's more, return the next state (observation)
                of the "Maze" environment.

        Parameters:
            action: The action index received from the DQN agent.
            arg: The additional arguments.

        Return:
            The tuple of (observation, reward, terminal, info).
            -------------------------------------------------
            state: Next state of the environment.
            reward: The reward of current action.
            terminal: Flag that indicates whether current image is finished.
            info: Extra information.(Optional, default is type.None)
        '''

        # Check whether switched the image or not.
        if not self._reset:
            raise Exception('You must call the @Method{reset} first '
                            'before calling this @Method{step} !!!')

        # Check the executability of current image-label pair.
        if self._terminal:
            raise Exception('The process of current image-label pair is finished, it '
                            'can not be @Method{step} any more !!! '
                            'You should call the @Method{reset} to reset the '
                            'environment to get the next image.')

        # # Check the validity of actions firstly.
        # if not isinstance(action, (int, np.int64)):
        #     raise TypeError('The action should be of @Type{Python.int} or @Type{Numpy.int64} !!!')
        # if action not in range(self._acts_num):
        #     raise Exception('The value of action should be in range(0, self._acts_num) !!!')

        # Check the validity of actions firstly.
        if not isinstance(action, (list, np.ndarray)) or len(action) != self._mask_num:
            raise TypeError('The action should be of mask-dim @Type{Python.list} or @Type{Numpy.ndarray} !!!')
        min_actval, max_actval = min(action), max(action)
        if not isinstance(min_actval, (int, np.int64)):
            raise TypeError('The min_actval should be of @Type{Python.int} or @Type{Numpy.int64} !!!')
        if min_actval not in range(self._acts_num):
            raise Exception('The value of min_actval should be in range(0, self._acts_num) !!!')
        if not isinstance(max_actval, (int, np.int64)):
            raise TypeError('The max_actval should be of @Type{Python.int} or @Type{Numpy.int64} !!!')
        if max_actval not in range(self._acts_num):
            raise Exception('The value of max_actval should be in range(0, self._acts_num) !!!')

        # Execute the actions.
        wrong_opt, procLab_list, vis_data, test_prev = self._execute_action(action=action, arg=arg)
        # wrong_opt, optConv_list, vis_data = self._execute_action(action=action, arg=arg)

        # Only calculate the reward when the label is not NoneType.
        if self._label is not None:
            # Compute the reward for given action.
            reward = self._compute_reward(action=action, arg=(wrong_opt, procLab_list))
        else:
            # Simply return NoneType.
            reward = None

        # Judge the whether it is terminal or not.
        terminal = self._terminal_judge(action=action, arg=wrong_opt)

        # Write the segmentation to the file system in "MHA" form when "Real" testing.
        if not self._train_mode and self._test_flag:
            self._adapter.write_result(result=self._segmentation.copy(),
                                       name=str(arg))

        # Generate the next state or NoneType according to the terminal flag.
        if terminal:
            # Return the fake state (Coz still have to pass through the
            #   tensorflow model).
            state = self._fake_state
            # The extra information.
            info = 'The processing of current image is over.\n ' \
                   'action - {} reward - {}'.format(
                       action, reward
                   )
            # Reset the environment flags.
            self.__reset_all_flags()
        else:
            # Generate next state.
            state = self._gen_state()
            # The extra information.
            info = 'action - {}, reward - {}.'.format(
                action, reward
            )

        # Package some additional information if needed.
        info = self._pack_additional_info(info, arg=(procLab_list,))

        # Assign the terminal flag to holder.
        self._terminal = terminal

        # Record the process.
        if self._anim_recorder is not None:
            # Generate the temp label for animation according to the raw label.
            if self._label is not None:
                # Use the raw label.
                anim_label = np.argmax(self._label, axis=-1)    # [w, h]
            else:
                # All black is OK.
                anim_label = np.zeros(self._proc_size)
            # Record the process to local file system.
            self._anim_recorder.record((self._4mod.copy(),
                                        anim_label,
                                        vis_data,
                                        self._segmentation.copy(),
                                        np.argmax(test_prev, axis=-1)
                                        ),
                                       arg=terminal)  # Need copy !!!

        # Release memory.
        gc.collect()

        # Return the tuple of (observation, reward, terminal, info).
        return state, reward, terminal, info


    def render(self):
        r'''
            Render method. That is, record the procedure into file system.

        :return:
        '''

        # Check whether can render or not.
        if self._terminal:
            # Real terminal.
            pass
        else:
            # Not yet finished.
            raise Exception('The processing of current image is not yet finish, '
                            'can not @Method{Render} !!!')

        # Save the process into filesystem.
        if self._anim_recorder is not None:
            self._anim_recorder.show(train_mode=self._train_mode)

        # Release the memory.
        gc.collect()



    def _gen_state(self):
        r'''
            Generate the state for next time step.

        :return: The state for next time step.
        '''

        # Firstly switch to the next image-label batch.
        self._last_ofCMHA = self.__switch2_next_imglabel(is_training=self._train_mode,
                                                         test_flag=self._test_flag)

        # Extracts the deep features of raw image by pass through the
        #   raw image into the FEN.
        self._feature_maps, self._FEN_lvl_feats = self.__extract_feats(image=self._4mod)

        # Package the features of FEN into the history.
        if len(self._feats_history) >= self._history_len:
            self._oldest_feats = self._feats_history.popleft()
        self._feats_history.append(self._feature_maps.copy())

        # The state including: FEN output feature maps.
        state = (self._feature_maps.copy(),  # FEN feature maps
                 )

        # Finish generation of state.
        return state


    def __switch2_next_imglabel(self, is_training, test_flag):
        r'''
            Switch to the next processing image.

        :param is_training: Indicating whether it is training mode or not.
        :param test_flag: Indicating whether it's real testing phrase.
        '''

        # Check the validity of parameters.
        if not isinstance(is_training, bool):
            raise TypeError('The is_training must be of @Type{bool} !!!')
        if not isinstance(test_flag, bool):
            raise TypeError('The test_flag must be of @Type{bool} !!!')

        # Fake class weights.
        clazz_weights = None

        # Get the next image-label pair according to the mode.
        if is_training and not test_flag:
            # Get the next train sample pair.
            img, label, finish_CMHA, self._cur_Cweights = self._adapter.next_image_pair(mode='Train', batch_size=1)
        elif not is_training and not test_flag:
            # Get the next validate image-label pair.
            img, label, finish_CMHA = self._adapter.next_image_pair(mode='Validate', batch_size=1)
        elif not is_training and test_flag:
            # Get the next test image-label pair.
            img, label, finish_CMHA = self._adapter.next_image_pair(mode='Test', batch_size=1)
        else:
            raise Exception('The training and test flag can not both be True !!!')

        # Get the original size.
        org_size = img.shape[:-1]     # (width, height)
        # Reshape the original size to target size.
        if org_size != self._proc_size:
            img = cv2.resize(img, self._proc_size)
            # Resize only when label is not None.
            if label is not None:
                label = cv2.resize(label, self._proc_size)
        else:
            pass
        # Record the four modalities and ground truth. The four modalities is
        #   lately used as the input to network.
        self._4mod = img        # [width, height, modalities]
        # Assign the label.
        self._label = label     # [width, height, cls_dim]

        # print('Maze Environment ------ Switch to next image-label pair !!!')

        # Finish switching to next processing image. And return the
        #   "Finish current MHA" flag.
        return finish_CMHA


    def __extract_feats(self, image):
        r'''
            Extract the feature maps from image for lately usage.

        :param image: The image that waiting for extracting features.
        '''

        # Check the validity.
        if not isinstance(image, np.ndarray) or image.ndim != 3:
            raise TypeError('The array should be 3-D numpy.ndarray !!!')

        # Get the input and output holders, respectively.
        tf_sess, TP_flag, FEN_image, FEN_out, FEN_feats_dict = self._feats_ext_util

        # Use the tensorflow holder ("Feature Extract" Network) to extract features of current image.
        fetches = [FEN_out]
        fetches.extend(FEN_feats_dict)
        pack_outs = tf_sess.run(fetches, feed_dict={
            TP_flag: False,
            FEN_image: [image],  # [b*t, mw, mh, c]
        })  # [2, b, w, h, c]
        fen_out = pack_outs[0][0]  # [w, h, c]
        fen_lvl_feats = [lfeats[0] for lfeats in pack_outs[1:]]    # [l, w, h, c]

        # Finish. And return the FEN out and features of each level.
        return fen_out, fen_lvl_feats


    def _pack_additional_info(self, info, arg):
        r'''
            Package some additional information for each time step. Including:
                1) Current processing image.
                2) Current label.

        :param info:
        :return:
        '''

        # When in train phrase. The additional information will includes:
        #   1) the current processing image. (4-modality image)
        #   2) the current label. (clazz-dim array)
        #   3) the previous result. (previous segmentation)
        #   4) the operation-conv kernel used in last time-step.
        #   5) the GT operation-conv kernel (label) for current time-step.
        if self._train_mode:
            # Get the previous result and operation-conv kernel.
            proc_lab, = arg
            if proc_lab is None:
                proc_lab = np.zeros_like(self._prev_result, dtype=np.float32)
                # opt_conv = self._opt_kernel.nope_conv

            # Use it.
            pinfo = (self._4mod.copy(), self._label.copy(), proc_lab, self._cur_Cweights.copy())

        # Directly use raw info.
        else:
            # Compute Dice if label is given.
            if self._label is not None:
                # Calculate the truth DICE.
                cls_DICE = []
                segmentation = to_categorical(self._segmentation, num_classes=self._cls_num)
                for cls in range(self._cls_num):
                    c_dice = self.__cal_DICE(predict=segmentation[:, :, cls],
                                             GT=self._label[:, :, cls])
                    cls_DICE.append(c_dice)
                # Compute the BRATS metric.
                cls_lab = np.argmax(self._label, axis=-1)
                dataR1, labelR1 = eval.Region1(data=self._segmentation.copy(), label=cls_lab)
                dataR2, labelR2 = eval.Region2(data=self._segmentation.copy(), label=cls_lab)
                dataR3, labelR3 = eval.Region3(data=self._segmentation.copy(), label=cls_lab)
                BRATS_m1 = eval.DICE_coef(dataR1, labelR1)
                BRATS_m2 = eval.DICE_coef(dataR2, labelR2)
                BRATS_m3 = eval.DICE_coef(dataR3, labelR3)

                # Package the metric
                pinfo = (cls_DICE, [BRATS_m1, BRATS_m2, BRATS_m3])

            # Return the raw information.
            else:
                pinfo = info
            # pinfo = info

        # Finish processing.
        return pinfo


    def _execute_action(self, action, arg):
        r'''
            Execute action specified by DQN agent for current process image.
            --------------------------------
            Action list:
                0: Left Conv
                1: Right Conv
                2: Up Conv
                3: Bottom Conv
                4: Dilate Conv
                5: Erode Conv
                6: Nope Conv
                7: No-attention Conv

        :param action: The action predicted by DQN.
        :param arg: Additional argument.
        :return:
        '''

        # Wrong situation flag.
        wrong = False

        # Iteratively select the operation conv-kernel to each class.
        procLab_list = []
        for cls, act in enumerate(action):
            # # Select different conv-kernel according to the action.
            # if act == 0:
            #     # Left-conv.
            #     conv_kernel = self._opt_kernel.left_conv
            # elif act == 1:
            #     # Right-conv.
            #     conv_kernel = self._opt_kernel.right_conv
            # elif act == 2:
            #     # Up-conv.
            #     conv_kernel = self._opt_kernel.up_conv
            # elif act == 3:
            #     # Bottom-conv.
            #     conv_kernel = self._opt_kernel.bottom_conv
            # elif act == 4:
            #     # Dilate-conv.
            #     conv_kernel = self._opt_kernel.dilate_conv
            # elif act == 5:
            #     # Erode-conv.
            #     conv_kernel = self._opt_kernel.erode_conv
            # elif act == 6:
            #     # nope-conv.
            #     conv_kernel = self._opt_kernel.nope_conv
            # elif act == 7:
            #     # noatt-conv
            #     conv_kernel = self._opt_kernel.noatt_conv
            # else:
            #     raise Exception('Unknown action !!!')

            # Add the processed label into list.
            proc_lab = self.__apply_2prev_label(
                act=act,
                prev_lab=self._prev_result[:, :, cls]
            )
            procLab_list.append(proc_lab)
        # Transpose to the right dimension order.
        procLab_list = np.asarray(procLab_list).transpose([1, 2, 0])

        # Apply the conv-kernel to the "UN Segmentation" branch.
        #   What's more, get the visualization data.
        vis_data, test_prev = self.__draw_segmentation(array=self._segmentation,
                                                       proc_lab=procLab_list,
                                            prev_res=self._prev_result,
                                            reset_prev=True,
                                            )

        # Package the current action into the history in the one-hot form.
        if len(self._ohact_history) >= self._history_len:
            self._ohact_history.popleft()
        one_hot_action = to_categorical(action, num_classes=self._acts_num)
        # one_hot_action = np.zeros(self._acts_num, dtype=np.float32)
        # one_hot_action[action] = 1.
        self._ohact_history.append(one_hot_action)

        # Increase the execution steps of current image.
        self._step += 1

        # Finish the execution of action, and return the wrong flag.
        #   What's more, return the visualization data.
        return wrong, procLab_list, vis_data, test_prev


    def __draw_segmentation(self, array, proc_lab, prev_res, reset_prev):
        r'''
            Draw the current segmentation result into the result holder.
                Which actually use the tensorflow model to predict
                the segmentation result.

        :param array: The array that used to record the segmentation result.
        '''

        # Check the validity.
        if not isinstance(array, np.ndarray) or array.ndim != 2:
            raise TypeError('The array should be 2-D numpy.ndarray !!!')

        # Get the FEN output, UN select mask and UN output, respectively.
        tf_sess, \
            TP_flag, \
            FEN_feats_dict, \
            FEN_outs, \
            UN_select_mask, \
            UN_upfet, \
            UN_result_out = self._segment_util

        # Use the tensorflow holder ("Up-sample" Network) to generate the segmentation,
        #   which is actually a classification probability map.
        feed_dict = {
            TP_flag: False,
            FEN_outs: [self._feature_maps]     # [b*t, mw, mh, c]
        }
        # Add processed label.
        feed_dict[UN_select_mask] = [proc_lab]  # [b*t, mw, mh]
        # Add the feats of each level.
        for FEN_1vl_ftensor, FEN_lvl_feats in zip(FEN_feats_dict, self._FEN_lvl_feats):
            if FEN_1vl_ftensor is FEN_outs:
                continue    # Can not repeatly add the same holder.
            feed_dict[FEN_1vl_ftensor] = [FEN_lvl_feats]
        un_pack_outs = tf_sess.run([UN_upfet, UN_result_out], feed_dict=feed_dict)
        # Get elements, respectively.
        upfets_val = un_pack_outs[0][0]         # [w, h, c]
        cls_prob_map = un_pack_outs[1][0, 0]    # [w, h, cls_dim]

        # Select the predicted class with the max probability.
        array[:] = np.argmax(cls_prob_map, axis=-1)      # [w, h]

        # Assign value for visualization.
        vis_array = upfets_val

        # Test prev.
        pad_dim = self._cls_num - self._mask_num
        if pad_dim != 0:
            test_prev = np.concatenate((np.zeros((self._proc_size[0], self._proc_size[1], 1), dtype=np.float32),
                                        prev_res + 1),
                                       axis=-1)
        else:
            test_prev = prev_res.copy()
        # test_prev = prev_res.copy()
        # test_prev = proc_lab.copy()

        # Iteratively set the previous segmentation result (Binary).
        if reset_prev and prev_res is not None:
            cls_st = self._cls_num - self._mask_num
            cls_end = self._cls_num
            prev_res[:] = to_categorical(array, num_classes=self._cls_num)[:, :, cls_st: cls_end]
            # prev_res[:] = to_categorical(array, num_classes=self._cls_num)
            # prev_res[:] = np.asarray(array > 0, dtype=np.float32)

        # Finish. And return the mediate result.
        return vis_array, test_prev
        # return vis_array


    def _compute_reward(self, action, arg):
        r'''
            Compute the reward of input action predicted by DQN agent.

        :param action: The action predicted by DQN.
        :param arg: Additional argument. Here is the external argument from
            @Method{step} and the wrong operation flag.
        :return:
        '''

        # Get the useful information from the additional arguments first.
        wrong_opt, procLab_list = arg

        # Negative reward if wrong operation.
        if wrong_opt:
            reward = -1.

        # Compute reward for each action.
        else:
            # Class-based reward list.
            Crewards = []
            # Iteratively compute the each class.
            for cls in range(self._cls_num - self._mask_num, self._cls_num):
                # Get the clazz-related label.
                cur_lab = self._label[:, :, cls]
                # Compute the DICE of operation.
                proc_lab = self.__apply_2prev_label(
                    act=action[cls + self._mask_num - self._cls_num],
                    prev_lab=self._prev_lab[:, :, cls + self._mask_num - self._cls_num]
                )
                # proc_lab = procLab_list[:, :, cls + self._mask_num - self._cls_num]
                opt_DICE = self.__cal_DICE(predict=proc_lab, GT=cur_lab)
                # Compute the DICE of non-opt.
                diff_lab = self.__apply_2prev_label(
                    act=6,
                    prev_lab=self._prev_lab[:, :, cls + self._mask_num - self._cls_num]
                )
                non_DICE = self.__cal_DICE(predict=diff_lab, GT=cur_lab)
                # Append difference into list.
                er = opt_DICE - non_DICE
                Crewards.append(er)

            # Translate the list to numpy array.
            Crewards = np.asarray(Crewards, dtype=np.float32)
            # Re-assign the reward value.
            Crewards = np.maximum(Crewards, 0.)
            # Crewards[np.argmax(Crewards)] = 1.

            # The external reward is actually the class-based reward.
            reward = Crewards

        # # Normalize
        # reward /= 155.

        # Iteratively assign the previous label.
        self._prev_lab[:] = self._label[:, :, self._cls_num - self._mask_num: self._cls_num]

        # Return the reward.
        return reward


    def __apply_2prev_label(self, act, prev_lab):

        # Operation distance.
        margin = 1  # 16
        # The position coordinate.
        xlist, ylist = np.where(prev_lab == 1)
        # lab after processing.
        proc_lab = np.zeros_like(prev_lab)

        # Select different conv-kernel according to the action.
        if act == 0:
            # Left-conv.
            pxs = xlist - margin
            pxs = np.maximum(0, pxs)
            pys = ylist
            proc_lab[pxs, pys] = 1
        elif act == 1:
            # Right-conv.
            pxs = xlist + margin
            pxs = np.minimum(self._proc_size[0]-1, pxs)
            pys = ylist
            proc_lab[pxs, pys] = 1
        elif act == 2:
            # Up-conv.
            pxs = xlist
            pys = ylist - margin
            pys = np.maximum(0, pys)
            proc_lab[pxs, pys] = 1
        elif act == 3:
            # Bottom-conv.
            pxs = xlist
            pys = ylist + margin
            pys = np.minimum(self._proc_size[1]-1, pys)
            proc_lab[pxs, pys] = 1
        elif act == 4:
            # Dilate-conv.
            # proc_lab = morphology.dilation(prev_lab, selem=morphology.disk(margin))
            pass
        elif act == 5:
            # Erode-conv.
            # proc_lab = morphology.erosion(prev_lab, selem=morphology.disk(margin))
            pass
        elif act == 6:
            # nope-conv.
            proc_lab[:] = prev_lab
        elif act == 7:
            # noatt-conv
            pass
        else:
            raise Exception('Unknown action !!!')

        # Finish. And then return the processed prev label.
        return proc_lab


    def __cal_DICE(self, predict, GT):
        r'''
            Calculate the Dice metric of the prediction and Ground Truth.

        :param predict:
        :param GT:
        :return:
        '''

        # TP, FP, FN.
        _tp = eval.true_positive(predict=predict, grand_truth=GT)
        _fp = eval.false_positive(predict=predict, grand_truth=GT)
        _fn = eval.false_negative(predict=predict, grand_truth=GT)

        # Denominator
        denominator = _fp + 2 * _tp + _fn

        # Numerator
        numerator = 2 * _tp

        # Compute the Dice value.
        if denominator == 0:
            DICE = 1.
        else:
            DICE = numerator / denominator      # 2 * TP / (FP + 2 * TP + FN)

        # Finish.
        return DICE


    def _terminal_judge(self, action, arg):
        r'''
            Judge whether the processing of current image is finished.

        :param action: The action predicted by DQN.
        :param arg: Additional argument. Here is the wrong operation flag.
        :return:
        '''

        # Initial flag.
        terminal = False

        # Get wrong operation flag.
        wrong_opt = arg

        # Terminated when executed wrong operation.
        if wrong_opt:
            terminal = True

        # Terminated when finish the segmentation of current MHA.
        if self._last_ofCMHA:
            terminal = True

        # Finished and return the flag.
        return terminal


    def __reset_all_flags(self):
        r'''
            Reset the flags used in this class. Call this method when finish
                the processing of current image.

        :return:
        '''

        # Set the reset flag to false.
        self._reset = False

        # Finished.
        return


    def __verify_data(self, adapter):
        r'''
            Verify the data supported by adapter.

        :param adapter:
        :return:
        '''

        # Firstly check the type of adapter.
        if not isinstance(adapter, Adapter):
            raise TypeError('The adapter should be @Type{Adapter} !!!')

        # Get a train image-label pair to verify.
        img, label, finish_CMHA, clazz_weights = adapter.next_image_pair(mode='Train', batch_size=1)

        # Check the type and dimension of the data.
        if not isinstance(img, np.ndarray) or not isinstance(label, np.ndarray):
            raise TypeError('The type of image and label both should be'
                            '@Type{numpy.ndarray} !!!')

        # Now we only support single image processing.
        if img.ndim != 3 or label.ndim != 3:    # [width, height, modalities], [width, height, cls]
            raise Exception('The dimension of the image and label both should'
                            'be 3 !!! img: {}, label: {}\n'
                            'Now we only support single image processing ...'.format(img.ndim, label.ndim))

        # Check the shape consistency.
        shape_img = img.shape[:-1]
        shape_label = label.shape[:-1]
        shape_consistency = shape_img == shape_label
        if not shape_consistency:
            raise Exception('The shape of image and label are not satisfy consistency !!! '
                            'img: {}, label: {}'.format(shape_img, shape_label))

        # Check whether the class number of label satisfy the action number or not.
        label_cls_num = label.shape[-1]
        if self._cls_num != label_cls_num:
            raise Exception('the class number of label can not satisfy the action number !!! '
                            'label_cls_number: {}, cls_num: {}'.format(label_cls_num, self._cls_num))

        # Check the validity of finish current MHA flag.
        if not isinstance(finish_CMHA, bool):
            raise TypeError('The finish_CMHA must be of bool type !!!')

        if not isinstance(clazz_weights, np.ndarray) or len(clazz_weights) != self._cls_num:
            raise Exception('The class weights should be of '
                            '{}-dimension numpy array !!!'.format(len(clazz_weights)))

        # Finish.
        return


    def __verify_tfutil(self, FE_util, segment_util):
        r'''
            Verify the tensorflow utility.

        :param FEN_util:
        :param UN_util:
        :return:
        '''

        # Check the validity of FE util.
        if not isinstance(FE_util, tuple) or len(FE_util) != 5:
            raise TypeError('The FE_util must be a tuple containing 5 elements !!!')
        # Further check the validity of each argument.
        _1, _2, _3, _4, _5 = FE_util
        if not isinstance(_1, (tf.Session, tf.InteractiveSession, tf_debug.LocalCLIDebugWrapperSession)) or \
                not isinstance(_2, tf.Tensor) or \
                not isinstance(_3, tf.Tensor) or \
                not isinstance(_4, tf.Tensor) or \
                not isinstance(_5, list):
            print(_1, _2, _3, _4, _5)
            raise TypeError('The 1st argument must be tensorflow.Session or tensorflow.InteractiveSession!!! '
                            'And the 2nd-4th argument must be tensorflow.Tensor !!! '
                            'And the 5th argument must be list !!!')

        # Check the validity of segment util.
        if not isinstance(segment_util, tuple) or len(segment_util) != 7:
            raise TypeError('The segment_util must be a tuple containing 7 elements !!!')
        # Further check the validity of input arguments.
        _1, _2, _3, _4, _5, _6, _7 = segment_util
        if not isinstance(_1, (tf.Session, tf.InteractiveSession, tf_debug.LocalCLIDebugWrapperSession)) or \
                not isinstance(_2, tf.Tensor) or \
                not isinstance(_3, list) or \
                not isinstance(_4, tf.Tensor) or \
                not isinstance(_5, tf.Tensor) or \
                not isinstance(_6, tf.Tensor) or \
                not isinstance(_7, tf.Tensor):
            print(_1, _2, _3, _4, _5, _6, _7)
            raise TypeError('The 1st argument must be tensorflow.Session or tensorflow.InteractiveSession!!! '
                            'And the 2nd, 4-6th argument must be tensorflow.Tensor !!! '
                            'And the 3rd argument must be list !!!')

        # Finish.
        return

