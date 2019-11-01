import numpy as np
import moviepy.editor as mpy
import cv2



# Mask-related abstraction.
class MaskVisual:
    r'''
        Visualization of the procedure of operating mask of @class{ImSegEnv}
    '''

    def __init__(self, fps, vision_filename_mask):
        r'''
            The initialization method. Mainly declare the parameters used in
                @class{~matplotlib.pyplot}

        ----------------------------------------------------------------------------
        Parameters:
            fps: The FPS.
            vision_filename_mask: Specify where to save the visulization file.
        '''

        # The file name mask.
        self._anim_fn_mask = vision_filename_mask + 'anim-%05d.gif'
        self._result_fn_mask = vision_filename_mask + 'result-%05d.jpg'
        # The index of current animation used to save animation video.
        self._vision_index = -1

        # The file name mask used in "Inference" phrase.
        self._anim_fn_infer_mask = vision_filename_mask + 'inference/anim-%05d.gif'
        self._result_fn_infer_mask = vision_filename_mask + 'inference/result-%05d.jpg'
        # The index of current animation used in "Inference" phrase.
        self._infer_vision_index = -1

        # Visualization parameters.
        self._fps = fps

    def record(self, trace, arg=None):
        r'''
            Record the trace produced by DQN agent. Especially, the trace is
                a tuple consisting of (action_index, position_x, position_y,
                mask, information).

        ---------------------------------------------------------------------------
        Parameters:
            trace: The trace element indicating the agent's position and mask
                state of current action.
            arg: The additional arguments. Which is used to control the process
                of recording trace.
        '''
        raise NotImplementedError

    def reload(self, origin):
        r'''
            Reload the original image and action-position history list for next
                epoch.

        ----------------------------------------------------------------------------
        Parameters:
            origin: The original image (Ground Truth) of current epoch.
        '''
        raise NotImplementedError

    def show(self, mode):
        r'''
            Show the specific animation.

        ----------------------------------------------------------------------------
        Parameters:
            mode: The mode indicating the "Train" or "Inference" phrase.
        '''
        raise NotImplementedError



# Proposal-related abstraction.
class ProposalsVisual:
    r'''
        Visualization of the result of RPN.
    '''

    def __init__(self, vision_filename_mask):
        r'''
            Initialization method.

        :param vision_filename_mask:
        '''

        # The proposals result file name mask.
        self._prop_fn_mask = vision_filename_mask + 'prop-%05d.jpg'
        # The index of current proposals mask to save.
        self._prop_index = -1

        # The proposals result file name mask used in "Inference" phrase.
        self._prop_fn_mask = vision_filename_mask + 'inference/prop-%05d.jpg'
        # The index of current proposals mask used in "Inference" phrase.
        self._infer_prop_index = -1

    def prop_vis(self):
        r'''
            To generate the visualization result in the file system.
        :return:
        '''
        raise NotImplementedError



# Exploit the @class{~moviepy.editor}.
class MaskVisualVMPY(MaskVisual, ProposalsVisual):
    r'''
        This version will simply save the animation of the current epoch into
            file system.
    '''

    def __init__(self, image_width, image_height,
                 sight_stride=16,
                 fps=25,
                 vision_filename_mask='G:/Finley/dqn-anim/'):
        r'''
            The initialization method. Mainly declare the parameters used in
                @class{~moviepy.editor}

        ----------------------------------------------------------------------------
        Parameters:
            image_width: The width of image processed by DQN agent.
            image_height: The height of image processed by DQN agent.
            animation_filename_mask: The filename mask of generated animation.
            in_detail: Whether to show the detailed information.
        '''

        # Normal Initialization.
        MaskVisual.__init__(self, fps, vision_filename_mask)
        ProposalsVisual.__init__(self, vision_filename_mask)

        # The image width and height processing by Agent.
        self._image_width = image_width
        self._image_height = image_height
        # Whether to enable the GIF or not.
        self._gif_enable = None

        # Fake upper and lower, used to generate the frame (only one) for
        #   visualization of proposals.
        self._fake_upper_lower = np.zeros((image_width, image_height))

        # Declaration the data used for visualization.
        #   act_pos_history: The trace of the action chosen by DQN agent.
        #
        #       Specially the action-position history list of the
        #           current epoch used to visualization. The element
        #           is in the form of a tuple consisted of
        #           (action_index, position_x, position_y, mask, information).
        #
        #   trace_len: The length of trace.
        #   origin: The original image.
        #   label: The ground truth of original image.
        #   grad_label: The gradient image of label.
        self._act_pos_history = []
        self._trace_len = -1
        self._t1 = None
        self._t1c = None
        self._t2 = None
        self._f = None
        self._ground_truth = None
        # The sight boundary.
        self._sight_boundary = (image_width // sight_stride, image_height // sight_stride)

        # The normalization value.
        self._normal_value = 255

        # The changeless elements.
        self._first_row = None
        self._label = None      # Used to genenrate frame.

        # # The final result.
        # self._result = None

    def record(self, trace, arg=None):
        r'''
            Record the trace produced by DQN agent. Especially, the trace is
                a tuple consisting of (action_index, position_x, position_y,
                mask, information).

        ---------------------------------------------------------------------------
        Parameters:
            trace: The trace element indicating the agent's position and mask
                state of current action.
            arg: The additional arguments. Here we use it to control the
                recording of last frame when disable "GIF".
        '''

        # Add trace element into action-position history list.
        #   Condition: 1) Gif enable; 2) Trace is terminal.
        if self._gif_enable or (not self._gif_enable and arg is True):
            self._act_pos_history.append(trace)

    def reload(self, origin):
        r'''
            Reload the original image and action-position history list for next
                epoch.

        ----------------------------------------------------------------------------
        Parameters:
            origin: A tuple of (image, label, grad_label). Specially:
                image - The original image of current epoch.
                label - The ground truth of original image.
                grad_label - The gradient image of label.
        '''

        # Reload the trace information of DQN agent.
        self._act_pos_history.clear()

        # # Check validity.
        # if not isinstance(origin, tuple) or len(origin) != 3:
        #     raise TypeError('The origin must be a tuple containing 3 element !!!')
        #
        # # Get the holders.
        # modalities = origin[0]
        # # Check the validity of origin[0].
        # if not isinstance(modalities, np.ndarray):
        #     raise TypeError('The origin[0] must be of @Type{numpy.ndarray} !!!')
        # else:
        #     # Check the dimension.
        #     if modalities.ndim != 3:
        #         raise Exception('The origin[0] must be a 3-D array !!!')
        # # Reset the original image for visualization.
        # self._t1 = modalities[:, :, 0]
        # self._t1c = modalities[:, :, 1]
        # self._t2 = modalities[:, :, 2]
        # self._f = modalities[:, :, 3]
        #
        # # Check the validity of origin[1].
        # if not isinstance(origin[1], np.ndarray):
        #     raise TypeError('The origin[1] must be of @Type{numpy.ndarray} !!!')
        # else:
        #     # Check the dimension.
        #     if origin[1].ndim != 2:
        #         raise Exception('The origin[1] must be a 2-D array !!!')
        # # Reset the label and gradient image.
        # self._ground_truth = self._normalization(origin[1], self._normal_value)
        #
        # # Generate the changeless elements here, coz this method will only be called
        # #   once per image.
        # self._first_row, self._label = self._gen_changeless()

        # Get the GIF enable flag.
        GIF_flag = origin[0]
        # GIF_flag = origin[2]
        if not isinstance(GIF_flag, bool):
            raise TypeError('The origin[2] must be of bool type !!!')

        # Reset the gif flag.
        self._gif_enable = GIF_flag

    def _normalization(self, src, upper):
        r'''
            Normalization.

        :param src:
        :param upper:
        :return:
        '''

        denominator = np.max(src) - np.min(src)
        if denominator == 0:
            normal = np.zeros(np.shape(src))
        else:
            normal = (src - np.min(src)) / (np.max(src) - np.min(src)) * upper
        return normal

    def _gen_changeless(self):
        r'''
            Generate the changeless elements. That is,
                1. The first row.
                2. The label with bounding-box.

        :return:
        '''

        # # Get the holders for convenient processing.
        # frame_t1 = self._t1
        # frame_t1c = self._t1c
        # frame_t2 = self._t2
        # frame_f = self._f
        # # The label.
        # frame_label = self._ground_truth
        #
        # # Specify the border value.
        # border_value = self._normal_value // 3
        #
        # # Generate the first row according to the type of proposals.
        # if isinstance(self._proposals, np.ndarray):
        #     # Iteratively draw the proposals in the source image.
        #     for bbox in self._proposals:
        #         # Draw in the T1 image.
        #         frame_t1 = self._draw_bbox(frame_t1, bbox=bbox, border_value=border_value)
        #         # Draw in the T1C image.
        #         frame_t1c = self._draw_bbox(frame_t1c, bbox=bbox, border_value=border_value)
        #         # Draw in the T2 image.
        #         frame_t2 = self._draw_bbox(frame_t2, bbox=bbox, border_value=border_value)
        #         # Draw in the F image.
        #         frame_f = self._draw_bbox(frame_f, bbox=bbox, border_value=border_value)
        #         # Draw on the label image.
        #         frame_label = self._draw_bbox(frame_label, bbox=bbox, border_value=border_value, duplicate=True)
        # elif isinstance(self._proposals, tuple):
        #     # Only need to draw once. Coz the ot_proposal is the same
        #     #   as proposals in this case.
        #     pass
        # else:
        #     raise TypeError('The proposals should be either @Type{tuple} or @Type{numpy.ndarray} !!!')
        #
        # # Draw OT proposal in the source image.
        # t1 = self._draw_bbox(frame_t1, bbox=self._ot_proposal, border_value=border_value*2, transpose=False)
        # t1c = self._draw_bbox(frame_t1c, bbox=self._ot_proposal, border_value=border_value*2, transpose=False)
        # t2 = self._draw_bbox(frame_t2, bbox=self._ot_proposal, border_value=border_value*2, transpose=False)
        # f = self._draw_bbox(frame_f, bbox=self._ot_proposal, border_value=border_value*2, transpose=False)
        # # Draw OT proposal in the label image.
        # label = self._draw_bbox(frame_label,
        #                         bbox=self._ot_proposal,
        #                         border_value=border_value*2,
        #                         transpose=False,
        #                         duplicate=True)
        #
        # # First row. Concatenate related modalities.
        # first_row = np.concatenate((t1, t1c, t2, f), axis=1)

        first_row = np.concatenate((self._t1, self._t1c, self._t2, self._f, self._ground_truth), axis=1)
        label = self._ground_truth

        # Finish the generation.
        return first_row, label

    # def _draw_bbox(self, src, bbox, border_value, border_width=1, transpose=True, duplicate=False):
    #     r'''
    #         Draw the given bounding-box in the source image.
    #             Note that, this method will change the source image.
    #
    #     :param src:
    #     :param bbox:
    #     :param border_value:
    #     :param border_width:
    #     :return:
    #     '''
    #
    #     # Duplicate the source image and processing on the
    #     #   duplication if flag is true.
    #     if duplicate:
    #         raw = src.copy()
    #     else:
    #         raw = src
    #
    #     # Firstly get the four boundary.
    #     if transpose:
    #         # Need transpose. [x1, y1, x2, y2]
    #         left, up, right, bottom = bbox
    #     else:
    #         # Correct directions.
    #         left, right, up, bottom = bbox
    #
    #     # Assign the left boundary in the source image.
    #     raw[left: left+border_width, up: bottom] = border_value
    #     # Assign the right boundary in the source image.
    #     raw[right-border_width: right, up: bottom] = border_value
    #     # Assign the up boundary in the source image.
    #     raw[left: right, up: up+border_width] = border_value
    #     # Assign the bottom boundary in the source image.
    #     raw[left: right, bottom-border_width: bottom] = border_value
    #
    #     # Finish the drawing of bounding-box.
    #     return raw

    def _make_frame(self, t):
        r'''
            The frame-generation update method for @class{~moviepy.editor}. Simply
                generate a @class{~numpy.ndarray} that contains the original image
                and operated mask as the current frame.

        -------------------------------------------------------------------------------------
        Parameters:
            t: The seconds of current frame. The unit is (1 / fps) second.

        -------------------------------------------------------------------------------------
        Return:
            The @class{~numpy.ndarray} that contains the original image and operated
                mask as the current frame.
        '''

        # Get information about current trace.
        modalities, label, vis_data, segmentation, test_prev = self._act_pos_history[int(t * self._fps - 1)]

        # Reset the original image for visualization.
        _t1 = modalities[:, :, 0]
        _t1c = modalities[:, :, 1]
        _t2 = modalities[:, :, 2]
        _f = modalities[:, :, 3]
        # Normalize the four modalities for visualization.
        _t1 = self._normalization(_t1, self._normal_value)
        _t1c = self._normalization(_t1c, self._normal_value)
        _t2 = self._normalization(_t2, self._normal_value)
        _f = self._normalization(_f, self._normal_value)
        # Generate the ground truth.
        ground_truth = self._normalization(label, self._normal_value)

        # # Generate the four visualization image. - 60 * 60
        # vw, vh, vc = vis_data.shape
        # vis_arr = np.zeros((self._image_width, self._image_height, 4))
        # for k in range(4):
        #     vis_col = []
        #     for i in range(self._image_width // vw):
        #         vis_row = []
        #         for j in range(self._image_height // vh):
        #             chan = ((self._image_width // vw) * (self._image_height // vh)) * k + (self._image_width // vw) * i + j
        #             vis_row.append(vis_data[:, :, chan])
        #         vis_row = np.concatenate(vis_row, axis=1)
        #         vis_col.append(vis_row)
        #     vis_col = np.concatenate(vis_col, axis=0)
        #     vis_arr[:, :, k] = vis_col
        # envis_data = vis_arr

        # Generate the four visualization image. - 60 * 60
        vw, vh, vc = vis_data.shape
        vis_arr = np.zeros((self._image_width, self._image_height, 4))
        for k in range(4):
            vis_col = []
            for i in range(self._image_width // vw):
                vis_row = []
                for j in range(self._image_height // vh):
                    chan = ((self._image_width // vw) * (self._image_height // vh)) * k + (
                                                                                          self._image_width // vw) * i + j
                    if chan >= vc:
                        vis_row.append(np.zeros([vw, vh]))
                    else:
                        vis_row.append(vis_data[:, :, chan])
                vis_row = np.concatenate(vis_row, axis=1)
                vis_col.append(vis_row)
            vis_col = np.concatenate(vis_col, axis=0)
            vis_arr[:, :, k] = vis_col
        envis_data = vis_arr

        # Normalize the value.
        evd0 = self._normalization(envis_data[:, :, 0], self._normal_value)
        evd1 = self._normalization(envis_data[:, :, 1], self._normal_value)
        evd2 = self._normalization(envis_data[:, :, 2], self._normal_value)
        evd3 = self._normalization(envis_data[:, :, 3], self._normal_value)
        segmentation = self._normalization(segmentation, self._normal_value)

        # Fake
        fake = np.zeros_like(segmentation)

        # Concatenate the first row.
        first_row = np.concatenate((_t1, _t1c, _t2, _f, ground_truth, fake), axis=1)

        # Normalize test prev.
        test_prev = self._normalization(test_prev, self._normal_value)

        # Second row. Concatenate the visualization and segmentation.
        second_row = np.concatenate((evd0, evd1, evd2, evd3, segmentation, test_prev), axis=1)

        # Concatenate all the rows.
        frame = np.concatenate((first_row, second_row), axis=0)

        # Finally return the frame.
        return frame


    def show(self, train_mode):
        r'''
            Showing the animation of the current epoch. Actually the animation will be
                generated as a "gif" file in the file system, and it will not directly
                played in the IDE.

        ----------------------------------------------------------------------------
        Parameters:
            train_mode: The mode indicating "Train" or "Inference" phrase.
        '''

        # Check the validity of calling.
        valid, info = self._check_show_validity('mask')
        # Raise error if not valid.
        if not valid:
            raise Exception(info)

        # Auto-increase the index of animation file.
        if train_mode:
            self._vision_index += 1
        else:
            self._infer_vision_index += 1

        # Calculate the duration of current animation.
        duration = len(self._act_pos_history) / self._fps

        # Record animation if enabled.
        if self._gif_enable:
            # Generate the animation using defined frame-generation method.
            a1 = mpy.VideoClip(self._make_frame, duration=duration)
            # Specify the saving path.
            if train_mode:
                filename = self._anim_fn_mask % self._vision_index
            else:
                filename = self._anim_fn_infer_mask % self._infer_vision_index
            # Save the animation (GIF) into file system.
            a1.write_gif(filename, fps=self._fps)

        # And then save the final result into filesystem.
        if self._gif_enable:
            time_step = duration
        else:
            time_step = 0.
        result = self._make_frame(time_step)
        # Specify the saving path.
        if train_mode:
            result_file_name = self._result_fn_mask % self._vision_index
        else:
            result_file_name = self._result_fn_infer_mask % self._infer_vision_index
        # Save the result.
        cv2.imwrite(result_file_name, result)

        # Reset the changeless elements.
        self._reset_changeless()

        # Finish.
        return

    def prop_vis(self):
        r'''
            Save the proposal result to local file system.

        :return:
        '''

        # Check the validity of calling.
        valid, info = self._check_show_validity('prop')
        # Raise error if not valid.
        if not valid:
            raise Exception(info)

        # Increase index.
        self._prop_index += 1

        # Add a frame to history.
        self._act_pos_history.append((self._fake_upper_lower, self._fake_upper_lower))
        # Generate a frame.
        prop_result = self._make_frame(1 / self._fps)

        # Generate the saving path according to mode.
        result_file_name = self._prop_fn_mask % self._prop_index
        # Record it to file system.
        cv2.imwrite(result_file_name, prop_result)

        # Reset the changeless.
        self._reset_changeless()

        # Finish.
        return

    def _check_show_validity(self, mode):
        r'''
            Check the validity of calling show method.

        :return:
        '''

        # The flag and information.
        valid = True
        info = None

        # # Not call the reload in this case.
        # if self._first_row is None or self._label is None:
        #     valid = False
        #     info = 'One must call the @Method{reload} before calling @Method{show}'

        # Must call the @Method{record} in "Mask" mode.
        if mode == 'mask':
            # Not call the record in this case.
            if len(self._act_pos_history) == 0:
                valid = False
                info = 'One must call the @Method{record} before calling @Method{show}'

        # Finish.
        return valid, info

    def _reset_changeless(self):
        r'''
            Reset the changeless elements.

        :return:
        '''

        self._first_row = None
        self._label = None

        # Finish.
        return


