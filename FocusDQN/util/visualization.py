import numpy as np
import moviepy.editor as mpy
import cv2
from PIL import Image, ImageDraw, ImageFont



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

        # The file name mask used in "Train" phrase.
        self._train_anim_fn_mask = vision_filename_mask + 'train/anim-%05d.'
        self._train_result_fn_mask = vision_filename_mask + 'train/result-%05d.jpg'
        # The index of current animation used in "Train" phrase.
        self._train_vision_index = -1

        # The file name mask used in "Validate" phrase.
        self._val_anim_fn_mask = vision_filename_mask + 'validate/anim-%05d.'
        self._val_result_fn_mask = vision_filename_mask + 'validate/result-%05d.jpg'
        # The index of current animation used in "Validate" phrase.
        self._val_vision_index = -1

        # The file name mask used in "Inference" phrase.
        self._test_anim_fn_mask = vision_filename_mask + 'test/anim-%05d.'
        self._test_result_fn_mask = vision_filename_mask + 'test/result-%05d.jpg'
        # The index of current animation used in "Inference" phrase.
        self._test_vision_index = -1

        # Visualization parameters.
        self._fps = fps

    def record(self, trace):
        r'''
            Record the trace produced by DQN agent. Especially, the trace is
                a tuple consisting of (action_index, position_x, position_y,
                mask, information).

        ---------------------------------------------------------------------------
        Parameters:
            trace: The trace element indicating the agent's position and mask
                state of current action.
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

    def show(self, mode, gif_enable):
        r'''
            Show the specific animation.

        ----------------------------------------------------------------------------
        Parameters:
            mode: The mode indicating the "Train" or "Inference" phrase.
            gif_enable: Enable "GIF" or not.
        '''
        raise NotImplementedError



# Exploit the @class{~moviepy.editor}.
class MaskVisualVMPY(MaskVisual):
    r'''
        This version will simply save the animation of the current epoch into
            file system.
    '''

    def __init__(self,
                 image_height,
                 image_width,
                 result_categories,
                 vision_filename_mask,
                 fps=25):
        r'''
            The initialization method. Mainly declare the parameters used in
                @class{~moviepy.editor}

        ----------------------------------------------------------------------------
        Parameters:
            image_height: The height of image processed by DQN agent.
            image_width: The width of image processed by DQN agent.
            animation_filename_mask: The filename mask of generated animation.
            in_detail: Whether to show the detailed information.
        '''

        # Normal Initialization.
        MaskVisual.__init__(self, fps, vision_filename_mask)

        # The image height and width processing by Agent.
        self._image_height = image_height
        self._image_width = image_width

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
        self._label = None
        # The result categories.
        self._res_cate = result_categories

        # The normalization value.
        self._normal_value = 255
        # Text font.
        self._font = ImageFont.truetype("consola.ttf", 20, encoding="unic")

        return

    def record(self, trace):
        r'''
            Record the trace produced by DQN agent. Especially, the trace is
                a tuple consisting of (action_index, position_x, position_y,
                mask, information).

        ---------------------------------------------------------------------------
        Parameters:
            trace: The trace element indicating the agent's position and mask
                state of current action.
        '''

        # Add trace element into action-position history list.
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

        # Check validity.
        if not isinstance(origin, tuple) or len(origin) != 2:
            raise TypeError('The origin must be a tuple containing 2 element !!!')

        # Get the holders.
        modalities = origin[0]
        # Check the validity of origin[0].
        if not isinstance(modalities, np.ndarray):
            raise TypeError('The origin[0] must be of @Type{numpy.ndarray} !!!')
        else:
            # Check the dimension.
            if modalities.ndim != 3:
                raise Exception('The origin[0] must be a 3-D array !!!')
        # Reset the original image for visualization.
        self._t1 = self._normalization(modalities[:, :, 0], self._normal_value)
        self._t1c = self._normalization(modalities[:, :, 1], self._normal_value)
        self._t2 = self._normalization(modalities[:, :, 2], self._normal_value)
        self._f = self._normalization(modalities[:, :, 3], self._normal_value)

        # Check the validity of origin[1].
        if not isinstance(origin[1], np.ndarray):
            raise TypeError('The origin[1] must be of @Type{numpy.ndarray} !!!')
        else:
            # Check the dimension.
            if origin[1].ndim != 2:
                raise Exception('The origin[1] must be a 2-D array !!!')
        # Reset the label and gradient image.
        lab = origin[1]
        if lab is None:
            lab = np.zeros((self._image_height, self._image_width), dtype=np.int64)
        self._label = self._normalization(lab, self._normal_value, max_val=self._res_cate)
        return


    def _normalization(self, src, lower, upper=None, min_val=None, max_val=None):
        r'''
            Normalization.
        '''
        if upper is None:
            upper = lower
            lower = 0
        min_v = np.min(src) if min_val is None else min_val
        max_v = np.max(src) if max_val is None else max_val
        denominator = max_v - min_v
        if denominator == 0:
            normal = np.zeros(np.shape(src))
        else:
            normal = (src - min_v) / (max_v - min_v) * (upper - lower) + lower
        return normal

    def _transfer_2val(self, bbox, h, w):
        r'''
            Transfer the normalized value to real value.
        '''
        y1, x1, y2, x2 = bbox
        up = int(round(y1 * h))
        left = int(round(x1 * w))
        bottom = int(round(y2 * h))
        right = int(round(x2 * w))
        return up, left, bottom, right

    def _draw_bbox(self, src, bbox, border_value, border_width=1, duplicate=True):
        r'''
            Draw the given bounding-box in the source image.
                Note that, this method will change the source image.
        '''

        # Duplicate the source image and processing on the duplication if flag is true.
        if duplicate:
            raw = src.copy()
        else:
            raw = src

        # Firstly get the four boundary. [y1, x1, y2, x2]
        up, left, bottom, right = self._transfer_2val(bbox, h=self._image_height, w=self._image_width)

        # Assign the left boundary in the source image.
        raw[up: up + border_width, left: right] = border_value
        # Assign the right boundary in the source image.
        raw[bottom - border_width: bottom, left: right] = border_value
        # Assign the up boundary in the source image.
        raw[up: bottom, left: left + border_width] = border_value
        # Assign the bottom boundary in the source image.
        raw[up: bottom, right - border_width: right] = border_value

        # Finish the drawing of bounding-box.
        return raw

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
        cur_region, focus_bbox, reward, segmentation, ter_info = self._act_pos_history[int(t * self._fps)]

        # Normalization.
        segmentation = self._normalization(segmentation, self._normal_value, max_val=self._res_cate)

        # Draw the bounding-box on image and label.
        b_v1 = self._normal_value // 3
        _t1 = self._draw_bbox(self._t1, cur_region, b_v1)
        _t1c = self._draw_bbox(self._t1c, cur_region, b_v1)
        _t2 = self._draw_bbox(self._t2, cur_region, b_v1)
        _f = self._draw_bbox(self._f, cur_region, b_v1)
        _lab = self._draw_bbox(self._label, cur_region, b_v1)
        _pred = self._draw_bbox(segmentation, cur_region, b_v1)
        # Draw the "Focus Bounding-box" if given.
        if focus_bbox is not None:
            b_v1 = self._normal_value // 3 * 2
            _t1 = self._draw_bbox(_t1, focus_bbox, b_v1, duplicate=False)
            _t1c = self._draw_bbox(_t1c, focus_bbox, b_v1, duplicate=False)
            _t2 = self._draw_bbox(_t2, focus_bbox, b_v1, duplicate=False)
            _f = self._draw_bbox(_f, focus_bbox, b_v1, duplicate=False)
            _lab = self._draw_bbox(_lab, focus_bbox, b_v1, duplicate=False)
            _pred = self._draw_bbox(_pred, focus_bbox, b_v1, duplicate=False)

        # Scale up the focus region for better visualization.
        vis_bbox = focus_bbox
        SEG_stage = False
        if vis_bbox is None:
            vis_bbox = cur_region
            SEG_stage = True
        vb_y1, vb_x1, vb_y2, vb_x2 = self._transfer_2val(vis_bbox, h=self._image_height, w=self._image_width)
        dsize = _lab.shape
        # scale.
        _vis_lab = self._label[vb_y1: vb_y2, vb_x1: vb_x2]
        _vis_lab = cv2.resize(_vis_lab, dsize, interpolation=cv2.INTER_NEAREST)
        _vis_pred = segmentation[vb_y1: vb_y2, vb_x1: vb_x2]
        _vis_pred = cv2.resize(_vis_pred, dsize, interpolation=cv2.INTER_NEAREST)

        # Indicator image.
        _indt = Image.fromarray(np.zeros((self._image_height, self._image_width)))
        draw = ImageDraw.Draw(_indt)
        if SEG_stage:
            txt = u'<- Segment ->'
        else:
            txt = u'<- Focus ->'
        txt += u'\nx1: {}, x2: {}\ny1: {}, y2: {}'.format(vb_x1, vb_x2, vb_y1, vb_y2)
        if reward is not None:
            txt += u'\nreward: {}'.format(reward)
        if ter_info is not None:
            txt += u'\n' + ter_info
        tw, th = self._font.getsize_multiline(txt)
        draw.multiline_text(((self._image_width - tw) // 2, (self._image_height - th) // 2), txt,
                            font=self._font, align='center')
        _indt = np.asarray(_indt)
        _indt = self._normalization(_indt, self._normal_value)

        # Generate each row.
        row_1st = np.concatenate((_t1, _t1c, _t2), axis=1)
        row_2nd = np.concatenate((_f, _lab, _pred), axis=1)
        row_3rd = np.concatenate((_indt, _vis_lab, _vis_pred), axis=1)
        # Concatenate all the rows.
        frame = np.concatenate((row_1st, row_2nd, row_3rd), axis=0)

        frame = np.stack((frame, frame, frame), axis=-1)

        # Finally return the frame.
        return frame


    def show(self, mode, anim_type=None):
        r'''
            Showing the animation of the current epoch. Actually the animation will be
                generated as a "gif" file in the file system, and it will not directly
                played in the IDE.

        ----------------------------------------------------------------------------
        Parameters:
            mode: The mode indicating "Train", "Validate" or "Inference" phrase.
        '''

        # Check the validity of calling.
        if len(self._act_pos_history) == 0:
            raise Exception('Nothing to show !!! One must call the @Method{record} before calling @Method{show}')

        # Auto-increase the index of animation file.
        if mode == 'Train':
            self._train_vision_index += 1
        elif mode == 'Validate':
            self._val_vision_index += 1
        elif mode == 'Test':
            self._test_vision_index += 1
        else:
            raise ValueError('Unknown mode value !!!')

        # Calculate the duration of current animation.
        duration = len(self._act_pos_history) / self._fps

        # Record animation if enabled.
        if anim_type is not None:
            # Generate the animation using defined frame-generation method.
            a1 = mpy.VideoClip(self._make_frame, duration=duration)
            # Specify the saving path.
            if mode == 'Train':
                filename = self._train_anim_fn_mask % self._train_vision_index
            elif mode == 'Validate':
                filename = self._val_anim_fn_mask % self._val_vision_index
            elif mode == 'Test':
                filename = self._test_anim_fn_mask % self._test_vision_index
            else:
                raise ValueError('Unknown mode value !!!')
            # Save the animation (GIF) into file system.
            if anim_type == 'gif':
                filename += u'gif'
                a1.write_gif(filename, fps=self._fps)
            elif anim_type == 'video':
                filename += u'mp4'
                a1.write_videofile(filename, fps=self._fps, codec='mpeg4', audio=False)
            else:
                raise ValueError('Unknown animation type !!!')

        # Get result.
        result = self._make_frame(duration - 1.0/self._fps)
        # Specify the saving path.
        if mode == 'Train':
            result_file_name = self._train_result_fn_mask % self._train_vision_index
        elif mode == 'Validate':
            result_file_name = self._val_result_fn_mask % self._val_vision_index
        elif mode == 'Test':
            result_file_name = self._test_result_fn_mask % self._test_vision_index
        else:
            raise ValueError('Unknown mode value !!!')
        # Save the result.
        cv2.imwrite(result_file_name, result)

        # Finish.
        return
