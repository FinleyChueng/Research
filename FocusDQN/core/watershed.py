import numpy as np
from skimage import measure, morphology
import cv2
import logging



# The basic class.
class WatershedBase:
    r'''
        The basic class used to implements the variants of "Watershed Algorithm",
            which supplies many basic functions.
    '''

    def __init__(self,
                 anim_recorder=None,
                 log_level=logging.INFO,
                 log_dir=None,
                 logger_name='Watershed'):
        r'''
            Initialization.

        Parameters:
            anim_recorder: Used to record the process.
            log_level: Specify the log level.
        '''

        # animation recorder.
        self._anim_recorder = anim_recorder
        # config logger.
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        # handler not exists.
        if not logger.handlers:
            # input log into screen.
            ch = logging.StreamHandler()
            ch.setLevel(log_level)
            # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            formatter = logging.Formatter('%(name)s - %(message)s')
            ch.setFormatter(formatter)
            logger.addHandler(ch)
            # input log into local file.
            if log_dir is not None:
                fh = logging.FileHandler(log_dir)
                fh.setLevel(logging.DEBUG)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
        self._logger = logger

    def _pre_process(self, origin):
        r'''
            Pre-process function.

        Parameters:
            origin: The original image.

        Return:
            The standardized gradient image of original image.
        '''

        # get the gradient image of original image.
        grad_img32f = cv2.Laplacian(origin, cv2.CV_32F)
        # transfer the negative value to positive.
        abs_lap32f = np.absolute(grad_img32f)

        # normalization.
        grad = (abs_lap32f - np.min(abs_lap32f)) / (np.max(abs_lap32f) - np.min(abs_lap32f)) * 255

        # covert the float to int.
        grad = np.asarray(np.ceil(grad), 'int')

        return grad

    def _filter_conn_region(self, prop, origin):
        r'''
            Filter the connected region according to the @variable{prop} supplied by
                @method{measure.regionprops()}

        Parameters:
            prop: The prop of the connected region. The prop is supplied by
                @method{measure.regionprops()}
            origin: The original processing image (especially the gradient image or
                duplication image).

        Return:
            The filtered connected region.
        '''

        # Firstly, get the bounding-box in origin image, and then obtain the
        #   relative region in input image.
        x_t, y_t, x_e, y_e = prop.bbox
        conn_region = origin[x_t: x_e, y_t: y_e].copy()

        # Secondly, get the coords of connected-region in origin image
        #   and generate the mask. The mask is lately used to filter
        #   the other fake (not in current prop) conn-region
        #   within the bounding-box of current conn-region.
        mask = np.zeros((x_e - x_t, y_e - y_t), dtype=np.int)
        poss = prop.coords.transpose()
        poss[0] -= x_t
        poss[1] -= y_t
        mask[poss[0], poss[1]] = 1

        # Finally, filter the conn-region to get the pure region in input image.
        conn_region *= mask

        return conn_region

    def _evaluate_merge_situation(self, diff_conn):
        r'''
            Judge whether the specific connected region is the "Two in one" or not.

            The relation of diff_number and situation can be described as below:
            ------------------------------------------------------------------------------------
            |   diff_number   |    situation                                                   |
            ------------------------------------------------------------------------------------
            |       1         |    occur the new conn-region.                                  |
            |       2         |    the origin region grows up on one side.                     |
            |       3         |    two origin regions merge into one region.                   |
            |                 |       or the origin region grows up on both side.              |
            |     >= 4        |    one possible situation: "2" and "3" occurs Simultaneously.  |
            |                 |       another situation: more than two regions merged into one.|
            ------------------------------------------------------------------------------------

        Parameters:
            diff_conn: The diff image of specific connected region.

        Return:
            the flag indicating whether current conn-region is a "Two-in-one" region.
        '''

        # Flag.
        merged = False
        # Count the conn-region number.
        diff_conn_label, diff_conn_num = measure.label(diff_conn, return_num=True)
        # If the conn-region number is more than 3, which means two possible situation:
        #   1) more than two (including two) previous connected region merged into one current region.
        #   2) the previous connected region grows up on two (or many) sides.
        # So we have to further judge.
        if diff_conn_num >= 3:
            # because of the "Squeeze" operation by @method{measure.regionprops()}, we
            #   have to padding the label if there's "1" in dimension.
            if diff_conn_label.shape[0] == 1:
                diff_conn_label = np.concatenate((diff_conn_label, np.zeros(diff_conn_label.shape, dtype=np.int)))
            elif diff_conn_label.shape[1] == 1:
                diff_conn_label = np.concatenate((diff_conn_label, np.zeros(diff_conn_label.shape, dtype=np.int)), axis=1)
            # We concerned the number of previous connected region in diff-image.
            diff_props = measure.regionprops(diff_conn_label)
            prev_connr_num = 0
            for prop in diff_props:
                if diff_conn[prop.coords[0][0], prop.coords[0][1]] == 1:
                    prev_connr_num += 1
                # If the number of previous connected region in diff-image is more than one, it means
                #   that there are at least two regions merged into one region. In this case, the
                #   "Dam construction" operation should be concerned.
                if prev_connr_num > 1:
                    merged = True
                    break
        else:
            # Only occurs new conn-region or the origin conn-region grows up.
            #   Do nothing.
            pass

        # Debug info. ----------------------------------------------------------
        # self._logger.debug('The merged is {}, and the diff-conn-num is {}'.format(merged, diff_conn_num))
        # ----------------------------------------------------------------------

        return merged

    def _dam_construction(self, prop, diff_conn, dam):
        r'''
            Used to construct dam on dam image according to the specific diff connected
                region and its prop.

        Parameters:
            prop: The prop of specific diff connected region, which is supplied by
                @method{measure.regionprops()}.
            diff_conn: The specific diff connected region.
            dam: The dam image, that is, a mask in some extend.

        Return:
            No returns, the dam will be constructed on the dam image.
        '''

        # because of the "Squeeze" operation by @method{measure.regionprops()}, we
        #   have to padding the diff-image if there's "1" in dimension.
        #
        # ********** This operation change the @variable{diff_conn}. *******************
        if diff_conn.shape[0] == 1:
            diff_conn = np.concatenate((diff_conn, np.zeros(diff_conn.shape, dtype=np.int)))
        elif diff_conn.shape[1] == 1:
            diff_conn = np.concatenate((diff_conn, np.zeros(diff_conn.shape, dtype=np.int)), axis=1)

        # Debug info. -------------------------------------------
        # self._logger.debug(diff_conn)
        # -------------------------------------------------------

        # a mask version of current connected region, that is, boundary.
        boundary = np.zeros(diff_conn.shape, dtype=np.int)
        boundary[np.where(diff_conn != 0)] = 1
        # re-make the diff-conn image in order to better dilation.
        proc_conn = diff_conn.copy()
        proc_conn[np.where(diff_conn == 2)] = -1
        # previous processing conn-region.
        prev_pc = proc_conn.copy()
        # the mask used to dilation.
        di_mask = morphology.square(3)
        # get the label image and connected number of original previous image.
        prev_pc_label, prev_pc_num = measure.label(proc_conn, return_num=True)
        # and then get the relative props.
        prev_pc_props = measure.regionprops(prev_pc_label)
        # sort the previous process-conn-region according to the position of
        #   first pixel in conn-region. That is, "coords[0][0]"
        prev_pc_props.sort(key=lambda prop: prop.coords[0][0])
        # recursively dilation to find the position of dam.
        while not (proc_conn == boundary).all():
            # dilation.
            proc_conn = morphology.dilation(proc_conn, selem=di_mask)
            # erase the error dilated pixels so that the dilated image is
            #   within the boundary.
            proc_conn[np.where((boundary - proc_conn) == -1)] = 0
            # restore the non-dilated pixels so that the diff info can be
            #   reserved.
            proc_conn[np.where((diff_conn - proc_conn) == 2)] = -1
            # compute conn-region number of dilated image.
            cur_pc_label, cur_pc_num = measure.label(proc_conn, return_num=True)
            # get the props of current process-conn-region.
            cur_pc_props = measure.regionprops(cur_pc_label)
            # current diff-conn-num < prev, means "Merging" occurs.
            if cur_pc_num < prev_pc_num:
                # sort the props of current process-conn-region in the same way
                #   as previous process-conn-region.
                cur_pc_props.sort(key=lambda prop: prop.coords[0][0])
                # list used to record the lately merged prev-conn-region.
                merged_prev_slcr = []
                # Use the "Ordered array merge" - likely-hood algorithm to find
                #   the lately merged previous conn-region.
                p_start = 0     # used to record the index of prev-conn-region precess.
                for c_idx in range(len(cur_pc_props)):
                    # used to indicating current conn-region contains how many
                    #   previous conn-region.
                    contain_num = 0
                    for p_idx in range(p_start, len(prev_pc_props)):
                        # get the position of conn-region pixels, both current and
                        #   previous conn-region.
                        c_conn_poss = cur_pc_props[c_idx].coords
                        p_conn_poss = prev_pc_props[p_idx].coords
                        # if the first pixel of previous conn-region is within the
                        #   current conn-region, we regard this as contains the whole
                        #   region.
                        if p_conn_poss[0] in c_conn_poss:
                            # "Contain", concern the next previous conn-region.
                            contain_num += 1
                            # only record the previous sea-level conn-region.
                            if prev_pc[p_conn_poss[0][0], p_conn_poss[0][1]] == 1:
                                # record prop into target list.
                                merged_prev_slcr.append(prev_pc_props[p_idx])
                        else:
                            # "Not Contain", concern the next current conn-region.
                            p_start = p_idx   # record index.
                            break
                    # if current cur-conn-region contains more than two prev-conn-region,
                    #   that means we find the target previous conn-region.
                    if len(merged_prev_slcr) >= 2:
                        # Debug info. ----------------------------------------------------
                        # self._logger.debug('The contain number is {}, the length of list is {}'.format(contain_num, len(merged_prev_slcr)))
                        # ----------------------------------------------------------------
                        # simply end the loop.
                        break
                    else:
                        # not found the target conn-region, clear the list.
                        merged_prev_slcr.clear()
                # as we found the target previous conn-region, we will use them to calculate
                #   the position of dam.
                # Specially, there should be a dam in each two-pconn, so we concern each pair
                #   of two.
                for mpslcr_idx in range(len(merged_prev_slcr) - 1):
                    # get the boundary of two prev-slcr, which is lately used to get the target
                    #   dam region.
                    now_mpslcr = merged_prev_slcr[mpslcr_idx]
                    next_mpslcr = merged_prev_slcr[mpslcr_idx + 1]
                    now_xt, now_yt, now_xe, now_ye = now_mpslcr.bbox
                    next_xt, next_yt, next_xe, next_ye = next_mpslcr.bbox
                    # get x-axis range and y-axis range.
                    dam_xt, dam_xe = self.__compute_merge_axis_range(now_xt, now_xe, next_xt, next_xe)
                    dam_yt, dam_ye = self.__compute_merge_axis_range(now_yt, now_ye, next_yt, next_ye)
                    # get the target dam region.
                    dam_region = prev_pc[dam_xt: dam_xe, dam_yt: dam_ye]
                    # the position of "-1" in dam-region is the real dam.
                    dam_poss = np.where(dam_region == -1)

                    # Debug info. -----------------------------------------------
                    # # print(now_mpslcr.bbox)
                    # # print(next_mpslcr.bbox)
                    # self._logger.debug(dam_region)
                    # # print(dam_xt, dam_xe, dam_yt, dam_ye)
                    # self._logger.debug(dam_poss)
                    # ------------------------------------------------------------

                    # first get the baseline in the original image.
                    base_xt, base_yt, _, _ = prop.bbox
                    # compute the real position of dam pixels in the original image.
                    dam_poss[0][:] += (base_xt + dam_xt)
                    dam_poss[1][:] += (base_yt + dam_yt)

                    # Debug info. ------------------------------------------------
                    # self._logger.debug(dam_poss)
                    # ------------------------------------------------------------

                    # set relative position in the dam-image to "1" as the "Dam".
                    dam[dam_poss] = 1
            # end of @if{diff-conn-num < prev}.
            # finally update previous info to current, and then begin the new epoch.
            prev_pc = proc_conn
            prev_pc_num = cur_pc_num
            prev_pc_props = cur_pc_props
        # end of @while{dilation}.

        return

    def __compute_merge_axis_range(self, no_pt, no_pe, ne_pt, ne_pe):
        r'''
            The function used to compute the range of "merge" situation at
                the specific axis.

        Parameters:
            no_pt: The start position of current boundary.
            no_pe: The end position of current boundary.
            ne_pt: The start position of next boundary.
            ne_pe: The end position of next boundary.

        Return:
            The range of "merge" situation at the specific axis, which is
                consisted of (start_pos, end_pos).
        '''

        # initial
        m_pt = -1
        m_pe = -1
        # calculate merge range according to the different situations.
        if no_pt >= ne_pe:
            m_pt = ne_pe - 1
            m_pe = no_pt + 1
        if no_pe <= ne_pt:
            m_pt = no_pe - 1
            m_pe = ne_pt + 1
        if no_pt < ne_pt and no_pe > ne_pt and no_pe < ne_pe:
            m_pt = ne_pt - 1
            m_pe = no_pe + 1
        if no_pt < ne_pe and no_pt > ne_pt and no_pe > ne_pe:
            m_pt = no_pt - 1
            m_pe = ne_pe + 1
        if no_pt >= ne_pt and no_pe <= ne_pe:
            m_pt = no_pt - 1
            m_pe = no_pe + 1
        if no_pt <= ne_pt and no_pe >= ne_pe:
            m_pt = ne_pt - 1
            m_pe = ne_pe + 1

        # return the range of merged region on one axis.
        return m_pt, m_pe
