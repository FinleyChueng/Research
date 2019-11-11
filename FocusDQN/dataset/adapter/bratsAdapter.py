import random

import numpy as np

from dataset.adapter.base import Adapter
# from dataset.bratstools.readMHAsingle import BRATS2015
from dataset.bratstools.readMHA_std import BRATS2015


class BratsAdapter(Adapter):
    r'''
        This class is used as an adapter to BRATS task.
        It can simply return the next train pair and test pair
            to caller.
    '''

    def __init__(self, need_normalization=False, enable_data_enhance=True):
        r'''
            The initialize method of this class.

        -----------------------------------------------
        Parameters:
            need_normalization: Whether need normalization or not.
            enable_data_enhance: Whether need data enhancement or not.
        '''

        # Normal initialization.
        self._need_normalize = need_normalization
        self._enable_data_enhance = enable_data_enhance

        # The actual class and parameters.
        self._train_brats = BRATS2015(train=True, test=False, validation_set_num=10)
        self._test_brats = BRATS2015(train=False, test=True)
        # self._train_brats = BRATS2015(train=True, test=False, validation_set_num=1)
        # self._train_brats = BRATS2015()
        # self._test_brats = BRATS2015()

        # The variables used in enhance.
        self._enhance_opt_list = []
        self._train_image = None    # Used to cache current training image
        self._train_label = None    # Used to cache current training label

    def next_image_pair(self, mode, batch_size):
        r'''
            Return the image pair consisted of (image, label).

        -----------------------------------------------
        Parameters:
            mode: Indicates the mode of adapter, whether to return
                the train image pair or test image pair.
            batch_size: The size of batch.

        ----------------------------------------------------------
        Return:
            A tuple of (image, label).
        '''

        if not isinstance(mode, str):
            raise TypeError('The mode should be @Type{str} !!!')

        if not isinstance(batch_size, int):
            raise TypeError('The batch_size should be @Type{int} !!!')

        # Return train or test pair according to the mode.
        if mode == 'Train':
            return self.__next_train_pair(batch_size)
        elif mode == 'Validate':
            return self.__next_test_pair(batch_size, validation=True)
        elif mode == 'Test':
            return self.__next_test_pair(batch_size, validation=False)
        else:
            raise Exception('Unknown dataset mode: {} !!!'.format(mode))

    def write_result(self, result, name, mode='Test'):
        r'''
            Save the segmentation result in "MHA" form to the file system.

        :param result:
        :param name:
        :param mode:
        :return:
        '''

        if not isinstance(result, np.ndarray) or result.ndim != 2:
            raise TypeError('The result should be an 2-D @Type{numpy.ndarray} !!!')

        if not isinstance(name, str):
            raise TypeError('The name should be @Type{str} !!!')

        if not isinstance(mode, str):
            raise TypeError('The mode should be @Type{str} !!!')

        # Return train or test pair according to the mode.
        if mode == 'Train':
            self._train_brats.save_train_Itk([result])
        elif mode == 'Validate':
            self._train_brats.saveItk([result], name='Finley_'+name)
        elif mode == 'Test':
            self._test_brats.saveItk([result], name='Finley_'+name)
        else:
            raise Exception('Unknown dataset mode: {} !!!'.format(mode))

        # Finish.
        return


    def __next_train_pair(self, batch_size):
        r'''
            Return the train pair consisted of (image, label).

        --------------------------------------------------------------------
        Parameters:
            batch_size: The batch size, indicating the number of next
                train pairs.

        ----------------------------------------------------------
        Return:
            A tuple of (image, label).
        '''

        # Indicates all angle are traversed. Switch to next image.
        if len(self._enhance_opt_list) == 0:
            # Get image, label and flag.
            image, label, MHA_idx, inst_idx, clazz_weights = self._train_brats.next_train_batch(batch_size)

            # Assign image. Note the data type.
            self._train_image = np.asarray(image, dtype=np.float32)
            # Normalize the train image if needed.
            if self._need_normalize:
                self._train_image = self.__normalize(self._train_image, 255)

            # Assign label. Note the data type.
            self._train_label = np.asarray(label, dtype=np.float32)

            if self._enable_data_enhance:
                # Reset the enhance operation list.
                self._enhance_opt_list = [0, 1, 2, 3]

        # Check whether to rotate the image and related label.
        if self._enable_data_enhance:
            # Generate rotate extend randomly.
            rotate_idx = random.randint(0, len(self._enhance_opt_list) - 1)
            rotate_extend = self._enhance_opt_list.pop(rotate_idx)
            if rotate_extend == 0:
                # Do not need rotation.
                four_modality = self._train_image.copy()
                ground_truth = self._train_label.copy()
                # pass
            else:
                # Rotate the image batch.
                four_modality = np.rot90(self._train_image.copy(), rotate_extend, (1, 2))
                # Rotate the label batch.
                ground_truth = np.rot90(self._train_label.copy(), rotate_extend, (1, 2))
        else:
            # Simply assign value.
            four_modality = self._train_image
            ground_truth = self._train_label

        # Return the batch pair according to the batch size.
        if batch_size == 1:
            # Shape: [width, height, modalities], [width, height, cls]
            return four_modality[0], ground_truth[0], MHA_idx, inst_idx[0], clazz_weights
        else:
            # Shape: [batches, width, height, modalities], [batches, width, height, cls]
            return four_modality, ground_truth, MHA_idx, inst_idx, clazz_weights


    def __next_test_pair(self, batch_size, validation):
        r'''
            Return the test pair consisted of (image, label).

        --------------------------------------------------------------------
        Parameters:
            batch_size: The batch size, indicating the number of next
                test pairs.

        ----------------------------------------------------------
        Return:
            A tuple of (image, label).
        '''

        # Get image, label and flag.
        if validation:
            test_image, test_label = self._train_brats.next_test_batch(batch_size)
        else:
            test_image, test_label = self._test_brats.next_test_batch(batch_size)

        # Assign test image. Note the data type.
        four_modality = np.asarray(test_image, dtype=np.float32)
        # Normalize the test image if needed.
        if self._need_normalize:
            four_modality = self.__normalize(four_modality, 255)
        # Assign label. Note the data type.
        if test_label is not None:
            ground_truth = np.asarray(test_label, dtype=np.float32)
        else:
            ground_truth = [None]

        # Return the batch pair according to the batch size.
        if batch_size == 1:
            # Shape: [width, height, modalities], [width, height, cls]
            return four_modality[0], ground_truth[0]
        else:
            # Shape: [batches, width, height, modalities], [batches, width, height, cls]
            return four_modality, ground_truth


    def __normalize(self, src, upper):
        r'''
            Normalize the src to (batches, modalities)-wise.

        :param src:
        :param upper:
        :return:
        '''

        # Firstly calculate the max and min of (batches, modalities)-wise.
        #   The raw src shape: [batches, width, height, modalities]
        bm_wise_max = np.max(src, axis=(1, 2))  # [batches, modalities]
        bm_wise_min = np.min(src, axis=(1, 2))  # [batches, modalities]

        # Expand the dimension of max to the original dimension.
        bm_wise_max = np.expand_dims(bm_wise_max, axis=1)   # [batches, 1, modalities]
        bm_wise_max = np.expand_dims(bm_wise_max, axis=1)   # [batches, 1, 1, modalities]
        # Expand the dimension of min to the original dimension.
        bm_wise_min = np.expand_dims(bm_wise_min, axis=1)   # [batches, 1, modalities]
        bm_wise_min = np.expand_dims(bm_wise_min, axis=1)   # [batches, 1, 1, modalities]

        # Now can calculate the normalized value of src.
        normal = (src - bm_wise_min) / (bm_wise_max - bm_wise_min) * float(upper)

        # Finish normalization.
        return normal
