# Adapter-related abstractions
class Adapter:

    @property
    def slices_3d(self):
        r'''
            The quantity of slices of 3-D data. 2-D data do not need to
                implement this property.
        '''
        return -1

    def next_image_pair(self, mode, batch_size):
        r'''
            Return the next image pair which is in local file system.
            The image pair is a tuple of (image, label).

        --------------------------------------------------------------------
        Parameters:
            mode: The dataset mode, indicating to get next train pair
                or test pair.
            batch_size: The batch size, indicating the number of next
                train or test pairs.

        --------------------------------------------------------------------
        Return:
            The train pair consisted of image and label.
        '''
        raise NotImplementedError

    def write_result(self, instance_id, result, name, mode):
        r'''
            Write the result according to the requirement of dataset.
                It usually save the result file in a special form.

        --------------------------------------------------------------------
        Parameters:
            instance_id: The id used to distinguish the sample instance.
            result: The result data, usually a 3-D or 4-D array.
            name: The file name of result.
            mode: The dataset mode, indicating the save mode.
        '''
        raise NotImplementedError

    def reset_position(self, offset, mode):
        r'''
            Reset the start position of data (instance) for iteration.

        --------------------------------------------------------------------
        Parameters:
            offset: The offset of data instance, that is, the start position.
            mode: The dataset mode, indicating the data iteration mode.
        '''
        raise NotImplementedError

    def precise_locate(self, position, mode):
        r'''
            Precisely locating and obtain the data instance according to
                the given position. It's just for convenient usage.

        --------------------------------------------------------------------
        Parameters:
            position: The position of the data instance.
            mode: The dataset mode, indicating the data iteration mode.
        '''
        raise NotImplementedError
