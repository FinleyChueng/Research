# Adapter-related abstractions
class Adapter:

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
