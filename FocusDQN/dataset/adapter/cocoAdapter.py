import cv2

from dataset.adapter.base import *
from dataset.cocotools.coco import COCO


class CocoAdapter(Adapter):

    def __init__(self, mode, ):
        # train set path.
        train_img_path = 'G:/Finley/dataset/COCO/train2017/'
        train_label_path = 'G:/Finley/dataset/COCO/annotations/stuff_train2017.json'
        train_stuff_path = 'G:/Finley/dataset/COCO/annotations/stuff_train2017_pixelmaps/'
        # test set path.
        test_img_path = 'G:/Finley/dataset/COCO/train2017/'
        test_label_path = 'G:/Finley/dataset/COCO/annotations/stuff_train2017.json'
        test_stuff_path = 'G:/Finley/dataset/COCO/annotations/stuff_train2017_pixelmaps/'
        # set the image and label path according to the mode.
        if mode == 'Train':
            self._img_path = train_img_path
            self._label_path = train_label_path
            self._stuff_path = train_stuff_path
        elif mode == 'Test':
            self._img_path = test_img_path
            self._label_path = test_label_path
            self._stuff_path = test_stuff_path
        else:
            raise Exception('Unknown Mode for COCO Adapter.')

        # initialize the COCO api.
        self._coco = COCO(self._label_path)
        self._id_list = self._coco.getImgIds()
        self._idx = 0
        self._batch_size = 1

    def next_image_pair(self):
        # get image and label according to the batch size.
        if self._batch_size == 1:
            # get image.
            img_info = self._coco.loadImgs(self._id_list[self._idx])[0]
            file_uri = self._img_path + img_info['file_name']
            img = cv2.imread(file_uri)
            # get relative label.
            label_name = img_info['file_name'][:-4] + '.png'
            label_uri = self._stuff_path + label_name
            label = cv2.imread(label_uri)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
            # increase the index.
            self._idx += 1
        else:
            # get images.
            img_infos = self._coco.loadImgs(self._id_list[self._idx, self._batch_size])
            file_uris = [self._img_path + info['file_name'] for info in img_infos]
            img = [cv2.imread(uri) for uri in file_uris]
            # get relative labels.
            label_names = [info['file_name'][:-4] + '.png' for info in img_infos]
            label_uris = [self._stuff_path + name for name in label_names]
            labels = [cv2.imread(uri) for uri in label_uris]
            label = [cv2.cvtColor(la, cv2.COLOR_BGR2GRAY) for la in labels]
            # increase the index.
            self._idx += self._batch_size

        return img, label
