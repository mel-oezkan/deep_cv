import os
import tensorflow as tf
import numpy as np
import cv2


class ResizedDataGenerator():

    """Generator to load resized images witout black bars
    """

    def __init__(self, image_ids=None):
        """
        image_ids : iterator of string of image_ids you want to train on"""
        self.image_ids = image_ids

        if self.image_ids is None:
            self.image_ids = np.array(
                [i[:-4] for i in os.listdir("../datasets/train/AOI_11_Rotterdam/Labels_128")])

    def load_sample(self, index):
        img_id = self.image_ids[index]
        sar_path = f"../datasets/train/AOI_11_Rotterdam/SAR-Intensity_128/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{img_id}.tif"
        label_path = f"../datasets/train/AOI_11_Rotterdam/Labels_128/{img_id}.tif"

        # create paths
        sar_path = f"../datasets/train/AOI_11_Rotterdam/SAR-Intensity_128/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{img_id}.tif"
        label_path = f"../datasets/train/AOI_11_Rotterdam/Labels_128/{img_id}.tif"

        # load and convert sar image
        sar_img = tf.image.convert_image_dtype(cv2.imread(
            sar_path, cv2.IMREAD_UNCHANGED), tf.float32)/255

        # load mask image
        mask_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

        # expand last dim
        mask_img = np.expand_dims(mask_img, 2)

        # convert
        mask_img = tf.image.convert_image_dtype(mask_img, tf.uint8)

        return sar_img, mask_img

    def in_range(self, idx: int):
        return idx > 0 and idx <= len(self.image_ids)

    def __len__(self):
        """Return length of dataset/self.path_array

        :return: length of self.path_array
        :rtype: integer
        """

        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple:
        """Generate one item from self.path_array

        :param idx: index of path in self.path_array
        :type idx: integer
        :return: image, label pair
        :rtype: tuple"""

        if not self.in_range(idx):
            raise IndexError('Given index is out of bounds')

        return self.load_sample(idx)

    def __call__(self, limit):
        # for all ids

        im_count = min(limit, len(self.image_ids))
        for img_id in range(im_count):
            yield self.load_sample(img_id)
