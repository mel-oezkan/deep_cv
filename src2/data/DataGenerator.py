import os
import random
import tensorflow as tf
import numpy as np
import cv2
import skimage
import pandas as pd

from src2.data import ROOT_DIR


class ResizedDataGenerator():

    """Generator to load resized images witout black bars.

    :param image_ids : iterator of string of image_ids you want to train on
    """

    def __init__(self, image_ids=None):
        """ Constructor function"""
       
        self.image_ids = image_ids

        if self.image_ids is None:
            self.image_ids = np.array(
                [i[:-4] for i in os.listdir("../datasets/train/AOI_11_Rotterdam/Labels_128")])

    def load_sample(self, index):
        img_id = self.image_ids[index]
        sar_path = f"{ROOT_DIR}/SAR-Intensity_128/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{img_id}.tif"
        label_path = f"{ROOT_DIR}/datasets/train/AOI_11_Rotterdam/Labels_128/{img_id}.tif"

        # create paths
        sar_path = f"{ROOT_DIR}/SAR-Intensity_128/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{img_id}.tif"
        label_path = f"{ROOT_DIR}/Labels_128/{img_id}.tif"

        # load and convert sar image
        sar_img = tf.image.convert_image_dtype(cv2.imread(
            sar_path, cv2.IMREAD_UNCHANGED), tf.float32)/255

        # load mask image
        mask_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        mask_img = np.expand_dims(mask_img, 2)
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


class DatasetGenerator:
    """This is a data-generator class. It takes a list of image paths and a
    summaray DataFrame to generate the input image and correspinding output
    mask.

    :param tile_ids: List of tile identifiers
    :type tile_ids: list
    :param summary: Summary dataframe containing all the polygons for each tile
    :type summary: pd.DataFrame
    :param tile_dir_path: Path that leads to the images. Should include the
        beginning of the filename as only tile_id's and '.tif' will be added
        to this string.
    :type tile_dir_path: string
    """

    def __init__(self,
                 tile_ids: list,
                 summary: pd.DataFrame,
                 tile_dir_path: str,
                 shuffle=False,
                 limit=None
                 ):
        """Constructor method"""
        self.tile_ids = np.array(tile_ids)
        self.summary = summary
        self.tile_dir_path = tile_dir_path
        self.limit = limit

        if shuffle:
            np.random.shuffle(self.tile_ids)

    def __len__(self) -> int:
        """Return length of dataset/self.path_array

        :return: length of self.path_array
        :rtype: integer
        """
        return self.tile_ids.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """Generate one item from self.path_array

        :param idx: index of path in self.path_array
        :type idx: integer
        :return: image, label pair
        :rtype: tuple"""
        # get tile identifier
        tile_id = self.tile_ids[idx]
        # create footprint mask
        label = create_labels.mask_from_id(tile_id, self.summary, edges=False)
        # load and resize image
        filename = self.tile_dir_path + tile_id + '.tif'
        image = skimage.io.imread(filename)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        # resize label
        label = np.expand_dims(label, axis=2)
        label = tf.image.resize(label, [256, 256])
        return image, label

    def __call__(self) -> tuple:
        """Yield one item from the dataset. Shuffle if end is reached

        :yield: one dataset item tuple
        :ytype: tuple
        """
        im_count = self.limit if self.limit else self.__len__()
        for i in range(im_count):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        """Shuffle self.path_array"""
        reidx = random.sample(population=list(range(self.__len__())),
                              k=self.__len__())
        self.tile_ids = self.tile_ids[reidx]
