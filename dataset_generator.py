import tensorflow as tf
import skimage
import create_labels
import random
import pandas as pd
import numpy as np
import cv2
import os


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


class Generator_resized_data():

    def __init__(self, image_ids=None):

        self.image_ids = image_ids

        if self.image_ids is None:

            self.image_ids = np.array(
                [i[:-4] for i in os.listdir("datasets/train/AOI_11_Rotterdam/Labels_128")])

    def __call__(self):

        for img_id in self.image_ids:

            sar_path = f"datasets/train/AOI_11_Rotterdam/SAR-Intensity_128/SN6_Train_AOI_11_Rotterdam_SAR-Intensity_{img_id}.tif"

            label_path = f"datasets/train/AOI_11_Rotterdam/Labels_128/{img_id}.tif"

            sar_img = tf.image.convert_image_dtype(cv2.imread(
                sar_path, cv2.IMREAD_UNCHANGED), tf.float32)/255

            mask_img = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

            mask_img = np.expand_dims(mask_img, 2)

            mask_img = tf.image.convert_image_dtype(mask_img, tf.int32)/255

            yield sar_img, mask_img
