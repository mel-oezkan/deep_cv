from numpy import tile
import tensorflow as tf
import skimage
import create_labels
import random


class DatasetGenerator:
    """This is a data-generator class. It takes a list of image paths and a
    summaray DataFrame to generate the input image and correspinding output
    mask.
    
    ::
    """

    def __init__(self, tile_ids, summary, tile_dir_path):
        """Constructor method"""
        self.tile_ids = tile_ids
        self.summary = summary
        self.tile_dir_path = tile_dir_path
    
    def __len__(self) -> int:
        """Return length of dataset/self.path_array
        
        :return: length of self.path_array
        :rtype: integer
        """
        return len(self.tile_ids)

    def __getitem__(self, idx: int) -> tuple:
        """Generate one item from self.path_array
        
        :param idx: index of path in self.path_array
        :type idx: integer
        :return: image, label pair
        :rtype: tuple"""
        # get tile identifier
        tile_id = self.tile_ids[idx]
        # create footprint mask
        label = create_labels.mask_from_id(tile_id, self.summary)
        # load and resize image
        filename = self.tile_dir_path + tile_id + '.tif'
        image = skimage.io.imread(filename)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        return image, label

    def __call__(self) -> tuple:
        """Yield one item from the dataset. Shuffle if end is reached
        
        :yield: one dataset item tuple
        :ytype: tuple
        """
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        """Shuffle self.path_array"""
        reidx = random.sample(population = list(range(self.__len__())),
                              k = self.__len__())
        self.tile_ids = self.tile_ids[reidx]
