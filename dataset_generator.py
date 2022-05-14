import tensorflow as tf
import skimage
import create_labels
import random


class DatasetGenerator:
    def __init__(self, path_array, summary):
        self.path_array = path_array
        self.summary = summary
    
    def __len__(self):
        return len(self.path_array)
    
    def __getitem__(self, idx):
        filename = self.path_array[idx]
        id = tf.strings.substr(tf.strings.split(filename, '.')[1], 74, -1)
        label = create_labels.mask_from_id(id, self.summary)

        st = tf.compat.as_str_any(filename)
        image = skimage.io.imread(st)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [256, 256])
        return image, label
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

            if i == self.__len__() - 1:
                self.on_epoch_end()
    
    def on_epoch_end(self):
        reidx = random.sample(population = list(range(self.__len__())),k = self.__len__())
        self.imgarr = self.imgarr[reidx]
