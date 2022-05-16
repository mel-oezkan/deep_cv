from random import shuffle
from imageio import save
import tensorflow as tf
from create_labels import *
import getopt
import sys

from dataset_generator import DatasetGenerator
from models.test_net import build_model, finalize_model


img_types = ['PAN', 'PS-RGB', 'PS-RGBNIR', 'RGBNIR', 'SAR-Intensity']
# Hyperparameters
IMG_TYPE = img_types[1]
BATCH_SIZE = 16
PREFETCH_SIZE = 24
EPOCHS = 2
MODEL_NAME = 'standart model'


def create_dataset(image_type: str=IMG_TYPE, max_images=None) -> tf.data.Dataset:
    """Create a dataset.
    
    :param image_type: type of image that should be read, defaults to
        file hyperparameter
    :type image_type: string, optional
    :param max_images: maximum number of images in final dataset, if None all
        available images are used, defaults to None
    :type max_images: integer or NoneType, optional
    :return: tensorflow dataset
    :rtype: tf.data.Dataset
    """
    # path to images, image id and '.tif'-ending will be added by the ds-generator
    img_path_prototype = f'./datasets/train/AOI_11_Rotterdam/{image_type}/SN6_Train_AOI_11_Rotterdam_{image_type}_'
    # load summary-document containing WKT-Polygon strings and image id's
    summary = load_summary()
    # list all image_ids that the generator should use
    img_ids = list(set(summary['ImageId']))
    img_ids = shuffle(img_ids)
    # limit length of img_ids to max_images
    if max_images and max_images < len(img_ids):
        img_ids = img_ids[:max_images]
    
    # create DatasetGenerator-object
    dg = DatasetGenerator(img_ids, summary, img_path_prototype)
    
    # create dataset from generator
    shape_in = (tf.float32, tf.int64)
    shape_out = (tf.TensorShape([256, 256, 3]), tf.TensorShape([68, 68, 1]))
    dataset = tf.data.Dataset.from_generator(dg, shape_in, shape_out)
    return dataset


def dataset_pipeline(dataset: tf.data.Dataset, batch_size: int=BATCH_SIZE,
                     prefetch_size: int=PREFETCH_SIZE, functions: list=None,
                     args: list=None) -> tf.data.Dataset:
    """Preprocess dataset.
    
    :param dataset: dataset to be preprocessed
    :type dataset: tf.data.Dataset
    :param batch_size: batch size of the resulting data, defaults to
        file-hyperparameter
    :type batch_size: integer, optional
    :param prefetch_size: amount of batches to be prefetched, defaults to
        file-hyperparameter
    :type batch_size: integer, optional
    :param functions: functions that should be mapped to the data,
        defaults to None
    :type functions: single function or list of functions, functions take and
        return input and target, optional
    :param args: list of arguments for each of the provided functions,
        defaults to None
    :type args: list of arguments of only one function was provided,
        list of lists of arguments if several functions where provided,
        optional
    :return: preprocessed dataset
    :rtype: tf.data.Dataset
    """
    # map functions provided as arguments to data
    # !! better way of doing the following using **kwargs!!
    if type(functions) not in [list, tuple] and functions:
        functions = [functions]
        args = [args]
    for func, arg in zip(functions, args) if functions else []:
        dataset = dataset.map(lambda *map_data: func(map_data, *arg))
    # cache the dataset
    dataset = dataset.cache()
    # shuffle, batch and prefetch the dataset
    dataset = dataset.shuffle(10_000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_size)
    return dataset


def train(model, train_dataset: tf.data.Dataset,
          val_dataset: tf.data.Dataset=None, epochs: int=EPOCHS,
          save_model=True, model_name: str=MODEL_NAME):
    """Trains generic UNet Model.
    
    :param model: model to be trained
    :type model: tf.Model
    :param train_dataset: dataset model should be trained on
    :type train_dataset: tf.data.Dataset
    :param val_dataset: dataset model should be valided on, defaults to None
    :type val_dataset: tf.data.Dataset
    :param epochs: epochs model should be trained for, defaults to file hyper-
        parameter
    :type epochs: interger, optional
    :param save_model: if true model will be saved when training is done,
        defaults to true
    :type save_model: boolean, optional
    :param model_name: name the saved model will be given, defaults to
        file hyperparameter
    :type model_name: 
    """ 
    # fit model to data
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
    if save_model:
        unet.save(f'model/{model_name}')

if __name__ == '__main__':
    #args = sys.argv[1:]
    #options, args = getopt.getopt(args, shortopts='t:', longopts=['type='])
    unet = build_model(nx=256, ny=256, channels=3, num_classes=1, layer_depth=5)
    finalize_model(unet)