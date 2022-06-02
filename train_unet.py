from gc import callbacks
from random import shuffle
import tensorflow as tf
from create_labels import *
import matplotlib.pyplot as plt
import random
import os
from IPython.display import clear_output


<<<<<<< HEAD
from dataset_generator import DatasetGenerator, Generator_resized_data
#from models.test_net import build_model, finalize_model
from src.Callbacks import DisplayTestCallback
from src.Losses import HybridLoss
from src.models.unet import UNetCompiled
=======
from dataset_generator import DatasetGenerator, ResizedDataGenerator
#from models.test_net import build_model, finalize_model
from models.unet_tensorflow import model
>>>>>>> 5f75df3bb841ce079f21ab92638974590d718f1e

model = UNetCompiled()
model.compile(loss=tf.keras.losses.MeanSquaredError())

img_types = ['PAN', 'PS-RGB', 'PS-RGBNIR', 'RGBNIR', 'SAR-Intensity']
# Hyperparameters
IMG_TYPE = img_types[1]
BATCH_SIZE = 16
PREFETCH_SIZE = 24
EPOCHS = 10
MODEL_NAME = 'standart model'


<<<<<<< HEAD
def create_resized_dataset(image_ids = None):
=======
def create_resized_dataset(image_ids=None):
>>>>>>> 5f75df3bb841ce079f21ab92638974590d718f1e
    """Creates tf.dataset of resized images without black bars

    image_ids : iterator of string of image_ids you want to train on
    """
<<<<<<< HEAD
    dataset = tf.data.Dataset.from_generator(
        Generator_resized_data(image_ids = image_ids),
        output_types=(tf.float32, tf.int8),
        output_shapes= (tf.TensorShape([128, 128, 4]), tf.TensorShape([128, 128, 1]))
    )
    return dataset
=======
    return tf.data.Dataset.from_generator(ResizedDataGenerator(image_ids=image_ids),
                                          output_types=(tf.float32, tf.int32),
                                          output_shapes=(tf.TensorShape(
                                              [128, 128, 4]), tf.TensorShape([128, 128, 1]))
                                          )
>>>>>>> 5f75df3bb841ce079f21ab92638974590d718f1e


def create_dataset(image_type: str = IMG_TYPE, max_images=None) -> tf.data.Dataset:
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
    img_ids = np.array(list(set(summary['ImageId'])))
    # shuffle
    reidx = random.sample(population=list(range(img_ids.shape[0])),
                          k=img_ids.shape[0])
    img_ids = img_ids[reidx]
    # limit length of img_ids to max_images
    if max_images and max_images < img_ids.shape[0]:
        img_ids = img_ids[:max_images]
    # create DatasetGenerator-object
    dg = DatasetGenerator(img_ids, summary, img_path_prototype)

    # create dataset from generator
    shape_in = (tf.float32, tf.int64)
    shape_out = (tf.TensorShape([128, 128, 3]), tf.TensorShape([128, 128, 1]))
    dataset = tf.data.Dataset.from_generator(dg, shape_in, shape_out)
    return dataset


def dataset_pipeline(dataset: tf.data.Dataset, batch_size: int = BATCH_SIZE,
                     prefetch_size: int = PREFETCH_SIZE, functions: list = None,
                     args: list = None) -> tf.data.Dataset:
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
          val_dataset: tf.data.Dataset = None,
          epochs: int = EPOCHS, save_model=True):
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
    """
    # fit model to data
    model.fit(train_dataset, validation_data=val_dataset,
              epochs=epochs, callbacks=[DisplayTestCallback(dataset.take(8))])
    if save_model:
        model.save(f'model/{model.name}')

    # def show_predictions(self):
    #     _, ax = plt.subplots(3, self.sample_count)

    #     preds = self.model.predict(self.val_x)
    #     if len(preds.shape) >= 3 and preds.ndim >= 2:
    #         for index, sample in enumerate(zip(self.val_x, preds, self.val_y)):
    #             ax[0, index].imshow(sample[0])
    #             ax[0, index].axis('off')
    #             ax[1, index].imshow(sample[1])
    #             ax[1, index].axis('off')
    #             ax[2, index].imshow(sample[2])
    #             ax[2, index].axis('off')

    #             plt.show()
    #             break


def display(display_list):
    plt.figure(figsize=(5, 5))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()
    input()
    plt.close()


def show_predictions(num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([mask[0], pred_mask[0]])#[image[0], mask[0], pred_mask[0]])


if __name__ == '__main__':
    #args = sys.argv[1:]
    #options, args = getopt.getopt(args, shortopts='t:', longopts=['type='])
    #unet.name = MODEL_NAME

    tf.keras.backend.clear_session()

    print(os.getcwd())
    ids = image_ids = np.array([i[:-4] for i in os.listdir("../datasets/train/AOI_11_Rotterdam/Labels_128")])
    dataset = create_resized_dataset(ids[:])
    dataset = dataset_pipeline(dataset)

    show_predictions()
    train(model, train_dataset=dataset)
