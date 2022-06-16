import tensorflow as tf
import datetime
from typing import List


def create_model(train_args):
    """Create a tensorflow model.

    :param model_args: contains all arguments necessary to create the model
    :type model_args: dictonary
    :return:
    :rtype:
    """
    # check that all relevant keys are present
    for arg in ['model_type', 'optimizer', 'loss']:
        assert arg in train_args.keys(), f"[{arg}] is missing in model_args!"
    model_type = train_args['model_type']

    # import correct model
    if model_type == 'unet':
        from src2.models.unet import create_model
    else:
        print(f"Model of type [{model_type}] not found.")

    # compile model
    model = create_model()
    model.compile(
        optimizer=train_args['optimizer'],
        loss=load_loss(train_args['loss']))

    return model


def load_loss(loss_name: str):
    # Todo1: Expand the loss function that loss parameters can be specified manually
    # Todo2: Include weighted loss
    """ Given a loss function in the config file
    imports it and initalizes it.

    :param loss_name: name of the loss function to be loaded
    :return the respective loss function
    """

    if loss_name == 'hybrid_loss':
        from src2.training.Losses import HybridLoss as Loss
    elif loss_name == 'dice_loss':
        from src2.training.Losses import DiceLoss as Loss
    elif loss_name == 'focal_loss':
        from src2.training.Losses import FocalLoss2d as Loss
    elif loss_name == 'mean_iou':
        from src2.training.Losses import MeanIOULoss as Loss
    elif loss_name == 'sparse_categorical_crossentropy':
        from tensorflow.keras.losses import SparseCategoricalCrossentropy as Loss
    elif loss_name == 'mean_square_error':
        from tensorflow.keras.losses import MeanSquaredError as Loss
    else:
        raise NotImplementedError(
            "Given Loss is not contained in the possible losses")

    loss_fnc = Loss()
    return loss_fnc


def load_callbacks(callbacks: List[str], val_data):

    labels = {0: 'background', 1: 'footprint'}

    callback_list = []
    for callback_name in callbacks:
        if callback_name == 'wandb':
            from wandb.keras import WandbCallback
            new_callback = WandbCallback(
                data_type='image',
                validation_data=val_data,
                labels=labels,
                output_type='segmentation_mask')

        elif callback_name == 'visualizer':
            pass

        else:
            continue
        callback_list.append(new_callback)

    return callback_list


def train_model(model, train_dataset, val_dataset, train_args):
    """Trains a given model.

    :param model: model to be trained
    :type model: tf.Model
    :param train_dataset: dataset model should be trained on
    :type train_dataset: tf.data.Dataset
    :param val_dataset: dataset model should be valided on, defaults to None
    :type val_dataset: tf.data.Dataset
    :return: 
    :rtype: 
    """
    # check that all relevant keys are present
    for arg in ['epochs', 'callbacks']:
        assert arg in train_args.keys(), f"[{arg}] is missing in model_args!"

    callbacks = load_callbacks(train_args['callbacks'], val_dataset)

    # fit model to data
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=train_args['epochs'],
        callbacks=callbacks
    )

    # save model
    if train_args['save_model']:
        time = datetime.datetime.now().strftime("%m.%d.%Y-%H:%M:%S")
        run_name = time + "-" + train_args['model_type']
        model.save(f'/src/runs/{run_name}')
