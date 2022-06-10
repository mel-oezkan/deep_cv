import tensorflow as tf
import datetime


def create_model(model_args):
    """Create a tensorflow model.

    :param model_args: contains all arguments necessary to create the model
    :type model_args: dictonary
    :return:
    :rtype:
    """
    # check that all relevant keys are present
    for arg in ['model_type', 'optimizer', 'loss']:
        assert arg in model_args.keys(), f"[{arg}] is missing in model_args!"
    model_type = model_args['model_type']

    # import correct model
    if model_type == 'unet':
        from models import unet as model
    else:
        print(f"Model of type [{model_type}] not found.")

    # compile model
    model.compile(optimizer=model_args['optimizer'], loss=model_args['loss'])
    return model


def train(model, train_dataset, val_dataset, train_args):
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

    # fit model to data
    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=train_args['epochs'],
        callbacks=train_args['callbacks']
    )

    # save model
    if train_args['save_model']:
        time = datetime.datetime.now().strftime("%m.%d.%Y-%H:%M:%S")
        model_name = time + "-" + train_args['model_type']
        model.save(f'/src/runs/{model_name}')
