from distutils.command.config import config
import json
import random
from xmlrpc.client import boolean
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
import numpy as np
import argparse
from src2.utils.configUtil import load_config

from src2.utils.dataUtil import load_data


tf.keras.backend.clear_session()

# -------------- Create and parse argumente parser --------------
parser = argparse.ArgumentParser(
    description="Student-t Variational Autoencoder for Robust Density Estimation.")

parser.add_argument(
    "--config_file", type=str, required=True,
    help="Path to the config file.")

parser.add_argument(
    "--log", type=boolean, default=True,
    help="Path to the config file.")


args = parser.parse_args()
config = load_config(args.config_file)

if args.log:
    wandb.init(project="test_project", entity="deep_cv")
    wandb.config = {
        "learning_rate": 0.001,
        "epochs": 2,
        "batch_size": 16,
        'buffer_size': 1000
    }

# -------------- Setting up Data Pipeline --------------

dataset = load_data(config["data"])

# -------------- Creating Test Samples --------------

# -------------- Setting up Training --------------


# print('Restric')
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5144)])
#     except RuntimeError as e:
#         print(e)

# model = BasicUnet.create_model()
# model.compile(
#     loss=tf.keras.losses.MeanSquareError()
# )

# -------------- Train Model --------------

# model_history = model.fit(
#     dataset,
#     epochs=wandb.config.get('epochs'),
#     callbacks=[
#         DisplayTestCallback(dataset.take(1)),
#         WandbCallback(
#             data_type='image',
#             input_type='image',
#             output_type='segmentation_mask',
#             predictions=10,
#             training_data=[val_imgs, val_label]
#         )
#     ]
# )
