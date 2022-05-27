import wandb

import matplotlib.pyplot as plt
import tensorflow as tf
from create_labels import *

import os

import matplotlib.pyplot as plt
import random

from dataset_generator import DatasetGenerator
from models.unet_tensorflow import model
from train_unet import DisplayCallback


wandb.init(project="test_project", entity="deep_cv")
wandb.config = {
    "learning_rate": 0.001,
    "epochs": 2,
    "batch_size": 128,
    'buffer_size': 1000
}


AOI_PATH = '/home/melih/Code/uni/sem6/space_net/train/AOI_11_Rotterdam'
image_type = 'PS-RGB'


img_path_prototype = f'{AOI_PATH}/{image_type}/SN6_Train_AOI_11_Rotterdam_{image_type}_'
summary = load_summary(
    f'{AOI_PATH}/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')


img_ids = list(set(summary['ImageId']))
reidx = random.sample(population=list(range(len(img_ids))), k=len(img_ids))
img_ids = np.array(img_ids)[reidx]

img_ids = img_ids[:1000]


dg = DatasetGenerator(img_ids, summary, img_path_prototype)
ot = (tf.float32, tf.int64)
os = (tf.TensorShape([256, 256, 3]), tf.TensorShape([256, 256, 1]))
dataset = tf.data.Dataset.from_generator(dg, ot, os).batch(8)


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5144)])
    except RuntimeError as e:
        print(e)


model_history = model.fit(
    dataset,
    epochs=wandb.config.get('batch_size'),
    callbacks=[DisplayCallback()]
)

# Optional
wandb.watch(model)
