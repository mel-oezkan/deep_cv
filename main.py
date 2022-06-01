import random
import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
from create_labels import *

from dataset_generator import DatasetGenerator
from src.Callbacks import DisplayTestCallback
from src.models.basicUnet.__index__ import BasicUnet


print('Initalize wandb project')
wandb.init(project="test_project", entity="deep_cv")
wandb.config = {
    "learning_rate": 0.001,
    "epochs": 2,
    "batch_size": 16,
    'buffer_size': 1000
}


tf.keras.backend.clear_session()


# -------------- Setting up Data Pipeline --------------
print('Setting up data pipeline')
AOI_PATH = '/home/melih/Code/uni/sem6/space_net/train/AOI_11_Rotterdam'
image_type = 'PS-RGB'


img_path_prototype = f'{AOI_PATH}/{image_type}/SN6_Train_AOI_11_Rotterdam_{image_type}_'
summary = load_summary(
    f'{AOI_PATH}/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv')


img_ids = list(set(summary['ImageId']))
data_gen = DatasetGenerator(
    img_ids, summary, img_path_prototype, shuffle=True, limit=200)

out_type = (tf.float32, tf.int64)
out_shape = (tf.TensorShape([256, 256, 3]), tf.TensorShape([256, 256, 1]))

dataset = tf.data.Dataset.from_generator(
    data_gen, out_type, out_shape
).batch(wandb.config.get('batch_size'))

# -------------- Creating Test Samples --------------

samples = random.choices(data_gen, k=10)
samples = list(zip(*samples))

val_imgs = np.asarray(samples[0])
val_label = np.asarray(samples[1])

# -------------- Setting up Training --------------


print('Restric')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5144)])
    except RuntimeError as e:
        print(e)

model = BasicUnet.create_model()
model.compile(
    loss=tf.keras.losses.MeanSquareError()
)

# -------------- Train Model --------------

model_history = model.fit(
    dataset,
    epochs=wandb.config.get('epochs'),
    callbacks=[
        DisplayTestCallback(dataset.take(1)),
        WandbCallback(
            data_type='image',
            input_type='image',
            output_type='segmentation_mask',
            predictions=10,
            training_data=[val_imgs, val_label]
        )
    ]
)
