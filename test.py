import matplotlib.pyplot as plt
import tensorflow as tf
tf.keras.backend.clear_session()

from create_labels import *
import os
from IPython.display import clear_output
import matplotlib.pyplot as plt
import random

from dataset_generator import DatasetGenerator,Generator_resized_data
from models.unet_tensorflow import model

gen= Generator_resized_data()

a,b = next(gen())

print(a.shape,b.shape)