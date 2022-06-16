import json
import pathlib
from ipywidgets import Box, Layout, widgets


def load_config(conf_path: str):
    with open(conf_path, 'r') as source:
        config = json.load(source)

    return config


class ConfigCreator():
    def __init__(self):
        self.layout = Layout(
            display='flex',
            flex_flow='column',
            align_items='stretch',
            width='100%'
        )

        self.data_options = {
            "dataset_name": ["reduced_data", ],
            "in_shape": [[128, 128, 4]],
            "out_shape": [128, 128, 1],
            "batch_size": 64,
            "augmentation": [
                "normalize",
                "flip_right",
                "flip_left"
            ],
        }

        self.initalize_data_params()
        self.initialize_train_params()

    def initalize_env_params(self):
        self.batch_size = widgets.IntText(
            value=64,
            description='Batch size:',
            disabled=False
        )

    def initalize_data_params(self):
        """
            "dataset_name": "reduced_data",
            "in_shape": [128, 128, 4],
            "out_shape": [128, 128, 1],
            "batch_size": 64,
            "augmentation": [
                "normalize",
                "flip_right",
                "flip_left"
            ],
            "debug": true
        """

        self.dataset_name = widgets.Dropdown(
            options=['reduced_data'],
            value='reduced_data',
            description='Dataset Name:'
        )

        self.in_shape = widgets.Dropdown(
            options=[[128, 128, 4], [256, 256, 4]],
            value=[128, 128, 4],
            description='In Shape:'
        )

        self.out_shape = widgets.Dropdown(
            options=[[128, 128, 1], [256, 256, 1]],
            value=[128, 128, 1],
            description='In Shape:'
        )

        self.batch_size = widgets.IntText(
            value=64,
            description='Batch size:',
            disabled=False
        )

        aug_options = [
            'normalize', 'flip_right',
            'flip_left', 'rand_brightness'
            'rand_hue'
        ]

        self.augmentations = widgets.SelectMultiple(
            value=aug_options.copy(),
            options=aug_options,
            description='Augmentations'
        )

    def initialize_train_params(self):
        """        
            "epochs": 12,
            "model_type": "unet",
            "lr-rate": 1e-5,
            "callbacks": [],
            "optimizer": "adam",
            "loss": "hybrid_loss"
        """

        self.epochs = widgets.IntText(
            value=12,
            description='Epochs:',
            disabled=False
        )

        self.model_type = widgets.Dropdown(
            options=['unet'],
            value='unet',
            description='Model Name:'
        )

        self.lr = widgets.FloatText(
            value=1e-4,
            description='Learning rate:',
            disabled=False
        )

        callback_options = [
            'WandbCallback'
        ]

        self.callbacks = widgets.SelectMultiple(
            value=callback_options.copy(),
            options=callback_options,
            description='Callbacks',
        )

        self.optimizer = widgets.Dropdown(
            options=['adam'],
            value='adam',
            description='Optimizer:'
        )

        self.loss = widgets.Dropdown(
            options=[
                'hybrid_loss',
                'dice_loss',
                'focal_loss',
                'mean_iou',
                'sparse_categorical_crossentropy',
                'mean_square_error'
            ],
            value="hybrid_loss",
            description='Loss function:'
        )

    def show_data_inputs(self):
        items = [
            self.dataset_name,
            self.in_shape,
            self.out_shape,
            self.batch_size,
            self.augmentations
        ]

        return Box(children=items, layout=self.layout)

    def show_train_inputs(self):
        items = [
            self.epochs,
            self.model_type,
            self.lr,
            self.callbacks,
            self.optimizer,
            self.loss,
        ]

        return Box(children=items, layout=self.layout)

    def debug_values(self):
        data_params = {
            'dataset_name': self.dataset_name.value,
            'in_shape': self.in_shape.value,
            'out_shape': self.out_shape.value,
            'batch_size': self.batch_size.value,
            'augmentation': self.augmentations.value
        }

        train_params = {
            "epochs": self.epochs,
            "model_type": self.model_type,
            "lr-r": self.lr,
            "callbacks": self.callbacks,
            "optimizer": self.optimizer,
            "loss": self.loss,
        }

        data = {
            'data': data_params,
            'training': train_params
        }

        print(data)

    def write_params(self, conf_name):

        data_params = {
            'dataset_name': self.dataset_name.value,
            'in_shape': self.in_shape.value,
            'out_shape': self.out_shape.value,
            'batch_size': self.batch_size.value,
            'augmentation': self.augmentations.value
        }

        train_params = {
            "epochs": self.epochs,
            "model_type": self.model_type,
            "lr-r": self.lr,
            "callbacks": self.callbacks,
            "optimizer": self.optimizer,
            "loss": self.loss,
        }

        data = {
            'data': data_params,
            'training': train_params
        }

        conf_path = pathlib.Path(f'./configs/{conf_name}.json')
        with open(conf_path, 'w') as conf_file:
            json.dump(data, conf_file)
