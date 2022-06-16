import wandb
from wandb.keras import WandbCallback

import tensorflow as tf
import argparse

from src2.utils.config_util import load_config
from src2.data.dataset import load_data
from src2.training.train_model import create_model, train_model
from src2.utils.environment import init_environment


# -------------- Create and parse argumente parser --------------

# can all this be moved away?

parser = argparse.ArgumentParser(
    description="Student-t Variational Autoencoder for Robust Density Estimation.")

parser.add_argument(
    "--config_file", type=str, required=True,
    help="Path to the config file.")

parser.add_argument(
    "--log", type=bool, default=False,
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


# -------------- Initalize environment --------------
print("Initalize Environment")
init_environment(config["env"])


# -------------- Setting up Data Pipeline --------------
print("Setting up Data pipeline")
datasets = load_data(config["data"])


# -------------- Setting up Training --------------
print("Creating the model")
model = create_model(config["training"])

# -------------- Train Model --------------
print("Train the new Model")
train_model(
    model,
    datasets['test'],
    datasets['val'],
    config["training"]
)

# -------------- Evaluate Model --------------
