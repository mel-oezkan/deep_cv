import wandb
from wandb.keras import WandbCallback

# Initialise a W&B run
wandb.init(config={"hyper": "parameter"})

model = None
X_train, y_train = [], []
X_test, y_test = [], []

# Add the WandbCallback to your Keras callbacks
model.fit(
    X_train, y_train,  
    validation_data=(X_test, y_test),
    callbacks=[WandbCallback()])