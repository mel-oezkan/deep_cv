import json


def load_config(conf_path: str):
    with open(conf_path, 'r') as source:
        config = json.load(source)

    return config
