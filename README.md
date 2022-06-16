# deep_cv

This repo is made for the course "Projects in deep computer vision" given by the lecturers Ulf Krumnack, Ph.D., and Axel Schaffland from the University of Osnabrück.

Within this repo we tried to get a first impression on how bigger machine learning projects are done. At the same time we also wanted to playaround with the SpaceNet6 dataset. For that we created methods to reduce the size and complexity of the datasets. Such that the original SAR and RGB images were resized to (128x128x4) and (128x128x4) respectively.

## Structuing of the Project

- separated pipeline into parts
- created config system for easier testing and handling

## Development

Crete a config file containing the training parameters e.g. `config.json`.

To run the training step execute the file `main.py` file. This will run the whole project and automatically save all results in a separate log file.
