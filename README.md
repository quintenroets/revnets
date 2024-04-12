# Revnets
![Python version](https://img.shields.io/badge/python-3.10+-brightgreen)
![Operating system](https://img.shields.io/badge/os-linux%20%7c%20macOS-brightgreen)
![Coverage](https://img.shields.io/badge/coverage-86%25-brightgreen)

Reverse engineer internal parameters of black box neural networks

## Usage

Run
```shell
revnets
```

The entry point of this command is the main function in revnets.main.main.
This command will start an experiment to recover the weights of a trained neural networks.
Each experiment consists of three configurable components:
1) pipelines: architecture + dataset that produces target networks that will be recovered
    - All possible networks are defined in [revnets.pipelines](revnets/pipelines)
    - The networks used in the experiment are configured with the option network in the config file
2) reconstructions: the techniques used to recover neural network weights
    - All possible techniques are defined in [revnets.reconstructions](revnets/reconstructions)
    - The networks used in the experiment are configured in the [init file](revnets/reconstructions/__init__.py)
3) evaluations: the methods used to evaluate a weight reconstruction
    - All possible evaluations are defined in [revnets.evaluations](revnets/evaluations)
    - The evaluations used in the experiment are configured in the [init file](revnets/evaluations/__init__.py)

Hyperparameters and other options are specified in a config file located at [config.yaml](~/Documents/Scripts/assets/revnets/config/config.yaml)

An example config file is provided at [config.yaml](examples/config.yaml)

For all possible options, see [config.py](revnets/models/config.py)

## Installation
```shell
pip install git+https://github.com/quintenroets/revnets.git
```

### Installation for development
Clone the project and run
```shell
pip install -e .
```
