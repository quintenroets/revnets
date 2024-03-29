# Revnets

Project to recover the weight of a blackbox neural network

## Installation

Make sure you are using python3.10+

```shell
pip install git+https://github.com/quintenroets/revnets
```

### Installation for development
Clone the project and run 
```shell
pip install -e .
```


## Usage
```shell
revnets
```

The entry point of this command is the main function in revnets.main.main. 
This command will start an experiment to recover the weights of a trained neural networks. 
Each experiment consists of three configurable components:
1) networks: the trained networks that will be recovered
    - All possible networks are defined in [revnets.networks](revnets/networks)
    - The networks used in the experiment are configured with the option network in the config file
2) reconstructions: the techniques used to recover neural network weights
    - All possible techniques are defined in [revnets.reconstructions](revnets/reconstructions)
    - The networks used in the experiment are configured in the [init file](revnets/reconstructions/__init__.py)
3) evaluations: the methods used to evaluate a weight reconstruction
    - All possible evaluations are defined in [revnets.evaluations](revnets/evaluations)
    - The evaluations used in the experiment are configured in the [init file](revnets/evaluations/__init__.py)

### Usage
Hyperparameters and other options are specified in a config file located at [config.yaml](~/Documents/Scripts/assets/revnets/config/config.yaml)

An example config file is provided at [config.yaml](examples/config.yaml)

For all possible options, see [config.py](revnets/utils/config.py)
