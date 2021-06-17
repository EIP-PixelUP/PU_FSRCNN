# PixelUp FSRCNN Implementation


## Introduction

Python version:		>= 3.9

This folder contains various Python scripts allowing for handling datasets, setting up and training the neural network, and checking its results on images.
This README will explain the steps to get a trained model ready for upscaling.
All scripts may be used individually for their own specific task, but the order in which they are documented
represents the order in which to execute them to result in a trained model.

* download_dataset
* prepare
* train
* check
* test
* upscale



## Usage

### Install the environment using Pipenv

``` sh
pipenv install
```

### Enter the environment using Pipenv

``` sh
pipenv shell
```

### Download the dataset

``` sh
./download_dataset.py
```

### Prepare the dataset for training

``` sh
./prepare.py
```

### Train the model

If you want, you can set the hyperparameters in the file `config.py`.

``` sh
./train.py
```

