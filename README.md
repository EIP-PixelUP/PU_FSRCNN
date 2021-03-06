# PixelUp FSRCNN Implementation


## Introduction

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
* export_onnx


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
python download_dataset.py
```

### Prepare the dataset for training

``` sh
python prepare.py
```

### Train the model

If you want, you can set the hyperparameters in the file `config.py`.

``` sh
python train.py
```

### Test the model 

``` sh
python test.py --image PATH_TO_IMAGE
```

### Upscale an image

``` sh
python upscale.py --image PATH_TO_IMAGE [ --output PATH_TO_OUTPUT_IMAGE ]
```

### Export the ONNX model

``` sh
python export_onnx.py 
```

It generates the file `fsrcnn.onnx`. You can then use the other scripts with ONNX by adding the `--onnx` flag.

``` sh
python test.py --image PATH_TO_IMAGE --onnx
python upscale.py --image PATH_TO_IMAGE --onnx
```
