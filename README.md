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



## Download_dataset

**Usage:**
		python	download_dataset.py
This will use the Datasets.json file to get a list of all the datasets to download, and will download them to the *datasets* folder

**Tools:**
		download_dataset(	Name	Format	Url	)
Utilitary function permitting the download of any dataset using its Format (zip,h5..etc) and URL


## Prepare

**Usage:**
		python	prepare.py
This will take the datasets present in the *datesets* folder and will package them into .h5 dataset format files.

