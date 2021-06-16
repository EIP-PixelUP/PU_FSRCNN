#!/usr/bin/env python3

from zipfile import ZipFile
from PIL import Image
import numpy as np
from typing import Union
from collections.abc import Iterator
from collections import namedtuple
import h5py
from tqdm import tqdm


def get_images_from_zip(zip_path) -> Iterator[Image]:
    with ZipFile(zip_path) as zip_dataset:
        for image_info in zip_dataset.infolist():
            if image_info.is_dir():
                continue
            with zip_dataset.open(image_info) as image_data:
                yield Image.open(image_data)


def extract_y(images: Iterator[Image]) -> Iterator[Image]:
    for image in images:
        yield image.convert("YCbCr").getchannel("Y")


def augment_images(images: Iterator[Image]) -> Iterator[Image]:
    for image in images:
        for scale in [1.0, 0.9, 0.8, 0.7, 0.6]:
            for rotation in [0, 90, 180, 270]:
                yield image\
                    .resize((int(image.width * scale),
                             int(image.height * scale)),
                            resample=Image.BICUBIC)\
                    .rotate(rotation, expand=True)


TrainingDatum = namedtuple("TrainingDatum", ["input", "output"])
TestingDatum = namedtuple("TestingDatum", ["input", "output"])


def create_patches(images: Iterator[Image],
                   *, scale, patch_size) -> Iterator[Union[TrainingDatum, TestingDatum]]:
    for image in images:
        low_res_size = (image.width // scale, image.height // scale)
        high_res_size = (image.width // scale * scale,
                         image.height // scale * scale)

        high_res_image = image.crop((0, 0, high_res_size[0], high_res_size[1]))
        low_res_image = high_res_image.resize(
            low_res_size, resample=Image.BICUBIC)

        yield TestingDatum(np.array(low_res_image), np.array(high_res_image))

        for x in range(0, low_res_image.width - patch_size, scale):
            for y in range(0, low_res_image.height - patch_size, scale):
                yield TrainingDatum(
                    np.array(low_res_image.crop(
                        (x, y, x+patch_size, y+patch_size))),
                    np.array(high_res_image.crop(
                        (x*scale, y*scale,
                         (x+patch_size)*scale, (y+patch_size)*scale)))
                )


def load_dataset(zip_file, *, scale, patch_size, augment=False) -> Iterator[TrainingDatum]:
    images = get_images_from_zip(zip_file)
    images = extract_y(images)
    images = augment_images(images) if augment else images
    images = create_patches(images, scale=scale, patch_size=patch_size)
    return images


def save_dataset(zip_file, dataset_file, *, scale, patch_size, augment=False):
    train_inputs = []
    train_outputs = []
    test_data = []
    for datum in tqdm(load_dataset(zip_file, scale=scale, patch_size=patch_size, augment=augment)):
        if isinstance(datum, TrainingDatum):
            train_inputs.append(datum.input)
            train_outputs.append(datum.output)
        else:
            test_data.append(datum)
    with h5py.File(dataset_file, "w") as f:
        test = f.create_group("test")
        test_input_group = test.create_group("input")
        test_output_group = test.create_group("output")
        for i, datum in enumerate(test_data):
            test_input_group.create_dataset(str(i), data=datum.input)
            test_output_group.create_dataset(str(i), data=datum.output)
        train = f.create_group("train")
        train.create_dataset("input", data=train_inputs)
        train.create_dataset("output", data=train_outputs)


if __name__ == "__main__":
    save_dataset("datasets/General-100.zip", "datasets/General-100.h5", scale=2, patch_size=10)
