#!/usr/bin/env python3

from zipfile import ZipFile
from PIL import Image
import numpy as np
from collections.abc import Iterator
from collections import namedtuple


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


def create_patches(images: Iterator[Image],
                   *, scale, patch_size) -> Iterator[TrainingDatum]:
    for image in images:
        low_res_size = (image.width // scale, image.height // scale)
        high_res_size = (image.width // scale * scale,
                         image.height // scale * scale)

        high_res_image = image.crop((0, 0, high_res_size[0], high_res_size[1]))
        low_res_image = high_res_image.resize(
            low_res_size, resample=Image.BICUBIC)

        for x in range(0, low_res_image.width - patch_size, scale):
            for y in range(0, low_res_image.height - patch_size, scale):
                yield TrainingDatum(
                    low_res_image.crop((x, y, x+patch_size, y+patch_size)),
                    high_res_image.crop(
                        (x*scale, y*scale,
                         (x+patch_size)*scale, (y+patch_size)*scale))
                )


def load_dataset(zip_file) -> Iterator[TrainingDatum]:
    images = get_images_from_zip(zip_file)
    images = extract_y(images)
    images = augment_images(images)
    images = create_patches(images, scale=2, patch_size=10)
    return images
