#!/usr/bin/env python3

import torch
import argparse
from PIL import Image
import numpy as np
from model import FSRCNN
from config import model_settings
from pathlib import Path

usage =\
    """
        python upscale.py imagePath     [outputPath]

                        imagePath       :       Path to the image to upscale
(optional)      outputPath      :       Path for the upscaled image
"""


class Upscaler:
    def __init__(self, weigths, scale):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = FSRCNN(**model_settings).to(self.device)
        self.model.load_state_dict(torch.load('result.pth'))
        self.model.eval()

        self.scale = scale

    def upscaleImage(self, imagePath: Path):
        with torch.no_grad():
            original = Image.open(imagePath)
            hr_size = (original.width * self.scale,
                       original.height * self.scale)
            y, cb, cr = original.convert("YCbCr").split()
            #####
            array_y = np.array(y)[np.newaxis, np.newaxis].astype(
                np.float32) / 255.0
            tensor_y = torch.from_numpy(array_y).to(self.device)
            result_y = self.model(tensor_y)
            result_y = (result_y.numpy()[0, 0] * 255.0).astype(np.uint8)
            result_image_y = Image.fromarray(result_y)
            #####
            cb_hr = cb.resize(hr_size, resample=Image.BICUBIC)
            cr_hr = cr.resize(hr_size, resample=Image.BICUBIC)
            new_image = Image.merge("YCbCr", (result_image_y, cb_hr, cr_hr))
            new_path = imagePath.with_stem(imagePath.stem + "_upscaled")
            new_image.convert("RGB").save(new_path)

            return new_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', dest='onnx', type=bool,
                        help='Should use ONNX format')
    parser.add_argument('--image', dest='imagePath',
                        type=str, help="Image path to upscale")
    args = parser.parse_args()
    upscaler = Upscaler("result.pth", 2)
    image = upscaler.upscaleImage(Path(args.imagePath))
    image.show()
