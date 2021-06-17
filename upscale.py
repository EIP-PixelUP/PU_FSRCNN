#!/usr/bin/env python3

import torch
import argparse
from PIL import Image
import numpy as np
from model import FSRCNN
from config import model_settings
from pathlib import Path
import onnxruntime

usage =\
    """
        python upscale.py imagePath     [outputPath]

                        imagePath       :       Path to the image to upscale
(optional)      outputPath      :       Path for the upscaled image
"""


class Upscaler:
    def __init__(self, weigths, scale, *, onnx: Path):
        self.onnx = onnx
        self.scale = scale

        if onnx:
            self.session = onnxruntime.InferenceSession("fsrcnn.onnx")
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = FSRCNN(**model_settings).to(self.device)
            self.model.load_state_dict(torch.load(weigths))
            self.model.eval()

    def _upscale(self, data: np.ndarray):
        if self.onnx:
            input_name = self.session.get_inputs()[0].name
            return self.session.run(None, {input_name: data})[0]
        else:
            tensor = torch.from_numpy(data).to(self.device)
            result = self.model(tensor)
            return result.numpy()

    def upscaleImage(self, image: Image):
        with torch.no_grad():
            original = image
            hr_size = (original.width * self.scale,
                       original.height * self.scale)
            y, cb, cr = original.convert("YCbCr").split()
            #####
            array_y = np.array(y)[np.newaxis, np.newaxis].astype(
                np.float32) / 255.0
            result_y = self._upscale(array_y)
            result_y = (result_y[0, 0] * 255.0).astype(np.uint8)
            result_image_y = Image.fromarray(result_y)
            #####
            cb_hr = cb.resize(hr_size, resample=Image.BICUBIC)
            cr_hr = cr.resize(hr_size, resample=Image.BICUBIC)
            new_image = Image.merge("YCbCr", (result_image_y, cb_hr, cr_hr))
            return new_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', action="store_true",
                        help='Should use ONNX format')
    parser.add_argument('--image', dest='imagePath',
                        type=str, help="Image path to upscale")
    args = parser.parse_args()
    image_path = Path(args.imagePath)
    upscaler = Upscaler("result.pth", 2, onnx=args.onnx)
    image = upscaler.upscaleImage(Image.open(image_path))
    new_path = image_path.with_stem(image_path.stem + "_upscaled")
    image.convert("RGB").save(new_path)
    image.show()
