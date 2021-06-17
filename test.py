#!/usr/bin/env python3

import check
import upscale
from PIL import Image
import argparse


def runTest(image: Image, scale: int = 2, *, onnx: bool = False):
    low_res_size = (image.width // scale, image.height // scale)
    high_res_size = (image.width // scale * scale,
                     image.height // scale * scale)
    high_res_image = image.crop((0, 0, high_res_size[0], high_res_size[1]))
    low_res_image = high_res_image.resize(
        low_res_size, resample=Image.BICUBIC)
    handler = upscale.Upscaler('result.pth', scale, onnx=onnx)
    upscaledImage = handler.upscaleImage(low_res_image)
    correspondance = check.compareImages(upscaledImage.convert("RGB"), high_res_image.convert("RGB"))
    print("After downscaling and upscaling by a factor ",
          scale, ", the accuracy on this image is:", sep='')
    print("\t", correspondance, "%")
    return correspondance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', action="store_true",
                        help='(optional) Should use ONNX format')
    # parser.add_argument('--scale', dest='scale', type=int,
    #                     help='(optional) The scale to use for down and upscaling')
    parser.add_argument('--image', dest='imagePath',
                        type=str, help="Image path to upscale", required=True)
    args = parser.parse_args()
    runTest(Image.open(args.imagePath), onnx=args.onnx)
