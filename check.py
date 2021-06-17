#!/usr/bin/env python3

from math import log10, sqrt
import numpy as np
import argparse
from PIL import Image


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def compareImages(firstImage: Image, secondImage: Image):
    first = np.array(firstImage)
    second = np.array(secondImage)
    return PSNR(first, second)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', dest='original',
                        type=str, help="Original image", required=True)
    parser.add_argument('--upscaled', dest='upscaled',
                        type=str, help="Upscaled image", required=True)
    args = parser.parse_args()
    try:
        val = compareImages(Image.open(args.original),
                            Image.open(args.upscaled))
        print("PSNR level of correspondance: ", val, "%")
    except ValueError as e:
        print("The two images must be of the same dimensions. Error: ", str(e))
    except Exception as e:
        print("Following error occured: ", str(e))
