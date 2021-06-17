#!/usr/bin/env python3

import sys
import torch
from PIL import Image
import numpy as np
from prepare import extract_y, augment_images, create_patches
from model import FSRCNN
from config import model_settings

usage =\
"""
	python upscale.py imagePath	[outputPath]

			imagePath	:	Path to the image to upscale
(optional)	outputPath	:	Path for the upscaled image
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

model = FSRCNN(**model_settings).to(device)
model.load_state_dict(torch.load('result.pth'))
model.eval()
torch.no_grad()

def upscaleImage(imagePath : str):
	Pimage = [Image.open(imagePath)]
	Pimage = extract_y(Pimage)
	Pimage = augment_images(Pimage)
	Pimage = create_patches(Pimage, scale=2, patch_size=10)
	image = None
	for img in Pimage:
		image = np.expand_dims(img[0] / 255., 0).astype(np.float32)
	tensor = torch.from_numpy(image).to(device)
	result = model(tensor)
	return result

if __name__ == "__main__":
	try:
		if len(sys.argv) != 2:
			print(usage, file=sys.stderr)
			sys.exit(84)
		elif sys.argv[1] == '-h' or sys.argv[1] == '--help':
			print(usage)
			sys.exit(0)
		image = upscaleImage(sys.argv[1])
		print(image)
	except Exception as e:
		print("Following error occured while trying to upscale image: ", str(e), file=sys.stderr)
		sys.exit(84)
