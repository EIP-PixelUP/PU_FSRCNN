#!/usr/bin/env python3

import onnx
import torch

from config import model_settings
from model import FSRCNN

model = FSRCNN(**model_settings)
model.load_state_dict(torch.load('result.pth'))
model.eval()

inputs = torch.ones(1, 1, 10, 10)

torch.onnx.export(model, inputs, "fsrcnn.onnx", verbose=True,
                  input_names=["input_image"], dynamic_axes={"input_image": [2, 3]})


onnx_model = onnx.load("fsrcnn.onnx")
onnx.checker.check_model(onnx_model)
