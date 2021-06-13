#!/usr/bin/env python3

import sys
import requests


request = requests.get("https://eip.epitech.eu/2023/pixelup/data/General-100.zip")
if request.status_code != 200:
	print("Request failed, status code: ", request.status_code, file=sys.stderr)
	sys.exit(84)

try:
	open('Dataset_General100.zip', 'wb').write(request.content)
except Exception as e:
	print("Saving file failed. Error: ", str(e))