#!/usr/bin/env python3

import sys
import json
import requests


def download_dataset(name, format, url):
    request = requests.get(url)
    if request.status_code != 200:
        print("Request for", name, "failed, status code: ",
              request.status_code, file=sys.stderr)
        return
    try:
        open('datasets/'+name+"."+format, 'wb').write(request.content)
    except Exception as e:
        print("Saving file for", name, "failed. Error: ", str(e))


if __name__ == "__main__":
    try:
        f = open("Datasets.json", 'r')
        datasets = json.load(f)
        for dataset in datasets['datasets']:
            print("Downloading file: ",
                  dataset['name'] + '.' + dataset['format'])
            download_dataset(
                dataset['name'], dataset['format'], dataset['url'])
    except Exception as e:
        print("Opening and reading Datasets.json failed with this error: ", str(e))
