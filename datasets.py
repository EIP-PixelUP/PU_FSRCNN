#!/usr/bin/env python3
#
# From https://github.com/yjn870/FSRCNN-pytorch

import h5py
import numpy as np
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        return (np.expand_dims(self.h5_file["input"][idx] / 255., 0).astype(np.float32),
                np.expand_dims(self.h5_file["output"][idx] / 255., 0).astype(np.float32))

    def __len__(self):
        return len(self.h5_file["input"])


class TestDataset(Dataset):
    def __init__(self, h5_file):
        super(TestDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        return (np.expand_dims(self.h5_file["input"][str(idx)][()] / 255., 0).astype(np.float32),
                np.expand_dims(self.h5_file["output"][str(idx)][()] / 255., 0).astype(np.float32))

    def __len__(self):
        return len(self.h5_file["input"])
