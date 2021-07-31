import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class NumbersDataset(Dataset):

    def __init__(self, config, path):
        super(NumbersDataset, self).__init__()

        self.config = config

        data = pd.read_csv(path)

        self.collection = []
        for index, row in data.iterrows():
            melspec = row["spect_path"]
            label = row["number"]
            self.collection.append([melspec, label])

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, item):
        spec_path, label = self.collection[item]

        # read spectrogram
        melspec = torch.from_numpy(np.load(spec_path))

        # prepare label
        label = torch.tensor([int(i) for i in str(int(label))], dtype=torch.long)

        input_length = melspec.shape[1]
        label_length = len(label)

        return melspec, label, input_length, label_length