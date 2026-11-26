import re
from os import listdir
from os.path import join

import torch
from torch.utils.data import Dataset as TorchDataset

from encoder import ENCODER_FACTORY


class Dataset(TorchDataset):
    """Ciphertext dataset. Every .bin file is one sample, the folder name is the class.

    All files are read once and cached in memory (the whole dataset is only a few MB).
    """

    def __init__(self, root="_dataset"):
        self.samples = []  # (file_path, label, key_id)
        self.classes = sorted(ENCODER_FACTORY.keys())

        data, labels, key_ids = [], [], []
        for cipher_name in self.classes:
            class_path = join(root, cipher_name)
            for filename in sorted(listdir(class_path)):
                if not filename.endswith(".bin"):
                    continue
                fp = join(class_path, filename)
                with open(fp, "rb") as f:
                    raw = f.read()

                match = re.search(r'_key(\d+)\.bin', filename)
                key_id = int(match.group(1)) if match else -1
                label = self.classes.index(cipher_name)

                self.samples.append((fp, label, key_id))
                data.append(torch.tensor(list(raw), dtype=torch.long))
                labels.append(label)
                key_ids.append(key_id)

        self.data = torch.stack(data)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.key_ids = torch.tensor(key_ids, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.data[idx], int(self.labels[idx])

    def get_key_id(self, idx):
        """Return key_id for sample at index idx"""
        return self.samples[idx][2]
