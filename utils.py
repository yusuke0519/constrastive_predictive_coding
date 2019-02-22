# # -*- coding: utf-8 -*-
import numpy as np
import torch.utils.data as data


class Subset(data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def split_dataset(dataset, train_size=0.9):
    random_state = np.random.RandomState(1234)
    all_size = len(dataset)
    train_size = int(all_size * train_size)
    indices = np.arange(all_size)
    random_state.shuffle(indices)
    idx_train = indices[:train_size]
    idx_valid = indices[train_size:]
    train_dataset = Subset(dataset, idx_train)
    valid_dataset = Subset(dataset, idx_valid)
    return train_dataset, valid_dataset
