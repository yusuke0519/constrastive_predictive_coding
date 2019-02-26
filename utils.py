# # -*- coding: utf-8 -*-
"""Utility files."""
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


def split_dataset(dataset, train_size=0.9, shuffle=False, drop_first=True):
    """Split dataet into train and valid dataset.

    Parameter
    ---------
    train_size : float
        training dataset size
    shuffle : bool (default False)
        whether shuffle the dataset before partitioning (Note that, if it is True, then train valid has some overwrap)
    drop_first : bool (default False)
        wheter drop first dataset.T samples (Note that, it should be true to avoid dataset overrap)

    """
    all_size = len(dataset)
    train_size = int(all_size * train_size)
    indices = np.arange(all_size)
    if shuffle:
        random_state = np.random.RandomState(1234)
        random_state.shuffle(indices)
    idx_train = indices[:train_size]
    idx_valid = indices[train_size:]
    if drop_first:
        idx_valid = idx_valid[dataset.T:]
    train_dataset = Subset(dataset, idx_train)
    valid_dataset = Subset(dataset, idx_valid)
    return train_dataset, valid_dataset
