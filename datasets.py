# # -*- coding: utf-8 -*-
import os
import zipfile

import wget
import numpy as np
import pandas as pd
from sampling import sampling

from sklearn.preprocessing import StandardScaler
import torch.utils.data as data


# # -*- coding: utf-8 -*-
import torch


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class DomainDatasetBase(data.ConcatDataset):
    """ Base class for multi domain dataset class
    subclasses must have these class variables:
        all_domain_key: domain keys for the dataset
        SingleDataset: dataset class for one domain dataset
    """
    SingleDataset = None

    def __init__(self, domain_keys, require_domain=True, datasets=None):
        """ Base class for multi domain dataset class
        Args:
            domain_keys: list or str
            require_domain: Boolean
        """
        assert isinstance(domain_keys, list) or isinstance(domain_keys, str)
        if isinstance(domain_keys, list):
            self.domain_keys = domain_keys
        elif isinstance(domain_keys, str):
            self.domain_keys = [x for x in domain_keys.split(',')]
        self.require_domain = require_domain
        self.domain_dict = dict(zip(self.domain_keys, range(len(self.domain_keys))))

        if datasets is None:
            datasets = []
            for domain_key in self.domain_keys:
                datasets += [self.get_single_dataset(domain_key, **self.domain_specific_params())]
        super(DomainDatasetBase, self).__init__(datasets)

    def get_single_dataset(self, domain_key, **kwargs):
        return self.SingleDataset(domain_key, **kwargs)

    def domain_specific_params(self):
        return {}

    @classmethod
    def get_disjoint_domains(cls, domain_keys):
        if isinstance(domain_keys, str):
            domain_keys = [x for x in domain_keys.split(',')]

        all_domain_keys = cls.get('all_domain_key')[:]
        for domain_key in domain_keys:
            all_domain_keys.remove(domain_key)
        return all_domain_keys

    @classmethod
    def get(cls, name):
        if hasattr(cls, name):
            return getattr(cls, name)
        elif hasattr(cls.SingleDataset, name):
            return getattr(cls.SingleDataset, name)


CONFIG = {}
CONFIG['url'] = 'http://www.opportunity-project.eu/system/files/Challenge/OpportunityChallengeLabeled.zip'
CONFIG['out'] = os.path.expanduser('~/.torch/datasets/Opportunity')
CONFIG['column_names'] = 'challenge_column_names.txt'
CONFIG['label_legend'] = 'challenge_label_legend.txt'


class _SingleUserSingleADL(data.Dataset):
    path = CONFIG['out']
    all_domain_key = ['S1', 'S2', 'S3', 'S4']
    all_adls = ["ADL1", "ADL2", "ADL3", "ADL4", "ADL5", "Drill"]
    all_targets = ['Gestures', 'Locomotion']
    data_keys = ['Acce', 'Iner']
    input_shape = (1, 113, None)

    def __init__(self, domain_key, adl_id, target_key, l_sample, interval, T=1):
        assert domain_key in self.all_domain_key
        assert adl_id in self.all_adls
        assert target_key in self.all_targets

        if not os.path.exists(self.path):
            self.download()

        self.domain_key = domain_key
        self.adl_id = adl_id
        self.target_key = target_key
        self.interval = interval
        self.T = T

        col_names = load_col_names(self.path)

        file_format = os.path.join(self.path, "{which}-{adl_id}.dat")
        df = pd.read_csv(
            file_format.format(which=self.domain_key, adl_id=adl_id), sep=' ', header=None, index_col=1)

        df.columns = [col_names[1:]]
        tgt_columns = [
            x for x in col_names[1:]
            for y in self.data_keys + [target_key]
            if x.find(y) > 0]
        df = df[tgt_columns].fillna(method='ffill').fillna(method='bfill')  # Nullを補完
        self.df = df
        self.X = df.values[:, :-1]
        self.X = StandardScaler().fit_transform(self.X)

        # label
        self.label_dict = label2ids(self.path, target_key)
        y = df.values[:, -1]
        self.y = y

        X = sampling(self.X, 'sliding', dtype='np', l_sample=l_sample, interval=interval)
        X = X.swapaxes(1, 2)# [:, np.newaxis, :, :]
        y = sampling(self.y, 'sliding', dtype='np', l_sample=l_sample, interval=interval)

        from collections import Counter
        Y = []
        for _y in y:
            if _y.max() == 0:
                Y.append(len(self.label_dict))
            else:
                _y = _y[_y != 0]
                _y = Counter(_y).most_common()[0][0]
                _y = self.label_dict[_y]
                Y.append(_y)
        self.X = X
        self.Y = Y
        self.length = len(Y)

    def download(self):
        print("Downloading dataset (It takes a while)")
        wget.download(CONFIG['url'], out=self.path+'.zip', bar=None)
        zf = zipfile.ZipFile(self.path + '.zip', 'r')
        zf.extractall(path=self.path)
        zf.close()

        # Delete double space
        command = 'grep -rl "  " {}/*.dat | xargs sed -i -e "s/  / /g"'.format(self.path)
        os.system(command)

    def __getitem__(self, index):
        X = self.X[index:index+self.T][..., np.newaxis].swapaxes(0, -1)
        Y = np.array(self.Y[index:index+self.T]).reshape(1, -1)
        return X, Y  # , self.domain_key

    def __len__(self):
        return self.length - (self.T - 1)


class OppG(DomainDatasetBase):
    SingleDataset = _SingleUserSingleADL
    target = "Gestures"
    num_classes = 18

    def __init__(self, domain_keys, require_domain=True, datasets=None, l_sample=30, interval=15, T=1):
        self.interval = interval
        self.l_sample = l_sample
        self.T = T
        super(OppG, self).__init__(domain_keys, require_domain, datasets=datasets)

    def get_single_dataset(self, domain_key, **kwargs):
        datasets = []
        for adl_id in self.SingleDataset.all_adls:
            dataset = self.SingleDataset(
                domain_key=domain_key, adl_id=adl_id, target_key=self.target, **kwargs
            )
            datasets.append(dataset)
        return data.ConcatDataset(datasets)

    def domain_specific_params(self):
        return {'l_sample': self.l_sample, 'interval': self.interval, 'T': self.T}


def load_col_names(dirname):
    file_name = os.path.join(dirname, CONFIG['column_names'])
    allLines = open(file_name).read().replace('\r', '').split('\n')
    column_name = np.array(
        [x.split(':')[1].split(';')[0] for x in allLines if x.startswith('Column:')])
    return column_name


def label2ids(dirname, target_key):
    file_name = os.path.join(dirname, CONFIG['label_legend'])
    allLines = open(file_name).read().replace('\r', '') .split('\n')
    label_name = np.array([int(x.replace('\t', '  ').split('   -   ')[0]) for x in allLines if x.find(target_key) > 0])
    label_ids = range(len(label_name))
    return dict(zip(label_name, label_ids))
