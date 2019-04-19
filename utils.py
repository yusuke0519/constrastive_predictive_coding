# # -*- coding: utf-8 -*-
"""Utility files."""
from future.utils import iteritems

import itertools
from sacred.commandline_options import CommandLineOption
from sacred.observers import MongoObserver

import numpy as np
import torch.utils.data as data
from torch.utils.data import Sampler, SubsetRandomSampler
from torch._six import int_classes as _int_classes



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


# # -*- coding: utf-8 -*-
def flatten_dict(tgt_dict):
    """Flatten the nested dictionary.

    Parameter
    ---------
    tgt_dict : nested dictionary

    """
    flat_dict = {}
    for k, v in iteritems(tgt_dict):
        if isinstance(v, dict):
            for k2, v2 in iteritems(v):
                flat_dict["{}.{}".format(k, k2)] = v2
        else:
            flat_dict[k] = v
    return flat_dict


def param_iterator(param_dict):
    params = []
    for key, values in iteritems(param_dict):
        if isinstance(values, list):
            params.append((key, values))
        else:
            params.append((key, [values]))
    param_names, param_variations = zip(*params)
    for i, param_variation in enumerate(itertools.product(*param_variations)):
        yield dict(zip(param_names, param_variation))


# def params_to_str(params):
#     return os.path.join(*[str(param) for param in params])


class CheckCompleteOption(CommandLineOption):
    """Custom Documentation."""
    short_flag = 'x'
    arg = 'T'  # T/F (True/False)

    @classmethod
    def apply(cls, args, run):
        db_name = run.config['db_name']
        print(run.config)
        results = find_by_config(
            run.config, status=['COMPLETED', 'RUNNING'], db_name=db_name, seed=run.config['seed'], exact_search=True)
        results = list(results)
        nb_results = len(results)
        have_record = nb_results > 0
        print("CheckCompleteOption", nb_results, run.config['seed'])

        run.info['complete'] = have_record
        if not have_record:
            run.observers.append(MongoObserver.create(url='mongodb://localhost:27017', db_name=db_name))
        elif args == 'F':
            run.observers.append(MongoObserver.create(url='mongodb://localhost:27017', db_name=db_name))
            run.info['complete'] = False


def find_by_config(config, prefix='config', status='COMPLETED', url=None, db_name=None, seed=None, collection_name='runs', fields=None, exact_search=False):
    from pymongo import MongoClient
    if url is None:
        url = 'mongodb://localhost:27017'
    if db_name is None:
        db_name = 'TEST_DB'
    if fields is None:
        fields = {"config": 1, "info": 1}
    elif isinstance(fields, list):
        fields = dict(zip(fields, [1] * len(fields)))

    client = MongoClient(url)
    db = client[db_name]

    search_query = {}
    for k, v in iteritems(config):
        if not isinstance(v, dict):
            continue
        for k2, v2 in iteritems(v):
            if isinstance(v2, list) and not exact_search:
                search_query['{}.{}.{}'.format(prefix, k, k2)] = {'$in': v2}
            else:
                search_query['{}.{}.{}'.format(prefix, k, k2)] = v2

    if seed is not None:
        if isinstance(seed, int):
            search_query['{}.{}'.format(prefix, "seed")] = seed
        elif isinstance(seed, list):
            search_query['{}.{}'.format(prefix, "seed")] = {'$in': seed}
    print(search_query)
    if status is None:
        return db[collection_name].find(search_query, fields)

    if isinstance(status, str):
        status = [status]
    search_query['status'] = {'$in': status}
    return db[collection_name].find(search_query, fields)


def update_default_config(config_key, config_value, url=None, db_name=None):
    """Add new attribute to the target mongo collection.

    Warning
    -------
    * You need to use prefix 'config' to add the sacred attribute
    Example
    -------
    > update_default_config("config.option1", "default", db_name='db_name')
    """
    from pymongo import MongoClient
    if url is None:
        url = 'mongodb://localhost:27017'
    if db_name is None:
        db_name = 'TEST_DB'

    client = MongoClient(url)
    db = client[db_name]
    db.runs.update_many({config_key: {"$exists": False}}, {'$set': {config_key: config_value}})


class SplitBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        for _sampler in sampler:
            if not isinstance(_sampler, Sampler):
                raise ValueError("sampler should be an instance of "
                                 "torch.utils.data.Sampler, but got sampler={}"
                                 .format(_sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.num_sampler = len(sampler)
        self.which_sampler = [int(batch_size/self.num_sampler*(i+1)) for i in range(self.num_sampler)]
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        sampler_idx = 0
        
        for i in range(self.batch_size):
            idx = self.sampler[sampler_idx].__iter__().__next__()
            batch.append(idx)
            if len(batch) == self.which_sampler[sampler_idx]:
                sampler_idx += 1
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def get_split_samplers(dataset, ids):
    assert len(dataset.datasets) == len(ids)
    size = [0] + dataset.cummulative_sizes
    sampler = []
    for i in ids:
        sampler.append(SubsetRandomSampler(range(size[i], size[i+1])))
    return sampler
