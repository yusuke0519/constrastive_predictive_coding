# # -*- coding: utf-8 -*-
"""Utils to handle saved information in mongodb."""

from future.utils import iteritems

import pandas as pd
from pymongo import MongoClient


def expand_config(config):
    """Expand the config dict produced by sacred (#sacred).

    Parameters
    ----------
    config : dict
        a config dict we want to expand

    Returns
    -------
    s : dict
        a config dict which flatten the given dictionary

    """
    s = {}
    for k, v in iteritems(config):
        if isinstance(v, dict):
            for k2, v2 in iteritems(v):
                s["{}.{}".format(k, k2)] = v2
        else:
            s[k] = v
    for k, v in iteritems(s):
        if isinstance(v, list):
            s[k] = '-'.join([str(x) for x in v])
    return s


def expand_log(log):
    """Expand the log dictionary produced by sacred (#sacred).

    Parameters
    ----------
    log : dict
        a log dict we want to expand

    Returns
    -------
    s : dict
        a expanded dict for easy use

    """
    s = {}
    # expand the log dictionary (flatten the list in the dictionary)
    for k, v in iteritems(log):
        if isinstance(v, list):  # if one uses SummaryWriter for logging, it records a touple of (step, time, value)
            s[k] = [value[2] for value in v]
    return s


class SacredRecords(pd.DataFrame):
    """Wrap results produced by sacred (extended from pd.DataFrame)."""

    @classmethod
    def from_mongo(cls, query_dict, url=None, db_name=None, fields=None, exact_search=False, status="COMPLETED"):
        extractor = MongoExtractor(url, db_name)
        summary = []
        for result in extractor.find(query_dict, fields, exact_search, status):
            s = pd.Series(expand_config(result['config']))
            try:
                s2 = pd.Series(expand_log(result['info']))
                s = s.append(s2)
            except Exception:
                print("Error occured")
            summary.append(s)
        return cls(summary)

    def find_best_steps(self, valid_metric, valid_func, metric_names, skip_first=0):
        def _find_best(x):
            if isinstance(x[metric_name], list):
                return x[metric_name][valid_func(x[valid_metric][skip_first:])+skip_first]
            else:
                return None
        for metric_name in metric_names:
            self['best-{}'.format(metric_name)] = self.apply(lambda x: _find_best(x), axis=1)

    @property
    def _constructor(self):
        return SacredRecords


class MongoExtractor(MongoClient):
    """Retrieve results produced by sacred from mongo db.

    Attributes
    ----------
    COLLECTION_NAME : str
    PREFIX : str
    DEFAULT_FIELDS : dict
        it determin which fields would be included
    db : MongoDB.db
        target database

    """

    COLLECTION_NAME = 'runs'  # it is predifined by sacred
    PREFIX = 'config'  # it is predifined by sacred
    DEFAULT_FIELDS = {'config': 1, 'info': 1}  # retrieve 'config' and 'info' fields only

    def __init__(self, url=None, db_name=None):
        """Connect to the database.

        Parameters
        ----------
        url : str (default: mongodb://localhost:27017)
        db_name : str (default: TEST_DB)

        """
        if url is None:
            url = 'mongodb://localhost:27017'
        if db_name is None:
            db_name = 'TEST_DB'
        super(MongoExtractor, self).__init__(url)
        self.db = self[db_name]

    def find(self, query_dict, fields=None, exact_search=False, status="COMPLETED"):
        """Execute find query and return cursor.

        Parameters
        ----------
        config : dict
            nested dictionary ({'dataset': dataset_dict, 'optim': optim_dict, 'model': model_dict}, etc)
        fields : dict
        exact_search : bool (default: False)
        status : str (default: "COMPLETED")

        Returns
        -------
        cursor : MondoDB.Cursor object

        """
        if fields is None:
            fields = MongoExtractor.DEFAULT_FIELDS
        if isinstance(fields, list):
            fields = dict(zip(fields, [1] * len(fields)))

        # convert config dictionary to sacred format
        search_query = {}
        for k, v in iteritems(query_dict):
            if isinstance(v, list) and not exact_search:
                search_query['{}.{}'.format(MongoExtractor.PREFIX, k)] = {'$in': v}
            elif v is None:
                pass
            else:
                search_query['{}.{}'.format(MongoExtractor.PREFIX, k)] = v

        if status is None:  # return all results regardress the status
            return self.db[MongoExtractor.COLLECTION_NAME].find(search_query, fields)

        if isinstance(status, str):
            status = [status]
        search_query['status'] = {'$in': status}
        return self.db[MongoExtractor.COLLECTION_NAME].find(search_query, fields)
