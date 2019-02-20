# # -*- coding: utf-8 -*-

import numpy as np
import pandas as pd



def get_segment(x, start, stop):
    if isinstance(x, pd.DataFrame):
        return x.iloc[start:stop]
    if isinstance(x, np.ndarray):
        return x[start:stop, ...]
    raise Exception("length segment mode requires pd.Dataframe or np.ndarray, not {}.".format(type(x)))
    
    
def get_timesegment(x, start, stop):
    if isinstance(x, pd.DataFrame):
        return x.loc[start:stop]
    raise Exception("time segment mode requires pd.Dataframe, not {}.".format(type(x)))
    
    
def sliding(x, l_sample, interval):
    start = 0
    stop = start + l_sample
    next_segment = get_segment(x, start, stop)
    while next_segment.shape[0] == l_sample:
        yield next_segment
        start += interval
        stop += interval
        next_segment = get_segment(x, start, stop)


def sliding_many(x, l_sample, interval, T=1):
    """
    applying sliding window procedure to the data x
    
    Parameter
    ---------
    x : np.array
        sequences
    l_sample : int
        the length of each segments
    interval : int
        the overlap length between adjacent segments
    T : int
        the number of segments for each sample
    """
    start = 0
    stop = start + l_sample
    next_segment = get_segment(x, start, stop)
    segments = []
    segments.append(next_segment)
    while next_segment.shape[0] == l_sample:
        if len(segments) == T:
            yield segments
            segments = []
        start += interval
        stop += interval
        next_segment = get_segment(x, start, stop)
        segments.append(next_segment)

        
def clip_segment_between(x, start, l_sample):
    for t in start:
        yield get_segment(x, t, t+l_sample)
            
            
def clip_time_between(x, start, stop):
    assert len(start) == len(stop), "start and stop must have same length"
    
    for t1, t2 in zip(start, stop):
        yield get_timesegment(x, t1, t2)
        
        
def sampling(x, func, dtype='list', **kwargs):
    if isinstance(x, pd.DataFrame) & ((dtype == 'ndarray') | (dtype == 'np')):
        raise Exception("The combination of x {0} and dtype {1} is not supprted.".format(type(x), dtype))
    if isinstance(func, str):
        func = {
            'sliding': sliding,
            'clips': clip_segment_between,
            'clipt': clip_time_between}.get(func)
        assert func is not None
    X = list(func(x, **kwargs))
    if dtype == 'list':
        return X
    elif (dtype == 'ndarray') | (dtype == 'np'):
        return np.array(X)
    elif (dtype == 'panel') | (dtype == 'pd'):
        index = range(0, len(X))
        return pd.Panel(dict(zip(index, X)))
    else:
        raise Exception("The dtype {} is not supported".format(dtype))


if __name__ == '__main__':
    from datasets.extend_pandas import Accelerations
    df = Accelerations.from_params('sophia2012', route='route1', subject='goto', term='term1', sensor='undersheet')

    # example of sliding window
    X_list = sampling(df.values, sliding, dtype='list', l_sample=400, interval=200)
    X_array = sampling(df.values, sliding, dtype='np', l_sample=400, interval=200)

    # example of clip_time_between
    start = [0, 2, 4, 6, 8]
    stop = [1, 3, 10, 9, 10.2]
    X = sampling(df, clip_time_between, dtype='list', start=start, stop=stop)


