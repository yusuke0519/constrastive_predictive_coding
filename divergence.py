# # -*- coding: utf-8 -*-
import itertools
from torch.utils import data


def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    return ((x1-x2)**2).sum().sqrt()


def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    ss1 = (sx1**k).mean(0)
    ss2 = (sx2**k).mean(0)
    return l2diff(ss1, ss2)


class CMD(object):
    def __init__(self, n_moments=5):
        self.n_moments = n_moments

    def __call__(self, x1, x2):
        mx1 = x1.mean(dim=0)
        mx2 = x2.mean(dim=0)
        sx1 = x1 - mx1
        sx2 = x2 - mx2
        dm = l2diff(mx1, mx2)
        scms = dm

        for i in range(self.n_moments-1):
            # moment diff of centralized samples
            scms += moment_diff(sx1, sx2, i+2)
        return scms


def pairwise_divergence(datasets, func, criterion, batch_size=128, num_batch=None):
    divergence = 0
    num_total_batch = 0
    if num_batch is None:
        num_batch = max([len(dataset) for dataset in datasets])

    for dataset1, dataset2 in itertools.combinations(datasets, 2):
        loader1 = data.DataLoader(dataset1, shuffle=False, batch_size=batch_size)
        loader2 = data.DataLoader(dataset2, shuffle=False, batch_size=batch_size)

        for num_iter, ((X1, Y1), (X2, Y2)) in enumerate(zip(loader1, loader2)):
            divergence += criterion(func(X1.float().cuda()), func(X2.float().cuda())).item()
            if ((num_iter+1) % num_batch) == 0:
                break
        num_total_batch += (num_iter+1)
    return divergence/num_total_batch
