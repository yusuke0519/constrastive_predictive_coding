# # -*- coding: utf-8 -*-
"""Main implementation of contrastive predictive coding."""

import torch
from torch import nn


def get_context(X, g_enc, c_enc):
    """Get a context form stepwise encoder and autoregressive encoder.

    Parameter
    ---------
    g_enc : step wise encoder
    c_enc : autoregressive encoder summalizing whole sequences
    """
    z_context = []
    nb_context = X.shape[-1]

    h = torch.zeros(X.shape[0], c_enc.hidden_size).cuda()
    for i in range(nb_context):
        z_context.append(g_enc(X[..., i]))

    o, h = c_enc(torch.stack(z_context))
    c = h[-1]
    return c


class CPCModel(nn.Module):
    """CPC container.

    Function
    --------
    forward : give a probability score for input sequence X

    """

    def __init__(self, g_enc, c_enc, predictor):
        """Initialize.

        Parameter
        ---------
        g_enc : step wise encoder
        c_enc : autoregressive encoder
        predictor : network to predict future z from a context c

        """
        super(CPCModel, self).__init__()
        self.g_enc = g_enc
        self.c_enc = c_enc
        self.predictor = predictor

    def get_context(self, X):
        return get_context(X, self.g_enc, self.c_enc)

    def get_score_of(self, X, K, predictions):
        score = [None] * K
        for i in range(K):
            z = self.g_enc(X[..., i])
            score[i] = torch.bmm(z.unsqueeze(1), predictions[i].unsqueeze(2)).squeeze(2)
        return score

    def get_predictions(self, c, K):
        predictions = [None] * K
        for i in range(K):
            predictions[i] = self.predictor(c, i)

        return predictions

    def forward(self, X_j, X_m, L, K):
        """Return probability that X comes from joint distributions.

        Parameter
        ---------
        X_j : float tensor
            Observations of the joint distributions p(x, c)
        X_m : float tensor
            Observations of the marginal distributions p(x)p(c)
        L : int
            Context size
        K : int
            Predict size
        """
        c = self.get_context(X_j[..., :L])
        predictions = self.get_predictions(c, K)
        score_j = self.get_score_of(X_j[..., L:], K, predictions)
        if isinstance(X_m, list):
            score_m = [None] * len(X_m)
            for i, X in enumerate(X_m):
                score_m[i] = self.get_score_of(X, K, predictions)
        else:
            score_m = self.get_score_of(X_m, K, predictions)

        return score_j, score_m
