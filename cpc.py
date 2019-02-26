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
        c = get_context(X_j[..., :L], self.g_enc, self.c_enc)
        score_j = [None] * K
        score_m = [None] * K
        for i in range(K):
            z_j = self.g_enc(X_j[..., L+i])
            z_m = self.g_enc(X_m[..., i])
            z_p = self.predictor(c, i)
            score_j[i] = torch.sigmoid(torch.bmm(z_j.unsqueeze(1), z_p.unsqueeze(2)).squeeze(2))
            score_m[i] = torch.sigmoid(torch.bmm(z_m.unsqueeze(1), z_p.unsqueeze(2)).squeeze(2))
        return score_j, score_m
