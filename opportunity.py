# # -*- coding: utf-8 -*-
"""Network architecture of each cmponents for opportunity datasets."""

import torch
from torch import nn


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Encoder(nn.Module):
    """Correspond to g_enc in the CPC paper."""

    def __init__(self, input_shape, hidden_size=400, activation='relu'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_shape = input_shape
        linear_size = 20 * input_shape[1] * 2

        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'lrelu':
            activation = nn.LeakyReLU

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 50, kernel_size=(1, 5)),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(50, 40, kernel_size=(1, 5)),
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Conv2d(40, 20, kernel_size=(1, 3)),
            activation(),
            nn.Dropout(0.5),
            Flatten()
        )
        self.output_size = linear_size
        
        if self.hidden_size is not None:
            self.fc = nn.Sequential(
                nn.Linear(linear_size, self.hidden_size),
                activation(),
                nn.Dropout(0.5),
            )
            self.output_size = hidden_size

    def forward(self, input_data):
        feature = self.conv(input_data)
        if hasattr(self, "fc"):
            feature = self.fc(feature)
        return feature

    def output_shape(self):
        return (None, self.output_size)


class ContextEncoder(nn.Module):
    """An utoregressive models to emmbedding observations into a context vector.

    We use GRU here.
    Caution: in the original paper, they say, "The output of the GRU at every timestep is used as the context c",
    but this code only uses final output of the GRU.
    """

    def __init__(self, input_shape, hidden_size=200, num_layers=2):
        super(ContextEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = 200
        self.gru = nn.GRU(input_shape[1], hidden_size=self.hidden_size, num_layers=self.num_layers)

    def forward(self, X):
        h0 = torch.zeros(self.num_layers, X.shape[1], self.hidden_size).cuda()
        return self.gru(X, h0)


class Predictor(nn.Module):
    """Predict the k step forward future using a context vector c, and k dependent weight matrix."""

    def __init__(self, input_shape, hidden_size, max_steps):
        """."""
        super(Predictor, self).__init__()
        self.max_steps = max_steps
        self.linears = nn.ModuleList([nn.Linear(input_shape[1], hidden_size) for i in range(max_steps)])

    def forward(self, c, k):
        """Predict the k step forward future from the context vector c.

        Parameter
        ---------
        c : torch.Variable
            context vector
        k : int
            the number of forward steps, which is used to determine the weight matrix
        """
        return self.linears[k](c)
