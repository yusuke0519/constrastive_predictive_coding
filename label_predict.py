# # -*- coding: utf-8 -*-
"""Code for label prediction.

%TODO: missing docstring
"""

import os
import argparse
import numpy as np
import pandas as pd

from torch import nn
import torch.utils.data as data
from torch import optim
import torch
from collections import OrderedDict
from sklearn import metrics

from datasets import OppG
from opportunity import Encoder, ContextEncoder, Predictor
from utils import split_dataset
from cpc import CPCModel, get_context


class Classifier(nn.Module):
    def __init__(self, num_classes, g_enc, c_enc=None, finetune_g=False, finetune_c=False):
        super(Classifier, self).__init__()
        self.g_enc = g_enc
        self.c_enc = c_enc
        self.finetune_g = finetune_g
        self.finetune_c = finetune_c
        self.num_classes = num_classes

        if c_enc is None:
            input_size = g_enc.output_shape()[1]
        else:
            input_size = c_enc.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(input_size, num_classes),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, X):
        if self.c_enc is None:
            z = self.g_enc(X[..., 0])
            return self.classifier(z)
        else:
            c = get_context(X, self.g_enc, self.c_enc)
            return self.classifier(c)

    def parameters(self):
        parameters = list(self.classifier.parameters())
        if self.finetune_g:
            parameters += self.g_enc.parameters()
        if self.finetune_c:
            parameters += self.c_enc.parameters()
        return parameters


def validate_label_prediction(classifier, dataset, batch_size=128, nb_batch=None):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    if nb_batch is None:
        nb_batch = len(loader)

    classifier.eval()
    ys = []
    pred_ys = []
    loss = 0
    criterion = nn.NLLLoss()
    for batch_idx, (X, Y) in enumerate(loader):
        y = Y[:, 0, 0].long().cuda()
        pred_y = classifier(X.float().cuda())
        loss += criterion(pred_y, y).item()
        ys.append(y.cpu().numpy())
        pred_y = np.argmax(pred_y.detach().cpu().numpy(), axis=1)
        pred_ys.append(pred_y)

        if (batch_idx + 1) == nb_batch:
            break
    loss /= (batch_idx+1)

    y = np.concatenate(ys)
    pred_y = np.concatenate(pred_ys)

    classifier.train()
    result = OrderedDict()
    result['accuracy'] = metrics.accuracy_score(y, pred_y)
    result['f1macro'] = metrics.f1_score(y, pred_y, average='macro')
    result['loss'] = loss
    return result


def label_predict(L, K, g_enc_size, num_gru, pretrain, finetune_g):
    # Load dataset
    print("Load datasets ...")
    dataset_joint = OppG('S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L)
    train_dataset_joint, valid_dataset_joint = split_dataset(dataset_joint, shuffle=False, drop_first=True)
    train_loader_joint = data.DataLoader(dataset_joint, batch_size=128, shuffle=True)

    # Test dataset for label prediction
    test_dataset = OppG('S1', 'Gestures', l_sample=30, interval=15, T=K+L)

    folder_name = 'models/{}-{}/{}-{}'.format(g_enc_size, num_gru, L, K)
    os.makedirs(folder_name, exist_ok=True)
    # Parameters
    """
    Case1: pretrain=True and finetune_g=False
    => Shallow classifier over unsupervisly learned representations
    Case2: pretrain=False and finetune_g=True
    => Fully supervised learning
    Case3: pretrain=True and finetune_g=True
    => Unsup+Sup learning
    Case4: pretrain=False and finetune_g=False
    => Baseline with random representations (to clarify the effect of CPC, not an architecture)
    """
    num_batch = 20000  # the number of batch size to train
    monitor_per = 100  # output the result per monitor_each iterations

    # parameter for models
    context_size = g_enc_size / 2

    # parameter of label train
    g_enc = Encoder(input_shape=dataset_joint.get('input_shape'), hidden_size=g_enc_size).cuda()
    c_enc = ContextEncoder(input_shape=g_enc.output_shape(), num_layers=num_gru, hidden_size=context_size).cuda()
    predictor = Predictor((None, c_enc.hidden_size), g_enc.output_shape()[1], max_steps=K).cuda()
    model = CPCModel(g_enc, c_enc, predictor).cuda()
    if pretrain:
        model.load_state_dict(torch.load('{}-{}.pth'.format(folder_name, 10000)))

    classifier = Classifier(
        num_classes=dataset_joint.get('num_classes'),
        g_enc=g_enc, finetune_g=finetune_g).cuda()
    # optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    train_results = []
    valid_results = []
    test_results = []
    for num_iter in range(num_batch):
        optimizer.zero_grad()
        X, Y = train_loader_joint.__iter__().__next__()
        if classifier.c_enc is None:
            y = Y[:, 0, 0].long().cuda()
        else:
            y = Y[:, 0, L].long().cuda()
        pred_y = classifier(X[..., :L].float().cuda())
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()

        if ((num_iter + 1) % monitor_per) != 0:
            continue
        print(num_iter+1)
        train_results.append(validate_label_prediction(classifier, train_dataset_joint, nb_batch=100))
        print(train_results[-1])
        valid_results.append(validate_label_prediction(classifier, valid_dataset_joint, nb_batch=100))
        print(valid_results[-1])
        test_results.append(validate_label_prediction(classifier, test_dataset, nb_batch=100))
        print(test_results[-1])
    folder_name = '{}/label_predict'.format(folder_name)
    os.makedirs(folder_name, exist_ok=True)
    pd.DataFrame(train_results).to_csv(os.path.join(folder_name, '{}-{}-train.csv'.format(pretrain, finetune_g)))
    pd.DataFrame(valid_results).to_csv(os.path.join(folder_name, '{}-{}-valid.csv'.format(pretrain, finetune_g)))
    pd.DataFrame(test_results).to_csv(os.path.join(folder_name, '{}-{}-test.csv'.format(pretrain, finetune_g)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-K', metavar='K', type=int, help='an integer for the accumulator')
    parser.add_argument('-L', metavar='L', type=int, help='an integer for the accumulator')
    parser.add_argument('--hidden', metavar='hidden', type=int, help='an integer for the accumulator')
    parser.add_argument('--gru', metavar='gru', type=int, help='an integer for the accumulator')
    args = parser.parse_args()
    print(args)

    # Parameters
    K = args.K  # maximum prediction steps (sequence length of future sequences)
    L = args.L  # context size

    # parameter for models
    g_enc_size = args.hidden
    num_gru = args.gru

    label_predict(L, K, g_enc_size, num_gru, True, False)  # CPC only
    label_predict(L, K, g_enc_size, num_gru, True, True)  # CPC + Finetune
    label_predict(L, K, g_enc_size, num_gru, False, True)  # Supervised
    label_predict(L, K, g_enc_size, num_gru, False, False)  # Random feature
