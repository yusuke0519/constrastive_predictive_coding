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
# from utils import split_dataset
from cpc import CPCModel, get_context


# Should not be repeated with main.py
BASE_PATH = 'CPC/{}-{}-{}'


class Classifier(nn.Module):
    def __init__(self, num_classes, g_enc, c_enc=None, finetune_g=False, finetune_c=False, hiddens=None):
        super(Classifier, self).__init__()
        if hiddens is None:
            hiddens = []
        assert isinstance(hiddens, list), "variable hiddens must be a list type object"
        self.g_enc = g_enc
        self.c_enc = c_enc
        self.finetune_g = finetune_g
        self.finetune_c = finetune_c
        self.num_classes = num_classes

        if c_enc is None:
            input_size = g_enc.output_shape()[1]
        else:
            input_size = c_enc.hidden_size

        layers = []
        for i in hiddens:
            layers.append(nn.Linear(input_size, i))
            layers.append(nn.ReLU(True))
            layers.append(nn.Dropout(0.5))
            input_size = i
        layers.append(nn.Linear(input_size, num_classes))
        layers.append(nn.LogSoftmax(dim=-1))

        self.classifier = nn.Sequential(*layers)

    def forward(self, X):
        if not self.finetune_g:
            self.g_enc.eval()  # It is necesaryy to deactivate dropout

        if self.c_enc is None:
            z = self.g_enc(X[..., -1])
            y_pred = self.classifier(z)
        else:
            # TODO: c_enc should be eval mode if it will not retrain
            # becuse if it haves the droout, BN etc,, then it gives stochastic foward output
            # on train mode
            # if not self.finetune_c:
            #     self.c_enc.eval()
            c = get_context(X, self.g_enc, self.c_enc)
            y_pred = self.classifier(c)
            # self.c_enc.train()
        self.g_enc.train()
        return y_pred

    def parameters(self):
        parameters = list(self.classifier.parameters())
        if self.finetune_g:
            parameters += self.g_enc.parameters()
        if self.finetune_c:
            parameters += self.c_enc.parameters()
        return parameters


def validate_label_prediction(classifier, dataset, L, batch_size=128, nb_batch=None):
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    if nb_batch is None:
        nb_batch = len(loader)

    classifier.eval()
    ys = []
    pred_ys = []
    loss = 0
    criterion = nn.NLLLoss()
    for batch_idx, (X, Y) in enumerate(loader):
        y = Y[:, 0, L-1].long().cuda()
        pred_y = classifier(X[..., :L].float().cuda())
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

    precision = metrics.precision_score(y, pred_y, average=None)
    recall = metrics.recall_score(y, pred_y, average=None)
    for i, (pre, rec) in enumerate(zip(precision, recall)):
        result['precision-{}'.format(i)] = pre
        result['recall-{}'.format(i)] = rec
    return result


def label_predict(
        L, K, g_enc_size, context_size, num_gru, pretrain,  # parameters for model
        finetune_g, use_c_enc=False, finetune_c=False,
        num_batch=10000, iteration_at=10000):
    """Train label classifier.

    TODO: should be separate out the data loading function and classifier building function.
    This function should be focus on the training given classifier with given configurations
    (like datasets or hyperparameters). The name might be just a train() then.

    The ideal argments migth be similar to validate_label_prediction function, with additional
    arguments for hyperparameters.

    Parameter
    ---------
    pretrain : bool
        wheter use CPC pretrained model.
    finetune_g : bool
        wheter train g_encoder.
    use_c_func : bool (default False)
        wheter use context encoder (autoregressive encoder summarize several inputs).
    finetune_c : bool (default False)
        wheter train context encoder. It should be false if use_c_func is False, and True
        if use_c_func and finetune_g is both True.


    Example
    -------
    Case1: pretrain=True and finetune_g=False
    => Shallow classifier over unsupervisly learned representations
    Case2: pretrain=False and finetune_g=True
    => Fully supervised learning
    Case3: pretrain=True and finetune_g=True
    => Unsup+Sup learning
    Case4: pretrain=False and finetune_g=False
    => Baseline with random representations (to clarify the effect of CPC, not an architecture)
    """
    if not use_c_enc and finetune_c:
        print("Invalid combination of parameters")
        return None
    if use_c_enc and finetune_g and not finetune_c:
        print("Invalid combination of parameters", use_c_enc, finetune_g, finetune_c)
        return None

    # Load dataset
    # TODO: it should be easy to change the dataset
    all_adls = ['Drill', 'ADL1', 'ADL2', 'ADL3', 'ADL4', 'ADL5']
    valid_adls = ['ADL1', 'ADL2', 'ADL3', 'ADL4', 'ADL5']
    train_adls = list(set(all_adls) - set(valid_adls))
    dataset_name = '-'.join(valid_adls)
    print("Load datasets ...")
    train_dataset = OppG(
        'S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L, adl_ids=train_adls)
    valid_dataset = OppG(
        'S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L, adl_ids=valid_adls)
    train_loader_joint = data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_dataset = OppG('S1', 'Gestures', l_sample=30, interval=15, T=K+L)

    print("Train: {}, Valid: {}, Test: {}".format(
        len(train_dataset), len(valid_dataset), len(test_dataset))
    )

    folder_name = BASE_PATH.format(g_enc_size, context_size, num_gru)
    folder_name = '{}/{}-{}'.format(folder_name, L, K)
    # save_folder_name = '{}/{}-{}/{}'.format(folder_name, L, K, dataset_name)
    # print(save_folder_name)
    os.makedirs(folder_name, exist_ok=True)
    monitor_per = 100  # output the result per monitor_each iterations

    # parameter of label train
    g_enc = Encoder(input_shape=train_dataset.get('input_shape'), hidden_size=g_enc_size).cuda()
    c_enc = ContextEncoder(input_shape=g_enc.output_shape(), num_layers=num_gru, hidden_size=context_size).cuda()
    predictor = Predictor((None, c_enc.hidden_size), g_enc.output_shape()[1], max_steps=K).cuda()
    model = CPCModel(g_enc, c_enc, predictor).cuda()
    if pretrain:
        model.load_state_dict(torch.load('{}-{}.pth'.format(folder_name, iteration_at)))

    if use_c_enc:
        classifier = Classifier(
            num_classes=train_dataset.get('num_classes'),
            g_enc=g_enc, c_enc=c_enc, finetune_g=finetune_g, finetune_c=finetune_c).cuda()
    else:
        classifier = Classifier(
            num_classes=train_dataset.get('num_classes'),
            g_enc=g_enc, finetune_g=finetune_g).cuda()
    print(classifier)
    # optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    train_results = []
    valid_results = []
    test_results = []
    folder_name = '{}/{}/label_predict'.format(dataset_name, folder_name)
    os.makedirs(folder_name, exist_ok=True)
    import time
    start_time = time.time()
    for num_iter in range(num_batch):
        optimizer.zero_grad()
        X, Y = train_loader_joint.__iter__().__next__()
        y = Y[:, 0, L-1].long().cuda()
        print(y.shape)
        pred_y = classifier(X[..., :L].float().cuda())
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()
        print(time.time() - start_time)
        if ((num_iter + 1) % monitor_per) != 0:
            continue
        print(num_iter+1)
        train_results.append(validate_label_prediction(classifier, train_dataset, L=L, nb_batch=None))
        print('train', train_results[-1]['accuracy'], train_results[-1]['f1macro'])
        valid_results.append(validate_label_prediction(classifier, valid_dataset, L=L, nb_batch=None))
        print('valid', valid_results[-1]['accuracy'], valid_results[-1]['f1macro'])
        test_results.append(validate_label_prediction(classifier, test_dataset, L=L, nb_batch=None))
        print('test', test_results[-1]['accuracy'], test_results[-1]['f1macro'])
        pd.DataFrame(train_results).to_csv(
            os.path.join(folder_name, '{}-{}-{}-{}-train.csv'.format(pretrain, finetune_g, use_c_enc, finetune_c)))
        pd.DataFrame(valid_results).to_csv(
            os.path.join(folder_name, '{}-{}-{}-{}-valid.csv'.format(pretrain, finetune_g, use_c_enc, finetune_c)))
        pd.DataFrame(test_results).to_csv(
            os.path.join(folder_name, '{}-{}-{}-{}-test.csv'.format(pretrain, finetune_g, use_c_enc, finetune_c)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-K', metavar='K', type=int, help='an integer for the accumulator')
    parser.add_argument('-L', metavar='L', type=int, help='an integer for the accumulator')
    parser.add_argument('--hidden', metavar='hidden', type=int, help='an integer for the accumulator')
    parser.add_argument('--context', metavar='context', type=int, default=200, help='an integer for the accumulator')
    parser.add_argument('--gru', metavar='gru', type=int, help='an integer for the accumulator')
    # parser.add_argument('--pretrain', metavar='pretrain', type=int, help='an integer for the accumulator')
    args = parser.parse_args()
    print(args)

    # Parameters
    K = args.K  # maximum prediction steps (sequence length of future sequences)
    L = args.L  # context size

    # parameter for models
    g_enc_size = args.hidden
    num_gru = args.gru
    context_size = args.context

    label_predict(L, K, g_enc_size, context_size, num_gru, False, True, True)  # CPC only
    label_predict(L, K, g_enc_size, context_size, num_gru, True, True, True)  # CPC only
    label_predict(L, K, g_enc_size, context_size, num_gru, True, False, False, True)  # CPC only
    label_predict(L, K, g_enc_size, context_size, num_gru, True, False, True, True)  # CPC only
    # label_predict(L, K, g_enc_size, num_gru, True, False, True)  # CPC only
    # label_predict(L, K, g_enc_size, num_gru, True, False, True)  # CPC only
    # label_predict(L, K, g_enc_size, num_gru, False, True, True)  # CPC only
    # label_predict(L, K, g_enc_size, num_gru, True, True, True)  # CPC only
    # label_predict(L, K, g_enc_size, num_gru, True, False)  # CPC only
    # label_predict(L, K, g_enc_size, num_gru, True, True)  # CPC + Finetune
    # label_predict(L, K, g_enc_size, num_gru, False, True)  # Supervised
    # label_predict(L, K, g_enc_size, num_gru, False, False)  # Random feature
