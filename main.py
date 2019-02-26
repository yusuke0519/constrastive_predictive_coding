# # -*- coding: utf-8 -*-
"""Implementation of contrastive predictive coding (CPC).

References: TODO
"""
import os
import argparse
from datasets import OppG
from collections import OrderedDict
import pandas as pd


import torch
from torch import nn
import torch.utils.data as data
from torch.autograd import Variable
from torch import optim

from opportunity import Encoder, ContextEncoder, Predictor
from utils import split_dataset


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


def validate(dataset_joint, dataset_marginal, model, L, K, num_eval=10, batch_size=128):
    """Evaluate the model."""
    model.eval()

    loader_joint = data.DataLoader(dataset_joint, batch_size=batch_size, shuffle=False, drop_last=True)
    loader_marginal = data.DataLoader(dataset_marginal, batch_size=batch_size, shuffle=False, drop_last=True)

    if num_eval is None:
        num_eval = len(loader_joint)

    losses = [0] * K
    TP = [0] * K
    TN = [0] * K
    FP = [0] * K
    FN = [0] * K

    for i, ((X_j, _), (X_m, _)) in enumerate(zip(loader_joint, loader_marginal)):
        X_j = X_j.float().cuda()
        X_m = X_m.float().cuda()
        score_j_list, score_m_list = model(X_j, X_m, L, K)
        for k in range(K):
            losses[k] += (-1.0 * torch.log(score_j_list[k]).mean() - torch.log(1-score_m_list[k]).mean()).item()
            TP[k] += (score_j_list[k] > 0.5).sum()
            TN[k] += (score_m_list[k] < 0.5).sum()
            FP[k] += (score_m_list[k] > 0.5).sum()
            FN[k] += (score_j_list[k] < 0.5).sum()

        if i+1 == num_eval:
            break

    results = OrderedDict()

    for k in range(K):
        results['loss-{}'.format(k)] = losses[k] / (2*(i+1))
        results['accuracy-{}'.format(k)] = float(TP[k]+TN[k]) / float(FP[k]+FN[k]+TP[k]+TN[k])

    model.train()
    return results


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
    num_batch = 10000  # the number of batch size to train
    monitor_each = 100  # output the result per monitor_each iterations

    # parameter for models
    g_enc_size = args.hidden
    context_size = g_enc_size / 2
    num_gru = args.gru

    folder_name = 'models/{}-{}'.format(g_enc_size, num_gru)
    os.makedirs(folder_name, exist_ok=True)

    print("Load datasets ...")
    dataset_joint = OppG('S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L)
    train_dataset_joint, valid_dataset_joint = split_dataset(dataset_joint, shuffle=False, drop_first=True)
    train_loader_joint = data.DataLoader(dataset_joint, batch_size=128, shuffle=True)
    valid_loader_joint = data.DataLoader(dataset_joint, batch_size=128, shuffle=False)

    # marginal sample come from same datasets for simplicity
    # Same train-valid split with joint dataset
    dataset_marginal = OppG('S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K)
    train_dataset_marginal, valid_dataset_marginal = split_dataset(dataset_marginal, shuffle=False, drop_first=True)
    train_loader_marginal = data.DataLoader(dataset_marginal, batch_size=128, shuffle=True)
    valid_loader_marginal = data.DataLoader(dataset_marginal, batch_size=128, shuffle=False)

    # Model parameters
    print("Prepare models ...")
    # Model initialize
    g_enc = Encoder(input_shape=dataset_joint.get('input_shape'), hidden_size=g_enc_size).cuda()
    c_enc = ContextEncoder(input_shape=g_enc.output_shape(), num_layers=num_gru, hidden_size=context_size).cuda()
    predictor = Predictor((None, c_enc.hidden_size), g_enc.output_shape()[1], max_steps=K).cuda()
    model = CPCModel(g_enc, c_enc, predictor).cuda()
    print(model)

    # train
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.BCELoss()

    train_results = []
    valid_results = []

    for num_iter in range(num_batch):
        X_j, _ = train_loader_joint.__iter__().__next__()
        X_m, _ = train_loader_marginal.__iter__().__next__()

        optimizer.zero_grad()
        X_j = Variable(X_j.float()).cuda()
        X_m = Variable(X_m.float()).cuda()
        score_j_list, score_m_list = model(X_j, X_m, L, K)
        loss = 0
        for score_j, score_m in zip(score_j_list, score_m_list):
            loss += criterion(score_j, Variable(torch.ones((len(score_j), 1))).cuda())
            loss += criterion(score_m, Variable(torch.zeros((len(score_j), 1))).cuda())
        loss = loss / (2*K)
        loss.backward()
        optimizer.step()

        if (num_iter+1) % monitor_each != 0:
            continue
        print(num_iter+1, loss.item())
        train_result = validate(train_dataset_joint, train_dataset_marginal, model, L, K, num_eval=100)
        train_results.append(train_result)
        print("  train CPC: ", train_result)
        valid_result = validate(valid_dataset_joint, valid_dataset_marginal, model, L, K, num_eval=100)
        valid_results.append(valid_result)
        print("  valid CPC: ", valid_result)
        torch.save(model.state_dict(), '{}/{}-{}-{}.pth'.format(folder_name, L, K, num_iter+1))
    train_results = pd.DataFrame(train_results)
    valid_results = pd.DataFrame(valid_results)
    train_results.to_csv('{}/{}-{}-train.csv'.format(folder_name, L, K))
    valid_results.to_csv('{}/{}-{}-valid.csv'.format(folder_name, L, K))

    # label_prediction
    from label_predict import label_predict
    label_predict(L, K, g_enc_size, num_gru, True, False)  # CPC only
    label_predict(L, K, g_enc_size, num_gru, True, True)  # CPC + Finetune
    label_predict(L, K, g_enc_size, num_gru, False, True)  # Supervised
    label_predict(L, K, g_enc_size, num_gru, False, False)  # Random feature
