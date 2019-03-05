# # -*- coding: utf-8 -*-
"""Implementation of contrastive predictive coding (CPC).

References: TODO
"""
import os
import argparse
from collections import OrderedDict
import pandas as pd

import torch
from torch import nn
import torch.utils.data as data
from torch.autograd import Variable
from torch import optim

from datasets import OppG
from opportunity import Encoder, ContextEncoder, Predictor
# from utils import split_dataset
from cpc import CPCModel
from label_predict import label_predict


def validate(dataset_joint, dataset_marginal, model, L, K, num_eval=10, batch_size=128):
    """Evaluate the model."""
    model.eval()

    loader_joint = data.DataLoader(dataset_joint, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_marginal = data.DataLoader(dataset_marginal, batch_size=batch_size, shuffle=True, drop_last=True)

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


BASE_PATH = 'CPC/{}-{}-{}'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-K', metavar='K', type=int, help='an integer for the accumulator')
    parser.add_argument('-L', metavar='L', type=int, help='an integer for the accumulator')
    parser.add_argument('--hidden', metavar='hidden', type=int, help='an integer for the accumulator')
    parser.add_argument('--context', metavar='context', type=int, default=200, help='an integer for the accumulator')
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
    context_size = args.context
    num_gru = args.gru

    folder_name = BASE_PATH.format(g_enc_size, context_size, num_gru)
    os.makedirs(folder_name, exist_ok=True)
    if not os.path.exists('{}/{}-{}-{}.pth'.format(folder_name, L, K, num_batch)):
        print("CPC training ...")
        print("Load datasets ...")
        train_dataset_joint = OppG(
            'S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L, adl_ids=['Drill', 'ADL1', 'ADL2', 'ADL3'])
        valid_dataset_joint = OppG(
            'S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L, adl_ids=['ADL4', 'ADL5'])
        train_loader_joint = data.DataLoader(train_dataset_joint, batch_size=128, shuffle=True)

        # marginal sample come from same datasets for simplicity
        # Same train-valid split with joint dataset
        train_dataset_marginal = OppG(
            'S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K, adl_ids=['Drill', 'ADL1', 'ADL2', 'ADL3'])
        valid_dataset_marginal = OppG(
            'S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K, adl_ids=['ADL4', 'ADL5'])
        train_loader_marginal = data.DataLoader(train_dataset_marginal, batch_size=128, shuffle=True)

        # Model parameters
        print("Prepare models ...")
        # Model initialize
        g_enc = Encoder(input_shape=train_dataset_joint.get('input_shape'), hidden_size=g_enc_size).cuda()
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
            train_result = validate(train_dataset_joint, train_dataset_marginal, model, L, K, num_eval=None)
            train_results.append(train_result)
            print("  train CPC: ", train_result)
            valid_result = validate(valid_dataset_joint, valid_dataset_marginal, model, L, K, num_eval=None)
            valid_results.append(valid_result)
            print("  valid CPC: ", valid_result)
            torch.save(model.state_dict(), '{}/{}-{}-{}.pth'.format(folder_name, L, K, num_iter+1))
        train_results = pd.DataFrame(train_results)
        valid_results = pd.DataFrame(valid_results)
        train_results.to_csv('{}/{}-{}-train.csv'.format(folder_name, L, K))
        valid_results.to_csv('{}/{}-{}-valid.csv'.format(folder_name, L, K))

    # # label_prediction
    print("Train label classifier ...")
    label_predict(L, K, g_enc_size, context_size, num_gru, True, False)  # CPC only
    label_predict(L, K, g_enc_size, context_size, num_gru, True, True)  # CPC + Finetune
    label_predict(L, K, g_enc_size, context_size, num_gru, False, True)  # Supervised
    label_predict(L, K, g_enc_size, context_size, num_gru, False, False)  # Random feature
    label_predict(L, K, g_enc_size, context_size, num_gru, True, False, True)  # Finetune context embedding
    label_predict(L, K, g_enc_size, context_size, num_gru, False, True, True, True)  # Finetune context embedding
    label_predict(L, K, g_enc_size, context_size, num_gru, True, True, True, True)  # Finetune context embedding
    label_predict(L, K, g_enc_size, context_size, num_gru, True, False, True, True)  # Finetune context embedding
