# # -*- coding: utf-8 -*-
"""Sacred wrapper for unsupervised representation learning."""
import random
from collections import OrderedDict
import pandas as pd
# sacred
from sacred import Experiment
from sacred import Ingredient
# pytorch
from torch.utils import data
import torch
from torch import nn
from torch import optim
# original
from datasets import OppG
from opportunity import Encoder, ContextEncoder, Predictor
from cpc import CPCModel
from utils import CheckCompleteOption


def get_dataset(name, validation, test_domain, L, K):
    """Prepare datasets for train, valid and test with configurations.

    Parameter
    ---------
    name : str
    validation : list
    test_domain : str or list
    L : int
    K : int
    """
    all_adls = OppG.get('all_adls')
    all_domain = OppG.get('all_domain_key')
    train_adls = list(set(all_adls) - set(validation))
    train_domain = list(set(all_domain) - set([test_domain]))
    train_dataset_joint = OppG(
        train_domain, l_sample=30, interval=15, T=K+L, adl_ids=train_adls)
    valid_dataset_joint = OppG(
        train_domain, l_sample=30, interval=15, T=K+L, adl_ids=validation)

    # marginal sample come from same datasets for simplicity
    # Same train-valid split with joint dataset
    train_dataset_marginal = OppG(
        train_domain, l_sample=30, interval=15, T=K, adl_ids=train_adls)
    valid_dataset_marginal = OppG(
        train_domain, l_sample=30, interval=15, T=K, adl_ids=validation)
    return train_dataset_joint, valid_dataset_joint, train_dataset_marginal, valid_dataset_marginal


def get_model(input_shape, K, name, hidden, context, num_gru):
    """Prepare cpc model for training.

    Parameter
    ---------
    TODO

    """
    g_enc = Encoder(input_shape=input_shape, hidden_size=hidden).cuda()
    c_enc = ContextEncoder(input_shape=g_enc.output_shape(), num_layers=num_gru, hidden_size=context).cuda()
    predictor = Predictor((None, c_enc.hidden_size), g_enc.output_shape()[1], max_steps=K).cuda()
    model = CPCModel(g_enc, c_enc, predictor).cuda()
    print(model)
    return model


data_ingredient = Ingredient('dataset')
data_ingredient.add_config({
    "name": 'oppG',
    'validation': ['ADL4', 'ADL5'],
    'test_domain': 'S1',
    'L': 12,
    'K': 5,
})


method_ingredient = Ingredient('method')
method_ingredient.add_config({
    'name': 'CPC',
    'hidden': 1600,
    'context': 200,
    'num_gru': 1,
})


optim_ingredient = Ingredient('optim')
optim_ingredient.add_config({
    'lr': 0.0001
})


ex = Experiment(ingredients=[data_ingredient, method_ingredient, optim_ingredient])
ex.add_config({
    'num_batch': 10000,
    'batch_size': 128,
    'db_name': 'CPC'
})


def train_CPC(joint_loader, marginal_loader, model, optimizer):
    """Train cpc model with a batch."""
    criterion = nn.BCELoss()

    X_j, _ = joint_loader.__iter__().__next__()
    X_m, _ = marginal_loader.__iter__().__next__()

    K = X_m.shape[-1]
    L = X_j.shape[-1] - K

    optimizer.zero_grad()
    X_j = X_j.float().cuda()
    X_m = X_m.float().cuda()
    score_j_list, score_m_list = model(X_j, X_m, L, K)
    loss = 0
    for score_j, score_m in zip(score_j_list, score_m_list):
        loss += criterion(score_j, torch.ones((len(score_j), 1)).cuda())
        loss += criterion(score_m, torch.zeros((len(score_j), 1)).cuda())
    loss = loss / (2*K)
    loss.backward()
    optimizer.step()
    return loss


def validate(dataset_joint, dataset_marginal, model, num_eval=10, batch_size=128):
    """Evaluate the model."""
    model.eval()

    loader_joint = data.DataLoader(dataset_joint, batch_size=batch_size, shuffle=True, drop_last=True)
    loader_marginal = data.DataLoader(dataset_marginal, batch_size=batch_size, shuffle=True, drop_last=True)

    if num_eval is None:
        num_eval = len(loader_joint)

    K = dataset_marginal.T
    L = dataset_joint.T - K
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


@ex.automain
def run(_config, _seed):
    """Train a model with configurations."""
    print(_config)
    monitor_per = 100
    random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    datasets = get_dataset(**_config['dataset'])
    train_dataset_joint, _, _, _ = datasets
    model = get_model(
        input_shape=train_dataset_joint.get('input_shape'), K=_config['dataset']['K'], **_config['method']
    )
    optimizer = optim.Adam(model.parameters(), **_config['optim'])
    train_dataset_joint, valid_dataset_joint, train_dataset_marginal, valid_dataset_marginal = datasets
    train_loader_joint = data.DataLoader(train_dataset_joint, batch_size=_config['batch_size'], shuffle=True)
    train_loader_marginal = data.DataLoader(train_dataset_marginal, batch_size=_config['batch_size'], shuffle=True)

    train_results = []
    valid_results = []

    for num_iter in range(_config['num_batch']):
        loss = train_CPC(train_loader_joint, train_loader_marginal, model, optimizer)
        if (num_iter+1) % monitor_per != 0:
            continue
        print(num_iter+1, loss.item())
        train_result = validate(train_dataset_joint, train_dataset_marginal, model, num_eval=None)
        train_results.append(train_result)
        print("  train CPC: ", train_result)
        valid_result = validate(valid_dataset_joint, valid_dataset_marginal, model, num_eval=None)
        valid_results.append(valid_result)
        print("  valid CPC: ", valid_result)
        # torch.save(model.state_dict(), '{}/{}-{}-{}.pth'.format(folder_name, L, K, num_iter+1))
    train_results = pd.DataFrame(train_results)
    valid_results = pd.DataFrame(valid_results)
    # train_results.to_csv('{}/{}-{}-train.csv'.format(folder_name, L, K))
    # valid_results.to_csv('{}/{}-{}-valid.csv'.format(folder_name, L, K))
