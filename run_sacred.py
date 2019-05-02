# # -*- coding: utf-8 -*-
"""Sacred wrapper for unsupervised representation learning."""
from future.utils import iteritems
import os
import datetime
import random
from collections import OrderedDict
import numpy as np
from tensorboardX import SummaryWriter
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
from cpc import CPCModel, get_context
from utils import CheckCompleteOption  # TODO: Find a way to avoid this import
from utils import get_split_samplers, SplitBatchSampler
from divergence import CMD, pairwise_divergence
from vae import Inference, Generator, Normal, KullbackLeibler, VAE
from vae import validate as validate_vae


def get_dataset(name, validation, test_domain, L, K):
    """Prepare datasets for train, valid and test with configurations.

    Parameter
    ---------
    name : str
    validation : str or list
    test_domain : str or list
    L : int
    K : int

    Return
    ------
    {train/valid}_dataset_{joint/marginal} : torch.Dataset
        {train/valid} datasets from {joint/marginal} distributions
    """
    if isinstance(validation, str):
        validation = validation.split('-')
    all_adls = OppG.get('all_adls')
    all_domain = OppG.get('all_domain_key')
    train_adls = sorted(list(set(all_adls) - set(validation)))
    train_domain = sorted(list(set(all_domain) - set([test_domain])))
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
    test_dataset = OppG(test_domain, l_sample=30, interval=15, T=K+L)
    return train_dataset_joint, valid_dataset_joint, train_dataset_marginal, valid_dataset_marginal, test_dataset


def get_model(input_shape, K, name, hidden, context, num_gru, **kwargs):
    """Prepare cpc model for training.

    Parameter
    ---------
    TODO

    """
    g_enc = Encoder(input_shape=input_shape, hidden_size=hidden).cuda()
    c_enc = ContextEncoder(input_shape=g_enc.output_shape(), num_layers=num_gru, hidden_size=context).cuda()
    predictor = Predictor((None, c_enc.hidden_size), g_enc.output_shape()[1], max_steps=K).cuda()
    model = CPCModel(g_enc, c_enc, predictor, p=kwargs['mask_size'], num_mask=kwargs['num_mask']).cuda()
    return model


def verify_dataset(config, command_name, logger):
    """Add assersion rule for datasets."""
    REGISTERD_DATASETS = ['oppG']
    assert config['dataset']['name'] in REGISTERD_DATASETS, "Invalid dataset name {}".format(
            config['dataset']['name'])
    return config


def verify_method(config, command_name, logger):
    """Add assersion rules."""
    REGISTERD_METHOD = ['CPC', 'VAE']
    assert config['method']['name'] in REGISTERD_METHOD, "Invalid method name {}".format(
            config['method']['name'])

    if config['method']['name'] == 'CPC':
        REGISTERED_PARAM = {
            'sampler_mode': ['random', 'diff', 'same'],
        }
        for key, valid_list in iteritems(REGISTERED_PARAM):
            assert config['method'][key] in valid_list, "Invalid {} {}".format(key, config['method'][key])
    return config


data_ingredient = Ingredient('dataset')
data_ingredient.add_config({
    "name": 'oppG',
    'validation': 'ADL4-ADL5',
    'test_domain': 'S1',
    'L': 12,
    'K': 5,
})
data_ingredient.config_hook(verify_dataset)

method_ingredient = Ingredient('method')
method_ingredient.add_config({
    'name': 'CPC',
    'hidden': 1600,
    'context': 800,
    'num_gru': 1,
    'sampler_mode': 'random',
    'num_negative': 1,
    'cont_type': 'sigmoid',
    'mask_size': 1.0,
    'num_mask': 1
})


optim_ingredient = Ingredient('optim')
optim_ingredient.add_config({
    'lr': 0.0001,
    'num_batch': 10000,
    'batch_size': 128,
})

invariance_ingredient = Ingredient('invariance')
invariance_ingredient.add_config({

})


ex = Experiment(ingredients=[data_ingredient, method_ingredient, optim_ingredient])
ex.add_config({
    # NOTE: the arguments here will not used for CheckCompleteOption
    'db_name': 'CPC_test',
    'gpu': 0,
})


def train_CPC(joint_loader, marginal_loader, model, optimizer, num_negative):
    """Train cpc model with a batch."""
    criterion = nn.BCELoss()

    X_j, _ = joint_loader.__iter__().__next__()
    X_j = X_j.float().cuda()
    optimizer.zero_grad()
    if num_negative == 1:
        X_m, _ = marginal_loader.__iter__().__next__()

        K = X_m.shape[-1]
        L = X_j.shape[-1] - K

        X_m = X_m.float().cuda()
        score_j_list, score_m_list = model(X_j, X_m, L, K)
        loss = 0
        for score_j, score_m in zip(score_j_list, score_m_list):
            loss += criterion(torch.sigmoid(score_j), torch.ones((len(score_j), 1)).cuda())
            loss += criterion(torch.sigmoid(score_m), torch.zeros((len(score_j), 1)).cuda())
    else:
        X_m = [None] * num_negative
        for i in range(num_negative):
            X, _ = marginal_loader.__iter__().__next__()
            X_m[i] = X.float().cuda()
        K = X_m[0].shape[-1]
        L = X_j[0].shape[-1] - K
        score_j_list, score_m_list = model(X_j, X_m, L, K)
        loss = 0
        for k in range(K):
            score_j = score_j_list[k]
            loss += criterion(torch.sigmoid(score_j), torch.ones((len(score_j), 1)).cuda())
            for m in range(num_negative):
                score_m = score_m_list[m][k]
                loss += criterion(torch.sigmoid(score_m), torch.zeros((len(score_j), 1)).cuda())

    loss = loss / (2*K)
    loss.backward()
    optimizer.step()
    return loss


def validate(dataset_joint, dataset_marginal, model, num_eval=10, batch_size=64):
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
            losses[k] += (-1.0 * torch.log(torch.clamp(torch.sigmoid(score_j_list[k]), min=1e-8)).mean()
                          - torch.log(torch.clamp(1-torch.sigmoid(score_m_list[k]), min=1e-8)).mean()).item()
            TP[k] += (score_j_list[k] > 0.5).sum().item()
            TN[k] += (score_m_list[k] < 0.5).sum().item()
            FP[k] += (score_m_list[k] > 0.5).sum().item()
            FN[k] += (score_j_list[k] < 0.5).sum().item()

        if i+1 == num_eval:
            break
    results = OrderedDict()
    for k in range(K):
        results['loss-{}'.format(k)] = losses[k] / (2*(i+1))
        results['accuracy-{}'.format(k)] = float(TP[k]+TN[k]) / float(FP[k]+FN[k]+TP[k]+TN[k])

    model.train()
    return results


def get_feature_of(g_enc, c_enc, L):
    def _func(X):
        if c_enc is None:
            return g_enc(X[..., L-1])
        else:
            return get_context(X[..., :(L-1)], g_enc, c_enc)
    return _func


@ex.automain
def CPC(_config, _seed, _run):
    """Train a model with configurations."""
    if ('complete' in _run.info) and (_run.info['complete']):
        print("The task is already finished")
        return None
    log_dir = os.path.join(
        "logs", _config['db_name'], datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
    _run.info['log_dir'] = log_dir
    writer = SummaryWriter(log_dir)

    monitor_per = 100
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    torch.cuda.set_device(_config['gpu'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Prepare datasets
    datasets = get_dataset(**_config['dataset'])
    train_dataset_joint, valid_dataset_joint, train_dataset_marginal, valid_dataset_marginal, test_dataset = datasets
    if _config['method']['sampler_mode'] == 'random':
        print("Sample mode: Random")
        train_loader_joint = data.DataLoader(
            train_dataset_joint, batch_size=_config['optim']['batch_size'], shuffle=True)
        train_loader_marginal = data.DataLoader(
            train_dataset_marginal, batch_size=_config['optim']['batch_size'], shuffle=True)
    elif _config['method']['sampler_mode'] == 'diff':
        joint_sampler = get_split_samplers(train_dataset_joint, [0, 1, 2])
        joint_batch_sampler = SplitBatchSampler(joint_sampler, _config['optim']['batch_size'], True)
        train_loader_joint = data.DataLoader(train_dataset_joint, batch_sampler=joint_batch_sampler)

        marginal_sampler = get_split_samplers(train_dataset_marginal, [1, 2, 0])
        marginal_batch_sampler = SplitBatchSampler(marginal_sampler, _config['optim']['batch_size'], True)
        train_loader_marginal = data.DataLoader(train_dataset_marginal, batch_sampler=marginal_batch_sampler)

    elif _config['method']['sampler_mode'] == 'same':
        joint_sampler = get_split_samplers(train_dataset_joint, [0, 1, 2])
        joint_batch_sampler = SplitBatchSampler(joint_sampler, _config['optim']['batch_size'], True)
        train_loader_joint = data.DataLoader(train_dataset_joint, batch_sampler=joint_batch_sampler)

        marginal_sampler = get_split_samplers(train_dataset_marginal, [0, 1, 2])
        marginal_batch_sampler = SplitBatchSampler(marginal_sampler, _config['optim']['batch_size'], True)
        train_loader_marginal = data.DataLoader(train_dataset_marginal, batch_sampler=marginal_batch_sampler)
    else:
        raise Exception()
    if _config['method']['name'] == 'CPC':
        model = get_model(
            input_shape=train_dataset_joint.get('input_shape'), K=_config['dataset']['K'], **_config['method']
        )
        # TODO: enable to select various optimization options
        # optimizer = optim.Adam(model.parameters(), **_config['optim'])
        optimizer = optim.Adam(model.parameters(), lr=_config['optim']['lr'])
        print("----- Model information -----")
        print(model)
        print(optimizer)
        print("----- ------")

        divergence_criterion = CMD(n_moments=5)
        get_g_of = get_feature_of(model.g_enc, None, _config['dataset']['L'])
        get_c_of = get_feature_of(model.g_enc, model.c_enc, _config['dataset']['L'])
        for num_iter in range(_config['optim']['num_batch']):
            loss = train_CPC(
                train_loader_joint, train_loader_marginal, model, optimizer, _config['method']['num_negative'])
            if (num_iter+1) % monitor_per != 0:
                continue
            print(num_iter+1, loss.item())
            train_result = validate(train_dataset_joint, train_dataset_marginal, model, num_eval=None)
            train_result['cmdg'] = pairwise_divergence(
                train_dataset_joint.datasets, get_g_of,
                divergence_criterion
            )
            train_result['cmdc'] = pairwise_divergence(
                train_dataset_joint.datasets, get_c_of,
                divergence_criterion
            )
            print("  train CPC: ", train_result)
            valid_result = validate(valid_dataset_joint, valid_dataset_marginal, model, num_eval=None)
            valid_result['cmdg'] = pairwise_divergence(
                valid_dataset_joint.datasets, get_g_of,
                divergence_criterion
            )
            valid_result['cmdc'] = pairwise_divergence(
                valid_dataset_joint.datasets, get_c_of,
                divergence_criterion
            )
            print("  valid CPC: ", valid_result)
            writer.add_scalars('train', train_result, num_iter+1)
            writer.add_scalars('valid', valid_result, num_iter+1)

            # NOTE: I decided to desable add artifact feature for the moment
            # Instead, one can retrieve the model information by simply load in accordance with info['log_dir']
            # ex.add_artifact(model_path, name='model_{}'.format(num_iter+1))
        model_path = '{}/model_{}.pth'.format(log_dir, num_iter+1)
        torch.save(model.state_dict(), model_path)
    elif _config['method']['name'] == 'VAE':
        # Model parameters
        print("Prepare models ...")
        g_enc_size = _config['method']['hidden']
        g_enc = Encoder(input_shape=train_dataset_joint.get('input_shape'), hidden_size=None).cuda()
        q = Inference(g_enc, network_output=g_enc.output_shape()[1], z_size=g_enc_size).cuda()
        p = Generator(z_size=g_enc_size, g_enc=g_enc).cuda()

        # prior
        loc = torch.tensor(0.).cuda()
        scale = torch.tensor(1.).cuda()
        prior = Normal(loc=loc, scale=scale, var=["z"], dim=g_enc_size, name="p_prior")
        kl = KullbackLeibler(q, prior)
        model = VAE(q, p, regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr": _config['optim']['lr']})
        print(model)

        # train

        for num_iter in range(_config['optim']['num_batch']):
            x, _ = train_loader_joint.__iter__().__next__()
            _ = model.train({"x": x[..., 0].float().cuda()})

            if (num_iter+1) % monitor_per != 0:
                continue
            print(num_iter+1)
            train_result = validate_vae(train_dataset_joint, p, q, model, num_eval=None)
            print("  train VAE: ", train_result)
            valid_result = validate_vae(valid_dataset_joint, p, q, model, num_eval=None)
            print("  valid VAE: ", valid_result)
            test_result = validate_vae(test_dataset, p, q, model, num_eval=None)
            print("  test VAE: ", test_result)
            writer.add_scalars('train', train_result, num_iter+1)
            writer.add_scalars('valid', valid_result, num_iter+1)
            writer.add_scalars('test', test_result, num_iter+1)

        torch.save(p.state_dict(), '{}/p_{}.pth'.format(log_dir, num_iter+1))
        torch.save(q.state_dict(), '{}/q_{}.pth'.format(log_dir, num_iter+1))
    else:
        raise Exception()

    result = writer.scalar_dict
    for k, v in iteritems(result):
        ks = k.split('/')
        _run.info['{}-{}'.format(ks[-2], ks[-1])] = v


"""Memo.

#TODO
* (Pending) Convert results we already done by main.py.
* Enable to use various dataset.
* Add VAE option for comparison.
"""
