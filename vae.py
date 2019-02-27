# # -*- coding: utf-8 -*-
"""Implementation of contrastive predictive coding (CPC).

References: TODO
"""
import os
import argparse
import pandas as pd

import torch
from torch import nn
import torch.utils.data as data
from torch import optim
from torch.nn import functional as F
from pixyz.distributions import Normal
from pixyz.losses import KullbackLeibler
from pixyz.models import VAE

from datasets import OppG
from opportunity import Encoder, ContextEncoder, Predictor
from cpc import CPCModel

from label_predict import Classifier, validate_label_prediction


def label_predict(datasets, L, K, g_enc_size, num_gru, pretrain, finetune_g, mode='VAE', num_batch=10000, monitor_per=100, iteration_at=10000):
    """Train label classifier.

    Parameter
    ---------
    num_batch : int
        the number of batch size to train (default 20000)
    monitor_per : int
        output the result per monitor_each iterations (default=100)

    """
    # Load dataset
    print("Load datasets ...")
    train_dataset_joint, valid_dataset_joint, test_dataset = datasets
    train_loader_joint = data.DataLoader(train_dataset_joint, batch_size=128, shuffle=True)

    if mode == 'CPC':
        folder_name = 'models/{}-{}/{}-{}'.format(g_enc_size, num_gru, L, K)
    elif mode == 'VAE':
        folder_name = 'VAE/{}/{}-{}'.format(g_enc_size, L, K)
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

    # parameter of label train

    print("Build a model (mode is {}) ... ".format(mode))
    if mode == 'CPC':
        # parameter for models
        context_size = g_enc_size / 2
        g_enc = Encoder(input_shape=train_dataset_joint.get('input_shape'), hidden_size=g_enc_size).cuda()
        c_enc = ContextEncoder(input_shape=g_enc.output_shape(), num_layers=num_gru, hidden_size=context_size).cuda()
        predictor = Predictor((None, c_enc.hidden_size), g_enc.output_shape()[1], max_steps=K).cuda()
        model = CPCModel(g_enc, c_enc, predictor).cuda()
        if pretrain:
            model.load_state_dict(torch.load('{}-{}.pth'.format(folder_name, 1000)))
    elif mode == 'VAE':
        g_enc = Encoder(input_shape=train_dataset_joint.get('input_shape'), hidden_size=None).cuda()
        q = Inference(g_enc, network_output=g_enc.output_shape()[1], z_size=g_enc_size).cuda()
        if pretrain:
            q.load_state_dict(torch.load('{}-{}-q.pth'.format(folder_name, 100)))
        g_enc = nn.Sequential(q.network, q.network_mu)
        g_enc.output_shape = lambda: (None, g_enc_size)  # dummy function, may be there exists a better way

    classifier = Classifier(
        num_classes=train_dataset_joint.get('num_classes'),
        g_enc=g_enc, finetune_g=finetune_g).cuda()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    train_results = []
    valid_results = []
    test_results = []

    print("Start training...")
    for num_iter in range(num_batch):
        optimizer.zero_grad()
        X, Y = train_loader_joint.__iter__().__next__()
        y = Y[:, 0, L-1].long().cuda()
        pred_y = classifier(X[..., :L].float().cuda())
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()

        if ((num_iter + 1) % monitor_per) != 0:
            continue
        print(num_iter+1)
        train_results.append(validate_label_prediction(classifier, train_dataset_joint, L=L, nb_batch=None))
        print(train_results[-1])
        valid_results.append(validate_label_prediction(classifier, valid_dataset_joint, L=L, nb_batch=None))
        print(valid_results[-1])
        test_results.append(validate_label_prediction(classifier, test_dataset, L=L, nb_batch=None))
        print(test_results[-1])
    folder_name = '{}/label_predict'.format(folder_name)
    os.makedirs(folder_name, exist_ok=True)
    pd.DataFrame(train_results).to_csv(os.path.join(folder_name, '{}-{}-train.csv'.format(pretrain, finetune_g)))
    pd.DataFrame(valid_results).to_csv(os.path.join(folder_name, '{}-{}-valid.csv'.format(pretrain, finetune_g)))
    pd.DataFrame(test_results).to_csv(os.path.join(folder_name, '{}-{}-test.csv'.format(pretrain, finetune_g)))


def validate(dataset, p, q, model, num_eval=None, batch_size=128):
    p.eval()
    q.eval()
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    if num_eval is None:
        num_eval = len(loader)

    loss = 0
    for x, _ in loader:
        loss += model.test({"x": x[..., 0].float().cuda()}).item()

    loss = loss * loader.batch_size / len(loader.dataset)
    # print('Test loss: {:.4f}'.format(loss))
    p.train()
    q.train()
    return {'loss': loss}


class Inference(Normal):
    def __init__(self, network, network_output, z_size):
        super().__init__(cond_var=["x"], var=["z"], name="q")

        self.network = network
        self.network_mu = nn.Linear(network_output, z_size)
        self.network_sigma = nn.Linear(network_output, z_size)

    def forward(self, x):
        h = self.network(x)
        return {"loc": self.network_mu(h), "scale": F.softplus(self.network_sigma(h))}


class Geneator(Normal):
    def __init__(self, z_size, g_enc):
        super().__init__(cond_var=["z"], var=["x"], name="p")
        self.fc = nn.Linear(z_size, g_enc.output_shape()[1]).cuda()
        self.deconv1 = nn.ConvTranspose2d(20, 40, kernel_size=(1, 3), stride=(1, 2))
        self.deconv2 = nn.ConvTranspose2d(40, 50, kernel_size=(1, 5), stride=(1, 2))
        self.deconv3 = nn.ConvTranspose2d(50, 1, kernel_size=(1, 5), stride=(1, 2), output_padding=(0, 1))

    def forward(self, z):
        h = self.fc(z)
        h = h.view(-1, 20, 113, 2)
        h = self.deconv1(h)
        h = self.deconv2(h)
        h = self.deconv3(h)
        return {"loc": h, "scale": torch.tensor(1.0).cuda()}


def vae(datasets, g_enc_size, K, L, folder_name, num_batch=10000):
    monitor_each = 100  # output the result per monitor_each iterationss

    train_dataset_joint, valid_dataset_joint, test_dataset = datasets
    train_loader_joint = data.DataLoader(train_dataset_joint, batch_size=128, shuffle=True)

    # Model parameters
    print("Prepare models ...")
    g_enc = Encoder(input_shape=train_dataset_joint.get('input_shape'), hidden_size=None).cuda()
    q = Inference(g_enc, network_output=g_enc.output_shape()[1], z_size=g_enc_size).cuda()
    p = Geneator(z_size=g_enc_size, g_enc=g_enc).cuda()

    # prior
    loc = torch.tensor(0.).cuda()
    scale = torch.tensor(1.).cuda()
    prior = Normal(loc=loc, scale=scale, var=["z"], dim=g_enc_size, name="p_prior")
    from torch import optim
    kl = KullbackLeibler(q, prior)
    model = VAE(q, p, regularizer=kl, optimizer=optim.Adam, optimizer_params={"lr": 0.001})
    print(model)

    # train
    train_results = []
    valid_results = []
    test_results = []

    for num_iter in range(num_batch):
        x, _ = train_loader_joint.__iter__().__next__()
        _ = model.train({"x": x[..., 0].float().cuda()})

        if (num_iter+1) % monitor_each != 0:
            continue
        print(num_iter+1)
        train_result = validate(train_dataset_joint, p, q, model, num_eval=None)
        train_results.append(train_result)
        print("  train VAE: ", train_result)
        valid_result = validate(valid_dataset_joint, p, q, model, num_eval=None)
        valid_results.append(valid_result)
        print("  valid VAE: ", valid_result)
        test_result = validate(test_dataset, p, q, model, num_eval=None)
        test_results.append(test_result)
        print("  test VAE: ", test_result)
        torch.save(p.state_dict(), '{}/{}-{}-{}-p.pth'.format(folder_name, L, K, num_iter+1))
        torch.save(q.state_dict(), '{}/{}-{}-{}-q.pth'.format(folder_name, L, K, num_iter+1))
    train_results = pd.DataFrame(train_results)
    valid_results = pd.DataFrame(valid_results)
    train_results.to_csv('{}/{}-{}-train.csv'.format(folder_name, L, K))
    valid_results.to_csv('{}/{}-{}-valid.csv'.format(folder_name, L, K))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-K', metavar='K', type=int, help='an integer for the accumulator')
    parser.add_argument('-L', metavar='L', type=int, help='an integer for the accumulator')
    parser.add_argument('--hidden', metavar='hidden', type=int, help='an integer for the accumulator')
    parser.add_argument('--gru', metavar='gru', type=int, help='an integer for the accumulator')
    parser.add_argument('--mode', metavar='mode', default='VAE', type=str, help='VAE or CPC')
    parser.add_argument('-N', metavar='N', type=int, default=10000, help='an integer for the accumulator')
    args = parser.parse_args()
    print(args)

    # Parameters
    K = args.K  # maximum prediction steps (sequence length of future sequences)
    L = args.L  # context size

    # parameter for models
    g_enc_size = args.hidden

    print("Load datasets ...")
    train_dataset_joint = OppG(
        'S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L, adl_ids=['Drill', 'ADL1', 'ADL2', 'ADL3'])
    valid_dataset_joint = OppG(
        'S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L, adl_ids=['ADL4', 'ADL5'])
    test_dataset = OppG('S1', 'Gestures', l_sample=30, interval=15, T=K+L)
    datasets = (train_dataset_joint, valid_dataset_joint, test_dataset)

    if args.mode == 'VAE':
        folder_name = 'VAE/{}'.format(g_enc_size)
        os.makedirs(folder_name, exist_ok=True)
        if not os.path.exists('{}/{}-{}-{}-q.pth'.format(folder_name, L, K, args.N)):
            vae(datasets, g_enc_size, K, L, folder_name, args.N)
    label_predict(
        datasets=datasets, L=L, K=K, g_enc_size=400, num_gru=2,
        pretrain=True, finetune_g=False, mode='VAE', iteration_at=args.N)
    label_predict(
        datasets=datasets, L=L, K=K, g_enc_size=400, num_gru=2,
        pretrain=True, finetune_g=True, mode='VAE', iteration_at=args.N)
    # # label_prediction
    # label_predict(L, K, g_enc_size, num_gru, True, False)  # CPC only
    # label_predict(L, K, g_enc_size, num_gru, True, True)  # CPC + Finetune
    # label_predict(L, K, g_enc_size, num_gru, False, True)  # Supervised
    # label_predict(L, K, g_enc_size, num_gru, False, False)  # Random feature
