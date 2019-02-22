# # -*- coding: utf-8 -*-
import numpy as np
import argparse
from datasets import OppG
from collections import OrderedDict
import pandas as pd


import torch
from torch import nn
import torch.utils.data as data
from torch.autograd import Variable
from torch import optim


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


class Encoder(nn.Module):
    """
    Correspond to g_enc in the CPC paper
    """
    def __init__(self, input_shape, hidden_size=400, activation='relu'):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_shape = input_shape
        linear_size = 20 * input_shape[1] * 2

        if activation == 'relu':
            activation = nn.ReLU
        elif activation == 'lrelu':
            activation = nn.LeakyReLU
        
        self.feature = nn.Sequential(
            nn.Conv2d(input_shape[0], 50, kernel_size=(1, 5)), 
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Conv2d(50, 40, kernel_size=(1, 5)), 
            activation(),
            nn.MaxPool2d(kernel_size=(1, 2)), 
            nn.Conv2d(40, 20, kernel_size=(1, 3)), 
            activation(),
            nn.Dropout(0.5),
            Flatten(),
            nn.Linear(linear_size, self.hidden_size), 
            activation(),
            nn.Dropout(0.5),
        )

    def forward(self, input_data):
        feature = self.feature(input_data)
        return feature

    def output_shape(self):
        return (None, self.hidden_size)
    
    
class ContextEncoder(nn.Module):
    """
    Some autoregressive models to emmbedding observations into a context vector. We use GRU here. 
    
    Caution: in the original paper, they say, "The output of the GRU at every timestep is used as the context c", 
    but this code only uses final output of the GRU. 
    """
    
    def __init__(self, input_shape, hidden_size=200, num_layers=2):
        super(ContextEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = 200
        self.gru = nn.GRU(input_shape[1], hidden_size=self.hidden_size, num_layers=self.num_layers)
        
    def forward(self, X):
        h0 = Variable(torch.zeros(self.num_layers, X.shape[1], self.hidden_size)).cuda()
        return self.gru(X, h0)
    

class Predictor(nn.Module):
    """
    Predict the k step forward future using a context vector c, and k dependent weight matrix. 
    """
    def __init__(self, input_shape, hidden_size, max_steps):
        super(Predictor, self).__init__()
        self.max_steps = max_steps
        self.linears = nn.ModuleList([nn.Linear(input_shape[1], hidden_size) for i in range(max_steps)])
        
    def forward(self, c, k):
        """
        predict the k step forward future from the context vector c
        
        Parameter
        ---------
        c : torch.Variable
            context vector
        k : int
            the number of forward steps, which is used to determine the weight matrix 
        """
        
        return self.linears[k](c)
    

def get_context(X, g_enc, c_enc):
    z_context = []
    nb_context = X.shape[-1]

    h = Variable(torch.zeros(X.shape[0], c_enc.hidden_size).cuda(), requires_grad=False)
    for i in range(nb_context):
        z_context.append(g_enc(X[..., i]))

    o, h = c_enc(torch.stack(z_context))
    c = h[-1]
    return c


class CPCModel(nn.Module):
    def __init__(self, g_enc, c_enc, predictor):
        super(CPCModel, self).__init__()
        self.g_enc = g_enc
        self.c_enc = c_enc
        self.predictor = predictor
        
    def forward(self, X_j, X_m, L, K):
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
    

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
    

def split_dataset(dataset, train_size=0.9):
    random_state = np.random.RandomState(1234)
    all_size = len(dataset)
    train_size = int(all_size * train_size)
    indices = np.arange(all_size)
    random_state.shuffle(indices)
    idx_train = indices[:train_size]
    idx_valid = indices[train_size:]
    
    train_dataset = Subset(dataset, idx_train)
    valid_dataset = Subset(dataset, idx_valid)
    return train_dataset, valid_dataset


def validate(dataset_joint, dataset_marginal, model, L, K, num_eval=10, batch_size=128):
    """
    Evaluate the model
    """

    model.eval()
    criterion = nn.BCELoss()

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
        # results['TP-{}'.format(k)] = TP[k]
        # results['TN-{}'.format(k)] = TN[k]
        # results['FP-{}'.format(k)] = FP[k]
        # results['FN-{}'.format(k)] = FN[k]
        results['accuracy-{}'.format(k)] = float(TP[k]+TN[k]) / float(FP[k]+FN[k]+TP[k]+TN[k])

    model.train()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-K', metavar='K', type=int, help='an integer for the accumulator')
    parser.add_argument('-L', metavar='L', type=int, help='an integer for the accumulator')
    args = parser.parse_args()
    print(args)

    # Parameters
    K = args.K # maximum prediction steps (sequence length of future sequences)
    L = args.L # context size
    num_batch = 10000  # the number of batch size to train
    monitor_each = 100 # output the result per monitor_each iterations

    # parameter for models
    g_enc_size = 100
    context_size = 50
    num_gru = 2
    
    
    print("Load datasets ...")
    dataset_joint = OppG('S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K+L)
    train_dataset_joint, valid_dataset_joint = split_dataset(dataset_joint)
    train_loader_joint = data.DataLoader(dataset_joint, batch_size=128, shuffle=True)
    valid_loader_joint = data.DataLoader(dataset_joint, batch_size=128, shuffle=False)
    
    # marginal sample come from same datasets for simplicity
    # Same train-valid split with joint dataset
    dataset_marginal = OppG('S2,S3,S4', 'Gestures', l_sample=30, interval=15, T=K)
    train_dataset_marginal, valid_dataset_marginal = split_dataset(dataset_marginal)
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
    optimizer = optim.Adam(list(g_enc.parameters()) + list(c_enc.parameters()) + list(predictor.parameters()), lr=0.0001)
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
            loss += criterion(score_j, Variable(torch.ones((len(score_j), 1))).cuda()) + criterion(score_m, Variable(torch.zeros((len(score_j), 1))).cuda())
        loss = loss / (2*K)
        loss.backward()
        optimizer.step()
        
        if (num_iter+1) % monitor_each != 0:
            continue;
        print(num_iter+1, loss.item())
        train_result = validate(train_dataset_joint, train_dataset_marginal, model, L, K, num_eval=100)
        train_results.append(train_result)
        print("  train CPC: ", train_result)
        valid_result = validate(valid_dataset_joint, valid_dataset_marginal, model, L, K, num_eval=100)
        valid_results.append(valid_result)
        print("  valid CPC: ", valid_result)

        torch.save(model.state_dict(), 'models/{}-{}-{}.pth'.format(L, K, num_iter+1))
    train_results = pd.DataFrame(train_results)
    valid_results = pd.DataFrame(valid_results)
    train_results.to_csv('models/{}-{}-train.csv'.format(L, K))
    valid_results.to_csv('models/{}-{}-valid.csv'.format(L, K))

