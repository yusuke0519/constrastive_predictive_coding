# # -*- coding: utf-8 -*-
from future.utils import iteritems
import os
import datetime
import time
from copy import deepcopy
import random
import numpy as np

import torch
from torch import nn
from torch import optim
from torch.utils import data
from tensorboardX import SummaryWriter

from label_predict import Classifier, validate_label_prediction
from run_sacred import get_model, get_dataset
from run_sacred import data_ingredient, method_ingredient, optim_ingredient, get_feature_of
from sacred import Experiment, Ingredient
from sacred_wrap import MongoExtractor

from utils import flatten_dict
from divergence import CMD, pairwise_divergence


classifier_ingredient = Ingredient('classifier')
classifier_ingredient.add_config({
    'pretrain': False,
    'finetune_g': False,
    'use_c_enc': False,
    'finetune_c': False,
})

classifier_optim_ingredient = Ingredient('classifier_optim')
classifier_optim_ingredient.add_config({
    'lr': 0.001,
    'num_batch': 10000,
    'batch_size': 128,
    'monitor_per': 100,
})


def get_classifier(model, num_classes, finetune_g, use_c_enc, finetune_c, **kwargs):
    if use_c_enc:
        classifier = Classifier(
            num_classes=num_classes,
            g_enc=model.g_enc, c_enc=model.c_enc, finetune_g=finetune_g, finetune_c=finetune_c).cuda()
    else:
        classifier = Classifier(
            num_classes=num_classes,
            g_enc=model.g_enc, finetune_g=finetune_g).cuda()
    return classifier


ex = Experiment(
    ingredients=[
        data_ingredient, method_ingredient, optim_ingredient,
        classifier_ingredient, classifier_optim_ingredient
    ])

ex.add_config({
    # NOTE: the arguments here will not used for CheckCompleteOption
    'db_name': 'CPC_DG',
    'unsup_db_name': 'CPC_test',
    'gpu': 0,
})


@ex.automain
def label_predict(_config, _seed, _run):
    """Train a model with configurations."""
    if ('complete' in _run.info) and (_run.info['complete']):
        print("The task is already finished")
        return None
    log_dir = os.path.join(
        "logs", _config['db_name'], datetime.datetime.now().strftime('%Y%m%d%H%M%S%f'))
    _run.info['log_dir'] = log_dir
    random.seed(_seed)
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)
    # torch.cuda.set_device(_config['gpu'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    writer = SummaryWriter(log_dir)

    # load datasets and build model
    train_dataset, valid_dataset, _, _, test_dataset = get_dataset(**_config['dataset'])
    train_loader = data.DataLoader(train_dataset, batch_size=_config['classifier_optim']['batch_size'], shuffle=True)
    print("Dataset: {} train, {} valid, {} test".format(
        len(train_dataset), len(valid_dataset), len(test_dataset)))
    model = get_model(input_shape=train_dataset.get('input_shape'), K=_config['dataset']['K'], **_config['method'])

    if _config['classifier']['pretrain']:
        query = deepcopy(_config)
        del query['classifier']
        del query['classifier_optim']
        del query['db_name']
        del query['gpu']
        del query['unsup_db_name']
        query = flatten_dict(query)
        extractor = MongoExtractor(None, _config['unsup_db_name'])
        # TODO: Should check whether the len(list) is one or not
        result = list(extractor.find(query, ['config', 'info'], False, 'COMPLETED'))
        assert len(result) == 1, "There are too many or no results. Please check the query {}".format(query)
        result = result[0]
        path = os.path.join(result['info']['log_dir'], 'model_{}.pth'.format(_config['optim']['num_batch']))
        model = model.cpu()
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model = model.cuda()
    classifier = get_classifier(model, train_dataset.get('num_classes'), **_config['classifier'])
    print(classifier)
    # TODO: Select valid poarameter from dictionary
    # MEMO: It can be donw with inspect module.
    optimizer = optim.Adam(classifier.parameters(), lr=_config['classifier_optim']['lr'])
    criterion = nn.NLLLoss()

    # Training and evaluation
    L = _config['dataset']['L']
    monitor_per = _config['classifier_optim']['monitor_per']
    divergence_criterion = CMD(n_moments=5)
    get_g_of = get_feature_of(model.g_enc, None, _config['dataset']['L'])
    get_c_of = get_feature_of(model.g_enc, model.c_enc, _config['dataset']['L'])

    start_time = time.time()
    for num_iter in range(_config['classifier_optim']['num_batch']):
        optimizer.zero_grad()
        X, Y = train_loader.__iter__().__next__()
        y = Y[:, 0, L-1].long().cuda()
        pred_y = classifier(X[..., :L].float().cuda())
        loss = criterion(pred_y, y)
        loss.backward()
        optimizer.step()
        if ((num_iter + 1) % monitor_per) != 0:
            continue
        print(time.time() - start_time)
        train_result = validate_label_prediction(classifier, train_dataset, L=L, nb_batch=None)
        valid_result = validate_label_prediction(classifier, valid_dataset, L=L, nb_batch=None)
        test_result = validate_label_prediction(classifier, test_dataset, L=L, nb_batch=None)
        elapsed_time_before_cmd = time.time() - start_time
        # train_result['cmdg'] = pairwise_divergence(
        #     train_dataset.datasets, get_g_of,
        #     divergence_criterion, num_batch=None, batch_size=1280
        # )
        # train_result['cmdc'] = pairwise_divergence(
        #     train_dataset.datasets, get_c_of,
        #     divergence_criterion, num_batch=None, batch_size=1280
        # )
        valid_result['cmdg'] = pairwise_divergence(
            valid_dataset.datasets, get_g_of,
            divergence_criterion, num_batch=None, batch_size=128
        )
        valid_result['cmdc'] = pairwise_divergence(
            valid_dataset.datasets, get_c_of,
            divergence_criterion, num_batch=None, batch_size=128
        )
        test_result['cmdg'] = pairwise_divergence(
            [valid_dataset, test_dataset], get_g_of,
            divergence_criterion, num_batch=None, batch_size=128
        )
        test_result['cmdc'] = pairwise_divergence(
            [valid_dataset, test_dataset], get_c_of,
            divergence_criterion, num_batch=None, batch_size=128
        )
        model_path = '{}/model_{}.pth'.format(log_dir, num_iter+1)
        torch.save(model.state_dict(), model_path)
        writer.add_scalars('train', train_result, num_iter+1)
        writer.add_scalars('valid', valid_result, num_iter+1)
        writer.add_scalars('test', test_result, num_iter+1)
        elapsed_time = time.time() - start_time
        print("-------- #iter: {}, elapad time: {} ({}) --------".format(
            num_iter+1, elapsed_time, elapsed_time_before_cmd))
        print('train', ', '.join(['{}:{:.3f}'.format(k, v) for k, v in iteritems(train_result)]))
        print('valid', ', '.join(['{}:{:.3f}'.format(k, v) for k, v in iteritems(valid_result)]))
        print('test', ', '.join(['{}:{:.3f}'.format(k, v) for k, v in iteritems(test_result)]))
        start_time = time.time()

    # Post process
    result = writer.scalar_dict
    for k, v in iteritems(result):
        ks = k.split('/')
        _run.info['{}-{}'.format(ks[-2], ks[-1])] = v
