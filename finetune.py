import os
import pdb
import pickle
import json
import logging
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import itertools 
from torch.utils.data import random_split

from src.modules.trainer import OCRTrainer
from src.utils.utils import EarlyStopping, gmkdir
from src.models.crnn import CRNN, PretrainedCRNN
from src.options.ft_opts import base_opts
from src.data.ndli_dataset import NDLIDataset, NDLICollator
from src.criterions.ctc import CustomCTCLoss 
from src.utils.top_sampler import SamplingTop
from main import Learner

class LearnerFinetune(Learner):
    def __init__(self, model, optimizer, savepath=None, resume=False):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_name = os.path.join(savepath, 'finetuned.ckpt')
        self.cuda = torch.cuda.is_available() 
        self.cuda_count = torch.cuda.device_count()
        if self.cuda:
            self.model = self.model.cuda()
        self.epoch = 0
        if self.cuda_count > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.best_score = None
        self.log_name = os.path.join(savepath, 'loss_log.csv')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
            log_file.write('epoch, train_loss, val_loss, train_ca, val_ca, train_wa, val_wa\n')

    def freeze(self, index, boolean=False):
        layer = self.get_layer_groups()[index]
        for params in layer.parameters():
            params.requires_grad = boolean

    def freeze_all_but(self, index):
        n_layers = len(self.get_layer_groups())
        for i in range(n_layers):
            self.freeze(i)
        self.freeze(index, boolean=True)

    def unfreeze(self, index):
        self.freeze(index, boolean=True)

    def unfreeze_all(self):
        n_layers = len(self.get_layer_groups())
        for i in range(n_layers):
            self.unfreeze(i)

    def child(self, x):
        return list(x.children())
    
    def recursive_(self, child):
        if hasattr(child, 'children'):
            if len(self.child(child)) != 0:
                child = self.child(child)
                return self.recursive_(child)
        return child

    def get_layer_groups(self):
        children = []
        for child in self.child(self.model):
            children.extend(self.recursive_(child))
        children = [child for child in children if list(child.parameters())]
        return children

if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    data = NDLIDataset(args)
    args.collate_fn = NDLICollator()
    train_split = int(0.8*len(data))
    val_split = len(data) - train_split
    args.data_train, args.data_val = random_split(data, (train_split, val_split))
    print('Traininig Data Size:{}\nVal Data Size:{}'.format(
        len(args.data_train), len(args.data_val)))
    
    vocabfile = 'lookups/' + '%s.vocab.json'%args.source_lang
    with open(vocabfile, 'r') as f:
        vocab = json.load(f)
    args.alphabet = ''.join(list(vocab['v2i'].keys()))
    args.source_nClasses = len(args.alphabet)
    
    vocabfile = 'lookups/' + '%s.vocab.json'%args.target_lang
    with open(vocabfile, 'r') as f:
        vocab = json.load(f)
    args.alphabet = ''.join(list(vocab['v2i'].keys()))
    args.target_nClasses = len(args.alphabet)

    model = PretrainedCRNN(args)
    
    args.criterion = CustomCTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.alpha = 0
    args.noise = False
    savepath = os.path.join(args.save_dir, args.target_name)
    gmkdir(savepath)
    learner = LearnerFinetune(model, optimizer, savepath=savepath, resume=args.resume)
    layers = learner.get_layer_groups()
    learner.freeze_all_but(-1)
    learner.unfreeze(-2)
    learner.fit(args)
    