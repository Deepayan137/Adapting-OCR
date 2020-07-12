
import os
import pdb
import pickle
import json
import logging
import time 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from argparse import ArgumentParser

from src.modules.trainer import OCRTrainer
from src.utils.utils import EarlyStopping, gmkdir
from src.models.crnn import CRNN
from src.options.opts import base_opts
# from src.data.synth_dataset import SynthDataset, SynthCollator
# from src.data.pickle_dataset import PickleDataset
from src.data.ndli_dataset import NDLIDataset, NDLICollator
from src.criterions.ctc import CustomCTCLoss 
from src.utils.top_sampler import SamplingTop

class Learner(object):
    def __init__(self, model, optimizer, savepath=None, resume=False):
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_name = os.path.join(savepath, 'best.ckpt')
        self.cuda = torch.cuda.is_available() 
        self.cuda_count = torch.cuda.device_count()
        if self.cuda:
            self.model = self.model.cuda()
        self.epoch = 0
        if self.cuda_count > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.best_score = None
        if resume and os.path.exists(self.checkpoint_name):
            self.checkpoint = torch.load(self.checkpoint_name)
            self.epoch = self.checkpoint['epoch']
            self.best_score=self.checkpoint['best']
            self.load()
        else:
            print('checkpoint does not exist')
        self.log_name = os.path.join(savepath, 'loss_log.csv')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
            log_file.write('epoch, train_loss, val_loss, train_ca, val_ca, train_wa, val_wa\n')


    def fit(self, opt):
        opt.cuda = self.cuda
        opt.model = self.model
        opt.optimizer = self.optimizer
        self.saver = EarlyStopping(self.checkpoint_name, patience=15, verbose=True, best_score=self.best_score)
        opt.epoch = self.epoch
        trainer = OCRTrainer(opt)
        
        for epoch in range(opt.epoch+1, opt.epochs):
            train_result = trainer.run_epoch()
            val_result = trainer.run_epoch(validation=True)
            trainer.count = epoch
            info = '%d, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f\n'%(epoch, train_result['train_loss'], 
                val_result['val_loss'], train_result['train_ca'],  val_result['val_ca'],
                train_result['train_wa'], val_result['val_wa'])
            with open(self.log_name, "a") as log_file:
                log_file.write(info)
            self.val_loss = val_result['val_loss']
            print(self.val_loss)
            if self.checkpoint_name:
                self.save(epoch)
            if self.saver.early_stop:
                print("Early stopping")
                break

    def load(self):
        print('Loading checkpoint at {} trained for {} epochs'.format(self.checkpoint_name, self.checkpoint['epoch']))
        self.model.load_state_dict(self.checkpoint['state_dict'])
        if 'opt_state_dict' in self.checkpoint.keys():
            print('Loading optimizer')
            self.optimizer.load_state_dict(self.checkpoint['opt_state_dict'])

    def save(self, epoch):
        self.saver(self.val_loss, epoch, self.model, self.optimizer)

    
        

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
    
    vocabfile = 'lookups/' + '%s.vocab.json'%args.lang
    with open(vocabfile, 'r') as f:
        vocab = json.load(f)
    args.alphabet = ''.join(list(vocab['v2i'].keys()))

    args.nClasses = len(args.alphabet)
    model = CRNN(args)
    args.criterion = CustomCTCLoss()
    savepath = os.path.join(args.save_dir, args.name)
    gmkdir(savepath)
    gmkdir(args.log_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.alpha = 0
    args.noise = False
    learner = Learner(model, optimizer, savepath=savepath, resume=args.resume)
    learner.fit(args)