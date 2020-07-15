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
from tqdm import *
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
from src.utils.utils import AverageMeter, Eval, OCRLabelConverter
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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

    def get_accuracy(self, args):
        loader = torch.utils.data.DataLoader(args.data,
                    batch_size=args.batch_size,
                    collate_fn=args.collate_fn)
        model = args.model
        model.eval()
        converter = OCRLabelConverter(args.alphabet)
        evaluator = Eval()
        labels, predictions = [], []
        for iteration, batch in enumerate(tqdm(loader)):
            input_, targets = batch['img'].to(device), batch['label']
            labels.extend(targets)
            targets, lengths = converter.encode(targets)
            logits = model(input_).transpose(1, 0)
            logits = torch.nn.functional.log_softmax(logits, 2)
            logits = logits.contiguous().cpu()
            T, B, H = logits.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            probs, pos = logits.max(2)
            pos = pos.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(pos.data, pred_sizes.data, raw=False)
            predictions.extend(sim_preds)
        ca = np.mean((list(map(evaluator.char_accuracy, list(zip(predictions, labels))))))
        wa = np.nanmean((list(map(evaluator.word_accuracy_line, list(zip(predictions, labels))))))
        return ca, wa

if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    data = NDLIDataset(args)
    args.collate_fn = NDLICollator()
    train_split = int(0.9*len(data))
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
    args.alpha = 0
    args.noise = False
    savepath = os.path.join(args.save_dir, args.target_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    gmkdir(savepath)
    learner = LearnerFinetune(model, optimizer, 
            savepath=savepath, resume=args.resume)
    
    if args.mode == 'train':
        # layers = learner.get_layer_groups()
        # args.epochs = 2
        # args.schedule = False
        # learner.freeze_all_but(-1)
        # learner.unfreeze(-2)
        learner.fit(args)
        # for i in range(1,len(layers[:-2])):
        #     if i%3 != 0:
        #         print('Finetuning %d/%d'%(i, len(layers)))
        #         learner.unfreeze(i)
        #         learner.fit(args)
        # learner.unfreeze_all()
        args.epochs = 50
        args.schedule = True
        learner.fit(args)
    elif args.mode == 'test':
        args.data = NDLIDataset(args)
        args.collate_fn = NDLICollator()
        savepath = os.path.join(args.save_dir, args.target_name)
        resume_file = os.path.join(savepath, 'finetuned.ckpt')
        if os.path.isfile(resume_file):
            print('Loading model %s'%resume_file)
            checkpoint = torch.load(resume_file)
            model.load_state_dict(checkpoint['state_dict'])
            args.model = model
            ca, wa = learner.get_accuracy(args)
            print("Character Accuracy: %.2f\nWord Accuracy: %.2f"%(ca, wa))
        else:
            print("=> no checkpoint found at '{}'".format(resume_file))
            print('Exiting')
    


    # python -m finetune --source_name teluguocr --target_name tamilocr_frozen_wts --source_lang Telugu --target_lang Tamil --path /ssd_scratch/cvit/deep/data/tamil --imgdir train