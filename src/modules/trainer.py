import os
import logging
import numpy as np
from collections import OrderedDict
from argparse import ArgumentParser
from tqdm import *

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.nn.utils.clip_grad import clip_grad_norm_

from itertools import chain
from src.utils.utils import AverageMeter, Eval, OCRLabelConverter
from src.optim.optimizer import STLR
from src.utils.utils import gaussian

class OCRTrainer(object):
    def __init__(self, opt):
        super(OCRTrainer, self).__init__()
        self.data_train = opt.data_train
        self.data_val = opt.data_val
        self.model = opt.model
        self.criterion = opt.criterion
        self.optimizer = opt.optimizer
        self.schedule = opt.schedule
        self.alpha = opt.alpha
        self.converter = OCRLabelConverter(opt.alphabet)
        self.evaluator = Eval()
        print('Scheduling is {}'.format(self.schedule))
        self.scheduler = STLR(self.optimizer, T_max=opt.epochs)
        self.batch_size = opt.batch_size
        self.count = opt.epoch
        self.epochs = opt.epochs
        self.cuda = opt.cuda
        self.collate_fn = opt.collate_fn
        self.noise = opt.noise
        self.init_meters()

    def init_meters(self):
        self.avgTrainLoss = AverageMeter("Train loss")
        self.avgTrainCharAccuracy = AverageMeter("Train Character Accuracy")
        self.avgTrainWordAccuracy = AverageMeter("Train Word Accuracy")
        self.avgValLoss = AverageMeter("Validation loss")
        self.avgValCharAccuracy = AverageMeter("Validation Character Accuracy")
        self.avgValWordAccuracy = AverageMeter("Validation Word Accuracy")

    def forward(self, x):
        logits = self.model(x)
        return logits.transpose(1, 0)

    def loss(self, logits, targets, pred_sizes, target_sizes):
        loss = self.criterion(logits, targets, pred_sizes, target_sizes)
        return loss

    def step(self):
        self.max_grad_norm = 0.05
        clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
    
    def schedule_lr(self):
        if self.schedule:
            self.scheduler.step()

    def mixup_data(self, x, y, lengths, alpha):
        y = self.evaluator.format_target(y, lengths)
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, [y[i] for i in index]
        lengths_b = torch.LongTensor([lengths[i] for i in index])
        y_a, y_b = torch.LongTensor(torch.LongTensor(list(chain((*y_a))))), \
        torch.LongTensor(torch.LongTensor(list(chain((*y_b)))))
        return mixed_x, y_a, y_b, lengths, lengths_b, lam

    def mixup_criterion(self, logits, y_a, y_b, l_a, l_b, pred_sizes, lam):
        return lam * self.loss(logits, y_a, pred_sizes, l_a) + \
        (1 - lam) * self.loss(logits, y_b, pred_sizes, l_b)

    def _run_batch(self, batch, report_accuracy=False, validation=False):
        input_, targets = batch['img'].cuda(), batch['label']
        targets, lengths = self.converter.encode(targets)
        if not validation:
            if self.noise:
                input_ = gaussian(input_)
            input_, targets_a, targets_b, lengths_a, lengths_b, lam = self.mixup_data(input_, targets, 
                lengths, self.alpha)
        else:
            input_, targets_a, targets_b, lengths_a, lengths_b, lam = self.mixup_data(input_, targets, 
                lengths, 0)
        logits = self.forward(input_)
        logits = logits.contiguous().cpu()
        logits = torch.nn.functional.log_softmax(logits, 2)
        T, B, H = logits.size()
        pred_sizes = torch.LongTensor([T for i in range(B)])
        targets_a = targets_a.view(-1).contiguous()
        targets_b = targets_b.view(-1).contiguous()
        loss = self.mixup_criterion(logits, targets_a, targets_b, lengths_a, lengths_b, pred_sizes, lam)
        if report_accuracy:
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = self.converter.decode(preds.data, pred_sizes.data, raw=False)
            ca = np.mean((list(map(self.evaluator.char_accuracy, list(zip(sim_preds, batch['label']))))))
            wa = np.mean((list(map(self.evaluator.word_accuracy, list(zip(sim_preds, batch['label']))))))
        return loss, ca, wa

    def run_epoch(self, validation=False):
        if not validation:
            loader = self.train_dataloader()
            pbar = tqdm(loader, desc='Epoch: [%d]/[%d] Training'%(self.count, 
                self.epochs), leave=True)
            self.model.train()
        else:
            loader = self.val_dataloader()
            pbar = tqdm(loader, desc='Validating', leave=True)
            self.model.eval()
        outputs = []
        for batch_nb, batch in enumerate(pbar):
            if not validation:
                output = self.training_step(batch)
            else:
                output = self.validation_step(batch)
            # pbar.set_description('%.2f'%output['loss'].item())
            pbar.set_postfix(output)
            outputs.append(output)
        # self.count+=1
        self.schedule_lr()
        if not validation:
            result = self.train_end(outputs)
        else:
            result = self.validation_end(outputs)
        return result

    def training_step(self, batch):
        loss, ca, wa = self._run_batch(batch, report_accuracy=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.step()
        output = OrderedDict({
            'loss': abs(loss.item()),
            'train_ca': ca.item(),
            'train_wa': wa.item()
            })
        return output

    def validation_step(self, batch):
        loss, ca, wa = self._run_batch(batch, report_accuracy=True, validation=True)
        output = OrderedDict({
            'val_loss': abs(loss.item()),
            'val_ca': ca.item(),
            'val_wa': wa.item()
            })
        return output

    def train_dataloader(self):
        # logging.info('training data loader called')
        loader = torch.utils.data.DataLoader(self.data_train,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                shuffle=True,
                num_workers=5)
        return loader
        
    def val_dataloader(self):
        # logging.info('val data loader called')
        loader = torch.utils.data.DataLoader(self.data_val,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                num_workers=5)
        return loader

    def train_end(self, outputs):
        for output in outputs:
            self.avgTrainLoss.add(output['loss'])
            self.avgTrainCharAccuracy.add(output['train_ca'])
            self.avgTrainWordAccuracy.add(output['train_wa'])

        train_loss_mean = abs(self.avgTrainLoss.compute())
        train_ca_mean = self.avgTrainCharAccuracy.compute()
        train_wa_mean = self.avgTrainWordAccuracy.compute()

        result = {'train_loss': train_loss_mean, 'train_ca': train_ca_mean,
        'train_wa': train_wa_mean}
        # result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': train_loss_mean}
        return result

    def validation_end(self, outputs):
        for output in outputs:
            self.avgValLoss.add(output['val_loss'])
            self.avgValCharAccuracy.add(output['val_ca'])
            self.avgValWordAccuracy.add(output['val_wa'])

        val_loss_mean = abs(self.avgValLoss.compute())
        val_ca_mean = self.avgValCharAccuracy.compute()
        val_wa_mean = self.avgValWordAccuracy.compute()

        result = {'val_loss': val_loss_mean, 'val_ca': val_ca_mean,
        'val_wa': val_wa_mean}
        # result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': val_loss_mean}
        return result


