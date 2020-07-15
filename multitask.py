import os
import pdb
import pickle
import json
import logging
import time 

import numpy as np
from tqdm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from argparse import ArgumentParser
from sklearn.metrics import f1_score

from src.modules.trainer import OCRTrainer
from src.utils.utils import EarlyStopping, gmkdir, AverageMeter
from src.models.crnn import CRNN
from src.options.opts import base_opts
from src.data.ndli_dataset import NDLIDataset, NDLICollator
from torch.nn.utils.clip_grad import clip_grad_norm_

if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    
    args.lang = 'Telugu'
    args.imgdir = 'telugu'
    data_lang1 = NDLIDataset(args)
   
    args.lang = 'Tamil'
    args.imgdir = 'tamil'
    data_lang2 = NDLIDataset(args)

    args.lang = 'Hindi'
    args.imgdir = 'hindi'
    data_lang3 = NDLIDataset(args)

    args.collate_fn = NDLICollator()
    
    args.data = torch.utils.data.ConcatDataset([data_lang1, 
        data_lang2, data_lang3])

    train_split = int(0.9*len(args.data))
    val_split = len(args.data) - train_split
    args.data_train, args.data_val = random_split(args.data, (train_split, val_split))
    print('Traininig Data Size:{}\nVal Data Size:{}'.format(
        len(args.data_train), len(args.data_val)))
    from src.models.layers import *
    from src.utils.utils import LangConverter
    model = LangIdentifier(args).cuda()
    criterion = torch.nn.CrossEntropyLoss()
    savepath = os.path.join(args.save_dir, args.name)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    gmkdir(savepath)
    train_loader = torch.utils.data.DataLoader(args.data_train,
                shuffle=True,
                batch_size=args.batch_size,
                collate_fn=args.collate_fn)
    val_loader = torch.utils.data.DataLoader(args.data_val,
                batch_size = args.batch_size,
                collate_fn=args.collate_fn)
    
    converter = LangConverter()
    avgTrainLoss = AverageMeter("Train loss")
    avgTrainAccuracy = AverageMeter("Train Accuracy")
    avgTrainF1= AverageMeter("Train F1")
    def validation():
        losses, correct, f1 = [], [], []
        for i, batch in enumerate(val_loader):
            input_, targets = batch['img'].cuda(), batch['lang']
            label = converter.encode(targets)
            logits = model(input_)
            logits = logits.contiguous().cpu()
            loss = criterion(logits, label)
            _, prediction = logits.max(1)
            batch_correct = torch.sum(prediction==label).item()
            batch_f1 = f1_score(prediction, label, average='macro')
            losses.append(loss.item())
            correct.append(batch_correct)
            f1.append(batch_f1)
        correct = np.sum(correct)
        accuracy = correct/len(args.data_val)
        f1 = np.mean(f1)
        loss = np.mean(losses)
        return loss, accuracy, f1

    best_loss = np.Inf
    save_it = False
    pdb.set_trace()
    for epoch in range(args.epochs):
        accs, f1s, losses = [], [], []
        for i, batch in enumerate(tqdm(train_loader)):
            input_, targets = batch['img'].cuda(), batch['lang']
            bs = input_.size(0)
            labels = converter.encode(targets)
            logits = model(input_)
            logits = logits.contiguous().cpu()
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = 0.05
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            _, predictions = logits.max(1)
            correct = torch.sum(predictions==labels).item()
            f1 = f1_score(labels, predictions, average='macro')
            accuracy = (correct/bs)*100
            avgTrainLoss.add(loss.item())
            avgTrainAccuracy.add(accuracy)
            avgTrainF1.add(f1)


        mean_loss = avgTrainLoss.compute()
        mean_accuracy = avgTrainAccuracy.compute()
        mean_f1 = avgTrainF1.compute()
        print('Epoch: [%d]/[%d] Loss: %.4f Accuracy: %.4f F1: %.4f'%(epoch, args.epochs, mean_loss,
            mean_accuracy, mean_f1))
        val_loss, val_accuracy, val_f1 = validation()
        print('Val Loss: %.4f Val Accuracy: %.4f Val F1: %.4f'%(val_loss, val_accuracy, val_f1))
        if val_loss < best_loss:
            best_loss = val_loss
            save_it = True

        if save_it:
            print('Saving')
            save_filename = 'latest.pth'
            save_path = os.path.join(savepath, save_filename)
            torch.save(model.state_dict(), save_path)
            save_it = False
        else:
            print('No not saving')