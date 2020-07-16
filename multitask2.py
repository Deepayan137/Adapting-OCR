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
from itertools import cycle
from sklearn.metrics import f1_score
from collections import OrderedDict

from src.modules.trainer import OCRTrainer
from src.optim.optimizer import STLR
from src.utils.utils import AverageMeter, Eval, OCRLabelConverter, gmkdir
from src.models.crnn import CRNN
from src.options.opts import base_opts
from src.data.ndli_dataset import NDLIDataset, NDLICollator
from torch.nn.utils.clip_grad import clip_grad_norm_
from src.models.layers import MultiLangCRNN, MultiLangCRNN_ver2
from src.criterions.ctc import CustomCTCLoss 

if __name__ == '__main__':
    parser = ArgumentParser()
    base_opts(parser)
    args = parser.parse_args()
    
    args.lang = 'Telugu'
    args.imgdir = 'telugu/train'
    data_lang1 = NDLIDataset(args)
    
    vocabfile = 'lookups/' + '%s.vocab.json'%args.lang
    with open(vocabfile, 'r') as f:
        vocab = json.load(f)
    args.alphabet1 = ''.join(list(vocab['v2i'].keys()))
    args.nClasses_lang1 = len(args.alphabet1)

    args.lang = 'Tamil'
    args.imgdir = 'tamil/train'
    data_lang2 = NDLIDataset(args)
    vocabfile = 'lookups/' + '%s.vocab.json'%args.lang
    with open(vocabfile, 'r') as f:
        vocab = json.load(f)
    args.alphabet2 = ''.join(list(vocab['v2i'].keys()))
    args.nClasses_lang2 = len(args.alphabet2)

    train_split = int(0.9*len(data_lang2))
    val_split = len(data_lang2) - train_split
    data_lang2_train, data_lang2_val = random_split(data_lang2, (train_split, val_split))
    dataset_size = max(len(data_lang1), len(data_lang2_train))    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    args.collate_fn = NDLICollator()
    train_loader_lang1 = torch.utils.data.DataLoader(data_lang1,
                shuffle=True,
                batch_size=args.batch_size,
                collate_fn=args.collate_fn,
                num_workers=5)
    train_loader_lang2 = torch.utils.data.DataLoader(data_lang2_train,
                shuffle=True,
                batch_size=args.batch_size,
                collate_fn=args.collate_fn,
                num_workers=5)
    val_loader_lang2 = torch.utils.data.DataLoader(data_lang2_val,
                batch_size = args.batch_size,
                collate_fn=args.collate_fn)

    
    model = MultiLangCRNN_ver2(args).cuda()
    criterion = CustomCTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = STLR(optimizer, T_max=args.epochs)
    avgTrainLoss = AverageMeter("Train loss")
    avgTrainAccuracy = AverageMeter("Train Accuracy")
    avgValLoss = AverageMeter("Val Loss")
    avgValAccuracy = AverageMeter("Val Accuracy")
    converter1 = OCRLabelConverter(args.alphabet1)
    converter2 = OCRLabelConverter(args.alphabet2)
    evaluator = Eval()
    savepath = os.path.join(args.save_dir, args.name)
    gmkdir(savepath)
    log_name = os.path.join(savepath, 'loss_log.csv')
    with open(log_name, "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)
        log_file.write('epoch, train_loss, val_loss, train_ca, val_ca, train_wa, val_wa\n')
    best_loss = np.Inf
    saveit = False
    
    for epoch in range(args.epochs):
        if len(data_lang2_train) > len(data_lang1):
            zip_dataset = zip(cycle(train_loader_lang1), train_loader_lang2)
        else:
            zip_dataset = zip(train_loader_lang1, cycle(train_loader_lang2))
        pbar = tqdm(zip_dataset, desc='Training')

        for i, (batch_lang1, batch_lang2) in enumerate(pbar):
            input_lang1, targets_lang1 = batch_lang1['img'].cuda(), batch_lang1['label']
            targets_lang1, lengths_lang1 = converter1.encode(targets_lang1)

            input_lang2, targets_lang2 = batch_lang2['img'].cuda(), batch_lang2['label']
            targets_lang2, lengths_lang2 = converter2.encode(targets_lang2)

            logits1, logits2 = model(input_lang1, input_lang2)
            logits1, logits2 = logits1.contiguous().cpu(), logits2.contiguous().cpu()
            logits1, logits2 = torch.nn.functional.log_softmax(logits1, 2),\
                torch.nn.functional.log_softmax(logits2, 2)
            T1, B1, H1 = logits1.size()
            T2, B2, H2 = logits2.size()
            pred_sizes1 = torch.LongTensor([T1 for i in range(B1)])
            pred_sizes2 = torch.LongTensor([T2 for i in range(B2)])
            targets_lang1 = targets_lang1.view(-1).contiguous()
            targets_lang2 = targets_lang2.view(-1).contiguous()
            loss1 = criterion(logits1, targets_lang1, pred_sizes1, lengths_lang1)
            loss2 = criterion(logits2, targets_lang2, pred_sizes2, lengths_lang2)
            loss = (loss1 + loss2)/2 
            optimizer.zero_grad()
            loss.backward()
            max_grad_norm = 0.05
            clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            probs, preds = logits2.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter2.decode(preds.data, pred_sizes2.data, raw=False)
            ca = np.mean((list(map(evaluator.char_accuracy, list(zip(sim_preds, batch_lang2['label']))))))
            wa = np.mean((list(map(evaluator.word_accuracy_line, list(zip(sim_preds, batch_lang2['label']))))))
            avgTrainLoss.add(loss.item())
            avgTrainAccuracy.add(wa)
            output = OrderedDict({
            'loss': loss.item(),
            'train ca': ca,
            'train wa': wa
            })

            pbar.set_postfix(output)
            

        for i, batch in enumerate(tqdm(val_loader_lang2)):
            input_lang2, targets_lang2 = batch_lang2['img'].cuda(), batch_lang2['label']
            targets_lang2, lengths_lang2 = converter2.encode(targets_lang2)
            feat = model.feat_extract(input_lang2)
            logits = model.classify_lang2(feat)
            T2, B2, H2 = logits.size()
            pred_sizes2 = torch.LongTensor([T2 for i in range(B2)])
            loss = criterion(logits, targets_lang2, pred_sizes2, lengths_lang2)
            probs, preds = logits.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter2.decode(preds.data, pred_sizes2.data, raw=False)
            ca = np.mean((list(map(evaluator.char_accuracy, list(zip(sim_preds, batch_lang2['label']))))))
            wa = np.mean((list(map(evaluator.word_accuracy_line, list(zip(sim_preds, batch_lang2['label']))))))
            avgValLoss.add(loss.item())
            avgValAccuracy.add(wa)
        scheduler.step()
        mean_train_loss = avgTrainLoss.compute()
        mean_val_loss = avgValLoss.compute()
        mean_train_accuracy = avgTrainAccuracy.compute()
        mean_val_accuracy = avgValAccuracy.compute()
        info = '%d, %.4f, %.4f, %.4f, %.4f'%(epoch, mean_train_loss, mean_val_loss,
            mean_train_accuracy, mean_val_accuracy)
        with open(log_name, "a") as log_file:
                log_file.write(info)
        print('''Epoch: [%d]/[%d] Train Loss: %.4f Train Acc: %.4f\n
            Val Loss: %.4f Val Acc: %.4f'''%(epoch, args.epochs, mean_train_loss,
                mean_train_accuracy, mean_val_loss, mean_val_accuracy))
        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            save_it = True

        if save_it:
            print('Saving')
            save_filename = 'best.pth'
            save_path = os.path.join(savepath, save_filename)
            torch.save(model.state_dict(), save_path)
            save_it = False
        else:
            print('No not saving')


