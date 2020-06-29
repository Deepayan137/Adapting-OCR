import os
import pdb
import sys
import math
import pickle
import random
import numpy as np
import Levenshtein as lev
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random as rand
from copy import deepcopy
from tqdm import *

from src.utils.utils import Eval, OCRLabelConverter
from src.utils.lm import LM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SamplingTop(object):
    def __init__(self, args):
        self.model = args.model.to(device)
        self.percent = args.percent
        self.converter = OCRLabelConverter(args.alphabet)
        self.evaluator = Eval()
        self.batch_size = args.batch_size
        self.data = args.data_train
        self.collate_fn = args.collate_fn
        self.target_folder = os.path.join(args.path, args.imgdir)

    def _to_cpu(self, item):
        return item.detach().cpu().numpy()

    def get_loader(self, data, sampler=None):

        return torch.utils.data.DataLoader(self.data,
                batch_size=32,
                collate_fn=self.collate_fn,
                shuffle=False,
                sampler=sampler)


    def get_samples(self, train_on_pred=False, combine_scoring=False):
        loader = self.get_loader(self.data)
        n_samples = int(self.percent * len(self.data))
        dest_file = os.path.join(self.target_folder, 'English.data.pkl')
        units = self.forward(loader)
        # random.shuffle(units)
        units = sorted(units, key=lambda x: x[0], reverse=True)
        scores, images, predictions, truths = list(zip(*units))
        if combine_scoring: 
            pretrained_lm = LM()
            lambd = 0.635 # only for sentences not words
            print('LM scoring..... takes time')
            lm_scores = pretrained_lm.score_sentences(predictions)
            combine_scores = [lambd*scores[i] + (1-lambd)*lm_scores[i] for i in range(len(scores))] 
            zipped = list(zip(combine_scores, images, predictions))
            zipped = sorted(zipped, key=lambda x: x[0], reverse=True)
            _, images, predictions = list(zip(*zipped))

        if train_on_pred:
            curated  = list(zip(images[:n_samples], predictions[:n_samples]))
            wa = np.nanmean((list(map(self.evaluator.word_accuracy_line, 
            list(zip(predictions[:n_samples], truths[:n_samples]))))))
            print(wa)
        else:
            curated = list(zip(images[:n_samples], truths[:n_samples]))
        return self.write_toPickle(curated)


    def forward(self, loader):
        # self.model.eval()
        units, labels = [], []
        ind = []
        pbar = tqdm(loader, desc='Scoring', leave=True)
        for iteration, batch in enumerate(pbar):
            input_, targets = batch['img'].to(device), batch['label']
            images = input_.squeeze(1).detach().cpu().numpy()*255
            targets, lengths = self.converter.encode(targets)
            logits = self.model(input_).transpose(1, 0)
            logits = torch.nn.functional.log_softmax(logits, 2)
            logits = logits.contiguous().cpu()
            T, B, H = logits.size()
            pred_sizes = torch.LongTensor([T for i in range(B)])
            probs, pos = logits.max(2)
            pos = pos.transpose(1, 0).contiguous()
            preds = self.converter.decode(pos.view(-1).data, pred_sizes.data, raw=False)
            scores = self.evaluator._blanks(self._to_cpu(probs.transpose(1, 0)), 
                    self._to_cpu(pos))
            # scores = np.exp(np.sum(self._to_cpu(probs), 1))
            units.extend(list(zip(scores, images, preds, batch['label'])))
        units = [units[i] for i in range(len(units)) if len(units[i][2].split())>2]
        return units

    def write_toFolder(self, curated):
        for image, label in curated:
            im = Image.fromarray(image[0].astype(np.uint8))
            dest = os.path.join(self.target_folder, '%s.jpg'%label)
            im.save(dest)
        print('Confident samples written')
        return True

    def write_toPickle(self, curated):
        data = dict(train=curated)
        dest_file = os.path.join(self.target_folder, 'English.data.pkl')
        with open(dest_file, 'wb') as data_file:
            pickle.dump(data, data_file)