import torch
import numpy as np
from collections import namedtuple
import pdb
from functools import wraps
from time import time as _timenow
from sys import stderr
import os
import pickle
import math
from PIL import Image
from operator import eq
import cv2
import random
import logging
import json
import math
from warnings import warn
import re
import string
from textdistance import levenshtein as lev

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def similarity(word1, word2):
    return lev.normalized_distance(word1, word2)

def gmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def corrupt(x):
    if random.random() > 0.5:
        noise = np.random.binomial(1, 1.0 - 0.2, size=x.size())
        result = x.clone()
        result  *= noise
        return result
    return x

def gaussian(images):
    if random.random() > 0.5:
        mean, var = 0, 0.1
        stddev = var**2
        noise = images.data.new(images.size()).normal_(mean, stddev)
        return images + noise
    return images

def time(f):
    @wraps(f)
    def _wrapped(*args, **kwargs):
        start = _timenow()
        result = f(*args, **kwargs)
        end = _timenow()
        print('[time] {}: {}'.format(f.__name__, end - start),
              file=stderr)
        return result

    return _wrapped


def split(samples, **kwargs):
    total = len(samples)
    indices = list(range(total))
    if kwargs['random']:
        np.random.shuffle(indices)
    percent = kwargs['split']
    # Split indices
    current = 0
    train_count = np.int(percent * total)
    train_indices = indices[current:current + train_count]
    current += train_count
    test_indices = indices[current:]
    train_subset, test_subset = [], []
    for i in train_indices:
        train_subset.append(samples[i])

    for i in test_indices:
        test_subset.append(samples[i])
    return train_subset, test_subset

def text_align(prWords, gtWords):
    row, col = len(prWords), len(gtWords)
    adjMat= np.zeros((row, col), dtype=float)
    for i in range(len(prWords)):
        for j in range(len(gtWords)):
            adjMat[i, j] = similarity(prWords[i], gtWords[j])
    pr_aligned=[]
    for i in range(len(prWords)):
        nn = list(map(lambda x:gtWords[x], np.argsort(adjMat[i, :])[:1])) 
        pr_aligned.append((prWords[i], nn[0]))
    return pr_aligned

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, save_file, patience=5, verbose=False, delta=0, best_score=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_file = save_file
        print(best_score)

    def __call__(self, val_loss, epoch, model, optimizer):
        
        score = -val_loss
        state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
                'best': score
                }
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, state)
        elif score < self.best_score - self.delta:

            self.counter += 1
            print(f'EarlyStopping counter: ({self.best_score:.6f} {self.counter} out of {self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, state)
            self.counter = 0

    def save_checkpoint(self, val_loss, state):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(state, self.save_file)
        self.val_loss_min = val_loss


class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.count = 0
        self.total = 0
        self.max = -1 * float("inf")
        self.min = float("inf")

    def add(self, element):
        # pdb.set_trace()
        self.total += element
        self.count += 1
        self.max = max(self.max, element)
        self.min = min(self.min, element)

    def compute(self):
        # pdb.set_trace()
        if self.count == 0:
            return float("inf")
        return self.total / self.count

    def __str__(self):
        return "%s (min, avg, max): (%.3lf, %.3lf, %.3lf)" % (self.name, self.min, self.compute(), self.max)

class Eval:
    def _blanks(self, max_vals,  max_indices):
        def get_ind(indices):
            result = []
            for i in range(len(indices)):
                if indices[i] != 0:
                    result.append(i)
            return result
        non_blank = list(map(get_ind, max_indices))
        scores = []

        for i, sub_list in enumerate(non_blank):
            sub_val = []
            if sub_list:
                for item in sub_list:
                    sub_val.append(max_vals[i][item])
            score = np.exp(np.sum(sub_val))
            if math.isnan(score):
                score = 0.0
            scores.append(score)
        return scores


    def _clean(self, word):
        regex = re.compile('[%s]' % re.escape('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~“”„'))
        return regex.sub('', word)

    def char_accuracy(self, pair):
        words, truths = pair
        words, truths = ''.join(words), ''.join(truths)
        sum_edit_dists = lev.distance(words, truths)
        sum_gt_lengths = sum(map(len, truths))
        fraction = 0
        if sum_gt_lengths != 0:
            fraction = sum_edit_dists / sum_gt_lengths

        percent = fraction * 100
        if 100.0 - percent < 0:
            return 0.0
        else:
            return 100.0 - percent

    def word_accuracy(self, pair):
        correct = 0
        word, truth = pair
        if self._clean(word) == self._clean(truth):
            correct = 1
        return correct

    def format_target(self, target, target_sizes):
        target_ = []
        start = 0
        for size_ in target_sizes:
            target_.append(target[start:start + size_])
            start += size_
        return target_

    def word_accuracy_line(self, pairs):
        preds, truths = pairs
        word_pairs = text_align(preds.split(), truths.split())
        word_acc = np.mean((list(map(self.word_accuracy, word_pairs))))
        return word_acc

class OCRLabelConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
        self.dict[''] = 0
    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        '''
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))
        '''
        length = []
        result = []
        for item in text:
            # item = item.decode('utf-8', 'strict')
            length.append(len(item))
            for char in item:
                if char in self.dict:
                    index = self.dict[char]
                else:
                    index = 0
                result.append(index)

        text = result
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(),
                                                                                                         length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(
                t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts