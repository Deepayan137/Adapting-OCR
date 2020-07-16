import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
import numpy as np
import os
import random

from src.models.crnn import BidirectionalLSTM

class BiLSTM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BiLSTM, self).__init__()
        self.out_dim = out_dim
        self.rnn = nn.LSTM(in_dim, out_dim, bidirectional=True)

    def forward(self, input):       
        out, _ = self.rnn(input)
        return out

class MultiLangCRNN_ver2(nn.Module):
    def __init__(self, args):
        super(MultiLangCRNN_ver2, self).__init__()
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = args.nChannels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        self.cnn = cnn
        self.rnn = nn.Sequential(BiLSTM(args.nHidden*2, args.nHidden),
                    BiLSTM(args.nHidden*2, args.nHidden))

        self.fc_lang1 = nn.Linear(args.nHidden*2, args.nClasses_lang1)
        self.fc_lang2 = nn.Linear(args.nHidden*2, args.nClasses_lang2)

    def feat_extract(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)
        return conv

    def classify_lang1(self, input):
        T, b, h =input.size()
        input = input.reshape(T * b, h)
        output = self.fc_lang1(input)  # [T * b, nOut]
        output = output.reshape(T, b, -1)
        return output

    def classify_lang2(self, input):
        T, b, h =input.size()
        input = input.reshape(T * b, h)
        output = self.fc_lang2(input)  # [T * b, nOut]
        output = output.reshape(T, b, -1)
        return output
    
    def forward(self, input1, input2):
        
        out1 = self.feat_extract(input1)
        out1 = self.classify_lang1(out1)

        out2 = self.feat_extract(input2)
        out2 = self.classify_lang2(out2)

        return out1, out2  

class MultiLangCRNN(nn.Module):
    def __init__(self, args):
        super(MultiLangCRNN, self).__init__()
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = args.nChannels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        self.cnn = cnn
        self.rnn1 = nn.Sequential(
            BidirectionalLSTM(args.nHidden*2, args.nHidden,args.nHidden),
            BidirectionalLSTM(args.nHidden, args.nHidden, args.nClasses_lang1))

        self.rnn2 = nn.Sequential(
            BidirectionalLSTM(args.nHidden*2, args.nHidden, args.nHidden),
            BidirectionalLSTM(args.nHidden, args.nHidden, args.nClasses_lang2))
        # self.fc_lang1 = nn.Linear(args.nHidden*2, args.nClasses_lang1)
        # self.fc_lang2 = nn.Linear(args.nHidden*2, args.nClasses_lang2)


    def feat_extract(self, input):
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        # output = self.stacked_rnn(conv)
        return conv

    def classify_lang1(self, input):
        out = self.rnn1(input)
        return out

    def classify_lang2(self, input):
        out = self.rnn2(input)
        return out
    
    def forward(self, input1, input2):
        
        out1 = self.feat_extract(input1)
        out1 = self.classify_lang1(out1)

        out2 = self.feat_extract(input2)
        out2 = self.classify_lang2(out2)

        return out1, out2

