class LangIdentifier(nn.Module):
    def __init__(self, args, leakyRelu=False):
        super(LangIdentifier, self).__init__()
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
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
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
        self.features = cnn
        self.classifier = nn.Sequential(
            nn.Linear(2*args.nHidden*26, 128), 
            nn.Linear(128, 3)
            )

    def forward(self, input):
        out = self.features(input)
        N, C, H, W = out.size()
        out = self.classifier(out.view(N, -1))
        return out 
        
class MultiTaskCRNN(nn.Module):
    def __init__(self, args):
        super(MultiTaskCRNN, self).__init__()
        model = LangIdentifier(args)
        savepath = os.path.join(args.save_dir, 'multitask')
        resume_file = os.path.join(savepath, 'latest.pth')
    
        print('Loading model %s'%resume_file)
        checkpoint = torch.load(resume_file)
        model.load_state_dict(checkpoint)
    
        self.cnn = model.features

        self.rnn = nn.Sequential()
        self.rnn = nn.Sequential(
            BidirectionalLSTM(args.nHidden*2, args.nHidden, args.nHidden),
            BidirectionalLSTM(args.nHidden, args.nHidden, args.nClasses))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)
        output = output.transpose(1,0) #Tbh to bth
        return output