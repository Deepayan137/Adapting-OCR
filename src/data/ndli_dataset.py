import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import lmdb
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
import os
import sys
import pdb
from data.create_ndli import createDataset

class NDLIDataset(Dataset):
    def __init__(self):
        super(NDLIDataset, self).__init__()
        path = '/ssd_scratch/cvit/deep/data/'
        imgdir = 'telugu'
        outputPath = os.path.join(path, imgdir)
        lmdbPath = os.path.join(path, imgdir,
                '%s.lmdb'%imgdir)
        if not os.path.exists(lmdbPath):
           createDataset(outputPath, imgdir) 
        self.env = lmdb.open(
            os.path.abspath(lmdbPath),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (opt.dataroot))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode('utf-8')).decode('utf-8'))
            self.nSamples = nSamples

        transform_list = [transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))]
        transform = transforms.Compose(transform_list)
        self.transform = transform
        
    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        envAug = False
        with eval('self.env'+'_aug'*envAug+'.begin(write=False)') as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode('utf-8'))

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)
            item = {'img': img, 'img_path': img_key, 'idx':index}

        
            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode('utf-8'))
            item['label'] = label.decode('utf-8')
        return item


class NDLICollator(object):
    def __call__(self, batch):

        img_path = [item['img_path'] for item in batch]
        width = [item['img'].shape[2] for item in batch]
        indexes = [item['idx'] for item in batch]
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)], dtype=torch.float32)
        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print(imgs.shape)
        item = {'img': imgs, 'img_path':img_path, 'idx':indexes}
        if 'label' in batch[0].keys():
            labels = [item['label'] for item in batch]
            item['label'] = labels
        return item


# data = NDLIDataset()
# print(data[0])