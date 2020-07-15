import os
import shutil
import sys
import random
from tqdm import *
import numpy as np
import pdb
path = sys.argv[1]
# path  = '/ssd_scratch/cvit/deep/data/Hindi'
dirs = os.listdir(path)

indices = np.arange(len(dirs))
random.shuffle(indices)
split_idx = int(0.9*len(indices))
train_indices = indices[:split_idx]
test_indices = indices[split_idx:]
splits = {'train':train_indices, 'test':test_indices}
for key in splits.keys():
	dest_dir = os.path.join(path, key)
	if not os.path.exists(dest_dir):
		os.makedirs(dest_dir)
	for index in tqdm(splits[key]):
		src_dir = os.path.join(path, dirs[index])
		if not os.path.exists(os.path.join(dest_dir, dirs[index])):
			shutil.move(src_dir, os.path.join(dest_dir, dirs[index]))
		else:
			print('exists')
	# print('Finished {}'.format(splits[key]))




