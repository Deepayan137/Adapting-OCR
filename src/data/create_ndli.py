import os
import subprocess
import glob
import numpy as np
from PIL import Image
from numpy import asarray
import sys
import shutil
import pdb
import re
from tqdm import *
import random
from progressbar import progressbar
import cv2
import lmdb
import io
class UnequalLength(BaseException):
    print('unequal length')
    pass


def image_prep(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


def parse_data(path):
    file_paths = list(map(lambda f: path + f, os.listdir(path)))

    def clean(base_name):
        return re.findall(r'\d+', base_name)[-1]

    def read(text_file):
        with open(text_file, 'r') as f:
            text = f.read()
        return text
    if path.endswith('Images/'):
        content = {clean(os.path.basename(x)): image_prep(x)
                   for x in (file_paths)}
        return content
    content = {clean(os.path.basename(x)): read(x) for x in file_paths}
    return content


def images_and_truths(image, plot, text):

    def resizeImage(line_crop):
        try:

            height, width = np.shape(line_crop)
            if height is 0 or width is 0:
                line_crop = np.zeros((32, 32))
            height, width = np.shape(line_crop)
            ratio = 32/height
            resized_image = cv2.resize(line_crop, None, fx=ratio, fy=ratio,
                                       interpolation=cv2.INTER_CUBIC)
            return np.array((resized_image), dtype='uint8')

        except Exception as e:
            print(e)

    def extract_units(unit):
        unit = list(map(lambda x: int(x), unit))
        x1, y1, w, h = unit
        x2 = x1+w
        y2 = y1+h
        line_crop = image[int(y1):int(y2), int(x1):int(x2)]
        return line_crop
    li = plot.split()
    units = [li[i:i+4] for i in range(0, len(li), 4)]
    croppedImages = list(map(extract_units, units))
    allTruths = [s.strip() for s in text.splitlines()]
    
    unitImages = [resizeImage(croppedImages[i])
                  for i in range(len(croppedImages))]
    unitTruths = [allTruths[i]
                  for i in range(len(allTruths))]
    if len(unitImages) == len(unitTruths):
        pairs = list(zip(unitImages, unitTruths))
        return pairs
    return None


def read_book(**kwargs):
    pairwise = []
    book_path = kwargs["book_path"]

    def dirs(f): return os.path.join(book_path, f)
    folder_paths = map(dirs, ['Images/', 'Annotations/', 'Segmentations/'])
    images, text, plots = list(map(parse_data, folder_paths))
    keys = [key for key in images.keys()]
    tsize, pno = [], []
    for key in keys:
        try:
            pairs = images_and_truths(
                images[key], plots[key], text[key])
            if pairs:
                pairwise.extend(pairs)
        except Exception as e:
            print('\n Key does not exist')
            print(key)
            print(e)
        except UnequalLength:
            pass
    return pairwise

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if type(k) == str:
                k = k.encode()
            if type(v) == str:
                v = v.encode()
            txn.put(k, v)

def createDataset(path, language):
    books = os.listdir(path)
    all_pairs = []
    for book in tqdm(books[:5]):
        book_path = os.path.join(path, book)
        pairs = read_book(book_path=book_path)
        all_pairs.extend(pairs)
    # if os.path.exists(outputPath):
    #     shutil.rmtree(outputPath)
    #     os.makedirs(outputPath)
    # else:
    #     os.makedirs(outputPath)
    outputPath = os.path.join(path, '%s.lmdb'%language)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    nSamples = len(all_pairs)
    for i in tqdm(range(nSamples)):
        image, label = all_pairs[i]
        image = Image.fromarray(image)
        imgByteArr = io.BytesIO()
        image.save(imgByteArr, format='tiff')
        wordBin = imgByteArr.getvalue()
        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt

        cache[imageKey] = wordBin
        cache[labelKey] = label
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1

    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    env.close()
    print('Created dataset with %d samples' % nSamples)

# createDataset()