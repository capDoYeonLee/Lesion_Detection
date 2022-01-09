import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import cv2
import base64
import time
import math
import datetime
import os
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from glob import glob

import torch
import torchvision
import torch.distributed as dist
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from collections import defaultdict, deque




train_files = sorted(glob('./Datasets/train/*'))
test_files = sorted(glob('./Datasets/test/*'))



train_json_list = []
for file in tqdm(train_files[:10]):
    with open(file, "r") as json_file:
        train_json_list.append(json.load(json_file))

test_json_list = []
for file in tqdm(test_files[:10]):
    with open(file, "r") as json_file:
        test_json_list.append(json.load(json_file))


label_count = {}
for data in train_json_list:
    for shape in data['shapes']:
        try:
            label_count[shape['label']]+=1
        except:
            label_count[shape['label']]=1
            

print(label_count)


plt.figure(figsize=(25,30))
for i in range(5):
    plt.subplot(1,5,i+1)
    # base64 형식을 array로 변환
    img = Image.open(BytesIO(base64.b64decode(train_json_list[i]['imageData'])))
    img = np.array(img, np.uint8)
    title = []
    for shape in train_json_list[i]['shapes']:
        points = np.array(shape['points'], np.int32)
        print(points)
        cv2.polylines(img, [points], True, (0,255,0), 3)
        title.append(shape['label'])
    title = ','.join(title)
    plt.imshow(img)
    plt.subplot(1,5,i+1).set_title(title)
plt.show()    