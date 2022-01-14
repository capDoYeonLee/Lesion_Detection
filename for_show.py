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
from pyrsistent import b
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



# glob을 사용해서 경로안에 있는 모든 파일을 sorted로 정렬해줌 
# train_files에 뭐가 있는지 궁금하면 print(train_files)를 작성해서 볼 수 있음.
# 보면 json file이 순서대로 정렬된것을 볼 수 있음.
train_files = sorted(glob('../Datasets/train/*'))
test_files = sorted(glob('../Datasets/test/*'))



# for문을 사용해서 train_files에서 json data를 가져옴.
# json data를 train_json_list라는 리스트에 넣어줌. 
# tqdm은 내 코드가 얼마나 진행되었는지 진행상황을 보여줌 
train_json_list = []
for file in tqdm(train_files[:100]):
    with open(file, "r") as json_file:
        train_json_list.append(json.load(json_file))

test_json_list = []
for file in tqdm(test_files[:100]):
    with open(file, "r") as json_file:
        test_json_list.append(json.load(json_file))

# data를 보면 shape에서 4가지의 label이 있다. 
# 아래 코드는 각각 label이 몇개가 있는지 보여준다 -> ex) 01_ulcer찍힌 data(사진)이 몇개가 있는지 count해줌.
label_count = {}
for data in train_json_list:
    for shape in data['shapes']:
        try:
            label_count[shape['label']]+=1
        except:
            label_count[shape['label']]=1
            

print(label_count)


# 아래 코드는 data를 통해 opencv 활용. 
# overlay된 사진을 보여줌. 


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

