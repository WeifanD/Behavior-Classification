#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
#import pandas as pd
from sklearn import svm,tree,neural_network as nn,ensemble
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import *
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from time import time
from sklearn.externals import joblib
import os, sys # sys.argv[0]

#path = sys.argv[1]

def process_image(img_dir):
    img = Image.open(img_dir).resize((128,128),Image.ANTIALIAS)
    img = img.convert('L')
    img = np.array(img, dtype=int).reshape(1,-1) #(1,64)
    img = np.abs(img-255)
    img = np.array(img).reshape(-1, 128*128)
    # print(x.shape) # (3, 256)
    return img

model = joblib.load('./model/train_model_2.m')
print(model)
# x_test = load_image('./img/test/')
img_lst = os.listdir('./img/test_0703/')
img_lst = sorted([int(x.split('.')[0]) for x in img_lst])
print('%d images' %len(img_lst))

for img in img_lst:
    x_test = process_image('./img/test_0703/' + str(img) + '.jpg')
    # print('x_test.shape:{}'.format(x_test.shape))
    y_hat = model.predict(x_test)
    print(img, '-', y_hat)

