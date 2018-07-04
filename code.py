#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
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
from sklearn.ensemble import RandomForestClassifier
import re

params = [{'C':np.logspace(0, 3, 7),'gamma':np.logspace(-5, 0, 11)},
          {'max_depth': range(1,6)},
          {'hidden_layer_sizes':(5,2)},
          {'n_estimators': range(1000,1500,50), 'max_depth': range(3,4)}
          ]
# 'ibfgs' solver for small data set with high efficiency
model = [svm.SVC(kernel='rbf'),
         tree.DecisionTreeClassifier(random_state=1),
         nn.MLPClassifier(solver = 'lbfgs', alpha = 1e-4, random_state=1),
         ensemble.GradientBoostingClassifier(n_estimators = 1500, random_state = 3, subsample = 0.5,
           learning_rate = 0.01, min_samples_leaf = 1)]

def load_image(img_file, label):
    img_dir = img_file + str(label) + '/'
    img_lst = os.listdir(img_dir)
    img_lst = sorted([int(x.split('.')[0]) for x in img_lst])
    print('{}----{} images'.format(label,len(img_lst)))

    x = []
    for img in img_lst:
        # add param 'Image.ANTIALIAS' (80.0 -> 100.0)
        img_path = img_dir + str(img) + '.jpg'
        img = Image.open(img_path).resize((128,128),Image.ANTIALIAS)
        img = img.convert('L')
        # img.show()

        img = np.array(img, dtype=int).reshape(1,-1) #(1,64)
        # print(img.shape) # (1, 256)
        img = np.abs(img-255)
        x.append(img)
    x = np.array(x).reshape(-1,128*128)
    # print(x.shape) # (128, 256)
    # print('\nx.shape:{}'.format(x.shape))

    if label == 1:
        y = np.ones(len(img_lst)).reshape(-1, 1)
    elif label == 0:
        y = np.zeros(len(img_lst)).reshape(-1, 1)
    elif label == 'test':
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1]).reshape(-1, 1)
        print('len(y): ', len(y))

    data = np.concatenate((x, y), axis=1)
    # print('\ny.shape:{}'.format(label.shape))
    # print(label,y)
    return data

def prepare_xy(data_train, data_test):
    print('\nLoad File Start...')
    print('\ndata_train.shape:{}'.format(data_train.shape))
    x, y = np.split(data_train, (-1, ), axis=1) #最后一列进行行拆分
    print(x.shape,y.shape)
    y = y.ravel().astype(np.int)
    print(y)

    x_test, y_test = np.split(data_test, (-1,), axis=1)  # 最后一列进行行拆分
    print(x_test.shape, y_test.shape)
    y_test = y_test.ravel().astype(np.int)
    print(y_test)

    # x, x_test, y, y_test = train_test_split(x, y, test_size=0.1, random_state=40)
    # print(y,y_test)
    images = x.reshape(-1, 128, 128)
    print(images.shape) # (39, 16, 16)
    images_test = x_test.reshape(-1, 128, 128)
    return x, x_test, y, y_test, images, images_test

def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print(tip + '正确率：%.2f%%' % (100*np.mean(acc)))

def save_image(im, i):
    im *= 15.9375
    im = 255 - im
    a = im.astype(np.uint8)
    output_path = '.\\HandWritten'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    Image.fromarray(a).save(output_path + ('\\%d.png' % i))

def show_input():
    matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 9), facecolor='w')
    for index, image in enumerate(images[:16]):
        plt.subplot(4, 8, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'train: %i' % y[index])
    for index, image in enumerate(images_test[:16]):
        plt.subplot(4, 8, index + 17)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        # save_image(image.copy(), index)
        plt.title(u'test: Unknown')
        # plt.title(u'test: %i' % y_test[index])
    plt.tight_layout()
    plt.show()

def train_model(model, params):
    t_start = time()
    model = GridSearchCV(model, param_grid=params, cv=5) #param_grid=params
    print('\nStart Learning...')
    t0 = time()
    model.fit(x, y)
    t1 = time()
    t = t1 - t0
    model.cv_results_
    print('训练+CV耗时：%d分钟%.3f秒' % (int(t / 128), t - 128 * int(t / 128)))
    print('最优参数：\t', model.best_params_)
    # clf.fit(x, y)

    print('Learning is OK...')
    print('x_test.shape:{}'.format(x_test.shape))
    print('训练集精确率：', round(precision_score(y, model.predict(x))*100),
          '\t训练准确率：', round(accuracy_score(y, model.predict(x)) * 100))
    y_hat = model.predict(x_test)
    print(y_hat)
    print('测试集精确率：', round(precision_score(y_test, y_hat)*100),
          '\t测试集准确率：', round(accuracy_score(y_test, y_hat)*100),
          '\t测试集AUC：', round(roc_auc_score(y_test, y_hat)*100))
    print('y_hat:{}'.format(y_hat))
    print('y  \t:{}'.format(y_test))

    # print('saving model {}'.format(re.findall(re.compile(r'(.*)\('), str(model))))
    joblib.dump(model, './model/train_model_2.m')
    print('model saved.')
    return y_hat

def check_target(model):
    print('\nCheck Target...')
    err_images = images_test[y_hat == 1]
    # print(err_images)
    plt.figure(figsize=(10, 8), facecolor='w')
    for index, image in enumerate(err_images):
        if index >= 25:
            break
        plt.subplot(5, 5, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(u'Pred: 1')
    plt.tight_layout()
    plt.savefig(str(model)[:5]+'.png')
    plt.show()


# number of images, number of pixels, numbe149133135r of classes
if __name__ == "__main__":
    data_0 = load_image('./img/', 0)
    data_1 = load_image('./img/', 1)
    data_test = load_image('./img/', 'test')

    data_train = np.concatenate((data_0, data_1), axis=0)

    x, x_test, y, y_test, images, images_test = prepare_xy(data_train, data_test)
    # show_input()

    params = {'max_depth':range(3,6)}
    model = ensemble.GradientBoostingClassifier(n_estimators = 250, random_state = 3, subsample = 0.5,
           learning_rate = 0.1, min_samples_leaf = 1, max_depth=4) #max_depth 3/5-93% -> 4-100%
    print(model)
    y_hat = train_model(model, params)
    # check_target(model)

    ## model comparison
    # for model, params in zip(model, params):
    #     print(model)
    #     y_hat = train_model(model, params)
    #     check_result(model)


    # err_images = images_test[y_test != y_hat]
    # err_y_hat = y_hat[y_test != y_hat]
    # err_y = y_test[y_test != y_hat]
    # print(err_y_hat)
    # print(err_y)
    # plt.figure(figsize=(10, 8), facecolor='w')
    # for index, image in enumerate(err_images):
    #     if index >= 12:
    #         break
    #     plt.subplot(3, 4, index + 1)
    #     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     plt.title(u'错分为：%i，真实值：%i' % (err_y_hat[index], err_y[index]))
    # plt.tight_layout()
    # plt.show()

