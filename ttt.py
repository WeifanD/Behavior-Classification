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

model = [svm.SVC(kernel='rbf'),
         tree.DecisionTreeClassifier(random_state=1),
         nn.MLPClassifier(solver = 'lbfgs', alpha = 1e-4, random_state=1),
         ]
print(re.findall(re.compile(r'(.*)\('), str(model[1])))
