import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import sklearn
import keras

from itertools import product

from keras.models import Sequential
from keras.layers import Dense

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except:
    from sklearn.manifold import TSNE

from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve

### Load data
def load_data(path):
    train = pd.read_csv(path + '/train.csv').drop('id', axis=1, inplace=False)
    test = pd.read_csv(path + '/test.csv').drop('id', axis=1, inplace=False)
    train_Y = LabelEncoder().fit_transform(train.target)
    train_Y_binarized = LabelBinarizer().fit_transform(train.target)
    train.drop('target', axis=1, inplace=True)
    return train.values, train_Y, test.values, train_Y_binarized

train, labels, test, binarized = load_data('./data')
X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.3)

### Logistic regression
model = LogisticRegression()
model.fit(X_train[:2000], y_train[:2000])
#measure_metrics(model, 'logreg', X_test, y_test)
