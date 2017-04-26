import matplotlib
import math

matplotlib.use('Agg')
import numpy as np
import scipy as sp
import pandas as pd
import seaborn as sn
import sklearn
import keras
import tensorflow as tf
from tqdm import tqdm
from tensorflow.python import debug

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from itertools import product

from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers import Dense, BatchNormalization, Activation

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    from sklearn.manifold import TSNE

from ladder import LadderNetwork, hyperparameters
from utils import mlp

f = pd.DataFrame(columns=['model', *list(map(lambda x: 'f1_class_' + str(x),
                                             range(1, 11))), 'f1_mean', 'roc_auc'])
path = './experiments'


def measure_metrics(name, X, y_true, predict, predict_prob, j=[0]):
    y_pred = predict(X)
    y_prob = predict_prob(X)
    if len(y_true.shape) > 1:
        y_true_bin = y_true
        _, y_true = np.where(y_true == 1)
    else:
        y_true_bin = LabelBinarizer().fit_transform(y_true)
    m = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    roc_auc = roc_auc_score(y_true_bin, y_prob, average='macro')
    fig, ax = plt.subplots(1, figsize=(15, 10))
    sn.heatmap(m, annot=True, fmt='d', ax=ax)
    sn.plt.title('Confusion matrix')
    sn.plt.ylabel('True label')
    sn.plt.xlabel('Predicted label')
    fig.savefig(path + '/{0}_confusion_matrix.png'.format(name))
    plt.close(fig)
    f.loc[j[0]] = ([name, *list(map(lambda x: '%.5f' % x, f1)), f1.mean(), roc_auc])
    print(f.iloc[j[0]])
    j[0] += 1
    # precision, recall = {}, {}
    # tprs, fprs = {}, {}
    # for clas in range(y_prob.shape[1]):
    #     precision[clas], recall[clas], _ = precision_recall_curve(y_true_bin[:, clas], y_prob[:, clas])
    #     fprs[clas], tprs[clas], _ = roc_curve(y_true_bin[:, clas], y_prob[:, clas])
    # prec1, rec1, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
    # fprs1, tprs1, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    # plt.style.use('seaborn-white')
    # jet = plt.get_cmap('Accent')
    # cNorm = colors.Normalize(vmin=0, vmax=10)
    # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    # fig, ax = plt.subplots(1, figsize=(15, 10))
    # for (i, prec), (j, rec) in zip(precision.items(), recall.items()):
    #     pass
    #     c = scalarMap.to_rgba(i)
    #     ax.plot(rec, prec, label='class_%d' % i, color=c)
    # ax.plot(rec1, prec1, label='mean', color=scalarMap.to_rgba(10))
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, loc='lower center', ncol=3, fancybox=True, shadow=True, prop={'size': 15})
    # plt.title('Precision-Recall curve')
    # fig.savefig(path + '/{0}_prec_rec_curve.png'.format(name))
    # plt.close(fig)
    # fig, ax = plt.subplots(1, figsize=(15, 10))
    # for (i, tp), (j, fp) in zip(tprs.items(), fprs.items()):
    #     c = scalarMap.to_rgba(i)
    #     ax.plot(fp, tp, label='class_%d' % i, color=c)
    # ax.plot(fprs1, tprs1, label='mean', color=scalarMap.to_rgba(10))
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles, labels, loc='lower center', ncol=3, fancybox=True, shadow=True, prop={'size': 15})
    # plt.title('ROC curve')
    # plt.xlabel('False Positive')
    # plt.ylabel('True Positive')
    # fig.savefig(path + '/{0}_roc_curve.png'.format(name))
    # plt.close(fig)


### Load data
def load_data(path):
    train = pd.read_csv(path + '/train.csv').drop('id', axis=1, inplace=False)
    test = pd.read_csv(path + '/test.csv').drop('id', axis=1, inplace=False)
    train_Y = LabelEncoder().fit_transform(train.target)
    train_Y_binarized = LabelBinarizer().fit_transform(train.target)
    train.drop('target', axis=1, inplace=True)
    return train.values, train_Y, test.values, train_Y_binarized


# train, labels, test, binarized = load_data('./data')
from mnist import train_data, train_labels, test_labels, test_data

perm = np.random.permutation(len(train_data))
X_train = train_data.reshape((-1, 784))[perm]
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = test_data.reshape((-1, 784))
X_test = (X_test - X_test.mean()) / X_test.std()
y_train = train_labels[perm]
y_test = test_labels
y_bin_train = LabelBinarizer().fit_transform(train_labels)[perm]
y_bin_test = LabelBinarizer().fit_transform(test_labels)
# X_train, X_test, y_train, y_test, y_bin_train, y_bin_test = train_test_split(train, labels, binarized, test_size=0.3)

# ### Logistic regression
# print('Logistic Regression:')
# model = LogisticRegression()
# print('fitting...')
# model.fit(X_train[: 1000], y_train[:1000])
# print('calculating metrics...')
# measure_metrics('logreg', X_test, y_test, model.predict, model.predict_proba)
### MLP
layers = [
    (784, None),
    (128, tf.nn.relu),
    (10, tf.nn.softmax)
]
print('Ladder:')
def score_l(ladder, x, y):
    pred = ladder.predict(x)
    mean = f1_score(np.where(y == 1)[1], pred, average=None).mean()
    return mean
params = {
    'denoise_cost_init': [0.1, 0.5, 1, 5, 7],
    'denoise_cost_param': [0.1, 0.5, 1, 2, 2.71, 5, 7, 10],
    'denoise_cost': ['hyperbolic_decay', 'exponential_decay']
}
grid = GridSearchCV(estimator=LadderNetwork(layers, **hyperparameters),
                    param_grid=params,
                    scoring=score_l,
                    fit_params={'epochs': 20},
                    n_jobs=1,
                    pre_dispatch=1,
                    refit=True)
grid.fit(X_train, y_bin_train)
print('--- BEST ESTIMATOR --- ')
print(grid.best_estimator_)
print('--- BEST PARAMS ---')
print(grid.best_params_)
print('--- RESULTS ---')
print(grid.cv_results_)
exit(0)
#ladder = LadderNetwork(layers, **hyperparameters)
#ladder.log_all('./stat')
## ladder.session = debug.LocalCLIDebugWrapperSession(ladder.session)
## ladder.session.add_tensor_filter("has_inf_or_nan", debug.has_inf_or_nan)
# ladder.fit(X_train, y_bin_train, batch_size=64, epochs=15, ratio=0.)
# measure_metrics('ladder0', X_test, y_bin_test, ladder.predict, ladder.predict_proba)
# for i in range(1, 10):
#     ladder.fit(X_train, y_bin_train[:1000], batch_size=64, epochs=1)
#     measure_metrics('ladder' + str(i), X_test, y_bin_test, ladder.predict, ladder.predict_proba)
# ladder.session.close()
# print('MLP:')
# fit, pred, prob = mlp(layers)
# fit(X_train[:1000], y_bin_train[:1000], epochs=25, bsize=64)
#
# measure_metrics('mlp', X_test, y_bin_test, pred, prob)
# f.to_csv('experiments/measures.csv', index=False)
