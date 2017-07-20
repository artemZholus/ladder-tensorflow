import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from input_data import SemiDataSet
try:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    from sklearn.manifold import TSNE

from ladder import LadderNetwork, hyperparameters
from utils import prepare_data
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
f = pd.DataFrame(columns=['model', *list(map(lambda x: 'f1_class_' + str(x),
                                             range(1, 11))), 'f1_mean', 'roc_auc'])
path = './experiments'
num_labeled = 500
if not os.path.exists(os.path.join(os.curdir, 'experiments')):
    os.makedirs('experiments')


def measure_metrics(name, X, y_true, predict, predict_prob, j=[0]):
    y_pred = predict(X)
    y_prob = predict_prob(X)
    if len(y_true.shape) > 1:
        y_true_bin = y_true
        _, y_true = np.where(y_true == 1)
    else:
        y_true_bin = LabelBinarizer().fit_transform(y_true)
    if len(y_pred.shape) > 1:
        y_pred_bin = y_pred
        _, y_pred = np.where(y_pred == 1)
    else:
        y_pred_bin = LabelBinarizer().fit_transform(y_pred)
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
    precision, recall = {}, {}
    tprs, fprs = {}, {}
    for clas in range(y_prob.shape[1]):
        precision[clas], recall[clas], _ = precision_recall_curve(y_true_bin[:, clas], y_prob[:, clas])
        fprs[clas], tprs[clas], _ = roc_curve(y_true_bin[:, clas], y_prob[:, clas])
    prec1, rec1, _ = precision_recall_curve(y_true_bin.ravel(), y_prob.ravel())
    fprs1, tprs1, _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
    plt.style.use('seaborn-white')
    jet = plt.get_cmap('Accent')
    cNorm = colors.Normalize(vmin=0, vmax=10)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    fig, ax = plt.subplots(1, figsize=(15, 10))
    for (i, prec), (j, rec) in zip(precision.items(), recall.items()):
        pass
        c = scalarMap.to_rgba(i)
        ax.plot(rec, prec, label='class_%d' % i, color=c)
    ax.plot(rec1, prec1, label='mean', color=scalarMap.to_rgba(10))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', ncol=3, fancybox=True, shadow=True, prop={'size': 15})
    plt.title('Precision-Recall curve')
    fig.savefig(path + '/{0}_prec_rec_curve.png'.format(name))
    plt.close(fig)
    fig, ax = plt.subplots(1, figsize=(15, 10))
    for (i, tp), (j, fp) in zip(tprs.items(), fprs.items()):
        c = scalarMap.to_rgba(i)
        ax.plot(fp, tp, label='class_%d' % i, color=c)
    ax.plot(fprs1, tprs1, label='mean', color=scalarMap.to_rgba(10))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='lower center', ncol=3, fancybox=True, shadow=True, prop={'size': 15})
    plt.title('ROC curve')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    fig.savefig(path + '/{0}_roc_curve.png'.format(name))
    plt.close(fig)

print('=== read data ===')
data_it = pd.read_csv('/home/azholus/data/q2norm_siders_unsupervised+supervised.csv', iterator=True, chunksize=500,
                      low_memory=False)
data = pd.concat(tqdm(data_it, total=374 * 2), ignore_index=True)


print('=== transform data ===')
gene_first = data.columns.tolist().index('pert_doseunit') + 1
gene_last = data.columns.tolist().index('ZW10')

genes = data.values[:,gene_first:gene_last]
genes = genes.astype(np.float32)
genes = pd.DataFrame(data=genes, columns=data.columns[gene_first:gene_last])
genes = pd.DataFrame(data=genes.values / genes.max().values, columns=data.columns[gene_first:gene_last])
side_ef = data[data.supervised].values[:, gene_last + 1:]
side_ef = side_ef.astype(np.float32)
side_ef = pd.DataFrame(data=side_ef, columns=data.columns[gene_last + 1:])

side_ef = pd.DataFrame(data=(side_ef.values > 0).astype(np.float32)[:,:26], columns=side_ef.columns[:26])
print(side_ef.values.shape)
data = genes.values
labels = side_ef.values
labeled_data = data[:len(labels)]
unlabeled_data = data[len(labels):]
X_train, X_test, y_train, y_test = train_test_split(labeled_data, labels, test_size=0.3)
X_train = np.vstack([X_train, unlabeled_data])
print(y_train.shape)
data = SemiDataSet(x=X_train, y=y_train, n_labeled=len(side_ef), n_classes=len(side_ef.columns))
# MLP
layers = [
    (len(genes.columns), None),
    (1000, tf.nn.relu),
    (500, tf.nn.relu),
    (250, tf.nn.relu),
    (250, tf.nn.relu),
    (250, tf.nn.relu),
    (len(side_ef.columns), tf.nn.sigmoid)
]
print('=== building ladder network ===')

# def score_l(ladder, x, y):
#     pred = ladder.predict(x)
#     mean = f1_score(np.where(y == 1)[1], pred, average=None).mean()
#     return mean
#
#
# params = {
#     'denoise_cost_init': [0.1, 0.5, 1, 5, 7],
#     'denoise_cost_param': [0.1, 0.5, 1, 2, 2.71, 5, 7, 10],
#     'denoise_cost': ['hyperbolic_decay', 'exponential_decay']
# }
# grid = GridSearchCV(estimator=LadderNetwork(layers, **hyperparameters),
#                     param_grid=params,
#                     scoring=score_l,
#                     fit_params={'epochs': 20},
#                     n_jobs=
#                     pre_dispatch=1,
#                     refit=True)
# grid.fit(X_train, y_bin_train)
# print('--- BEST ESTIMATOR --- ')
# print(grid.best_estimator_)
# print('--- BEST PARAMS ---')
# print(grid.best_params_)
# print('--- RESULTS ---')
# print(grid.cv_results_)
# exit(0)
ladder = LadderNetwork(layers, **hyperparameters)
ladder.log_all('./stat')
# ladder.session = tfdbg.LocalCLIDebugWrapperSession(ladder.session)
# ladder.session.add_tensor_filter("has_inf_or_nan", tfdbg.has_inf_or_nan)
print('=== training ===')
ladder.fit(data, test_x=X_test, test_y=y_test)
measure_metrics('ladder', X_test, y_test, ladder.predict, ladder.predict_proba)
ladder.session.close()
f.to_csv('experiments/measures.csv', index=False)
