import time
import tensorflow as tf
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import math
from utils import get_decay, join, split_lu, labeled, unlabeled
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn import datasets
from functools import wraps
from utils import conditional_context, SemiSupervisedDataset
import math
import os
import csv
from tqdm import tqdm
from input_data import read_data_sets

hyperparameters = {
    'learning_rate': 0.001,
    'denoise_cost_init': 1000,
    'denoise_cost': 'exponential_decay',
    'denoise_cost_param': 0.01,
    'denoise_cost_threshold': 0.1,
    'batch_size': 100,
    'noise_std': 0.2,
}


class LadderNetwork:
    def __init__(self, layers_, batch_size, learning_rate=1e-4, noise_std=0.3, denoise_cost=None,
                 denoise_cost_init=None, denoise_cost_param=None, denoise_cost_threshold=None):
        self.layers_ = layers_
        self.layers, self.activation = zip(*layers_)
        self.batch_size = batch_size
        self.graph = tf.Graph()
        self.session = tf.get_default_session()
        self.denoise_cost_init = denoise_cost_init
        self.denoise_cost_param = denoise_cost_param
        self.denoise_cost_threshold = denoise_cost_threshold
        if self.session is not None:
            self.session.close()
        self.session = None
        if not isinstance(denoise_cost, str):
            decay = denoise_cost
        else:
            decay = get_decay(denoise_cost)(denoise_cost_init, denoise_cost_param,
                                            len(self.layers), denoise_cost_threshold)
        self.learning_rate = learning_rate
        self.denoising_cost = decay
        self.noise_std = noise_std
        with self.graph.as_default():
            self.__init_placeholders()
            self.__init_weights()
            print("=== Corrupted Encoder ===")
            self.y_c, self.corr = self.build_encoder(self.inputs, noise_std)

            print("=== Clean Encoder ===")
            self.y, self.clean = self.build_encoder(self.inputs, 0.0)  # 0.0 -> do not add noise
            print('=== Decoder ===')
            self.build_decoder()
            print("=== Cost ===")
            self.build_cost()
        self.supervised_summary = None
        self.unsupervised_summary = None
        self.supervised_histograms = None
        self.unsupervised_histograms = None
        self.writer = None
        self.session = tf.InteractiveSession(graph=self.graph)
        self.session.run(tf.global_variables_initializer())
        self.initialized = False

    def __init_placeholders(self):
        with tf.name_scope('placeholders'):
            self.inputs = tf.placeholder(tf.float32, shape=(None, self.layers[0]))
            self.outputs = tf.placeholder(tf.float32)
            self.training = tf.placeholder(tf.bool)

    def __str__(self):
        return 'LadderNetwork(layers=[{0}], learning_rate={1}, noise_std={2}, denoise_cost={3},' \
               'denoise_cost_init={4}, denoise_cost_param={5})'.format(
            ', '.join(map(str, list(zip(self.layers, self.activation)))),
            str(self.learning_rate),
            str(self.noise_std),
            str(self.denoise_cost),
            str(self.denoise_cost_init),
            str(self.denoise_cost_param)
        )

    @staticmethod
    def bi(inits, size, name):
        return tf.Variable(inits * tf.ones([size]), name=name)

    @staticmethod
    def wi(shape, name):
        return tf.Variable(tf.random_normal(shape, name=name)) / math.sqrt(shape[0])

    @staticmethod
    def batch_normalization(batch, mean=None, var=None):
        if mean is None or var is None:
            mean, var = tf.nn.moments(batch, axes=[0])
        return (batch - mean) / tf.sqrt(var + tf.constant(1e-10))

    def update_batch_normalization(self, batch, l):
        """
        batch normalize + update average mean and variance of layer l
        """
        with tf.name_scope('update_bn'):
            mean, var = tf.nn.moments(batch, axes=[0])
            assign_mean = self.running_mean[l - 1].assign(mean)
            assign_var = self.running_var[l - 1].assign(var)
            self.bn_assigns.append(self.ewma.apply([self.running_mean[l - 1], self.running_var[l - 1]]))
            with tf.control_dependencies([assign_mean, assign_var]):
                return (batch - mean) / tf.sqrt(var + 1e-10)

    def __init_weights(self):
        shapes = list(zip(self.layers[:-1], self.layers[1:]))  # shapes of linear layers
        self.weights = {}
        with tf.name_scope('encoder_weights'):
            self.weights['W'] = [LadderNetwork.wi(s, "W_%d" % i) for i, s in enumerate(shapes)]
            self.weights['beta'] = \
                [LadderNetwork.bi(0.0, l, "beta_%d" % l) for l in self.layers[1:]]
            self.weights['gamma'] = \
                [LadderNetwork.bi(1.0, l, "gamma_%d" % l) for l in self.layers[1:]]
        with tf.name_scope('decoder_weights'):
            self.weights['V'] = [LadderNetwork.wi(s[::-1], "V") for s in shapes]

        self.ewma = tf.train.ExponentialMovingAverage(
            decay=0.99)  # to calculate the moving averages of mean and variance
        self.bn_assigns = []  # this list stores the updates to be made to average mean and variance
        self.running_mean = [tf.Variable(tf.constant(0.0, shape=[l]), trainable=False) for l in self.layers[1:]]
        self.running_var = [tf.Variable(tf.constant(1.0, shape=[l]), trainable=False) for l in self.layers[1:]]

    def build_encoder(self, inputs, noise_std):
        clean = True if noise_std == 0. else False
        with tf.name_scope('clean_encoder' if clean else 'corrupted_encoder'):
            L = len(self.layers) - 1
            h = inputs + tf.random_normal(tf.shape(inputs)) * noise_std  # add noise to input
            d = {}  # to store the pre-activation, activation, mean and variance for each layer
            # The data for labeled and unlabeled examples are stored separately
            d['labeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
            d['unlabeled'] = {'z': {}, 'm': {}, 'v': {}, 'h': {}}
            d['labeled']['z'][0], d['unlabeled']['z'][0] = split_lu(h)
            for l in range(1, L + 1):
                with tf.name_scope('layer_%d' % l):
                    print("Layer ", l, ": ", self.layers[l - 1], " -> ", self.layers[l])
                    d['labeled']['h'][l - 1], d['unlabeled']['h'][l - 1] = split_lu(h)
                    z_pre = tf.matmul(h, self.weights['W'][l - 1])  # pre-activation
                    z_pre_l, z_pre_u = split_lu(z_pre)  # split labeled and unlabeled examples

                    m, v = tf.nn.moments(z_pre_u, axes=[0])

                    # if training:
                    def training_batch_norm():
                        # Training batch normalization
                        # batch normalization for labeled and unlabeled examples is performed separately
                        if noise_std > 0:
                            # Corrupted encoder
                            # batch normalization + noise
                            z = join(LadderNetwork.batch_normalization(z_pre_l),
                                     LadderNetwork.batch_normalization(z_pre_u, m, v))
                            z += tf.random_normal(tf.shape(z_pre)) * noise_std
                        else:
                            # Clean encoder
                            # batch normalization + update the average mean and variance using batch mean and variance of labeled examples
                            z = join(self.update_batch_normalization(z_pre_l, l),
                                     LadderNetwork.batch_normalization(z_pre_u, m, v))
                        return z

                    # else:
                    def eval_batch_norm():
                        # Evaluation batch normalization
                        # obtain average mean and variance and use it to normalize the batch
                        mean = self.ewma.average(self.running_mean[l - 1])
                        var = self.ewma.average(self.running_var[l - 1])
                        z = self.batch_normalization(z_pre, mean, var)
                        # Instead of the above statement, the use of the following 2 statements containing a typo
                        # consistently produces a 0.2% higher accuracy for unclear reasons.
                        # m_l, v_l = tf.nn.moments(z_pre_l, axes=[0])
                        # z = join(batch_normalization(z_pre_l, m_l, mean, var), batch_normalization(z_pre_u, mean, var))
                        return z

                    # perform batch normalization according to value of boolean "training" placeholder:
                    with tf.name_scope('batchnorm'):
                        z = tf.cond(self.training, training_batch_norm, eval_batch_norm)

                    if l == L:
                        # use softmax activation in output layer
                        h = tf.nn.softmax(self.weights['gamma'][l - 1] * (z + self.weights["beta"][l - 1]))
                    else:
                        # use ReLU activation in hidden layers
                        h = tf.nn.relu(z + self.weights["beta"][l - 1])
                    d['labeled']['z'][l], d['unlabeled']['z'][l] = split_lu(z)
                    d['unlabeled']['m'][l], d['unlabeled']['v'][
                        l] = m, v  # save mean and variance of unlabeled examples for decoding
            d['labeled']['h'][l], d['unlabeled']['h'][l] = split_lu(h)
            return h, d

    @staticmethod
    def g_gauss(graph, z_c, u, size):
        """
        gaussian denoising function proposed in the original paper
        """
        with graph.name_scope('g_gauss'):
            wi = lambda inits, name: tf.Variable(inits * tf.ones([size]), name=name)
            a1 = wi(0., 'a1')
            a2 = wi(1., 'a2')
            a3 = wi(0., 'a3')
            a4 = wi(0., 'a4')
            a5 = wi(0., 'a5')

            a6 = wi(0., 'a6')
            a7 = wi(1., 'a7')
            a8 = wi(0., 'a8')
            a9 = wi(0., 'a9')
            a10 = wi(0., 'a10')

            mu = a1 * tf.sigmoid(a2 * u + a3) + a4 * u + a5
            v = a6 * tf.sigmoid(a7 * u + a8) + a9 * u + a10

            z_est = (z_c - mu) * v + mu
            return z_est

    @staticmethod
    def denoise_gauss(graph, z, u, size):
        """
        :param size: size of vector u
        :param z tensor of corrupted activations of the l-th layer of noisy encoder
        :param u tensor of activations on the l+1 - th layer of the decoder (assumedly batch-normalized)
        :return z_hat denoised activations
        """
        with graph.name_scope('gaussian_denoise'):
            # assert z.get_shape() == u.get_shape() -- always false
            w = lambda init, name: tf.Variable(initial_value=init * tf.ones([size]), name=name)
            b_0 = w(0., 'b_0')
            w_0z = w(1., 'w_0z')
            w_0u = w(0., 'w_0u')
            w_0zu = w(0., 'w_0zu')
            w_sigma = w(1., 'w_sigma')
            b_1 = w(0., 'b_1')
            w_1z = w(1., 'w_1z')
            w_1u = w(0., 'w_1u')
            w_1zu = w(0., 'w_1zu')
            activation = b_0 + w_0z * z + w_0u * u + w_0zu * z * u + \
                         w_sigma * tf.sigmoid(b_1 + w_1z * z + w_1u * u + w_1zu * z * u)
        return activation

    def build_decoder(self):
        with tf.name_scope('decoder'):
            z_est = {}
            self.d_cost = []  # to store the denoising cost of all layers
            L = len(self.layers) - 1
            for l in range(L, -1, -1):
                with tf.name_scope('layer_%d' % l):
                    print("Layer ", l, ": ", self.layers[l + 1] if l + 1 < len(self.layers) else None, " -> ",
                          self.layers[l],
                          ", denoising cost: ", self.denoising_cost[l])
                    z, z_c = self.clean['unlabeled']['z'][l], self.corr['unlabeled']['z'][l]
                    m, v = self.clean['unlabeled']['m'].get(l, 0), self.clean['unlabeled']['v'].get(l, 1 - 1e-10)
                    if l == L:
                        u = unlabeled(self.y_c)
                    else:
                        u = tf.matmul(z_est[l + 1], self.weights['V'][l])
                    u = LadderNetwork.batch_normalization(u)
                    z_est[l] = LadderNetwork.g_gauss(self.graph, z_c, u, self.layers[l])
                    z_est_bn = (z_est[l] - m) / v
                    # append the cost of this layer to d_cost
                    with tf.name_scope('loss'):
                        self.d_cost.append(
                            (tf.reduce_mean(tf.reduce_sum(tf.square(z_est_bn - z), 1)) / self.layers[l])
                            * self.denoising_cost[l])

    def build_cost(self):
        with tf.name_scope('loss'):
            self.u_cost = tf.add_n(self.d_cost)

            y_N = labeled(self.y_c)
            self.cost = -tf.reduce_mean(tf.reduce_sum(self.outputs * tf.log(y_N), 1))  # supervised cost
            self.loss = self.cost + self.u_cost  # total cost

        with tf.name_scope('prediction'):
            pred_cost = -tf.reduce_mean(tf.reduce_sum(self.outputs * tf.log(self.y), 1))  # cost used for prediction
            correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.outputs, 1))  # no of correct predictions
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)

        learning_rate = tf.Variable(self.learning_rate, trainable=False)
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # add the updates of batch normalization statistics to train_step
        bn_updates = tf.group(*self.bn_assigns)
        with tf.control_dependencies([train_step]):
            self.train_step = tf.group(bn_updates)

    def __check_valid(self):
        assert self.session is not None
        assert self.session.graph is self.graph or self.graph is tf.get_default_graph()
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())
            self.initialized = True

    def train_on_batch(self, x, y, step=[0]):
        scalar_log_period = 5
        hist_log_period = 10
        if self.scalar_summary is not None and step[0] % scalar_log_period == 0:
            nodes = [self.scalar_summary, self.train_step]
            if step[0] % hist_log_period == 0:
                nodes += [self.histogram_summary]
            res = self.session.run(nodes, feed_dict={self.inputs: x, self.outputs: y, self.training: 1})
            for node in res:
                if node is not None:
                    self.writer.add_summary(node, step[0])
            self.writer.flush()
        else:
            self.session.run(self.train_step, feed_dict={self.inputs: x, self.outputs: y,
                                                         self.training: 1})
        step[0] += 1

    def acc(self, x, y):
        return self.session.run(self.accuracy, feed_dict={self.inputs: x, self.outputs: y, self.training: 0})

    def log_all(self, dir_path):
        """
        creates all logging instances
        logs supervised cost and layer-wise and total unsupervised costs
        also logs histograms of each layer of encoder and decoder
        :return: log writer
        """
        self.__log_all_costs()
        self.__log_all_histograms()
        self.writer = tf.summary.FileWriter(dir_path, self.graph)
        return self.writer

    def predict_proba(self, X, verbose=False):
        iters = int(math.ceil(len(X) / self.batch_size))
        iters = tqdm(range(iters)) if verbose else range(iters)
        out = np.zeros((len(X), self.layers[-1]))
        for i in iters:
            out[self.ith_slice(i, X)] = self.session.run(self.y, feed_dict={
                self.inputs: X[self.ith_slice(i, X)],
                self.training: 0
            })
        return out

    def __log_all_histograms(self):
        encoder = []
        for attr in ['W', 'beta', 'gamma']:
            for layer in range(len(self.layers) - 1):
                encoder.append(tf.summary.histogram('encoder_layer_{0}_{1}'.format(str(layer), attr),
                                                    self.weights[attr][layer]))
        decoder = []
        for layer in range(len(self.layers) - 1):
            decoder.append(tf.summary.histogram('decoder_layer_{0}_weights'.format(str(layer)),
                                                self.weights['V'][layer]))
        self.histogram_summary = tf.summary.merge(encoder + decoder)

    def __log_all_costs(self):
        sup_summary = tf.summary.scalar('supervised cost', self.cost)
        unsup_summary = tf.summary.scalar('unsupervised cost', self.u_cost)
        denoise_summary = []
        for i, d_cost in enumerate(reversed(self.d_cost)):
            denoise_summary.append(tf.summary.scalar('denoising cost on layer {0}'.format(str(i)), d_cost))
        self.scalar_summary = tf.summary.merge([sup_summary, unsup_summary, *denoise_summary])

    def predict(self, X, verbose=False):
        out = self.predict_proba(X, verbose=verbose)
        return np.eye(self.layers[-1], self.layers[-1])[np.argmax(out, 1)]

    def ith_slice(self, i, data):
        return slice(i * self.batch_size, min((i + 1) * self.batch_size, len(data)))

    def fit(self, data, test_x, test_y, epochs=5, ratio=None, verbose=False):
        """
        fits model
        this method should be launched in the session scope
        with session.graph equals to self.graph

        :param data: train data
        :param epochs: num of epochs of learning
        :param ratio:
        :return: fitted model
        """
        for i in tqdm(list(range(0, epochs))):
            # for images, labels in data:
            images, labels = data.next_batch(self.batch_size)
            self.train_on_batch(images, labels)
            # self.session.run(self.train_step, feed_dict={self.inputs: images, self.outputs: labels, self.training: True})
            if i % 50 == 0:
                print("Epoch ", i, ", Accuracy: ",
                      self.session.run(self.accuracy,
                                       feed_dict={self.inputs: test_x,
                                                  self.outputs: test_y, self.training: False}),
                      "%")
                with open('train_log', 'a') as train_log:
                    # write test accuracy to file "train_log"
                    log_i = [i] + self.session.run([self.accuracy],
                                                   feed_dict={self.inputs: test_x,
                                                              self.outputs: test_y,
                                                              self.training: False})
                    train_log.write(','.join(map(str, log_i)) + '\n')
