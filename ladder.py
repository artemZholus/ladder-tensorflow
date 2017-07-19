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

hyperparameters = {
    'learning_rate': 0.001,
    'denoise_cost_init': 1,
    'denoise_cost': 'hyperbolic_decay',
    'denoise_cost_param': 1,
    'batch_size': 100,
    'noise_std': 0.2,
}
import tensorflow as tf

import math
import os
import csv
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

layer_sizes = [784, 1000, 500, 250, 250, 250, 10]

L = len(layer_sizes) - 1  # number of layers

num_examples = 100
num_epochs = 150
num_labeled = 100

starter_learning_rate = 0.02

decay_after = 15  # epoch after which to begin learning rate decay

batch_size = 100
num_iter = int((num_examples / batch_size) * num_epochs)  # number of loop iterations



class LadderNetwork(BaseEstimator):
    def __init__(self, layers_, batch_size, learning_rate=1e-4, noise_std=0.3, denoise_cost=None,
                 denoise_cost_init=None, denoise_cost_param=None):
        self.layers_ = layers_
        self.layers, self.activation = zip(*layers_)
        self.batch_size = batch_size
        self.graph = tf.Graph()
        self.session = tf.get_default_session()
        self.denoise_cost_init = denoise_cost_init
        self.denoise_cost_param = denoise_cost_param
        if self.session is not None:
            self.session.close()
        self.session = None
        if not isinstance(denoise_cost, str):
            decay = denoise_cost
        else:
            decay = get_decay(denoise_cost)(denoise_cost_init, denoise_cost_param, len(self.layers))
        self.learning_rate = learning_rate
        self.denoise_cost = decay
        self.noise_std = noise_std
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.float32, shape=(None, self.layers[0]))
            self.outputs = tf.placeholder(tf.float32)
            self.learning_phase = tf.placeholder(tf.bool)
            self.ewma = tf.train.ExponentialMovingAverage(decay=0.99)
            self.bn_assigns = []
            self.running_mean = []
            self.running_var = []
            self.__build()
        self.supervised_summary = None
        self.unsupervised_summary = None
        self.supervised_histograms = None
        self.unsupervised_histograms = None
        self.writer = None
        self.session = tf.InteractiveSession(graph=self.graph)
        self.initialized = False

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

    def __check_valid(self):
        assert self.session is not None
        assert self.session.graph is self.graph or self.graph is tf.get_default_graph()
        if not self.initialized:
            self.session.run(tf.global_variables_initializer())
            self.initialized = True

    def __build(self):
        print('- Building Ladder Network...')
        self.clean, self.corrupted = self.__build_encoder()
        self.decoder = self.__build_decoder(self.corrupted, self.clean)
        self.supervised_cost = self.__build_supervised_learning_rule()
        self.unsupervised_cost, self.denoising_costs = \
            self.__build_unsupervised_learning_rule()
        self.cost = self.unsupervised_cost + self.supervised_cost
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        bn_updates = tf.group(*self.bn_assigns)
        with tf.control_dependencies([self.optimizer]):
            self.optimizer = tf.group(bn_updates)
        self.prediction = self.__build_prediction()
        self.__build_accuracy()
        self.predict_probas = self.__build_predict_proba()

        print('- Ladder Network built!')

    def __encoder_weights(self):
        with self.graph.name_scope('encoder'):
            encoder_weights = [None]
            encoder_bn_gamma = [None]
            encoder_bn_beta = [None]
            for i, shape in enumerate(zip(self.layers[:-1], self.layers[1:])):
                with tf.name_scope('running'):
                    self.running_mean.append(tf.Variable(tf.constant(0.0, shape=[shape[1]]),
                                                         name='mean_%d' % i, trainable=False))
                    self.running_var.append(tf.Variable(tf.constant(1.0, shape=[shape[1]]),
                                                        name='var_%d' % i, trainable=False))
                with self.graph.name_scope('weights_' + str(i)):
                    encoder_weights.append(tf.Variable(tf.random_normal(shape, name='weight_' + str(i))))
                    encoder_bn_beta.append(tf.Variable(tf.zeros([shape[1]]), name='beta_' + str(i)))
                    encoder_bn_gamma.append(tf.Variable(tf.ones([shape[1]]), name='gamma_' + str(i)))
        return LadderNetwork.Encoder(encoder_weights, encoder_bn_beta, encoder_bn_gamma), \
               LadderNetwork.CorruptedEncoder(encoder_weights, encoder_bn_beta, encoder_bn_gamma)

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

    def __build_encoder(self):
        """
        name conventions:
            h - outputs of the clean encoder's layers
            h_tilde - output of the corrupted encoder's layers
        :return: None
        """
        print('\t- Building Encoder...')

        def encoder_(input, encoder):
            z_pre = []
            h = []
            clean = not isinstance(encoder, LadderNetwork.CorruptedEncoder)
            clean_str = 'clean' if clean else 'corrupted'
            mean, variance = [], []
            labeled_preactivations = []
            labeled_activations = []
            unlabeled_preactivations = []
            unlabeled_activations = []
            if clean:
                h.append(input)
            else:
                h.append(input + tf.random_normal(tf.shape(input)) * self.noise_std)
            for l in range(1, len(self.layers)):
                for x in [labeled_preactivations, labeled_activations,
                          unlabeled_preactivations, unlabeled_activations]:
                    x.append(0)
                with self.graph.name_scope(clean_str + '_layer_' + str(l)):
                    h_pre = tf.matmul(h[-1], encoder.weights[l])
                    h_pre_l, h_pre_u = split_lu(h_pre, batch_size=self.batch_size)

                    m, v = tf.nn.moments(h_pre_u, [0])

                    def training_batch_norm():
                        if not clean:
                            z = join(LadderNetwork.batch_normalization(h_pre_l),
                                     LadderNetwork.batch_normalization(h_pre_u, m, v))
                            z += tf.random_normal(tf.shape(h_pre)) * self.noise_std
                        else:
                            z = join(self.update_batch_normalization(h_pre_l, l),
                                     LadderNetwork.batch_normalization(h_pre_u, m, v))
                        return z

                    def eval_batch_norm():
                        mean = self.ewma.average(self.running_mean[l - 1])
                        var = self.ewma.average(self.running_var[l - 1])
                        z = LadderNetwork.batch_normalization(h_pre, mean, var)
                        return z

                    with tf.name_scope('batch_norm'):
                        h_pre = tf.cond(self.learning_phase, training_batch_norm, eval_batch_norm)

                    if clean:
                        mean.append(m)
                        variance.append(v)
                    labeled_preactivations[-1], unlabeled_preactivations[-1] = \
                        split_lu(h_pre, batch_size=self.batch_size)
                    h_pre_l = encoder.gamma[l] * (h_pre + encoder.beta[l])
                    z_pre.append(h_pre)
                    h.append(self.activation[l](h_pre_l))
                    labeled_activations[-1], unlabeled_activations[-1] = split_lu(h[-1], batch_size=self.batch_size)
            if clean:
                return labeled_activations, unlabeled_activations, \
                       mean, variance, labeled_preactivations, unlabeled_preactivations
            return labeled_activations, unlabeled_activations, labeled_preactivations, unlabeled_preactivations

        print('\t\t- Initializing Encoder weights...')
        clean, corrupted = self.__encoder_weights()
        print('\t\t\t- Building clean encoder...')
        labeled_activations_clean, unlabeled_activations_clean, \
        mean, variance, labeled_preactivations_clean, unlabeled_preactivations_clean = encoder_(self.inputs, clean)
        print('\t\t\t- Building corrupted encoder...')
        labeled_activations_corrupted, unlabeled_activations_corrupted, \
        labeled_preactivations_corrputed, unlabeled_preactivations_corrupted = encoder_(self.inputs, corrupted)
        clean.labeled_activations = labeled_activations_clean
        clean.unlabeled_activations = unlabeled_activations_clean
        clean.unlabeled_preactivations = unlabeled_preactivations_clean
        clean.labeled_preactivations = labeled_preactivations_clean
        clean.mean = mean
        clean.variance = variance
        corrupted.labeled_activations = labeled_activations_corrupted
        corrupted.unlabeled_activations = unlabeled_activations_corrupted
        corrupted.labeled_preactivations = labeled_preactivations_corrputed
        corrupted.unlabeled_preactivations = unlabeled_preactivations_corrupted
        print('\t- Encoder built!')
        return clean, corrupted

    class Decoder:
        def __init__(self, weights):
            self.weights = weights
            self.activations = None
            self.encoder = None
            self.bn_activations = None

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

    def __decoder_weights(self) -> Decoder:
        with self.graph.name_scope('decoder'):
            decoder_weights = []
            for i, shape in enumerate(zip(self.layers[1:], self.layers[:-1])):
                decoder_weights.append(
                    tf.Variable(tf.random_normal(shape, name='decoder_layer_' + str(i)))
                )
        return LadderNetwork.Decoder(decoder_weights)

    class Encoder:
        def __init__(self, weights, beta, gamma):
            self.weights = weights
            self.beta = beta
            self.gamma = gamma
            self.mean = None
            self.variance = None
            self.labeled_activations = None
            self.unlabeled_activations = None
            self.labeled_preactivations = None
            self.unlabeled_preactivations = None

    class CorruptedEncoder(Encoder):
        def __init__(self, weights, beta, gamma):
            super().__init__(weights, beta, gamma)
            self.decoder = None

    def __build_decoder(self, corrupted_encoder: CorruptedEncoder, clean_encoder):
        print('\t- Building decoder...')
        print('\t\t- Initializing decoder weights...')
        decoder = self.__decoder_weights()
        decoder.encoder = corrupted_encoder

        def decoder_():
            z_pre = corrupted_encoder.unlabeled_preactivations
            z_hat = {}
            z_hat_n = {}
            for l in reversed(range(len(self.layers) - 1)):
                with self.graph.name_scope('decoder_layer_' + str(l)):
                    if l == len(self.layers) - 2:
                        u = decoder.encoder.unlabeled_activations[-1]
                    else:
                        u = tf.matmul(z_hat[l + 1], decoder.weights[l + 1])
                    u = LadderNetwork.batch_normalization(u)
                    # z_hat[l] = LadderNetwork.Decoder.denoise_gauss(z_pre[l], u, self.layers[l + 1])
                    z_hat[l] = LadderNetwork.Decoder.g_gauss(self.graph, z_pre[l], u, self.layers[l + 1])
                    z_hat_n[l] = (z_hat[l] - clean_encoder.mean[l]) / (clean_encoder.variance[l] + tf.constant(10e-8))
            return z_hat, z_hat_n

        z_hat, z_hat_n = decoder_()
        decoder.activations = z_hat
        decoder.bn_activations = z_hat_n
        print('\t- Decoder built!')
        return decoder

    def __supervised_cost(self):
        with self.graph.name_scope('supervised_cost'):
            cost = tf.nn.softmax_cross_entropy_with_logits(labels=self.outputs,
                                                           logits=self.corrupted.labeled_preactivations[-1])
            # why corrupted
        return cost

    def __unsupervised_cost(self):
        with self.graph.name_scope('unsupervised_cost'):
            denoise = []
            for l in reversed(range(len(self.layers) - 1)):
                with self.graph.name_scope('denoise_cost_' + str(l)):
                    denoise.append(
                        tf.reduce_mean(
                            tf.reduce_sum(
                                tf.square(self.decoder.bn_activations[l] - self.clean.unlabeled_preactivations[l]),
                                1
                            ) * self.denoise_cost[l]
                        )
                    )
                pass
            u_cost = tf.add_n(denoise)
        return u_cost, denoise

    def __build_unsupervised_learning_rule(self):
        print('\t- Building unsupervised learning rule...')
        with self.graph.name_scope('unsup_learning_rule'):
            u_cost, denoise_costs = self.__unsupervised_cost()
        print('\t- Unsupervised learning rule built!')
        return u_cost, denoise_costs

    def __build_supervised_learning_rule(self):
        print('\t- Building supervised learning rule...')
        with self.graph.name_scope('sup_learning_rule'):
            cost = tf.reduce_mean(self.__supervised_cost())
        print('\t- Supervised learning rule built!')
        return cost

    def __build_predict_proba(self):
        return self.clean.labeled_activations[-1]

    def __build_prediction(self):
        with self.graph.name_scope('prediction'):
            prediction = tf.argmax(self.clean.labeled_activations[-1], 1)
        return prediction

    def __build_accuracy(self):
        pass

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

    def __log_all_costs(self):
        sup_summary = tf.summary.scalar('supervised cost', self.supervised_cost)
        unsup_summary = tf.summary.scalar('unsupervised cost', self.unsupervised_cost)
        denoise_summary = []
        for i, d_cost in enumerate(self.denoising_costs):
            denoise_summary.append(tf.summary.scalar('denoising cost on layer {0}'.format(str(i)), d_cost))
        self.scalar_summary = tf.summary.merge([sup_summary, unsup_summary, *denoise_summary])

    def __log_all_histograms(self):
        encoder = []
        for attr in ['weights', 'beta', 'gamma']:
            for layer in range(1, len(self.layers)):
                encoder.append(tf.summary.histogram('encoder_layer_{0}_{1}'.format(str(layer), attr),
                                                    getattr(self.clean, attr)[layer]))
        decoder = []
        for layer in range(len(self.layers) - 1):
            decoder.append(tf.summary.histogram('decoder_layer_{0}_weights'.format(str(layer)),
                                                self.decoder.weights[layer]))
        self.histogram_summary = tf.summary.merge(encoder + decoder)

    def train_on_batch(self, X, y, step=[0]):
        self.__check_valid()
        if self.scalar_summary is not None:
            summary, hist, loss = self.session.run([self.scalar_summary,
                                                    self.histogram_summary,
                                                    self.optimizer],
                                                   feed_dict={self.inputs: X, self.outputs: y,
                                                              self.learning_phase: 1})
            if step[0] % 5 == 0:
                self.writer.add_summary(summary, step[0])
                self.writer.add_summary(hist, step[0])
                self.writer.flush()
            step[0] += 1
        else:
            loss = self.session.run(self.optimizer, feed_dict={self.inputs: X, self.outputs: y,
                                                               self.learning_phase: 1})
        return loss

    def predict_proba(self, X, verbose=False):
        self.__check_valid()
        iters = int(math.ceil(len(X) / self.batch_size))
        iters = tqdm(range(iters)) if verbose else range(iters)
        out = np.zeros((len(X), self.layers[-1]))
        for i in iters:
            out[self.ith_slice(i, X)] = self.session.run(self.predict_probas, feed_dict={
                self.inputs: X[self.ith_slice(i, X)],
                self.learning_phase: 0
            })
        return out

    def accuracy(self, x, y):
        pred = self.predict(x)
        return (np.argmax(pred, 1) == np.argmax(y, 1)).mean()

    def predict(self, X, verbose=False):
        out = self.predict_proba(X, verbose=verbose)
        return np.eye(self.layers[-1], self.layers[-1])[np.argmax(out, 1)]

    def ith_slice(self, i, data):
        return slice(i * self.batch_size, min((i + 1) * self.batch_size, len(data)))

    def fit(self, data, epochs=5, ratio=None, verbose=False):
        """
        fits model
        this method should be launched in the session scope
        with session.graph equals to self.graph

        :param data: train data
        :param epochs: num of epochs of learning
        :param ratio:
        :return: fitted model
        """
        self.__check_valid()
        iter = list(range(0, epochs))
        iter = tqdm(iter) if verbose else iter
        for i in iter:
            images, labels = data.train.next_batch(self.batch_size)
            self.session.run(self.optimizer,
                             feed_dict={self.inputs: images, self.outputs: labels, self.learning_phase: 1})
            if i % 50 == 0 and verbose:
                print("Epoch ", i, ", Accuracy: ",
                      self.accuracy(data.test.images, data.test.labels) * 100, "%")
        # for epoch_num in range(epochs):
        #     print('Epoch No. {0}'.format(str(epoch_num)))
        #     for i, (x, y) in tqdm(enumerate(data)):
        #         self.train_on_batch(x, y)
