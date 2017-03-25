import time
import tensorflow as tf
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from utils import get_decay, semisupervised_batch_iterator
from tqdm import tqdm
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn import datasets
from functools import wraps
from utils import conditional_context

hyperparameters = {
    'learning_rate': 0.0001,
    'denoise_cost_init': 0,
    'denoise_cost': 'hyperbolic_decay',
    'denoise_cost_param': 1,
    'noise_std': 0.05,
}


class LadderNetwork(BaseEstimator):
    def __init__(self, layers, learning_rate=1e-4, noise_std=0.05, denoise_cost=None,
                 denoise_cost_init=None, denoise_cost_param=None):
        self.layers, self.activation = zip(*layers)
        self.batch_size = -1
        self.graph = tf.Graph()
        self.session = tf.get_default_session()
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
            self.__build()
        self.supervised_summary = None
        self.unsupervised_summary = None
        self.supervised_histograms = None
        self.unsupervised_histograms = None
        self.writer = None
        self.session = tf.InteractiveSession(graph=self.graph)
        self.session.run(tf.global_variables_initializer())

    def __check_valid(self):
        assert self.session is not None
        assert self.session.graph is self.graph or self.graph is tf.get_default_graph()

    def __build(self):
        print('- Building Ladder Network...')
        self.clean, self.corrupted = self.__build_encoder()
        self.decoder = self.__build_decoder(self.corrupted, self.clean)
        self.supervised_optimizer, self.supervised_cost = self.__build_supervised_learning_rule()
        self.unsupervised_optimizer, self.unsupervised_cost, self.denoising_costs = \
            self.__build_unsupervised_learning_rule()
        self.prediction = self.__build_prediction()
        self.predict_probas = self.__build_predict_proba()
        print('- Ladder Network built!')

    def __encoder_weights(self):
        with self.graph.name_scope('encoder'):
            encoder_weights = [None]
            encoder_bn_gamma = [None]
            encoder_bn_beta = [None]
            for i, shape in enumerate(zip(self.layers[:-1], self.layers[1:])):
                with self.graph.name_scope('weights_' + str(i)):
                    encoder_weights.append(tf.Variable(tf.random_normal(shape, name='weight_' + str(i))))
                    encoder_bn_beta.append(tf.Variable(tf.zeros([shape[1]]), name='beta_' + str(i)))
                    encoder_bn_gamma.append(tf.Variable(tf.ones([shape[1]]), name='gamma_' + str(i)))
        return LadderNetwork.Encoder(encoder_weights, encoder_bn_beta, encoder_bn_gamma), \
               LadderNetwork.CorruptedEncoder(encoder_weights, encoder_bn_beta, encoder_bn_gamma)

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
            if clean:
                h.append(input)
            else:
                h.append(input + tf.random_normal(tf.shape(input)) * self.noise_std)
            for l in range(1, len(self.layers)):
                with self.graph.name_scope(clean_str + '_layer_' + str(l)):
                    h_pre = tf.matmul(h[-1], encoder.weights[l])
                    m, v = tf.nn.moments(h_pre, axes=[0])
                    #h_pre = (h_pre - m) / v
                    if clean:
                        mean.append(m)
                        variance.append(v)
                    if not clean:
                        h_pre += tf.random_normal(tf.shape(h_pre)) * self.noise_std
                    z_pre.append(h_pre)
                    #h.append(self.activation[l](encoder.gamma[l] * (h_pre + encoder.beta[l])))
                    h.append(self.activation[l](h_pre))
            if clean:
                return h, mean, variance, z_pre
            return h, z_pre

        print('\t\t- Initializing Encoder weights...')
        clean, corrupted = self.__encoder_weights()
        print('\t\t\t- Building clean encoder...')
        h, mean, variance, z_pre_cl = encoder_(self.inputs, clean)
        print('\t\t\t- Building corrupted encoder...')
        h_tilde, z_pre = encoder_(self.inputs, corrupted)
        clean.activations = h
        clean.mean = mean
        clean.variance = variance
        clean.preactivations = z_pre_cl
        corrupted.activations = h_tilde
        corrupted.preactivations = z_pre
        print('\t- Encoder built!')
        return clean, corrupted

    def __decoder_weights(self):
        with self.graph.name_scope('decoder'):
            decoder_weights = []
            for i, shape in enumerate(zip(self.layers[1:], self.layers[:-1])):
                decoder_weights.append(
                    tf.Variable(tf.random_normal(shape, name='decoder_layer_' + str(i)))
                )
        return LadderNetwork.Decoder(decoder_weights)

    def __build_decoder(self, corrupted_encoder, clean_encoder):
        print('\t- Building decoder...')
        print('\t\t- Initializing decoder weights...')
        decoder = self.__decoder_weights()
        decoder.encoder = clean_encoder

        def decoder_():
            z_pre = corrupted_encoder.preactivations
            z_hat = {}
            for l in reversed(range(len(self.layers) - 1)):
                with self.graph.name_scope('decoder_layer_' + str(l)):
                    if l == len(self.layers) - 2:
                        u = decoder.encoder.activations[-1]
                    else:
                        u = tf.matmul(z_hat[l + 1], decoder.weights[l + 1])
                    z_hat[l] = LadderNetwork.Decoder.denoise_gauss(z_pre[l], u, self.layers[l + 1])
                    #z_hat[l] = (z_hat[l] - clean_encoder.mean[l]) / (clean_encoder.variance[l] + tf.constant(10e-8))
            return z_hat

        z_hat = decoder_()
        decoder.activations = z_hat
        print('\t- Decoder built!')
        return decoder

    def __supervised_cost(self):
        with self.graph.name_scope('supervised_cost'):
            cost = -tf.reduce_sum(
                tf.reduce_sum(self.outputs * tf.log(self.corrupted.activations[-1]), 1))  # why corrupted
        return cost

    def __unsupervised_cost(self):
        with self.graph.name_scope('unsupervised_cost'):
            denoise = []
            for l in reversed(range(len(self.layers) - 1)):
                with self.graph.name_scope('denoise_cost_' + str(l)):
                    denoise.append(
                        tf.reduce_mean(
                            tf.reduce_sum(
                                tf.square(self.decoder.activations[l] - self.corrupted.preactivations[l]),
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
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(u_cost)
        print('\t- Unsupervised learning rule built!')
        return optimizer, u_cost, denoise_costs

    def __build_supervised_learning_rule(self):
        print('\t- Building supervised learning rule...')
        with self.graph.name_scope('sup_learning_rule'):
            cost = self.__supervised_cost()
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        print('\t- Supervised learning rule built!')
        return optimizer, cost

    def __build_predict_proba(self):
        return self.corrupted.activations[-1]

    def __build_prediction(self):
        with self.graph.name_scope('prediction'):
            prediction = tf.argmax(self.corrupted.activations[-1], 1)
        return prediction

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
        self.supervised_summary = tf.summary.merge([sup_summary])
        self.unsupervised_summary = tf.summary.merge([unsup_summary, *denoise_summary])

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
        self.supervised_histograms = tf.summary.merge(encoder)
        self.unsupervised_histograms = tf.summary.merge(decoder)

    def train_on_batch_supervised(self, X, y, step=[0]):
        self.__check_valid()
        if self.supervised_summary is not None:
            summary, hist, loss = self.session.run([self.supervised_summary,
                                                    self.supervised_histograms,
                                                    self.supervised_optimizer],
                                                   feed_dict={self.inputs: X, self.outputs: y})
            if step[0] % 5 == 0:
                self.writer.add_summary(summary, step[0])
                self.writer.add_summary(hist, step[0])
                self.writer.flush()
            step[0] += 1
        else:
            loss = self.session.run(self.supervised_optimizer, feed_dict={self.inputs: X, self.outputs: y})
        return loss

    def train_on_batch_unsupervised(self, X, step=[0]):
        self.__check_valid()
        if self.unsupervised_summary is not None:
            summary, hist, loss = self.session.run([self.unsupervised_summary,
                                                    self.unsupervised_histograms,
                                                    self.unsupervised_optimizer],
                                                   feed_dict={self.inputs: X})
            if step[0] % 5 == 0:
                self.writer.add_summary(summary, step[0])
                self.writer.add_summary(hist, step[0])
                self.writer.flush()
            step[0] += 1
        else:
            loss = self.session.run(self.unsupervised_optimizer, feed_dict={self.inputs: X})
        return loss

    def train_on_batch(self, batch, labels):
        with self.graph.as_default():
            self.__check_valid()
            if len(batch) > len(labels):
                supervised_batch = batch[:len(labels)]
            else:
                supervised_batch = batch
            unsupervised_batch = batch
            self.train_on_batch_supervised(supervised_batch, labels)
            self.train_on_batch_unsupervised(unsupervised_batch)

    def predict(self, X):
        self.__check_valid()
        return self.session.run(self.prediction, feed_dict={self.inputs: X})

    def predict_proba(self, X):
        self.__check_valid()
        return self.session.run(self.predict_probas, feed_dict={self.inputs: X})

    def fit(self, X, y, epochs=5, batch_size=128, verbose=False, unsupervised_batch=None):
        """
        fits model
        this method should be launched in the session scope
        with session.graph equals to self.graph

        :param X: data array of size (m, n)
        :param y: labels array of size (k, n), k should always be <= m
        :param epochs: num of epochs of learning
        :param batch_size: batch size of supervised part
                total batch size is batch_size * (1 + unsupervised_ratio)
        :param unsupervised_batch: int > 0 unsupervised batch size
                                or float in [0..1] ratio coefficient means how many unsupervised examples we
                                use against supervised one
        :return: fitted model
        """
        self.__check_valid()
        if isinstance(unsupervised_batch, int):
            assert unsupervised_batch > 0
            ratio = unsupervised_batch / batch_size
        if unsupervised_batch is None:
            ratio = len(X) / len(y)
        for epoch_num in range(epochs):
            print('Epoch No. {0}'.format(str(epoch_num)))
            for i, (unsupervised, (supervised, labels)) in tqdm(enumerate(semisupervised_batch_iterator(
                    X, y, batch_size, ratio))):
                self.train_on_batch_supervised(supervised, labels)
                #if unsupervised is not None:
                #    self.train_on_batch_unsupervised(unsupervised)
                if i % 10 == 0 and verbose:
                    print('iter: %d' % i)


    class Encoder:
        def __init__(self, weights, beta, gamma):
            self.weights = weights
            self.beta = beta
            self.gamma = gamma
            self.mean = None
            self.variance = None
            self.activations = None
            self.preactivations = None

    class CorruptedEncoder(Encoder):
        def __init__(self, weights, beta, gamma):
            super().__init__(weights, beta, gamma)
            self.preactivations = None
            self.decoder = None

    class Decoder:
        def __init__(self, weights):
            self.weights = weights
            self.activations = None
            self.encoder = None

        @staticmethod
        def denoise_gauss(z, u, size):
            """
            :param size: size of vector u
            :param z tensor of corrupted activations of the l-th layer of noisy encoder
            :param u tensor of activations on the l+1 - th layer of the decoder (assumedly batch-normalized)
            :return z_hat denoised activations
            """
            with tf.get_default_graph().name_scope('gaussian_denoise'):
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

def load_data(path):
    train = pd.read_csv(path + '/train.csv').drop('id', axis=1, inplace=False)
    test = pd.read_csv(path + '/test.csv').drop('id', axis=1, inplace=False)
    train_Y = LabelEncoder().fit_transform(train.target)
    train_Y_binarized = LabelBinarizer().fit_transform(train.target)
    train.drop('target', axis=1, inplace=True)
    return train.values, train_Y, test.values, train_Y_binarized


if __name__ == '__main__':
    layers = [
        (93, None),
        (1024, tf.nn.relu),
        (512, tf.nn.relu),
        (128, tf.nn.relu),
        (64, tf.nn.relu),
        (9, tf.nn.softmax)
    ]
    train, labels, test, bin = load_data('./data')
    ladder = LadderNetwork(layers, **hyperparameters)
    input('All done to start learning!')
    ladder.fit(train, bin, batch_size=16, unsupervised_batch=16, verbose=True)
