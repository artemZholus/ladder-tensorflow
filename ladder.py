import tensorflow as tf
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from utils import get_decay
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelBinarizer
from sklearn import datasets

class LadderNetwork(BaseEstimator):
    hyperparameters = {
        'learning_rate': 0.01,
        'denoise_cost_init': 10,
        'denoise_cost_decay': 'hyperbolic_decay',
        'denoise_cost_param': 1,
        'noise_std': 0.5
    }

    def __init__(self, layers):
        self.layers, self.activation = zip(*layers)
        self.clean = None
        self.corrupted = None
        self.decoder = None
        self.session = None
        self.batch_size = -1
        self.learning_rate = tf.constant(LadderNetwork.hyperparameters['learning_rate'])
        self.denoise_cost = tf.constant(
            get_decay(LadderNetwork.hyperparameters['denoise_cost_decay'])(
                LadderNetwork.hyperparameters['denoise_cost_init'],
                LadderNetwork.hyperparameters['denoise_cost_param'],
                len(self.layers)
            ))
        self.noise_std = tf.constant(LadderNetwork.hyperparameters['noise_std'])
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.layers[0]))
        self.outputs = tf.placeholder(tf.float32)
        self.clean, self.corrupted, self.decoder = self.__build()

    def __build(self):
        clean, corrupted = self.__build_encoder()
        decoder = self.__build_decoder(corrupted, clean)
        return clean, corrupted, decoder

    def __encoder_weights(self):
        with tf.name_scope('encoder'):
            encoder_weights = [None]
            encoder_bn_gamma = [None]
            encoder_bn_beta = [None]
            for i, shape in enumerate(zip(self.layers[:-1], self.layers[1:])):
                with tf.name_scope('weights_' + str(i)):
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
                with tf.name_scope(clean_str + '_layer_' + str(l)):
                    h_pre = tf.matmul(h[-1], encoder.weights[l])
                    m, v = tf.nn.moments(h_pre, axes=[0])
                    h_pre = (h_pre - m) / v
                    if clean:
                        mean.append(m)
                        variance.append(v)
                    if not clean:
                        h_pre += tf.random_normal(tf.shape(h_pre)) * self.noise_std
                    z_pre.append(h_pre)
                    h.append(self.activation[l](encoder.gamma[l] * (h_pre + encoder.beta[l])))
            if clean:
                return h, mean, variance, z_pre
            return h, z_pre
        clean, corrupted = self.__encoder_weights()
        h, mean, variance, z_pre_cl = encoder_(self.inputs, clean)
        h_tilde, z_pre = encoder_(self.inputs, corrupted)
        clean.activations = h
        clean.mean = mean
        clean.variance = variance
        clean.preactivations = z_pre_cl
        corrupted.activations = h_tilde
        corrupted.preactivations = z_pre
        return clean, corrupted

    def __decoder_weights(self):
        with tf.name_scope('decoder'):
            decoder_weights = []
            for i, shape in enumerate(zip(self.layers[1:], self.layers[:-1])):
                decoder_weights.append(
                    tf.Variable(tf.random_normal(shape, name='decoder_layer_' + str(i)))
                )
        return LadderNetwork.Decoder(decoder_weights)

    def __build_decoder(self, corrupted_encoder, clean_encoder):
        decoder = self.__decoder_weights()
        decoder.encoder = corrupted_encoder
        def decoder_():
            z_pre = corrupted_encoder.preactivations
            z_hat = {}
            for l in reversed(range(len(self.layers) - 1)):
                with tf.name_scope('decoder_layer_' + str(l)):
                    if l == len(self.layers) - 2:
                        u = decoder.encoder.activations[-1]
                    else:
                        u = tf.matmul(z_hat[l + 1], decoder.weights[l + 1])
                    z_hat[l] = LadderNetwork.denoise_gauss(z_pre[l], u, self.layers[l + 1])
                    z_hat[l] = (z_hat[l] - clean_encoder.mean[l]) / (clean_encoder.variance[l] + tf.constant(10e-8))
            return z_hat
        z_hat = decoder_()
        decoder.activations = z_hat
        return decoder

    def __supervised_cost(self):
        with tf.name_scope('supervised_cost'):
            cost = -tf.reduce_mean(tf.reduce_sum(self.outputs * tf.log(self.corrupted.activations[-1]), 1)) # why corrupted
        return cost

    def __unsupervised_cost(self):
        with tf.name_scope('unsupervised_cost'):
            denoise = []
            for l in reversed(range(len(self.layers) - 1)):
                with tf.name_scope('denoise_cost_' + str(l)):
                    denoise.append(
                        tf.reduce_mean(
                            tf.square(self.decoder.activations[l] - self.corrupted.preactivations[l]),
                            1
                        ) * self.denoise_cost[l]
                    )
                pass
            u_cost = tf.add_n(denoise)
        return u_cost

    def cost(self):
        cost = self.__supervised_cost()
        u_cost = self.__unsupervised_cost()
        return cost, u_cost

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
        :param z tensor of corrupted activations of the l-th layer of noisy encoder
        :param u tensor of activations on the l+1 - th layer of the decoder (assumedly batch-normalized)
        :return z_hat denoised activations
        """
        with tf.name_scope('gaussian_denoise'):
            #assert z.get_shape() == u.get_shape() -- always false
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
    def batch_norm(batch):
        with tf.name_scope('batch_norm'):
            mean, variance = tf.nn.moments(batch, axes=[0])
            norm = (batch - mean) / tf.sqrt(variance + tf.constant(1e-9))
        return norm

def load_data(path):
    train = pd.read_csv(path + '/train.csv').drop('id', axis=1, inplace=False)
    test = pd.read_csv(path + '/test.csv').drop('id', axis=1, inplace=False)
    train_Y = LabelBinarizer().fit_transform(train.target)
    train.drop('target', axis=1, inplace=True)
    return train, train_Y, test

if __name__ == '__main__':
    layers = [
        (93, None),
        (1024, tf.nn.relu),
        (512, tf.nn.relu),
        (128, tf.nn.relu),
        (9, tf.nn.softmax)
    ]
    nn = LadderNetwork(layers)
    cost, u_cost = nn.cost()
    session = tf.InteractiveSession()
    session.run(tf.global_variables_initializer())
    learning_rate = tf.Variable(LadderNetwork.hyperparameters['learning_rate'], trainable=False)
    supervised = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    unsupervised = tf.train.AdamOptimizer(learning_rate).minimize(u_cost)
    writer = tf.summary.FileWriter('./logs/', session.graph)
    input('done!')
