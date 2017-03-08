import time
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


hyperparameters = {
    'learning_rate': 0.01,
    'denoise_cost_init': 10,
    'denoise_cost_decay': 'hyperbolic_decay',
    'denoise_cost_param': 1,
    'noise_std': 0.05,
    'batch_size': 256
}

class LadderNetwork(BaseEstimator):
    def __init__(self, layers, verbose=True, **params):
        self.layers, self.activation = zip(*layers)
        self.session = None
        self.batch_size = -1
        self.params = params
        self.learning_rate = tf.constant(params['learning_rate'])
        self.denoise_cost = tf.constant(
            get_decay(params['denoise_cost_decay'])(
                params['denoise_cost_init'],
                params['denoise_cost_param'],
                len(self.layers)
            ))
        self.noise_std = tf.constant(params['noise_std'])
        self.inputs = tf.placeholder(tf.float32, shape=(None, self.layers[0]))
        self.outputs = tf.placeholder(tf.float32)
        self.__build()
        self.session = tf.InteractiveSession()
        print('- Initializing weights...')
        self.session.run(tf.global_variables_initializer())
        print('- Weights initialized!')
        self.supervised_summary = None
        self.unsupervised_summary = None
        if verbose:
            print('- Creating logging instances...')
            self.writer = self.__log_all_costs()
            print('- Logging instances created!')
        else:
            self.writer = None
        self.accuracy = self.supervised_accuracy()

    def __build(self):
        print('- Building Ladder Network...')
        self.clean, self.corrupted = self.__build_encoder()
        self.decoder = self.__build_decoder(self.corrupted, self.clean)
        self.supervised_optimizer, self.supervised_cost = self.__build_supervised_learning_rule()
        self.unsupervised_optimizer, self.unsupervised_cost, self.denoising_costs = \
            self.__build_unsupervised_learning_rule()
        print('- Ladder Network built!')

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
        with tf.name_scope('decoder'):
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
                    z_hat[l] = LadderNetwork.Decoder.denoise_gauss(z_pre[l], u, self.layers[l + 1])
                    z_hat[l] = (z_hat[l] - clean_encoder.mean[l]) / (clean_encoder.variance[l] + tf.constant(10e-8))
            return z_hat

        z_hat = decoder_()
        decoder.activations = z_hat
        print('\t- Decoder built!')
        return decoder

    def __supervised_cost(self):
        with tf.name_scope('supervised_cost'):
            cost = -tf.reduce_mean(
                tf.reduce_sum(self.outputs * tf.log(self.clean.activations[-1]), 1))  # why corrupted
        return cost

    def __unsupervised_cost(self):
        with tf.name_scope('unsupervised_cost'):
            denoise = []
            for l in reversed(range(len(self.layers) - 1)):
                with tf.name_scope('denoise_cost_' + str(l)):
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
        with tf.name_scope('unsup_learning_rule'):
            u_cost, denoise_costs = self.__unsupervised_cost()
            optimizer = tf.train.AdamOptimizer(self.params['learning_rate']).minimize(u_cost)
        print('\t- Unsupervised learning rule built!')
        return optimizer, u_cost, denoise_costs

    def __build_supervised_learning_rule(self):
        print('\t- Building supervised learning rule...')
        with tf.name_scope('sup_learning_rule'):
            cost = self.__supervised_cost()
            optimizer = tf.train.AdamOptimizer(self.params['learning_rate']).minimize(cost)
        print('\t- Supervised learning rule built!')
        return optimizer, cost

    def train_on_batch_supervised(self, X, y, step=[0]):
        if self.supervised_summary is not None:
            summary, _ = self.session.run([self.supervised_summary, self.supervised_optimizer],
                                          feed_dict={self.inputs: X, self.outputs: y})
            if step[0] % 5 == 0:
                self.writer.add_summary(summary, step[0])
                self.writer.flush()
            step[0] += 1
        else:
            self.session.run(self.supervised_optimizer, feed_dict={self.inputs: X, self.outputs: y})

    def train_on_batch_unsupervised(self, X, step=[0]):
        if self.unsupervised_summary is not None:
            summary, _ = self.session.run([self.unsupervised_summary, self.unsupervised_optimizer],
                                          feed_dict={self.inputs: X})
            if step[0] % 5 == 0:
                self.writer.add_summary(summary, step[0])
                self.writer.flush()
            step[0] += 1
        else:
            self.session.run(self.unsupervised_optimizer, feed_dict={self.inputs: X})
        pass

    def supervised_accuracy(self):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.clean.activations[-1], 1), tf.argmax(self.outputs, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) * tf.constant(100.0)
        return accuracy

    def __log_all_costs(self):
        sup_summary = tf.summary.scalar('supervised cost', self.supervised_cost)
        unsup_summary = tf.summary.scalar('unsupervised cost', self.unsupervised_cost)
        denoise_summary = []
        for i, d_cost in enumerate(self.denoising_costs):
            denoise_summary.append(tf.summary.scalar('denoising cost on layer {0}'.format(str(i)), d_cost))
        self.supervised_summary = tf.summary.merge([sup_summary])
        self.unsupervised_summary = tf.summary.merge([unsup_summary, *denoise_summary])
        return tf.summary.FileWriter('./logs/train_summary', self.session.graph)

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
            with tf.name_scope('gaussian_denoise'):
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
    train_Y = LabelBinarizer().fit_transform(train.target)
    train.drop('target', axis=1, inplace=True)
    return train.values, train_Y, test.values

if __name__ == '__main__':
    layers = [
        (93, None),
        (1024, tf.nn.relu),
        (512, tf.nn.relu),
        (128, tf.nn.relu),
        (9, tf.nn.softmax)
    ]
    train, labels, test = load_data('./data')
    batch_size = hyperparameters['batch_size']
    ladder = LadderNetwork(layers, **hyperparameters)
    input('All done to start learning!')
    for epoch in range(30):
        print('epoch %d' % epoch)
        perm = np.random.permutation(len(train) - 300)
        train = train[perm]
        labels = labels[perm]
        test = test[np.random.permutation(len(test) - 300)]
        sup_batch = len(train) // 2000
        unsup_batch = len(test) // 2000
        for i in range(1900):
            X = train[i * sup_batch:(i + 1) * sup_batch]
            Y = labels[i * sup_batch:(i + 1) * sup_batch]
            Z = np.vstack([test[i * unsup_batch: (i + 1) * unsup_batch], X])
            if i % 50 == 0:
                print('step %d' % i)
                print('total cost: %d' % (ladder.unsupervised_cost + ladder.supervised_cost).eval(
                      feed_dict={ladder.inputs: X, ladder.outputs: Y}))
            ladder.train_on_batch_supervised(X, Y)
            ladder.train_on_batch_unsupervised(Z)
