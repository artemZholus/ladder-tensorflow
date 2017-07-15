import numpy as np
import math
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer


def get_decay(decay_type):
    return eval(decay_type)


def hyperbolic_decay(mul, pow, n):
    """
    hyperbolic decay function.
    Output looks like [mul/1^pow, mul/2^pow, ..., mul/n^pow].

    :param mul: multiplication coefficient
    :param pow: power coefficient
    :param n: length of decay
    :return: decay coeffs.
    """
    return (np.power(1 / np.arange(1, n + 1), pow) * mul).tolist()


def exponential_decay(mul, base, n):
    """
    linear decay function
    Output looks like [lo, lo + (hi - lo) / n, 2 * (hi - lo) / n, ..., hi]

    :param hi: upper bound
    :param lo: lower bound
    :param n:
    :return:
    """
    return (mul * np.power(base, np.arange(n))).tolist()


def conditional_context(statement, func, context):
    if statement:
        with context:
            return func()
    else:
        return func()


class SemiSupervisedDataset:
    def __init__(self, X, y, batch_size, shuffle=True, include_supervised=False):
        self.batch_size = batch_size
        self.supervised_x = X[:len(y)]
        self.supervised_y = y
        start = 0 if include_supervised else len(y)
        self.unsupervised_x = X[start:]
        if shuffle:
            perm = np.random.permutation(len(self.supervised_x))
            self.supervised_x = self.supervised_x[perm]
            self.supervised_y = self.supervised_y[perm]
            perm = np.random.permutation(len(self.unsupervised_x))
            self.unsupervised_x = self.unsupervised_x[perm]
        self.i_supervised = 0
        self.i_unsupervised = 0
        self.supervised_steps = int(math.ceil(len(self.supervised_y) / self.batch_size))
        self.unsupervised_steps = int(math.ceil(len(self.unsupervised_x) / self.batch_size))

    def reset_supervised(self):
        self.i_supervised = 0
        perm = np.random.permutation(len(self.supervised_x))
        self.supervised_x = self.supervised_x[perm]
        self.supervised_y = self.supervised_y[perm]

    def reset_unsupervised(self):
        self.i_unsupervised = 0
        perm = np.random.permutation(len(self.unsupervised_x))
        self.unsupervised_x = self.unsupervised_x[perm]

    def reset(self):
        self.reset_supervised()
        if self.i_unsupervised > self.unsupervised_steps:
            self.reset_unsupervised()

    def __iter__(self):
        self.reset()
        return self

    def __len__(self):
        return len(self.supervised_x)

    def __next__(self):
        while self.i_supervised < self.supervised_steps:
            if self.i_unsupervised >= self.unsupervised_steps:
                self.reset_unsupervised()
            start = self.i_supervised * self.batch_size
            end = (self.i_supervised + 1) * self.batch_size
            start_unsupervised = self.i_unsupervised * self.batch_size
            end_unsupervised = min((self.i_unsupervised + 1) * self.batch_size, len(self.unsupervised_x))
            X_batch = self.supervised_x[start:min(end, len(self.supervised_x))]
            y_batch = self.supervised_y[start:min(end, len(self.supervised_y))]
            Z_batch = self.unsupervised_x[start_unsupervised:end_unsupervised]
            X_batch = np.vstack([X_batch, Z_batch])
            self.i_supervised += 1
            self.i_unsupervised += 1
            return X_batch, y_batch
        raise StopIteration()


def prepare_data(train_data, train_labels, test_data, test_labels, num_labeled=100):
    perm = np.random.permutation(len(train_data))
    unsupervised_images = train_data.reshape((-1, 784))[perm]
    labels = np.zeros((num_labeled,))
    label_size = num_labeled // 10
    for j, i in enumerate(np.unique(train_labels)):
        labels[j * 10: (j + 1) * 10] = np.where(train_labels == i)[0][:label_size]
    labels = np.random.permutation(labels).astype(np.int32)
    train_labels = train_labels[labels]
    train_labels = LabelBinarizer().fit_transform(train_labels)
    labeled_images = unsupervised_images[labels]

    y_test = test_labels
    X_test = test_data.reshape((-1, 784))

    X_train = np.vstack([labeled_images, unsupervised_images])
    y_train = train_labels
    return X_train, y_train, X_test, y_test


def mlp(layers_):
    layers, activation = zip(*layers_)
    graph = tf.Graph()
    with graph.as_default(), tf.device('/gpu:0'):
        with graph.name_scope('io'):
            inputs = tf.placeholder(tf.float32, shape=(None, layers[0]))
            outputs = tf.placeholder(tf.float32)
        ls = [inputs]
        for i, (in_, out_, activ) in enumerate(zip(layers[:-1], layers[1:], activation[1:])):
            bn_beta = tf.Variable(tf.zeros([out_]))
            bn_gamma = tf.Variable(tf.ones([out_]))
            w = tf.Variable(tf.random_normal((in_, out_), name='weight_' + str(i)))
            pre = tf.matmul(ls[-1], w)
            m, v = tf.nn.moments(pre, [0])
            pre = (pre - m) / v
            pre = bn_gamma * pre + bn_beta
            if activ is not tf.nn.softmax:
                pre = activ(pre)
            ls.append(pre)

        with graph.name_scope('learning'):
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outputs,
                                                                          logits=ls[-1]))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        with graph.name_scope('prediction'):
            pred = tf.nn.softmax(ls[-1])
            cls  = tf.argmax(pred, 1)

    s = tf.InteractiveSession(graph=graph)
    s.run(tf.global_variables_initializer())
    def fit(x, y, epochs, bsize):
        for i in range(epochs):
            print('epoch ', i)
            perm = np.random.permutation(len(x) - 10)
            x = x[perm]
            y = y[perm]
            for j in tqdm(range(math.ceil(len(x) / bsize))):
                x_batch = x[j * bsize:min(((j + 1) * bsize), len(x))]
                y_batch = y[j * bsize:min(((j + 1) * bsize), len(y))]
                s.run(optimizer, feed_dict={inputs: x_batch, outputs: y_batch})

    def predict(x, bsize=50):
        ans = []
        for j in tqdm(range(math.ceil(len(x) / bsize))):
            x_batch = x[j * bsize:min(((j + 1) * bsize), len(x))]
            ans.append(s.run(cls, feed_dict={inputs: x_batch}))
        ans = np.array(ans)
        return ans.reshape((-1,))

    def predict_proba(x, bsize=50):
        ans = []
        for j in tqdm(range(math.ceil(len(x) / bsize))):
            x_batch = x[j * bsize:min(((j + 1) * bsize), len(x))]
            ans.append(s.run(pred, feed_dict={inputs: x_batch}))
        ans = np.array(ans)
        return ans.reshape((-1, ans.shape[-1]))
    return fit, predict, predict_proba

def join(l, u):
    return tf.concat([l, u] ,0)

def labeled(x, batch_size):
    return tf.slice(x, [0, 0], [batch_size, -1]) if x is not None else x

def unlabeled(x, batch_size):
    return tf.slice(x, [batch_size, 0], [-1, -1]) if x is not None else x

def split_lu(x, batch_size):
    return labeled(x, batch_size), unlabeled(x, batch_size)

