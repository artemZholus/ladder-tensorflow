import numpy as np
import math
import tensorflow as tf
from tqdm import tqdm


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


def semisupervised_batch_iterator(X, y, batch_size, ratio):
    """
    iterates over dataset with supervised batch size batch_size
    simultaneously yields unsupervised batches with size
    batch_size * ratio

    :param X: data, array (m, n)
    :param y: labels, array (k, n)
    :param batch_size:
    :param ratio:
    :return:
    """
    unsupervised_batch_size = int(ratio * batch_size)
    Z, X = X, X[:len(y)]
    perm = np.random.permutation(len(X))
    perm_unsupervised = np.random.permutation(len(Z))
    for batch_cnt in range(math.ceil(len(X) / batch_size)):
        start = batch_cnt * batch_size
        end = (batch_cnt + 1) * batch_size
        start_unsupervised = batch_cnt * unsupervised_batch_size
        end_unsupervised = min((batch_cnt + 1) * unsupervised_batch_size, len(Z))
        X_batch = X[perm[start:min(end, len(X))]]
        y_batch = y[perm[start:min(end, len(X))]]
        if ratio == 0.:
            yield None, (X_batch, y_batch)
        Z_batch = Z[perm_unsupervised[start_unsupervised:end_unsupervised]]
        if len(Z_batch) % batch_size == 0:
            Z_batch = Z_batch.reshape((-1, batch_size, Z_batch.shape[-1]))
        else:
            Z_b = []
            for k in range(math.ceil(ratio)):
                Z_b.append(Z_batch[k * batch_size:min((k + 1) * batch_size, len(Z_batch))])
            Z_batch = Z_b
        yield Z_batch, (X_batch, y_batch)


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