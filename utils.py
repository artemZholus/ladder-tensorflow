from functools import wraps
import numpy as np
import scipy as sp


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
    if ratio == 0.0:
        unsupervised_batch_num = 1
    else:
        unsupervised_batch_num = len(Z) // unsupervised_batch_size
    for batch_cnt in range(len(X) // batch_size):
        start = batch_cnt * batch_size
        end = (batch_cnt + 1) * batch_size
        start_unsupervised = ((batch_cnt % unsupervised_batch_num) * unsupervised_batch_size)
        end_unsupervised = ((batch_cnt + 1) % unsupervised_batch_num) * unsupervised_batch_size
        if len(Z) + unsupervised_batch_size > end_unsupervised > len(Z):
            end_unsupervised = len(Z)
        else:
            end_unsupervised %= len(Z)
        if end > len(X):
            end = len(X)
        X_batch = X[perm[start:end]]
        y_batch = y[perm[start:end]]
        if ratio == 0.0:
            Z_batch = None
        else:
            Z_batch = Z[perm_unsupervised[start_unsupervised:end_unsupervised]]
        assert len(Z_batch) == unsupervised_batch_size
        yield Z_batch, (X_batch, y_batch)
