import numpy as np
import scipy as sp


def get_decay(decay_type):
    return eval(decay_type)


def hyperbolic_decay(mult, pow, n):
    """
    hyperbolic decay function.
    Resulting list looks like [mult/1^pow, mult/2^pow, ..., mult/n^pow].

    :param mult: multiplication coefficient
    :param pow: power coefficient
    :param n: length of decay
    :return: decay coeffs.
    """
    return (np.power(1 / np.arange(1, n + 1), pow) * mult).tolist()
