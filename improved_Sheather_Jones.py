# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 16:00:11 2021

@author: roman

L'implémentation de ce fichier provient du repo github suivant : https://github.com/Neojume/pythonABC
"""

import numpy as np
import distributions as distr


"""

L'algorithme ici présent permet de calculer l'estimation de la bandwith h. Cette bandwith est nécessaire 
pour l'implémentation de l'estimation de la "probability density function" (PDF). 

Cette algorithme peut être utilisé lorsque suffisament de donnée sont disponible et doit être utilisé
si les données sont loin de la normal et multimodal. De plus, cette méthode à uniquement besoin des données pour fonctionner
ce qui est le cas ici. 

"""


def dnorm(x):
    return distr.normal.pdf(x, 0.0, 1.0)


def sj(x, h):
    '''
    Equation 12 of Sheather and Jones [1]_
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    '''
    phi6 = lambda x: (x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15) * dnorm(x)
    phi4 = lambda x: (x ** 4 - 6 * x ** 2 + 3) * dnorm(x)

    n = len(x)
    one = np.ones((1, n))

    lam = np.percentile(x, 75) - np.percentile(x, 25)
    a = 0.92 * lam * n ** (-1 / 7.0)
    b = 0.912 * lam * n ** (-1 / 9.0)

    W = np.tile(x, (n, 1))
    W = W - W.T

    W1 = phi6(W / b)
    tdb = np.dot(np.dot(one, W1), one.T)
    tdb = -tdb / (n * (n - 1) * b ** 7)

    W1 = phi4(W / a)
    sda = np.dot(np.dot(one, W1), one.T)
    sda = sda / (n * (n - 1) * a ** 5)

    alpha2 = 1.357 * (abs(sda / tdb)) ** (1 / 7.0) * h ** (5 / 7.0)

    W1 = phi4(W / alpha2)
    sdalpha2 = np.dot(np.dot(one, W1), one.T)
    sdalpha2 = sdalpha2 / (n * (n - 1) * alpha2 ** 5)

    return (distr.normal.pdf(0, 0, np.sqrt(2)) /
            (n * abs(sdalpha2[0, 0]))) ** 0.2 - h

def wmean(x, w):
    '''
    Weighted mean
    '''
    return sum(x * w) / float(sum(w))

def wvar(x, w):
    '''
    Weighted variance
    '''
    return sum(w * (x - wmean(x, w)) ** 2) / float(sum(w) - 1)


def hnorm(x, weights=None):
    '''
    Bandwidth estimate assuming f is normal. See paragraph 2.4.2 of
    Bowman and Azzalini[1]_ for details.
    References
    ----------
    .. [1] Applied Smoothing Techniques for Data Analysis: the
        Kernel Approach with S-Plus Illustrations.
        Bowman, A.W. and Azzalini, A. (1997).
        Oxford University Press, Oxford
    '''

    x = np.asarray(x)

    if weights is None:
        weights = np.ones(len(x))

    n = float(sum(weights))

    if len(x.shape) == 1:
        sd = np.sqrt(wvar(x, weights))
        return sd * (4 / (3 * n)) ** (1 / 5.0)

    # TODO: make this work for more dimensions
    # ((4 / (p + 2) * n)^(1 / (p+4)) * sigma_i
    if len(x.shape) == 2:
        ndim = x.shape[1]
        sd = np.sqrt(np.apply_along_axis(wvar, 1, x, weights))
        return (4.0 / ((ndim + 2.0) * n) ** (1.0 / (ndim + 4.0))) * sd


def hsj(x, weights=None):
    '''
    Sheather-Jones bandwidth estimator [1]_.
    References
    ----------
    .. [1] A reliable data-based bandwidth selection method for kernel
        density estimation. Simon J. Sheather and Michael C. Jones.
        Journal of the Royal Statistical Society, Series B. 1991
    '''

    h0 = hnorm(x)
    v0 = sj(x, h0)

    if v0 > 0:
        hstep = 1.1
    else:
        hstep = 0.9

    h1 = h0 * hstep
    v1 = sj(x, h1)

    while v1 * v0 > 0:
        h0 = h1
        v0 = v1
        h1 = h0 * hstep
        v1 = sj(x, h1)

    return h0 + (h1 - h0) * abs(v0) / (abs(v0) + abs(v1))