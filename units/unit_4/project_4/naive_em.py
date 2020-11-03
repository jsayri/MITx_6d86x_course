"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    # prob to belong to a mixture: p(i|j) = pj / (2 * pi * var)^(d/2) * exp(-1/(2 * var) * ||x-u||^2)
    gm_prob = lambda x, p, mu, var, d: np.multiply(np.divide(p, (2*np.pi*var)**(d/2)),
                                                   np.exp(np.divide(((xi-mu)**2).sum(axis=1), -2*var)))

    # variable initialization
    n, d = X.shape
    k, _ = mixture.mu.shape
    post = np.zeros((n, k)) # soft assignation (posterior prob : p(j|i))
    log_lh = 0 # init log-likelihood

    for ii, xi in enumerate(X):
        prior_xi = gm_prob(xi, mixture.p, mixture.mu, mixture.var, d)
        post[ii, :] = prior_xi / np.sum(prior_xi)

        # log-likelihood as the sum at each i and j
        log_lh += post[ii, :] @ np.log(prior_xi / post[ii, :])

    return post, log_lh


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # variable initialization
    n, d = X.shape
    _, k = post.shape

    mu_new = np.zeros((k, d))
    var_new = np.zeros(k)

    # get new parameters
    p_new = np.sum(post, axis=0) / n
    for jj in range(k):
        mu_new[jj] = post[:, jj] @ X / post[:, jj].sum()
        var_new[jj] = post[:, jj] @ ((X-mu_new[jj])**2).sum(axis=1) / (d * post[:, jj].sum())

    return GaussianMixture(mu_new, var_new, p_new)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """

    # initialization
    # new log-likelihood − old log-likelihood ≤ 10−6⋅|new log-likelihood|
    min_th = 10**-6
    ll, ll_old = 1, -np.inf
    ll_error = 1
    ll_vec = []

    while ll_error > min_th * abs(ll):

        # E-step
        post, ll = estep(X, mixture)

        # M-step
        mixture = mstep(X, post)

        # next iteration update
        ll_error = ll - ll_old
        ll_old = ll
        ll_vec.append(ll)

    return mixture, post, ll, ll_vec
