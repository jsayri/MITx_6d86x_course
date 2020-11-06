"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    # variable initialization
    n, d = X.shape
    k, _ = mixture.mu.shape
    post = np.zeros((n, k))  # soft assignation (posterior prob : p(j|i))
    log_lh = 0.  # init log-likelihood

    # gaussian model: exp(-1/(2 * var) * ||x-u||^2) / (2 * pi * var)^(d/2)
    N = lambda xi, mu, var, d: np.divide(np.exp(np.divide(((xi-mu)**2).sum(axis=1), -2*var)), (2*np.pi*var)**(d/2))
    logN = lambda xi, mu, var, d: -d/2*np.log((2 * np.pi * var)) + np.divide(((xi - mu)**2).sum(axis=1), -2 * var)
    # log prior expression: log(p_i) + log(N(xi, mu, var, d))
    # fui_fun = lambda xi, p, mu, var, d: np.log(p + 1e-16) + np.log(N(xi, mu, var, d) + 1e-16)
    fui_fun = lambda xi, p, mu, var, d: np.log(p + 1e-16) + logN(xi, mu, var, d)


    # iteration by cluster, matrix calculation
    for ii, xi in enumerate(X):
        # select non-zero elements
        cu_pos = xi > 0
        xcu = xi[cu_pos] # build non zero vector from xi
        # calculate log expression for probability at each cluster
        f_ui = fui_fun(xcu, mixture.p, mixture.mu[:, cu_pos], mixture.var, xcu.size)
        log_post = f_ui - logsumexp(f_ui)
        # calculate posterior prob for xi
        post[ii, :] = np.exp(log_post)

        # log-likelihood as the sum of each i and j
        log_lh += post[ii, :] @ (f_ui - log_post)

    return post, log_lh



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # variable initialization
    n, d = X.shape
    _, k = post.shape

    mu_old = mixture.mu
    mu_new = np.zeros((k, d))
    var_new = np.zeros(k)
    dcu = X > 0  # indicator matrix

    # get new parameters
    p_new = np.sum(post, axis=0) / n
    for jj in range(k):
        # update average 'mu'
        sum_post = (np.tile(post[:, jj], (d, 1)).T * dcu).sum(axis=0)
        if any(sum_post < 1): # avoid erratic results if few points
            mu_new[jj, sum_post < 1] = mu_old[jj, sum_post < 1]
            ok_idx = sum_post >= 1
            if ok_idx.sum() >= 1:
                mu_new[jj, ok_idx] = post[:, jj] @ X[:, ok_idx] / sum_post[ok_idx]

        else:
            mu_new[jj] = post[:, jj] @ X / sum_post
        # update variance 'var'
        dist_x_mu = ((X - mu_new[jj]*dcu)**2).sum(axis=1)
        var_new[jj] = post[:, jj] @ dist_x_mu / ((dcu.sum(axis=1) * post[:, jj]).sum())

    # corrections for numerical stability
    var_new[var_new < min_variance] = min_variance

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
    min_th = 10 ** -6
    ll, ll_old = 1, -np.inf
    ll_error = 1
    ll_vec = []
    while ll_error > min_th * abs(ll):
        # E-step
        post, ll = estep(X, mixture)

        # M-step
        mixture = mstep(X, post, mixture)

        # next iteration update
        ll_error = ll - ll_old
        ll_old = ll
        ll_vec.append(ll)

    return mixture, post, ll, ll_vec


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    # variable initialization
    p, mu, var = mixture.p, mixture.mu, mixture.var
    dcu = X > 0
    dhu = ~dcu
    X_pred = X.copy()

    # gaussian model: exp(-1/(2 * var) * ||x-u||^2) / (2 * pi * var)^(d/2)
    logN = lambda xi, mu, var, d: -d / 2 * np.log((2 * np.pi * var)) + np.divide(((xi - mu) ** 2).sum(axis=1), -2 * var)
    fui_fun = lambda xi, p, mu, var, d: np.log(p + 1e-16) + logN(xi, mu, var, d)

    for ii, xi in enumerate(X):
        # define non missing values
        xcu = xi[dcu[ii, :]]  # build non zero vector from xi

        # calculate posterior prob for each cluster
        f_ui = fui_fun(xcu, p, mu[:, dcu[ii, :]], var, xcu.size)
        post = np.exp(f_ui - logsumexp(f_ui))

        # estimate missing values 'xhu' with mu and posterior probability
        X_pred[ii, dhu[ii, :]] = post @ mu[:, dhu[ii, :]]

    return X_pred