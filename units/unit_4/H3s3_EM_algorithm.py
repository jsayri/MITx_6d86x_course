# Homework 4, 
# Section 3. EM Algorithm

import numpy as np

if __name__ == "__main__":

    # consider a mixture model as
    # p(x|th) = w1 * N(x,u1,s1) + w2 * N(x,u2,s2) = sum(wi * N(x,ui,si)) for i from 1 to 2
    # define initial inputs
    X = np.array([-1, 0, 4, 5, 6], dtype='float64')
    th0 = np.array([[0.5, 6, 1], [0.5, 7, 4]], dtype='float64') # k=0: w1,u1,s1; k=1: w2,u2,s2
    th = th0.copy()  # parameters init for iteration

    # gaussian function
    N = lambda x, u, s, d=1: 1 / (2 * np.pi * s) ** (d / 2) * np.exp(-1 / (2 * s) * np.linalg.norm(x - u) ** 2)

    # prob of x given params
    pxgp = np.zeros(X.shape)
    for ii, xi in enumerate(X):
        for wk, uk, sk in th0:
            pxgp[ii] += wk * N(xi, uk, sk)

    # Variables notation:
    # pxgp : prob of x given params, p(xi|theta)
    # pjgx : prob of being in group j given x(i), p(j|i)
    # log_lk: log-likelihood

    # calculate the log-likelihood
    log_lk = np.zeros(1)
    for ii, xi in enumerate(X):
        for wk, uk, sk in th0:
            pjgx = wk * N(xi, uk, sk) / pxgp[ii]
            log_lk += pjgx * np.log(pxgp[ii] / pjgx)

    print('log-likelihood of l(D,th) : {}\n'.format(round(float(log_lk), 10)))

    # iterative step
    iter_algorithm, iter_max = 0, 8
    while iter_algorithm < iter_max:
        print('Iteration {}'.format(iter_algorithm+1))

        # E-step
        print('E-step, expectation (weights)')
        print('display to which gaussian belong each point')
        # say to which gaussian belong each datapoint based on the initial parameter
        gg_m = np.zeros((X.shape[0], th.shape[0])) # p(j|i): matrix of prob
        x_group = np.zeros(X.shape[0])
        for ii, xi in enumerate(X):
            # calculate p(x|p) for a given xi
            pxgp = np.dot(th[:, 0], np.vectorize(N)(xi, th[:, 1], th[:, 2]))
            # get a matrix for p(y=k|xi) = p(j|i) 'gaussian mixture'
            for jj, (wk, uk, sk) in enumerate(th):
                gg_m[ii, jj] = wk * N(xi, uk, sk) / pxgp
            x_group[ii] = np.argmax(gg_m[ii, :]) # clusters formed
            print('x{}: {}'.format(ii, round(x_group[ii]+1)))

        # M-step
        print('\nM-step, maximization')
        # update parameters
        for jj, (wk, uk, sk) in enumerate(th):
            uk = np.dot(gg_m[:, jj], X) / np.sum(gg_m[:, jj])
            sk = np.dot(gg_m[:, jj], (X-uk)**2) / np.sum(gg_m[:, jj])
            wk = np.sum(gg_m[:, jj]) / X.shape[0]
            th[jj, :] = wk, uk, sk

        print('updated parameters (w, u, s):\n', th)
        print()
        # update for next iteration
        iter_algorithm += 1