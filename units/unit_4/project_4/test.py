import numpy as np
import em
import common

# EM routine for multiples cluster and seeds
def em_for_k_and_seeds(X, Ks, Seeds, verbose=True):
    '''
    Execution of EM algorithm for a matrix incomplete data. It will
    check a set of clusters (Ks) for multiples seeds
    Inputs
    X :         np.ndarray, matrix of incomplete data
    Ks :        list, values of clusters to test
    Seeds :     list, values of random seeds
    verbose :   boolean, display results, default 'True'

    Outputs
    opt_seed :  np.ndarray, value of optimal seed for each cluster
    '''

    # init test values
    log_lh_m = np.zeros((Ks.shape[0], Seeds.shape[0]))

    # Iterative execution on clusters & seeds
    for ii, k in enumerate(Ks):
        for jj, seed in enumerate(Seeds):
            # initialization for em algorithm
            mixture, post = common.init(X, k, seed)

            # execute EM algorithm with gaussian mixture
            mixture, post, cost, _ = em.run(X, mixture, post)

            # store each cost at each k and seed
            log_lh_m[ii, jj] = cost
            if verbose:
                print('k={}, seed={}, cost={}'.format(k, seed, round(cost, 5)))

            # display cost function
            # common.cost_plot(cost_vec)

    # get minimum for each cluster
    print('Optimal cost at each cluster for multiples seeds.')
    op_seed = np.zeros(len(Ks))
    for ii, k in enumerate(Ks):
        op_seed[ii] = np.argmax(log_lh_m[ii, :])
        print('k={}, max_cost={}, seed={}'.format(k, np.max(log_lh_m[ii, :]).round(6), op_seed[ii]))

    return op_seed



if __name__ == "__main__":
    X = np.loadtxt("test_incomplete.txt")
    X_gold = np.loadtxt("test_complete.txt")

    K = 4
    n, d = X.shape
    seed = 0


    # implementation of E-step
    # gmm_init, _ = common.init(X, K, seed) # gaussian initialization
    # post, log_lh = em.estep(X, gmm_init) # run e-step
    # expected result from test_solution.txt: LL=-152.16319226209848
    # print('Log-likelihood for E-step: {}'.format(log_lh))


    # implementation of M-step
    # g_mix_new = em.mstep(X, post, gmm_init)
    # print('Updated gaussian mixture\n', g_mix_new)


    # implementation of full run
    # gmm_init, post = common.init(X, K, seed) # gaussian initialization
    # gmm_end, post, cost = em.run(X, gmm_init, post)
    # print('result log-likelihood: \n', cost)


    # test algorithm with Netflix incomplete data
    X_netflix = np.loadtxt("netflix_incomplete.txt")
    X_netflix_gold = np.loadtxt("netflix_complete.txt")

    # scenarios' definition
    Ks = np.array([1, 12]) # clusters
    Seeds = np.array([0, 1, 2, 3, 4]) # random seeds
    # em algorithm execution
    opseeds = em_for_k_and_seeds(X_netflix, Ks, Seeds)


    # toy test, fill missing data matrix
    gmm_init, post = common.init(X, K, seed)  # gaussian initialization
    gmm_end, post, cost, _ = em.run(X, gmm_init, post)
    X_pred = em.fill_matrix(X, gmm_end)
    rmse_x = common.rmse(X_gold, X_pred)
    print('RMSE for toy data: {}'.format(rmse_x))


    # Netflix data, fill missing data matrix
    mixture, post = common.init(X_netflix, 12, 1)  # gaussian initialization
    mixture, post, cost, _ = em.run(X_netflix, mixture, post)
    X_pred = em.fill_matrix(X_netflix, mixture)
    rmse_netflix = common.rmse(X_netflix_gold, X_pred)
    print('RMSE for netlflix data: {}'.format(rmse_netflix))

