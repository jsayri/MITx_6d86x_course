import numpy as np
import kmeans
import common
import naive_em
import em

def run_section_2(X: np.ndarray) -> None:
    '''
    Routine that execute the section 2 of project 4
    '''
    # Section 2. K-means
    print('Section 1, K-means')

    # init test values
    Ks = np.array([1, 2, 3, 4])
    Seeds = np.array([0, 1, 2, 3, 4])
    cost_m = np.zeros((Ks.shape[0], Seeds.shape[0]))

    # Iterative execution on clusters & seeds
    for ii, k in enumerate(Ks):
        for jj, seed in enumerate(Seeds):
            # initialization for k-means and gaussians
            g_mix, post = common.init(X, k, seed)

            # execute k-means algorithm with gaussian mixture
            mixture, post, cost = kmeans.run(X, g_mix, post)

            # display graph
            title_case = 'Cluster K = ' + str(k) + ' and seed = ' + str(seed)
            # common.plot(X, g_mix, post, title_case)

            # store each cost at each k and seed
            cost_m[ii, jj] = cost
            # print('k={}, seed={}, cost={}'.format(k, seed, round(cost, 5)))

    # get minimum for each cluster
    print('Display minimal cost for each cluster considering multiples seeds')
    op_seed = np.zeros(len(Ks))
    for ii, k in enumerate(Ks):
        op_seed[ii] = np.argmin(cost_m[ii, :])
        print('k={}, min_cost={}, seed={}'.format(k, np.min(cost_m[ii, :]).round(6), op_seed[ii]))

    return op_seed

# Section 3. Expectation–maximization algorithm
def run_section_3(X: np.ndarray):
    '''
        Routine that execute the section 3 of project 4
    '''

    # X data is loaded (toydata) for test purpose

    # algorithm test based on instructions (K = 3)
    g_mix, _ = common.init(X, 3, 0) # gaussian initialization

    # check e-step execution
    post, log_lh = naive_em.estep(X, g_mix)
    # print('Log-likelihood for E-step: {}'.format(log_lh.round(8)))
    # with K=3 and a seed of 0, on the toy dataset, log likelihood of -1388.0818

    # check m-step execution
    g_mix_new = naive_em.mstep(X, post)
    # print('Updated gaussian mixture\n', g_mix_new)

    # execution of full naive em algorithm
    gmm_init, post_init = common.init(X, 3, 0)  # gaussian initialization
    gmm_final, post_end, log_lh, ll_vec = naive_em.run(X, gmm_init, post_init)

    # display clusters
    title_plot = 'GMM, naive approach, X: toydata, k=3'
    common.plot(X, gmm_final, post_end, title_plot)

    # display cost function
    common.cost_plot(ll_vec)


def run_section_4(X: np.ndarray):
    '''
    Routine that execute the section 3 of project 4
    Run EM algorithm for multiple cluster and seeds
    '''
    # Section 4. K-means
    print('Section 4, EM cluster optimal')

    # init test values
    Ks = np.array([1, 2, 3, 4])
    Seeds = np.array([0, 1, 2, 3, 4])
    cost_m = np.zeros((Ks.shape[0], Seeds.shape[0]))

    # Iterative execution on clusters & seeds
    for ii, k in enumerate(Ks):
        for jj, seed in enumerate(Seeds):
            # initialization for em algorithm
            g_mix, post = common.init(X, k, seed)

            # execute EM algorithm with gaussian mixture
            mixture, post, cost, _ = naive_em.run(X, g_mix, post)

            # display graph
            title_case = 'Cluster K = ' + str(k) + ' and seed = ' + str(seed)
            # common.plot(X, g_mix, post, title_case)

            # store each cost at each k and seed
            cost_m[ii, jj] = cost
            # print('k={}, seed={}, cost={}'.format(k, seed, round(cost, 5)))

    # get minimum for each cluster
    print('Display the maximal cost for each cluster considering multiples seeds, consider negative results...')
    op_seed = np.zeros(len(Ks))
    for ii, k in enumerate(Ks):
        op_seed[ii] = np.argmax(cost_m[ii, :])
        print('k={}, max_cost={}, seed={}'.format(k, np.max(cost_m[ii, :]).round(6), op_seed[ii]))

    return op_seed


def run_section_4_plots(X, optimal_seeds_kmeans, optimal_seeds_em):
    # init test values
    Ks = np.array([1, 2, 3, 4])

    # iteration as a function of cluster numbers
    for jj, (k, seed_km, seed_em) in enumerate(zip(Ks, optimal_seeds_kmeans, optimal_seeds_em)):

        # k-means algorithm
        g_mix, post = common.init(X, k, round(seed_km))
        mixture, post, cost = kmeans.run(X, g_mix, post)
        title_case = 'K-means algorithm, k=' + str(k) + ', seed=' + str(seed_km)
        fname = 'plots/k' + str(k) + '_kmeans'
        common.plot(X, g_mix, post, title_case, fname)

        # EM algorithm
        g_mix, post = common.init(X, k, round(seed_em))
        mixture, post, cost, _ = naive_em.run(X, g_mix, post)
        title_case = 'EM algorithm, k=' + str(k) + ', seed=' + str(seed_em)
        fname = 'plots/k' + str(k) + '_em'
        common.plot(X, mixture, post, title_case, fname)


def run_section_5(X: np.ndarray, best_seeds=None):
    # Section 4. K-means
    print('Section 5, Bayesian Information Criterion')

    # init test values
    Ks = np.array([1, 2, 3, 4])
    if best_seeds is None:
        best_seeds = np.zeros(len(Ks))

    # execute EM for multiples cluster number
    bic_m = np.zeros(len(Ks))

    for jj, k in enumerate(Ks):

        # EM algorithm
        g_mix, post = common.init(X, k, int(best_seeds[jj]))
        mixture, post, cost, _ = naive_em.run(X, g_mix, post)

        # Bayesian Information Criterion
        bic_m[jj] = common.bic(X, mixture, cost)
        print('bic(k={}: {}'.format(k, bic_m[jj]))

    print('best cluster is k={}, with bic={}'.format(Ks[np.argmax(bic_m)], np.max(bic_m).round(5)))


if __name__ == "__main__":

    # load toy data set
    X = np.loadtxt("toy_data.txt")

    # Run section 2 execution, k-means algorithm for toy data set
    # oseed_s2 = run_section_2(X)
    # oseed_s2 = np.zeros(4)
    # Run section 3 execution, EM algorithm for matrix completion
    # run_section_3(X)

    # Run section 4, execution, EM and k-means comparison
    oseed_s4 = run_section_4(X)
    # oseed_s4 = np.zeros(4)

    # Run section 4, part 2, comparison between methods, plot graphs
    # run_section_4_plots(X, oseed_s2, oseed_s4)

    # Run section 5, calculate BIC and get best k
    run_section_5(X, oseed_s4)