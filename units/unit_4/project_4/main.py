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
    for ii, k in enumerate(Ks):
        print('k={}, min_cost={}'.format(k, np.min(cost_m[ii,:])))


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



if __name__ == "__main__":

    # load toy data set
    X = np.loadtxt("toy_data.txt")

    # Run section 2 execution, k-means algorithm for toy data set
    # run_section_2(X)

    # Run section 3 execution, EM algorithm for matrix completaition
    run_section_3(X)
