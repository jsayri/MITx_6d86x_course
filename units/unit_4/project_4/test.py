import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0


# implementation of E-step
gmm_init, _ = common.init(X, K, seed) # gaussian initialization
post, log_lh = em.estep(X, gmm_init) # run e-step
# expected result from test_solution.txt: LL=-152.16319226209848
print('Log-likelihood for E-step: {}'.format(log_lh))


# implementation of M-step
g_mix_new = em.mstep(X, post, gmm_init)
print('Updated gaussian mixture\n', g_mix_new)


# implementation of full run



# check m-step execution
gmm_init, post_init = common.init(X, K, seed)  # gaussian initialization
gmm_final, post_end, log_lh, ll_vec = em.run(X, gmm_init, post_init)
