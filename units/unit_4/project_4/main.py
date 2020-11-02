import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

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
print('Section 1, K-means')
print('Display minimal cost for each cluster considering multiples seeds')
for ii, k in enumerate(Ks):
    print('k={}, min_cost={}'.format(k, np.min(cost_m[ii,:])))
