# Homework 4, 
# Section 1. K-means and K-medoids

import numpy as np

# k-medoids algorithm
def k_medoids(x, k, zk=None, d_type='l1'):
    '''
    K-medoids algorithm
    Inputs
    x :     numpy array with rows per sample
    k :     number of clusters
    zk:     numpy array with centers, one per row, default None *
    d_type  select the distance evaluation function, default 'l1' **

    Outputs
    clstrs  numpy array with a list of cluster assignation for x
    zk_new  np array with centroids selected by the algorithm

    Notes:
        * when zk=None the algorithm will randomly select zk array
        ** possibles option for d_type: 'l1', 'l2'
    '''
    # function set-up variables
    max_iter = 10 # maximum number of iteration allowed
    iter_count = 0 # iteration counter
    min_cost_error = 10 ** -6  # convergence rate constant
    cost_old = 100 # previous iteration cost value
    cost_val = 0 # current iteration cost value
    cost_error = 100 # cost error for iteration
    clstrs = np.zeros(x.shape[0]) # cluster vector

    # set distance function
    if d_type is 'l1':
        dist_fun = lambda x, z: np.linalg.norm(x - z, ord=1, axis=1)
    elif d_type is 'l2':
        dist_fun = lambda x, z: np.linalg.norm(x - z, ord=2, axis=1)
    else:
        raise ValueError(d_type, 'Wrong or missing value')

    # initialization of cluster centers
    if zk is None:
        # randomly select from the input dataset
        raise NotImplementedError

    # iterate until no change in cost
    while cost_error > min_cost_error and iter_count < max_iter:
        # assign clusters, Cj = {i | s.t. z(j) closest to x(i)}
        for ii, xi in enumerate(x):
            dist_xi_to_z = dist_fun(xi, zk)
            clstrs[ii] = np.argmin(dist_xi_to_z) # assign xi to cluster cj
            cost_val += np.min(dist_xi_to_z) # calculate cost

        # update centroids, z(j) exist {x(i), ..., x(n)} s.t. sum(dist(x(i), z(j)) for i in Cj is minimal
        for jj in range(0, k):
            xcj = x[clstrs == jj, :] # x that belong to cluster j
            dist_cj = np.array([np.sum(dist_fun(xcj, zij)) for zij in x]) # a z=x(i) for all x
            zk[jj] = x[np.argmin(dist_cj), :] # set the new zjj center for cluster k, first occurrence is selected

        # prepare for next iteration
        cost_error = abs(cost_old - cost_val)
        cost_old = cost_val
        cost_val = 0.0
        iter_count += 1

    return clstrs, zk

# k-means algorith
def k_means(x, z_init, k):
    '''
    K-means algorithm
    '''

def print_clusters(x_array, clusters_array, clusters_centers):
    for ii, c_center in enumerate(clusters_centers):
        x_cluster = x_array[clusters_array == ii, :]
        print('Cluster {}'.format(ii), ', center ', c_center.tolist())
        print('members: ', str(x_cluster.tolist()).replace('], [', ']; [').replace('[[', '[').replace(']]', ']'))
    print()


if __name__ == "__main__":
    # Define data set and algorithm configuration
    # dataset
    X = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])
    k = 2  # two clusters
    # center initialization
    ck = np.array([[-5, 2], [0, -6]])

    # Clustering 1: Execute k-medoids with l1 norm and defined centers.
    clusters_c1, zk_c1 = k_medoids(X, 2, ck)
    print_clusters(X, clusters_c1, zk_c1)

    # Clustering 2: Execute k-medoids with l2 norm and defined centers.
    clusters_c2, zk_c2 = k_medoids(X, 2, ck, d_type='l2')
    print_clusters(X, clusters_c2, zk_c2)

    # Clustering 3: Execute k-means with l1 norm.