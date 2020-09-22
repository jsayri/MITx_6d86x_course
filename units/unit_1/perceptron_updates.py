# execution of Perceptron algorithm to answer Homework1

import numpy as np

def perceptron_upd(x, y, T=5, nm_max = 100):
    '''
    Execute a version of the perceptron algorithm for the section 6 Perceptron Updates from homework1
    Perceptron algorithm with an offset different from origin (th_0 unlock!)
    Inputs
    x :     list, vector 2xm with point, i.e. x = [[1,1], [2,1], [-1,4]]
    y :     list, label vector with the outputs for each x, i.e. y = [1, -1, 1]
    T :     integer, number of repetitions for the algorithm, default 5
    nm_max :integer, max number of mistakes allowed
    '''

    th = np.zeros((1, x.shape[1]))  # initial theta
    # th_0 = np.zeros(1)              # initial theta_0
    th_list = []                    # theta list of updates
    nm = 0                          # number of mistakes counter
    for t in range(T):
        if nm == nm_max: break
        # print('iteration', t)
        for i in range(x.shape[0]):
            # print('vector i =', i)
            xi = x[i]   # get (x1, x2) at i
            yi = y[i]

            # if (yi * (np.dot(th, xi) + th_0)) <= 0:    # with th_0
            if (yi * (np.dot(th, xi))) <= 0:
                # check number of mistakes counter
                if nm >= nm_max:
                    break

                th = th + yi * xi
                # th_0 = th_0 + yi
                th_list.append(th.tolist()[0])

                # mistake counter
                nm += 1

    print('number of mistakes = ', len(th_list))
    return th, th_list

def feature_vector(d, n):
    '''
    Fill a feature feature vector following some specific conditions:
      - x_i^(t) = cos(pi*t), if i = t
      - x_i^(t) = 0, otherwise
    Inputs
    d : integer, dimension of the feature vector
    n : integer, number of feature vectors
    Ouput
    x : array, list of feature vectors as [x^(1), x^(2), ..., x^(n)]
        with a dimension of 1xd for each x^(i)
    '''
    x = np.zeros((n, d))
    for t in range(n):
        x_t = np.zeros(d)
        if t < d:
            x_t[t] = np.cos(np.pi * (t + 1)) # i=t

        x[t, :] = x_t
    return x


if __name__ == '__main__':

    # Homework 1, 6. Perceptron Updates
    # data set d=2, and two features vector
    x = np.array([[-1, 0], [0, 1]])
    y = np.array([1, 1])

    th, th_list = perceptron_upd(x, y)
    print('result for theta', th)
    print('variation of theta over iterations')
    print(th_list)

    print()

    # data set d=3, and three features vector
    x = feature_vector(3, 3)
    y = np.ones(x.shape[0])

    th, th_list = perceptron_upd(x, y)
    print('result for theta', th)
    print('variation of theta over iterations')
    print(th_list)

    # the other solution part consist into calculate a plane P that goes throught x and
    # compare it with theta
    print()