# execution of Perceptron algorithm to answer Homework1

import numpy as np

def perceptron_perf(x, y, T=5, nm_max = 100):
    '''
    Execute a version of the perceptron algorithm for the section 2 Perceptron Performance from homework1
    Perceptron algorithm with an offset different from origin (th_0 unlock!)
    Inputs
    x :     list, vector 2xm with point, i.e. x = [[1,1], [2,1], [-1,4]]
    y :     list, label vector with the outputs for each x, i.e. y = [1, -1, 1]
    T :     integer, number of repetitions for the algorithm, default 5
    nm_max :integer, max number of mistakes allowed
    '''

    th = np.zeros((1, x.shape[1]))  # initial theta
    th_0 = np.zeros(1)              # initial theta_0
    th_list = []                    # theta list of updates
    nm = 0                          # number of mistakes counter
    for t in range(T):
        if nm == nm_max: break
        # print('iteration', t)
        for i in range(x.shape[0]):
            # print('vector i =', i)
            xi = x[i]   # get (x1, x2) at i
            yi = y[i]

            if (yi * (np.dot(th, xi) + th_0)) <= 0:
                # check number of mistakes counter
                if nm >= nm_max:
                    break

                th = th + yi * xi
                th_0 = th_0 + yi
                th_list.append(th.tolist()[0])

                # mistake counter
                nm += 1




    print('number of mistakes = ', len(th_list))
    return th, th_0, th_list


if __name__ == '__main__':

    # Homework 1, 2. Perceptron Performance
    # data set (2a), with offset th_0 and th initialize to zero
    x = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
    y = np.array([1, 1, -1, -1, -1])

    # th, th_0, th_list = perceptron_perf(x, y, 10)
    # print('result for theta', th)
    # print('result for theta_0', th_0)
    # print('variation of theta over iterations')
    # print(th_list)

    print()


    # data set for problem 3b-4
    x = np.array([[-1, 1], [1, -1], [1, 1], [2, 2]])
    y = np.array([1, 1, -1, -1])

    th, th_0, th_list = perceptron_perf(x, y, 4, 5)
    print('result for theta', th)
    print('result for theta_0', th_0)
    print('variation of theta over iterations')
    print(th_list)

    print()