# execution of Perceptron algorithm to answer Homework1

import numpy as np

def perceptron_disp(x, y, T=5):
    '''
    Display the iteretions within the perceptron algorithm
    Inputs
    x :     list, vector 2xm with point, i.e. x = [[1,1], [2,1], [-1,4]]^T
    y :     list, label vector with the outputs for each x, i.e. y = [1, -1, 1]
    T :     integer, number of repetions for the algorithm, default 5
    '''

    th = np.zeros((1, x.shape[1]))    # initial theta
    th_list = []
    for t in range(T):
        # print('iteration', t)
        for i in range(x.shape[0]):
            # print('vector i =', i)
            xi = x[i]   # get (x1, x2) at i
            yi = y[i]

            if (yi * np.dot(th, xi)) <= 0:
                th = th + yi * xi
                th_list.append(th.tolist()[0])

    print('number of mistakes = ', len(th_list))
    return th, th_list


if __name__ == '__main__':

    # Homework 1
    # Perceptron algorithm iteration display
    x = np.array([[-1, -1], [1, 0], [-1, 1.5]])
    y = np.array([1, -1, 1])

    # case with x1
    x1 = x
    y1 = y
    th, th_list = perceptron_disp(x1, y1, 3)
    print('result for theta', th)
    print('variation of theta over iterations')
    print(th_list)

    print()
    # case with x2
    x2 = x[[1, 2, 0], :]
    y2 = y[[1, 2, 0]]
    th, th_list = perceptron_disp(x2, y2, 3)
    print('result for theta', th)
    print('variation of theta over iterations')
    print(th_list)

    print()
    # case with x3
    x3 = np.array([[-1, -1], [1, 0], [-1, 10]])
    y3 = np.array([1, -1, 1])
    th, th_list = perceptron_disp(x3, y3, 7)
    print('result for theta', th)
    print('variation of theta over iterations')
    print(th_list)

    print()
    # case with x4
    x4 = np.array([[1, 0], [-1, 10], [-1, -1]])
    y4 = np.array([-1, 1, 1])
    th, th_list = perceptron_disp(x4, y4, 7)
    print('result for theta', th)
    print('variation of theta over iterations')
    print(th_list)
