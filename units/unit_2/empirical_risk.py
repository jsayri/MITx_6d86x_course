# This is a sample Python script.

import numpy as np

def empirical_risk(x, y, th):
    '''Empirical risk function
        Rn = 1/n * sum(yi - th * xi) for i = 1,..,n
        Inputs
        x : feature matrix, rows are samples
        y : real values results, exist in reals
        th : theta vector, parameter for linear regression
        '''
    n = len(y)  # total number from training set (rows in feature matrix)
    return sum([(yi - np.dot(xi, th)) ** 2 / 2 for xi, yi in zip(x, y)]) / n

def hinge_empirical_risk(x, y, th):
    '''Empirical risk function for hinge loss function
        Rn = 1/n * sum(yi - th * xi) for i = 1,..,n
        Inputs
        x : feature matrix, rows are samples
        y : real values results, exist in reals
        th : theta vector, parameter for linear regression
        '''
    n = len(y)  # total number from training set (rows in feature matrix)
    z = lambda xi, yi, th : yi - np.dot(xi, th)
    return sum([1 - z(xi, yi, th) if z(xi, yi, th) < 1 else 0 for xi, yi in zip(x, y)]) / n


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # calculate the empirical risk for a set of values
    x = np.array([[1, 0, 1], [1, 1, 1], [1, 1, -1], [-1, 1, 1]])
    y = np.array([2, 2.7, -0.7, 2])
    th = np.array([0, 1, 2])

    # calculate empirical risk
    Rn = empirical_risk(x, y, th)
    print('Rn : {:.4f}'.format(Rn))

    # calculate hinge empirical risk
    Rn = hinge_empirical_risk(x, y, th)
    print('Rn : {:.4f}'.format(Rn))
