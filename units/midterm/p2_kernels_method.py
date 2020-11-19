import numpy as np

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    z = np.array([yi * (np.dot(xi, theta) + theta_0) for xi, yi in zip(feature_matrix, labels)])
    hl = np.sum(np.max([np.zeros(z.shape), 1-z], 0)) / z.size
    return hl


if __name__ == '__main__':
    # data points
    X = np.array([[0, 0], [2, 0], [1, 1], [0, 2], [3, 3], [4, 1], [5, 2], [1, 4], [4, 4], [5, 5]])
    Y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    nm = np.array([1, 65, 11, 31, 72, 30, 0, 21, 4, 15])

    # initial value
    th_init = [0, 0, 0]
    th_0_init = 0

    # feature mapping
    f = lambda x: np.array([x[:, 0]**2, np.sqrt(2)*x[:, 0]*x[:, 1], x[:, 1]**2])
    # np.array([X[:,0]**2, np.sqrt(2)*X[:,0]**2*X[:, 1]**2, X[:, 1]**2])

    # get new theta from for a linear transformation
    phiX = f(X).T
    th = th_init + (Y * nm) @ phiX
    th_0 = th_0_init + Y @ nm

    # Hinge loss for all point with current classifier
    hl = hinge_loss_full(phiX, Y, th, th_0)
    print('hinge losses: {}'.format(hl))
