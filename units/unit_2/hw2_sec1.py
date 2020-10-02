# Operations for homework 2
import numpy as np

# section 1. Collaborative filtering, kernels, linear regression

def get_sqrt_error(Y, X):
    e_sum = 0
    for Ya, Xa in zip(Y, X):
        for Yai, Xai in zip(Ya, Xa):
            if Yai != 0:
                e_sum += (Yai - Xai)**2
    return e_sum / 2


if __name__ == '__main__':
    U0 = np.array([6, 0, 3, 6])
    V0 = np.array([4, 2, 1])
    L = 1   # hyperparameter lambda

    X0 = np.multiply(U0.reshape((4,1)), V0)
    print(X0)

    Y = np.array([[5, 0, 7], [0, 2, 0], [4, 0, 0], [0, 3, 6]])

    # calculate squared error term for the current estimate X
    sqrt_error = sum([(Yai - Xai)**2 for Ya, Xa in zip(Y, X0) for Yai, Xai in zip(Ya, Xa) if not Yai == 0]) / 2
    sqrt_error2 = get_sqrt_error(Y, X0)
    print('square error: {}'.format(sqrt_error))

    # calculate the regularization terms for the current estimate X
    reg_error = L/2 * sum(U0**2) + L/2 * sum(V0**2)
    print('regularization error: {}'.format(reg_error))