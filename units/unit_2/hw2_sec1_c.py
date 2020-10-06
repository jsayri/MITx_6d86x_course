import numpy as np

# matrix factorization based on code from: http://www.quuxlabs.com/wp-content/uploads/2010/09/mf.py_.txt
# original code from Albert Au Yeung (2010)
# adapted to 6.86x factorization matrix nomenclature

def matrix_factorization(Y, U, V, K, steps=5000, alpha=0.0002, L=0.02):
    """
    Inputs:
        Y     : a matrix to be factorized, dimension N x M
        U     : an initial matrix of dimension N x K
        V     : an initial matrix of dimension M x K
        K     : the number of latent features
        steps : the maximum number of steps to perform the optimisation
        alpha : the learning rate
        L  : the regularization parameter
    Outputs:
        the final matrices U and V, after a given number of "steps"
    """
    V = V.T
    for step in range(steps):
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i][j] > 0:
                    eij = Y[i][j] - np.dot(U[i,:],V[:,j])
                    for k in range(K):
                        U[i][k] = U[i][k] + alpha * (2 * eij * V[k][j] - L * U[i][k])
                        V[k][j] = V[k][j] + alpha * (2 * eij * U[i][k] - L * V[k][j])
        eY = np.dot(U,V)
        e = 0
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if Y[i][j] > 0:
                    e = e + pow(Y[i][j] - np.dot(U[i,:],V[:,j]), 2)
                    for k in range(K):
                        e = e + (L/2) * ( pow(U[i][k],2) + pow(V[k][j],2) )
        if e < 0.001:
            break
    return U, V.T


if __name__ == "__main__":
    Y = np.array([
         [5, 0, 7],
         [0, 2, 0],
         [4, 0, 0],
         [0, 3, 6],
        ])

    U = np.array([6, 0, 3, 6]).reshape((4, 1))
    V = np.array([4, 2, 1]).reshape((3, 1))

    L = 1   # lambda, algorithm hyperparameter
    # nU, nV = matrix_factorization(Y, U, V, 1, steps=1, alpha=1, L=1)

    X = U * V.T

    # squared error
    eS = sum((Yai - Xai)**2 for Ya, Xa in zip(Y, X) for Yai, Xai in zip(Ya, Xa) if Yai > 0) / 2

    # Regularization error
    eR = L / 2 * sum(U ** 2) + L / 2 * sum(V ** 2)

    # calculate the new U with fixed V
