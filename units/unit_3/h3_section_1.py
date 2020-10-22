import numpy as np

W = np.array([[1, 0, -1], [0, 1, -1], [-1, 0, -1], [0, -1, -1]])
V = np.array([[1, 1, 1, 1, 0], [-1, -1, -1, -1, 2]])

X = np.array([3, 14])

Z = np.matmul(X, W[:, 0:-1].T) + W[:, -1]

fz = np.maximum(Z, np.zeros(Z.shape))

U = np.matmul(fz, V[:, 0:-1].T) + V[:, -1]

fu = np.maximum(U, np.zeros(U.shape))

O = np.exp(fu) / np.exp(fu).sum()

# output of neural networks
# asuming same V as before, calculate O for multiple cases
# a) f(z1) + f(z2) + f(z3) + f(z4) = 1
Ua = np.array([1, 1])
fua = np.maximum(Ua, np.zeros(Ua.shape))
Oa = np.exp(fua) / np.exp(fua).sum()

# b) f(z1) + f(z2) + f(z3) + f(z4) = 0
Ub = np.array([0, 2])
fub = np.maximum(Ub, np.zeros(Ub.shape))
Ob = np.exp(fub) / np.exp(fub).sum()

# c) f(z1) + f(z2) + f(z3) + f(z4) = 3
Uc = np.array([3, -1])
fuc = np.maximum(Uc, np.zeros(Uc.shape))
Oc = np.exp(fuc) / np.exp(fuc).sum()