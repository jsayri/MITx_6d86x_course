import numpy as np

# calculating cost
x1 = np.array([-1, 2])
x2 = np.array([-2, 1])
x3 = np.array([-1, 0])
z1 = np.array([-1, 1])
x4 = np.array([2, 1])
x5 = np.array([3, 2])
z2 = np.array([2, 2])

# euclidean distance
eu_d = lambda x, z: np.sum(x - z)**2

# Cost as the sum of the eucliean distances
C1 = eu_d(x1, z1) + eu_d(x2, z1) + eu_d(x3, z1)
C2 = eu_d(x4, z2) + eu_d(x5, z2)

print('Cost 1: ', C1)
print('Cost 2: ', C2)