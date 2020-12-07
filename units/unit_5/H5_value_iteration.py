import numpy as np

# homework 5, section 1. Value Iteration for Markov Decision Process
# b, If we initialize the value function with 0, enter the value of state B after:

s = ['a', 'b', 'c', 'd']
a = ['up', 'down']

# Transition matrix
T = np.zeros((len(s), len(a), len(s))) # states, actions, future states prob

g = .75

# Rewards (row 1: up, row 2: down)
R = np.array([[1, 1, 10, 0], [0, 1, 1, 10]])

# Transition matrix
T = np.zeros((len(s), len(a), len(s)))

# going up transition
T[0, 0, 1] = 1
T[1, 0, 2] = 1
T[2, 0, 3] = 1

# going down transition
T[3, 1, 2] = 1
T[2, 1, 1] = 1
T[1, 1, 0] = 1


# Value iteration algorithm
n_iter = 100

V_init = np.zeros(len(s))
V = np.zeros((n_iter, len(s))) # iteration matrix

V[0, :] = V_init # initialization
for k in range(1, n_iter):

    # previous iteration
    Vj = V[k-1, :]

    # at each transition
    for ii, ss in enumerate(s):
        Vdiscount = R[:, ii] + g * np.tile(Vj, (len(a), 1)).T
        V[k, ii] = np.max((T[ii].T * Vdiscount).sum(axis=0))

print('end execution')

print('V_{} = '.format(n_iter), V[-1, :])