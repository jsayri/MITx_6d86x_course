import numpy as np

# lecture 17, section 7. Value Iteration
# Another Example of Value Iteration (Software Implementation)

# states
s = ['s'+str(num) for num in range(0,5)]
a = ['ml', 'mr', 's'] # ml,mr: move left/right, s: stay

# Transition matrix
T = np.zeros((len(s), len(a), len(s))) # states, actions, future states prob

# IMPORTANT!!!
# Reward Note: The reward function is defined to be R(s,a,s′)=R(s), R(s=5)=1 and R(s)=0 otherwise

# If the agent chooses to move (either left or right) at any of the inner grid locations, such an action
# is successful with probability 1/3 and with probability 2/3 it fails to move
# move to the left
T[1:, 0, 0:4] = 1/3
T[1:, 0, 1:4] = 2/3

# move to the right
T[0:4, 1, 1:] = 1/3
T[0:4, 1, 0:4] = 2/3


# if the agent chooses to move left at the leftmost grid location, then the action ends up exactly the
# same as choosing to stay, i.e., staying at the leftmost grid location with probability 1/2, and ends
# up at its neighboring grid location with probability 1/2
T[0, 0, 0] = .5
T[0, 0, 1] = .5

# if the agent chooses to move right at the rightmost grid location, then the action ends up exactly the
# same as choosing to stay, i.e., staying at the rightmost grid location with probability 1/2, and ends
# up at its neighboring grid location with probability 1/2.
T[4, 1, 4] = .5
T[4, 1, 3] = .5

g = .5

# Value iteration algorithm
n_iter = 101

V_init = np.zeros(len(s))

V = np.zeros((n_iter, len(s))) # iteration matrix
V[0, :] = V_init # initialization
for k in range(1, n_iter):

    # previous iteration
    Vj = V[k-1, :]

    # at s=0
    # stay: T(0,s,0)*(R(0,s,0)+g*Vopt(0)) + T(0,s,1)*(R(0,s,1)+g*Vopt(1))
    V_s = .5 * (0 + g * Vj[0]) + .5 * (0 + g * Vj[1])
    # left: T(0,l,0)*(R(0,l,0)+g*Vopt(0)) + T(0,l,1)*(R(0,l,1)+g*Vopt(1))
    V_l = .5 * (0 + g * Vj[0]) + .5 * (0 + g * Vj[1])
    # right: T(0,r,0)*(R(0,r,0)+g*Vopt(0)) + T(0,r,1)*(R(0,l,1)+g*Vopt(1))
    V_r = 2/3 * (0 + g * Vj[0]) + 1/3 * (0 + g * Vj[1])

    V[k, 0] = np.max([V_s, V_l, V_r])

    # at s=1
    # stay: T(1,s,1)*(R(1,s,1) + g*Vopt(1)) + T(1,s,0)*(R(1,s,0) + g*Vopt(0)) + T(1,s,2)*(R(1,s,2) + g*Vopt(2))
    V_s = .5 * (0 + g * Vj[1]) + .25 * (0 + g * Vj[0]) + .25 * (0 + g * Vj[2])
    # left: T(1,l,1)*(R(1,l,1) + g*Vopt(1)) + T(1,l,0)*(R(1,l,0) + g*Vopt(0))
    V_l = 2 / 3 * (0 + g * Vj[1]) + 1 / 3 * (0 + g * Vj[0])
    # right: T(1,r,1)*(R(1,r,1) + g*Vopt(1)) + T(1,r,2)*(R(1,r,2) + g*Vopt(2))
    V_r = 2 / 3 * (0 + g * Vj[1]) + 1 / 3 * (0 + g * Vj[2])

    V[k, 1] = np.max([V_s, V_l, V_r])

    # at s=2
    # stay: T(2,s,2)*(R(2,s,2) + g*Vopt(2)) + T(2,s,1)*(R(2,s,1) + g*Vopt(1)) + T(2,s,3)*(R(2,s,3) + g*Vopt(3))
    V_s = .5 * (0 + g * Vj[2]) + .25 * (0 + g * Vj[1]) + .25 * (0 + g * Vj[3])
    # left: T(2,l,2)*(R(2,l,2) + g*Vopt(2)) + T(2,l,1)*(R(2,l,1) + g*Vopt(1))
    V_l = 2 / 3 * (0 + g * Vj[2]) + 1 / 3 * (0 + g * Vj[1])
    # right: T(2,r,2)*(R(2,r,2) + g*Vopt(2)) + T(2,r,3)*(R(2,r,3) + g*Vopt(3))
    V_r = 2 / 3 * (0 + g * Vj[2]) + 1 / 3 * (0 + g * Vj[3])

    V[k, 2] = np.max([V_s, V_l, V_r])

    # at s=3
    # stay: T(3,s,3)*(R(3,s,3) + g*Vopt(3)) + T(3,s,2)*(R(3,s,2) + g*Vopt(2)) + T(3,s,4)*(R(3,s,4) + g*Vopt(4))
    V_s = .5 * (0 + g * Vj[3]) + .25 * (0 + g * Vj[2]) + .25 * (0 + g * Vj[4])
    # left: T(3,l,3)*(R(3,l,3) + g*Vopt(3)) + T(3,l,2)*(R(3,l,2) + g*Vopt(2))
    V_l = 2 / 3 * (0 + g * Vj[3]) + 1 / 3 * (0 + g * Vj[2])
    # right: T(3,r,3)*(R(3,r,3) + g*Vopt(3)) + T(3,r,4)*(R(3,r,4) + g*Vopt(4))
    V_r = 2 / 3 * (0 + g * Vj[3]) + 1 / 3 * (0 + g * Vj[4])

    V[k, 3] = np.max([V_s, V_l, V_r])

    # at s=4
    # stay: T(4,s,4)*(R(4,s,4) + g*Vopt(4)) + T(4,s,3)*(R(4,s,3) + g*Vopt(3))
    V_s = .5 * (1 + g * Vj[4]) + .5 * (1 + g * Vj[3])
    # left: T(4,l,4)*(R(4,l,4) + g*Vopt(4)) + T(4,l,3)*(R(4,l,3) + g*Vopt(3))
    V_l = 2 / 3 * (1 + g * Vj[4]) + 1 / 3 * (1 + g * Vj[3])
    # right: T(4,r,4)*(R(4,r,4) + g*Vopt(4)) + T(4,r,3)*(R(4,r,3) + g*Vopt(3))
    V_r = .5 * (1 + g * Vj[4]) + .5 * (1 + g * Vj[3])

    V[k, 4] = np.max([V_s, V_l, V_r])

print('end execution')

print('V_{} = '.format(n_iter), V[-1, :])