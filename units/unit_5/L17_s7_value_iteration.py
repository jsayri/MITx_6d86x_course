import numpy as np

# lecture 17, section 7. Value Iteration
# Another Example of Value Iteration (Software Implementation)

# states
s = ['s'+str(num) for num in range(0,5)]
a = ['ml', 'mr', 's'] # ml,mr: move left/right, s: stay
ad = {'ml': 0, 'mr': 1, 's': 2} # action dictionary

# Transition matrix
T = np.zeros((len(s), len(a), len(s))) # states, actions, future states prob

# IMPORTANT!!!
# Reward Note: The reward function is defined to be R(s,a,s′)=R(s), R(s=5)=1 and R(s)=0 otherwise

#  If the agent chooses to stay at the location, such an action is successful with probability 1/2
T[0, 2, 0] = .5
T[1, 2, 1] = .5
T[2, 2, 2] = .5
T[3, 2, 3] = .5
T[4, 2, 4] = .5

# if the agent is at the leftmost or rightmost grid location it ends up at its neighboring grid
# location with probability 1/2
T[0, 2, 1] = .5
T[4, 2, 3] = .5

# if the agent is at any of the inner grid locations it has a probability 1/4 each of ending up
# at either of the neighboring locations.
T[1, 2, 0], T[1, 2, 2] = .25, .25
T[2, 2, 1], T[2, 2, 3] = .25, .25
T[3, 2, 2], T[3, 2, 4] = .25, .25

# If the agent chooses to move (either left or right) at any of the inner grid locations, such an
# action is successful with probability 1/3 and with probability 2/3 it fails to move, and
# move left, fails to move left
T[1, 0, 0], T[1, 0, 1] = 1/3, 2/3
T[2, 0, 1], T[2, 0, 2] = 1/3, 2/3
T[3, 0, 2], T[3, 0, 3] = 1/3, 2/3
T[4, 0, 3], T[4, 0, 4] = 1/3, 2/3

# move right, fails to move right
T[0, 1, 1], T[0, 1, 0] = 1/3, 2/3
T[1, 1, 2], T[1, 1, 1] = 1/3, 2/3
T[2, 1, 3], T[2, 1, 2] = 1/3, 2/3
T[3, 1, 4], T[3, 1, 3] = 1/3, 2/3

# if the agent chooses to move left at the leftmost grid location, then the action ends up exactly
# the same as choosing to stay, i.e., staying at the leftmost grid location with probability 1/2,
# and ends up at its neighboring grid location with probability 1/2
T[0, 0, 0], T[0, 0, 1] = .5, .5 # move left at the leftmost grid location
T[4, 1, 4], T[4, 1, 3] = .5, .5 # move right at the rightmost grid location


# The reward function is defined to be R(s,a,s′)=R(s), R(s=5)=1 and R(s)=0 otherwise.
# Reward table
R = np.zeros(len(s)) # reward success, rows: states, columns: all actions (R(s))
#Rs = np.zeros((len(s), len(a))) # reward success, rows: states, columns: actions (ml, mr, s)
#Rf = np.zeros((len(s), len(a))) # reward fails
R[4] = 1

# gamma value
g = .5

# Value iteration algorithm
n_iter = 100

V = np.zeros((n_iter, len(s))) # iteration matrix
for k in range(1, n_iter):

    # get previous iteration
    Vj = V[k-1, :]

    # at each state-transition Vk*[s]=max_a(sum_s'(T(s,a,s') * (R(s) + g * Vj(s)) ) )
    for ii, ss in enumerate(s):
        # Vdiscount = R + g * Vj # consider R(s,a,s') = R(s') (not in this problem)
        Vdiscount = R[ii] + g * Vj # consider R(s,a,s') = R(s)
        V[k, ii] = np.max(T[ii] @ Vdiscount.T)

print('end execution')

print('V_{} = '.format(n_iter), V[-1, :])