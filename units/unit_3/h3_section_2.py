import numpy as np


# LSTM states
# Calculate the values ht at each time-step and enter them below as an array [h0,h1,h2,h3,h4,h5].

X = np.array([0,0, 1, 1, 1, 0])
c_old = 0
h_old = 0
h = []

# aproximations
sig_aprox = lambda x: 1 if x>=1 else 0 if x<=-1 else x #1/(1+np.exp(-x))
tanh_aprox = lambda x: 1 if x>=1 else -1 if x<=-1 else x
# sigmoid argument equation
arg_eq = lambda Wh, Wx, b, h_old, x: Wh*h_old + Wx*x + b
# cell gate equation
cell_eq = lambda f, c_old, i, Wh, Wx, b, h_old, x: f * c_old + i * tanh_aprox(Wh*h_old + Wx*x + b)
# new state equation
h_eq = lambda o, c: np.round(o * tanh_aprox(c))

# function parameters
Wfh, Wfx, bf = 0, 0, -100
Wih, Wix, bi = 0, 100, 100
Woh, Wox, bo = 0, 100, 0
Wch, Wcx, bc = -100, 50, 0

# loop for sequence 1, X = [0,0, 1, 1, 1, 0]
print_states = True
print('sequence x = {}'.format(X))

for i, xi in enumerate(X):
    # time-step operations
    fi = sig_aprox(arg_eq(Wfh, Wfx, bf, h_old, xi))
    ii = sig_aprox(arg_eq(Wih, Wix, bi, h_old, xi))
    oi = sig_aprox(arg_eq(Woh, Wox, bo, h_old, xi))
    ci = cell_eq(fi, c_old, ii, Wch, Wcx, bc, h_old, xi)
    hi = h_eq(oi, ci)

    h.append(hi)  # store state
    h_old = hi  # update old state
    c_old = ci  # update old memory cell

    if print_states:
        print('State %d, with x%d = %d' % (i, i, xi))
        print('f{} = {}'.format(i, fi))
        print('i{} = {}'.format(i, ii))
        print('o{} = {}'.format(i, oi))
        print('c{} = {}'.format(i, ci))
        print('h{} = {}'.format(i, hi))
        print()

print('h states: {}'.format(h))


# loop for sequence 2, X = [1, 1, 0, 1, 1]
X = np.array([1, 1, 0, 1, 1])
c_old = 0
h_old = 0
h = []
print_states = False
print('sequence x = {}'.format(X))

for i, xi in enumerate(X):
    # time-step operations
    fi = sig_aprox(arg_eq(Wfh, Wfx, bf, h_old, xi))
    ii = sig_aprox(arg_eq(Wih, Wix, bi, h_old, xi))
    oi = sig_aprox(arg_eq(Woh, Wox, bo, h_old, xi))
    ci = cell_eq(fi, c_old, ii, Wch, Wcx, bc, h_old, xi)
    hi = h_eq(oi, ci)

    h.append(hi)  # store state
    h_old = hi  # update old state
    c_old = ci  # update old memory cell

    if print_states:
        print('State %d, with x%d = %d' % (i, i, xi))
        print('f{} = {}'.format(i, fi))
        print('i{} = {}'.format(i, ii))
        print('o{} = {}'.format(i, oi))
        print('c{} = {}'.format(i, ci))
        print('h{} = {}'.format(i, hi))
        print()

print('h states: {}'.format(h))

# Third execution
X = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0])
c_old = 0
h_old = 0
h = []
print_states = False
print('sequence x = {}'.format(X))

for i, xi in enumerate(X):
    # time-step operations
    fi = sig_aprox(arg_eq(Wfh, Wfx, bf, h_old, xi))
    ii = sig_aprox(arg_eq(Wih, Wix, bi, h_old, xi))
    oi = sig_aprox(arg_eq(Woh, Wox, bo, h_old, xi))
    ci = cell_eq(fi, c_old, ii, Wch, Wcx, bc, h_old, xi)
    hi = h_eq(oi, ci)

    h.append(hi)  # store state
    h_old = hi  # update old state
    c_old = ci  # update old memory cell

    if print_states:
        print('State %d, with x%d = %d' % (i, i, xi))
        print('f{} = {}'.format(i, fi))
        print('i{} = {}'.format(i, ii))
        print('o{} = {}'.format(i, oi))
        print('c{} = {}'.format(i, ci))
        print('h{} = {}'.format(i, hi))
        print()

print('h states: {}'.format(h))