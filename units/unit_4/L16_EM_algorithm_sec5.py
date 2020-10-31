import numpy as np

# Lecture 16, section 5

# Prob. Gaussian Mixture Model: An Example Update - E-Step 
# define points
x1 = .2
x2 = -.9
x3 = -1
x4 = 1.2
x5 = 1.8
X = np.array([x1, x2, x3, x4, x5])

u1, u2 = -3, 2
s1, s2 = 4, 4
p1, p2 = .5, .5

# gaussian function
N = lambda x, u, s, d=1: 1/(2*np.pi*s)**(d/2) * np.exp(-1/(2*s) * np.linalg.norm(x-u)**2)

# prob of x given params
pxgp = np.zeros(X.shape)
for ii, xi in enumerate(X):
    pxgp[ii] = p1 * N(xi, u1, s1) + p2 * N(xi, u2, s2)

# posteriour prob.
pjgi_m = np.zeros((len(X), 2))
for jj, uj, sj, pj in zip(range(0,2), [u1, u2], [s1, s2], [p1, p2]):
    print('Cluster ',jj+1, ', with u = {}, s^2 = {} and p = {}'.format(uj, sj, pj))
    for ii, xi in enumerate(X):
        pjgi = pj * N(xi, uj, sj) / pxgp[ii]
        print('p({}|{}) = '.format(jj+1, ii+1), round(pjgi, 5))
        pjgi_m[ii, jj] = pjgi

    print('model update:')
    nj_hat = np.sum(pjgi_m[:,jj])
    pj_hat = nj_hat / len(X)
    uj_hat = np.dot(pjgi_m[:,jj], X) / nj_hat
    sj_hat = np.dot(pjgi_m[:,jj], (X - uj_hat)**2) / (1 * nj_hat)

    print('p{0}_hat = {1} \nu{0}_hat = {2} \ns{0}_hat = {3}'.format(jj+1, round(pj_hat,5), round(uj_hat, 5), round(sj_hat, 5)))

    print()
print(pjgi_m)

