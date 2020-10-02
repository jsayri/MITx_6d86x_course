# from lecture 6, I want to check that the combination formula is working for
# some few examples with order 1, 2 and 3 and vectors x dimension 1 up to 4 such as
# x = [x1], x = [x1, x2, x3], etc

import numpy as np
import math

def binom(n, k):
    '''Calculate binomial coeffcient'''
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)

def get_all_comb_no_rep(n, r):
    '''
    Calculate the sum of all combination without repetition with the next formula
    sum(comb(n, ri)) for ri = 1 to r
    n :     number of elements to combine
    r :     posible combinations
    '''
    return sum([binom(n+ri-1, ri) for ri in range(1, r+1)])


def get_x_dim_d(d = 3):
    '''
    Define a list of vector x for a given dimension "d"
    Inputs
    d : scalar, dimension of the vector 'x', default d = 3
    '''

    return [str('x{}'.format(i+1)) for i in range(d)]

def get_comb_single(x, r):
    '''
    Return a list with the possible combination of a vector x with order "r"
    Inputs
    x :     list, vector with elements to combine, dimension "n"
    r :     scalar, order of the combination, default 2
    '''
    cum_list = []
    for i in range(len(x)):
        for j in range(len(x)):
            element = x[i]
            rr = r - 1
            while rr > 0:
                rr -= 1
                if j >= i:
                    element += x[j]
            if not (j < i) and element not in cum_list:
                cum_list.append(element)

    return cum_list

def get_comb_order(x, r):
    '''
    Get all possible polynomiak combinations for a vector of dimension "n" and order "r"
    combination as the concatenation of each possible order r = 2 means order 1 and order 2
    x :     list, vector with elements to combine, dimension "n"
    r :     scalar, order of the combination, default 2
    '''
    x_list = []
    for ri in range(r):
        x_list += get_comb_single(x, ri+1)
    return x_list

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # define initial vector
    d = 2
    r = 2
    x = get_x_dim_d(d)
    print(x)

    # case order 2
    x_single = get_comb_single(x, 2)
    print(x_single)

    # case order 1
    x_single = get_comb_single(x, 1)
    print(x_single)

    # not working for order 3... !!!!!!
    x_comb = get_comb_order(x, 2)
    print(x_comb)

    # combination without repetition formula
    print('All possible combination (formula) are: {}'.format(get_all_comb_no_rep(d, r)))
