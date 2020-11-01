# Homework 4, 
# Section 2. Maximum Likelihood Estimation

import numpy as np
import itertools

if __name__ == "__main__":
    # Unigram Model
    print('Unigram Model, solution')
    doc_sequence = ['A', 'B', 'A', 'B', 'B', 'C', 'A', 'B', 'A', 'A', 'B', 'C', 'A', 'C']
    dict_W = ['A', 'B', 'C']
    theta_array = np.zeros(len(dict_W))

    # what are the parameters' MLE values?
    for ii, w in enumerate(dict_W):
        theta_w = doc_sequence.count(w) / len(doc_sequence)
        print('Theta_{}'.format(w), ' =',round(theta_w, 3))
        theta_array[ii] = theta_w

    # most likely sequence
    test_sec = [['A', 'B', 'C'], ['B', 'B', 'B'], ['A', 'B', 'B'], ['A', 'A', 'C']]
    for secq in test_sec:
        sec_likelihood = np.prod([theta_array[dict_W.index(w)] for w in secq])
        print(secq, sec_likelihood)

    # MLE for bigrams models, demonstration by Lei Mao
    # https://leimao.github.io/blog/Maximum-Likelihood-Estimation-Ngram/

    # Bigram Model 3, implementation
    print('\nBigram Model 3, solution')
    test_sequence = 'AABCBAB'

    # train parameters from dict_W for bigrams with doc_sequence
    doc_sequence = 'ABABBCABAABCAC'
    theta_bi = np.zeros((3,3))
    for ii, w1 in enumerate(dict_W):
        for jj, w2 in enumerate(dict_W):
            # p(w2 | w1)
            theta_bi[ii, jj] = doc_sequence.count(w1+w2) / sum([doc_sequence.count(w1+wj) for wj in dict_W])
            print('p({}|{}) = {}'.format(w2, w1, round(theta_bi[ii, jj], 5)))

    # test sequence with the bigrams training set
    probs_sec = []
    for ii, w2 in enumerate(test_sequence):
        if ii == 0:
            pw2gw1 = 1 # assume uniform for p(w|null)
        else:
            w1 = test_sequence[ii-1]
            pw2gw1 = theta_bi[dict_W.index(w1), dict_W.index(w2)]

        probs_sec.append(pw2gw1)

    print('test sequence prob. bigrams: {}'.format(probs_sec))
    print('test sequence total prob {}'.format(np.prod(probs_sec)))
