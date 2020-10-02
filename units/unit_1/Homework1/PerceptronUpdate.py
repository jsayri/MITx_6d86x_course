#Author: Sophia Fakih
# Task 3: Perceptron Updates. Convergence of perceptron algorithm
#   Variables:
#       x = list. Inputs. Features matrix
#       y = list. Label vector. Outputs
#       T = scalar. Number of times the algorithm will be executed.
#       theta = separating hyperplane
#       theta_progression = progression of the separating hyperplane ( θ , in the list format described above)

import numpy as np

def perceptron(features, label, T):
    theta = np.zeros((1, features.shape[1]))        # 1 row, the same number of columns as 'x'
    theta_progression = []
    theta_0 = 0                                     #Through the origin
    for t in range(T):
        for i in range(len(features)):              #Up to number of row of
            xi = features[i]                        #Moving along x
            yi = label[i]                           #Moving along y
            if (yi * (np.matmul(theta, xi) + theta_0)) <= 0:
                theta = theta + yi*xi
                theta_progression.append(theta.tolist()[0])
    return theta.tolist()[0], theta_progression


if __name__ == "__main__":

    x = np.array([[-1, 0], [0, 1]])
    #print('Shape of x:', x.shape)
    #print(range(len(x)))
    y = np.array([1, 1])
    theta, theta_progression=perceptron(x, y, 5)
    print("Perceptron algorithm for d=2")
    print('theta:', theta)
    print('theta_progression:', theta_progression)
    print('mistakes:', len(theta_progression), '\n')
