# Author: Sophia Fakih
# Task 2: Perceptron Mistakes. Applying the perceptron algorithm through the origin
#   Variables:
#       x = list. Inputs. features_matrix
#       y = list. Label vector. Outputs
#       T = scalar. Number of times the algorithm will be executed.
#       theta = separating hyperplane
#       theta_progression = progression of the separating hyperplane ( θ , in the list format described above)

import numpy as np


def perceptron(features, label, T, times_mis=None):
    theta = np.zeros((1, features.shape[1]))           # 1 row, the same number of columns as 'x'
    theta_progression = []
    theta_0 = 0                                        #Initialize in cero
    count_times_mis = 0
    for t in range(T):
        for i in range(len(features)):                 #Up to number of row of
            if count_times_mis == times_mis:
                break
            else:
                xi = features[i]                        #Moving along x
                yi = label[i]                           #Moving along y
                if (yi * (np.matmul(theta, xi) + theta_0)) <= 0:
                    theta = theta + yi*xi                       #Update theta
                    theta_progression.append(theta.tolist()[0])    #Add current theta to the list to check progression
                    theta_0 = theta_0 + yi                          #Update Offset
                    count_times_mis += 1
    return theta.tolist()[0], theta_0, theta_progression

if __name__ == "__main__":

## 2.(a) Data set to find theta and theta_0 when it is misclasified 4 times
    x = np.array([[-4, 2], [-2, 1], [-1, -1], [2, 2], [1, -2]])
    #print('Shape of x:', x.shape)
    #print(range(len(x)))
    y = np.array([1, 1, -1, -1, -1])
    theta, theta_0, theta_progression = perceptron(x, y, 5, 4)
    print("Perceptron algorithm when it has misclassified 4 times")
    print('theta:', theta)
    print('theta_progression', theta_progression)
    print('number of mistakes', len(theta_progression))
    print('Offset: theta_0:', theta_0, '\n')

## 2.(b) Iteration order modified to find theta
    x1 = np.array([[-2, 1], [-1, -1], [2, 2], [1, -2], [-4, 2]])
    # print('Shape of x:', x.shape)
    # print(range(len(x)))
    y1 = np.array([1, -1, -1, -1, 1])
    theta1, theta1_0, theta1_progression = perceptron(x1, y1, 5)
    print("Perceptron algorithm when it has already converged with iteration order modified")
    print('theta:', theta1)
    print('theta_progression', theta1_progression)
    print('number of mistakes', len(theta1_progression))
    print('Offset: theta_0:', theta1_0, '\n')

#### 3.(b.4)
    x3 = np.array([[-1, 1], [1, -1], [1, 1], [2, 2]])
    y3 = np.array([1, 1, -1, -1])
    theta3, theta3_0, theta3_progression = perceptron(x3, y3, 5)
    print('theta:', theta3)
    print('theta_progression', theta3_progression)
    print('number of mistakes', len(theta3_progression))
    print('Offset: theta_0:', theta3_0)
    print()

