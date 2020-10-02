#Author: Sophia Fakih
# Task 1: Perceptron Mistakes. Applying the perceptron algorithm through the origin
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

    x = np.array([[-1, -1], [1, 0], [-1, 1.5]])
    #print('Shape of x:', x.shape)
    #print(range(len(x)))
    y = np.array([1, -1, 1])
    theta, theta_progression=perceptron(x, y, 3)
    print("Perceptron algorithm starting with x(1)")
    print('theta:', theta)
    print('theta_progression:', theta_progression)
    print('mistakes:', len(theta_progression), '\n')

    print("Perceptron algorithm starting with x(2)")
    x2 = np.array([[1, 0], [-1, 1.5], [-1, -1]])
    #print('Shape of x:', x.shape)
    #print(range(len(x)))
    y2 = np.array([-1, 1, 1])
    theta2, theta2_progression=perceptron(x2, y2, 3)
    print('theta:', theta2)
    print('theta_progression:', theta2_progression)
    print('mistakes:', len(theta2_progression), '\n')

    x3 = np.array([[-1, -1], [1, 0], [-1, 10]])
    y3 = np.array([1, -1, 1])
    theta3, theta3_progression = perceptron(x3, y3, 8)
    print("Perceptron algorithm with a new x(3) and starting with x(1)")
    print('theta:', theta3)
    print('theta_progression:', theta3_progression)
    print('mistakes:', len(theta3_progression), '\n')

    x4 = np.array([[1, 0], [-1, 1.5], [-1, 10]])
    y4 = np.array([-1, 1, 1])
    theta4, theta4_progression = perceptron(x4, y4, 5)
    print("Perceptron algorithm with a new x(3) and starting with x(2)")
    print('theta:', theta4)
    print('theta_progression:', theta4_progression)
    print('mistakes:', len(theta4_progression), '\n')