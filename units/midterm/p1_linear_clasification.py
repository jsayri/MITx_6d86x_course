import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    z = np.array([yi * (np.dot(xi, theta) + theta_0) for xi, yi in zip(feature_matrix, labels)])
    hl = np.sum(np.max([np.zeros(z.shape), 1-z], 0)) / z.size
    return hl

# Pegasos perceptron
def pegasos_single_step_update(xi, yi, L, eta, c_th, c_th_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        xi - A numpy array describing a single data point.
        yi - The correct classification of the feature vector.
        L - The lambda value being used to update the parameters.
        eta - Learning rate to update parameters.
        c_th - The current theta being used by the Pegasos
            algorithm before this update.
        c_th_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns:
        new_theta - numpy array with the value of the theta after update
        new_theta_0 - real valued number of theta_0 after updated
    """
    if (yi * (np.dot(xi, c_th) + c_th_0)) <= 1.0:
        new_theta = (1 - eta * L) * c_th + eta * yi * xi
        new_theta_0 = c_th_0 + eta * yi
    else:
        new_theta = (1 - eta * L) * c_th
        new_theta_0 = c_th_0
    return new_theta, new_theta_0



def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data for T iterations
    through the data set.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    Args:
        feature_matrix - numpy matrix, one data point per row.
        labels - numpy array with the correct classification for the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lambda value being used to update the Pegasos
            algorithm parameters.

    Returns:
        theta -  numpy array with the value of the linear classification
        theta_0 - the offset classification parameter
    """
    theta = np.zeros(feature_matrix.shape[1])
    theta_0 = 0
    counter = 0
    th_update = []
    th0_update = []

    for t in range(T):
        for xi, yi in zip(feature_matrix, labels):
            counter += 1  # update counter
            eta = 1 / np.sqrt(counter)  # update learning rate 'eta'
            theta, theta_0 = pegasos_single_step_update(xi, yi, L, eta, theta, theta_0)
            th_update.append(theta)
            th0_update.append(theta_0)
    return theta, theta_0, (th_update, th0_update)


def plot_history(data):
    """Plots the theta over iterations"""
    th = data[0]
    fig, ax = plt.subplots()
    ax.title.set_text('Theta cost')
    plt.plot(range(0, len(th)), th)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Theta')



if __name__ == '__main__':
    # data points
    X = np.array([[0, 0], [2, 0], [3, 0], [0, 2], [2, 2], [5, 1], [5, 2], [2, 4], [4, 4], [5, 5]])
    Y = np.array([-1, -1, -1, -1, -1, 1, 1, 1, 1, 1])
    nm = np.array([1, 9, 10, 5, 9, 11, 0, 3, 1, 1])

    # initial values
    theta_init = np.array([0, 0])
    theta_0_init = 0.

    # get theta from manual execution of perceptron
    theta = theta_init + (Y*nm) @ X
    theta_0 = theta_0_init + Y @ nm

    # by guess and math
    theta = np.array([1, 1])
    theta_0 = np.array([-5])

    # pegasos perceptron
    # theta, theta_0, _ = pegasos(X, Y, 200, 1)

    # maximum distance
    d = 1 / np.linalg.norm(theta)

    # Hinge loss for all point with current classifier
    hl = hinge_loss_full(X, Y, theta, theta_0)

    # for a new separator (div by 2)
    hl = hinge_loss_full(X, Y, theta/2, theta_0/2)
