import numpy as np


def hinge_loss(z):
    if z >= 1:
        return 0
    return 1 - z


def squared_loss(z):
    return (z**2)/2


def z(feature_vector, label, theta):
    return label - np.dot(feature_vector, theta)


class TrainingData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


theta = np.array([0, 1, 2])

all_data = [TrainingData(x=np.array([1, 0, 1]), y=2),
            TrainingData(x=np.array([1, 1, 1]), y=2.7),
            TrainingData(x=np.array([1, 1, -1]), y=-0.7),
            TrainingData(x=np.array([-1, 1, 1]), y=2)]

all_zs = [z(data.x, data.y, theta) for data in all_data]
all_hinge_loss = [hinge_loss(z) for z in all_zs]
all_squared_loss = [squared_loss(z) for z in all_zs]


print(sum(all_hinge_loss)/4)
print(sum(all_squared_loss)/4)




