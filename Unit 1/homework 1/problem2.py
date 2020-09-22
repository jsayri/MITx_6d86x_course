import numpy as np


class TrainingItem:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def converge(item_list, t):
    theta = np.zeros(item_list[0].x.shape)
    all_thetas = []
    changed_index = -1
    mistakes_found = 0
    converged = False

    for i in range(t):
        for (index, item) in enumerate(item_list):
            if error(item, theta):
                theta = (theta + item.x)*item.y
                all_thetas.append(theta.tolist())
                mistakes_found += 1
                print(f'Mistake {mistakes_found} found at X_{index + 1}: new theta: {theta}')

    print(f'{all_thetas}')

    return theta


def error(item, theta):
    agreement = np.dot(item.x, theta)*item.y
    return agreement <= 0


item_list = [TrainingItem(x=np.array([-1, -1]),  y=1),
             TrainingItem(x=np.array([1, 0]),    y=-1),
             TrainingItem(x=np.array([-1, 10]), y=1)]

converge(item_list)

# item_list_start_from_2 = [TrainingItem(x=np.array([1, 0]),    y=-1),
#                           TrainingItem(x=np.array([-1, 10]), y=1),
#                           TrainingItem(x=np.array([-1, -1]),  y=1)]
#
# converge(item_list_start_from_2)
