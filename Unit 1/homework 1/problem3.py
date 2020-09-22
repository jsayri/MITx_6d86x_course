import numpy as np


class TrainingItem:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Classifier:
    def __init__(self, x_shape):
        self.theta = np.zeros(x_shape)
        self.theta_0 = 0

    def __str__(self):
        return f'theta: {self.theta}, theta_0: {self.theta_0}'


def converge(item_list):
    cfr = Classifier(item_list[0].x.shape)

    all_thetas = []
    mistakes_found = 0
    changed_index = -1

    continue_loop = True
    converged = False

    while not converged:
        for (index, item) in enumerate(item_list):
            if error(item, cfr):
                cfr.theta = cfr.theta + item.y * item.x
                cfr.theta_0 = cfr.theta_0 + item.y

                all_thetas.append(cfr.theta.tolist())
                mistakes_found += 1
                changed_index = index

                print(f'Mistake {mistakes_found} found at X_{index + 1}: new classifier: {cfr}')

                # if mistakes_found == 4:
                #     continue_loop = False
                #     break
            else:
                if changed_index == index:
                    converged = True
                    break

    print(f'{all_thetas}')

    return cfr


def error(item, cfr):
    agreement = np.dot(item.x, cfr.theta) * item.y + item.y * cfr.theta_0
    return agreement <= 0


item_list = [TrainingItem(x=np.array([-1, 1]), y=1),
             TrainingItem(x=np.array([1, -1]), y=1),
             TrainingItem(x=np.array([1, 1]), y=-1),
             TrainingItem(x=np.array([2, 2]), y=-1)]

converge(item_list)
