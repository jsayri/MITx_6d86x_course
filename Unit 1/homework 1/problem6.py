import numpy as np


class TrainingItem:
    @staticmethod
    def get_training_items(d):
        return [TrainingItem(t, d) for t in range(1, d + 1)]

    def find_x(self, i):
        if i == self.t:
            return np.cos(np.pi * self.t)
        return 0

    def __init__(self, t, d):
        self.y = np.random.choice([-1, 1])
        self.t = t

        dimensions = range(1, d + 1)
        self.x = np.array(list(map(self.find_x, dimensions)))

    def __str__(self):
        return f'x: {self.x}. y: {self.y}. t: {self.t}'


class Classifier:
    def __init__(self, x_shape, offset=False):
        self.theta = np.zeros(x_shape)
        if offset:
            self.theta_0 = 0

    def __str__(self):
        if hasattr(self, 'theta_0'):
            return f'theta: {self.theta}, theta_0: {self.theta_0}'

        return f'theta: {self.theta}'

def converge(item_list):
    cfr = Classifier(item_list[0].x.shape)

    all_thetas = []
    mistakes_found = 0
    changed_index = -1

    converged = False

    while not converged:
        for (index, item) in enumerate(item_list):
            if error(item, cfr):
                cfr.theta = cfr.theta + item.y * item.x
                if hasattr(cfr, 'theta_0'):
                    cfr.theta_0 = cfr.theta_0 + item.y

                all_thetas.append(cfr.theta.tolist())
                mistakes_found += 1
                changed_index = index

                print(f'Mistake {mistakes_found} found at X_{index + 1}: new classifier: {cfr}')

            else:
                if changed_index == index:
                    converged = True
                    break

    print(f'{all_thetas}')

    return cfr


def error(item, cfr):
    agreement = np.dot(item.x, cfr.theta) * item.y
    if hasattr(cfr, 'theta_0'):
        agreement += item.y * cfr.theta_0
    return agreement <= 0


item_list = TrainingItem.get_training_items(3)
for item in item_list:
    print(item)

converge(item_list)
