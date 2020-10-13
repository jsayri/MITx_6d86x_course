import numpy as np

U_0 = np.array([6, 0, 3, 6])
V_0 = np.array([4, 2, 1]).transpose()

#answer to 1a:
print(f'1a: \n{np.outer(U_0, V_0)} \n')


def get_squared_error(matrix_1, matrix_2):
    squared_error = 0
    for row_1, row_2 in zip(matrix_1, matrix_2):
        for (item_1, item_2) in zip(row_1, row_2):
            if item_1 is None:
                continue

            current_error = ((item_1 - item_2)**2)/2
            squared_error += current_error

    return squared_error

def reg_value(value, l):
    return (l/2)*value**2

def get_regularization_term(U, V, l):
    reg_terms_U = [reg_value(item, l) for item in U]
    reg_terms_V = [reg_value(item, l) for item in V]

    return np.sum(reg_terms_U) + np.sum(reg_terms_V)


y = np.array([[5, None, 7],
              [None, 2, None],
              [4, None, None],
              [None, 3, 6]])

l = 1
x = np.outer(U_0, V_0)

#answer to 1b
print(f'1b: \nsquared error:{get_squared_error(y, x)}')
print(f'reg term: {get_regularization_term(U_0, V_0,  l)}\n')


def next_U(V, U_shape, y, l):
    numerator = np.zeros(U_shape)
    denominator = np.ones(U_shape) * l
    for U_index, row in enumerate(y):
        for V_index, label in enumerate(row):
            if label is None:
                continue

            numerator[U_index] += V[V_index] * label
            denominator[U_index] += V[V_index]**2

    return np.divide(numerator, denominator)

#answer to 1c
print(f'1c. U(1): {next_U(V_0, U_0.shape, y, l)}')

