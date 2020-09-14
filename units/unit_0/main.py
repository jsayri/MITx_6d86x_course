import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    return np.random.random((n, 1))


def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    #Your code here
    A = np.random.random((h, w))
    B = np.random.random((h, w))
    s = A + B
    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    #Your code here
    s = np.linalg.norm(A + B)
    return s


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    #Your code here
    out = np.tanh(np.matmul(weights.transpose(), inputs))
    return out

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x < y:
        s = x * y
    else:
        s = x/y
    return s


def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y
    """
    #Your code here
    f = np.vectorize(scalar_function)
    return f(x, y)


# Main function call (to test results)

# call function randomization
rslt1 = randomization(2)
print('Resuturn a nx1 vector with random values from a normal distribution')
print(rslt1)

# call function randomization
a, b, c = operations(2, 3)
print('Return a h x w vector with random values from a normal distribution')
print('Matrix A :')
print(a)
print('Matrix B :')
print(b)
print('Sum of A & B :')
print(c)

# call function norm
a, b, c = operations(3, 1)
c = norm(a, b)
print('Return a h x w vector with random values from a normal distribution')
print('vector a :')
print(a)
print('vector  b :')
print(b)
print('L2 norm of a+b :')
print(c)

# call function neural_networks
x = np.random.randint(0, 10, (2, 1))
w = np.random.random((2, 1))
z = neural_network(x, w)
print('inputs as x')
print(x)
print('weights as w')
print(w)
print('output as z')
print(z)

# call scalar function
print('result 1 scalar_function(1, 2): ', scalar_function(1, 2))
print('result 2 scalar_function(5, 4): ', scalar_function(5, 4))

# call vectorize function
print('result 3 vector_function([1, 2, 3], [2, 7, 1]): ', vector_function([1, 2, 3], [2, 7, 1]))
print('result 4 vector_function([1, 2, 3], [2, 7, 1]): ', vector_function(np.array([1, 2, 3]), np.array([2, 7, 1])))
print('result 5 vector_function([7, 3, 9], 3): ', vector_function(np.array([7, 3, 9]), np.array([3])))