import numpy as np

#pragma: coderesponse template
def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    #Your code here
    A = np.random.random([n,1])
    return A
    #raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
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
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A + B
    return A, B, s

    #raise NotImplementedError
#pragma: coderesponse end


#pragma: coderesponse template
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
    s = np.linalg.norm(A+B)
    return s

    #raise NotImplementedError
#pragma: coderesponse end



#pragma: coderesponse template
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
    z= np.tanh(np.matmul(weights.transpose(), inputs))
    return z

    #raise NotImplementedError
#pragma: coderesponse end



#pragma: coderesponse template
def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    #Your code here
    if x<= y:
      return x*y
    else:
      return x/y
    #raise NotImplementedError
#pragma: coderesponse end



#pragma: coderesponse template
def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    f = np.vectorize(scalar_function)
    return f(x, y)
    #raise NotImplementedError
#pragma: coderesponse end



######### TESTING FUNCTIONS  ##########
# 1 Function: Randomization
print("Test randomization: returns nx1 array")
print(randomization(3), '\n')

# 2 Function: Operations
print("Test operations: returns three arrays: A, B and A+B")
matrixA, matrixB, matrixSum = (operations(2,2))
print("A = ", matrixA)
print("B = ", matrixB)
print("Sum A + B =", matrixSum, '\n')

# 3 Function: norm
print("Test norm: returns norm of the sum of two arrays.")
#A = np.random.random([2,2])
#B = np.random.random([2,2])
print("A = ", matrixA)
print("B = ", matrixB)
import pdb; pdb.set_trace()
print("L2 norm of the sum A+B =", norm(matrixA,matrixB), '\n')

# 4 Function: neural_network
print("Test neural_network: inputs runs through a 1-layer neural network.")
inputA = np.random.random([2,1])
weightB = np.random.random([2,1])
print(neural_network(inputA,weightB), '\n')


# 5 Function: scalar_function
print("Test scalar_function: returns the f(x,y)")
print(scalar_function(10,5), '\n')

#6 Fuction: vector_function
print("Test vector_function: Make sure vector_function can deal with vector input x,y ")
print(vector_function(np.array([1, 5, 3]), np.array([4, 2, 6])))