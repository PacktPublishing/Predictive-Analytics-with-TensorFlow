import numpy as np

matrix = np.matrix(
    [[3, 6, 7],
     [2, 7, 9],
     [5, 8, 6]])

inverse = np.linalg.inv(matrix)
print(inverse)

I = inverse * matrix
print(I)

