import numpy as np

matrix = np.matrix(
    [[3, 6, 7],
     [2, 7, 9],
     [5, 8, 6]])

eigvals = np.linalg.eigvals(matrix)
print(eigvals)
