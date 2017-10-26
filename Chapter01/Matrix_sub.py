import numpy as np

A = np.matrix(
    [[1, 4],
     [2, 9]]
)

B = np.matrix(
    [[7, -9],
     [4, 6]]
)


C = A - B
print(C)