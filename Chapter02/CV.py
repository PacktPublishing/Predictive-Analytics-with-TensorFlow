import numpy as np
a = np.array([2.1, 2.5, 4.0, 3.6])
b = np.array([8, 12, 14, 10])
np.cov(a, b)

cv = np.cov(a, b)[0][1]
print(cv)

