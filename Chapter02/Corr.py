import numpy as np
import numpy as np
from statistics import variance, stdev
a = [1,2,3,4,5,6]
#expectation
np.mean(a)
#variance
a = np.array([1,2,3,4,5,6])
var = variance(a)
sd = stdev(a)
print(var)
print(sd)


a = np.array([2.1, 2.5, 4.0, 3.6])
b = np.array([8, 12, 14, 10])

corr = np.corrcoef(a, b)[0, 1]
print(corr)