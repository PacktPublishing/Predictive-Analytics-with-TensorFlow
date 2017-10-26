import numpy as np
from statistics import variance, stdev

returns = [20, 10, -15, 5]
ret_arr = np.array(returns)

# For sample
var = variance(ret_arr)
sd = stdev(ret_arr)
print(var)
print(sd)

# For population
mean = np.mean(ret_arr)
var = np.var(ret_arr)
sd = np.std(ret_arr)

print("Mean: ", mean)
print("Variance", var)
print("Standard deviation", sd)

