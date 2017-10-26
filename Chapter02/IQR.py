import numpy as np
from statistics import variance, stdev

data_points = np.array([35, 56, 43, 59, 63, 79, 35, 41, 64, 43, 93, 60, 77, 24, 82])

# Range
dt_rng = np.max(data_points, axis=0) - np.min(data_points, axis=0)
print ("Range:", dt_rng)

# Percentiles
print("Quantiles:")
for val in [20, 80, 100]:
    qntls = np.percentile(data_points,val)
print(str(val)+"%", qntls)

# IQR
q75, q25 = np.percentile(data_points, [75, 25])
print("Inter quartile range:", q75-q25)
