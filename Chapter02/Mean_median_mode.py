import numpy as np
from scipy import stats

data = np.array([3, 5, 9, 2, 7, 3, 6, 9, 3])

# Mean
dt_mean = np.mean(data);
print("Mean :", round(dt_mean, 2))

# Median
dt_median = np.median(data);
print("Median :", dt_median)

# Mode
dt_mode = stats.mode(data);
print("Mode :", dt_mode[0][0])