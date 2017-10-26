import csv
import numpy as np
import matplotlib.pyplot as plt
import time_series_preprocessor as tsp
import warnings

warnings.filterwarnings("ignore")

timeseries = tsp.load_series('input/international-airline-passengers.csv')
print(timeseries)
print(np.shape(timeseries))

plt.figure()
plt.plot(timeseries)
plt.title('Normalized time series')
plt.xlabel('ID')
plt.ylabel('Normalized value')
plt.legend(loc='upper left')
plt.show()

