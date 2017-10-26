from numpy import random
from matplotlib import pyplot as plt

X = random.random_integers(10, size = 1000)
print(X)

plt.hist(X)
plt.title("Frequency distribution")
plt.xlabel("Integers")
plt.ylabel("Value")
plt.show()

Y = random.normal(size=1000)
print(Y)

plt.hist(Y)
plt.title("Frequency distribution")
plt.xlabel("Integers")
plt.ylabel("Value")
plt.show()