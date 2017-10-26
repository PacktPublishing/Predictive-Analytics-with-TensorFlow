import numpy as np
from sklearn import decomposition
import csv

pca = decomposition.PCA(n_components=10)
rows = []
delimiter = ','
with open('data/housing.csv', 'r') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=delimiter)
	for row in csv_reader:
		rows.append([float(r) for r in row])
data = np.array(rows)

Y = pca.fit_transform(data)
print("Using SKLEARN for PCA on Housing dataset: ")
print("---------------------")
print(Y)
print("---------------------")

