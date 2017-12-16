"""
File with example code classifying different flower kinds.

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from perceptron import Perceptron
from decision_region import plot_decision_region

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values
plt.scatter(X[0:50, 0], X[0:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('Länge des Kelchblattes [cm]')
plt.ylabel('Länge des Blütenblattes [cm]')
plt.legend(loc='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochen')
plt.ylabel('Anzahl der Fehlklassifizierungen')
plt.show()

plot_decision_region(X, y, ppn)
