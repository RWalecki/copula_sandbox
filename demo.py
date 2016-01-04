import numpy as np
import copulas
import margins

F1 = margins.uniform(1,3)
F2 = margins.uniform(3,5)
C = copulas.Frank(5)

X = []
for i in range(5000):
    u1, v2 = np.random.uniform(size=2)
    u2 = C(u1, v2)
    X.append([ F1(u1), F2(u2) ])

X = np.array(X).T

import matplotlib.pyplot as plt
plt.scatter(X[0],X[1])
plt.show()
