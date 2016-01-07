import numpy as np
from Multivariate_Sampling import copulas
from Multivariate_Sampling import margins
from Multivariate_Sampling import MVD

M1 = margins.uniform()
M2 = margins.sigmoid()
C = copulas.frank()

mvd = MVD(copula=C,margin=[M1,M2])
mvd.copula_para = [1]
mvd.margin_para = [[3,3],[1,2]]

X = mvd.generate_x()

import matplotlib.pyplot as plt
plt.scatter(X.T[0],X.T[1])
plt.savefig('/tmp/tmp.pdf')

mvd.fit(X)
print mvd.margin_para
print mvd.copula_para
print

X = mvd.generate_x()
mvd.fit(X)
print mvd.margin_para
print mvd.copula_para
print

X = mvd.generate_x(10)
mvd.fit(X)
print mvd.margin_para
print mvd.copula_para

