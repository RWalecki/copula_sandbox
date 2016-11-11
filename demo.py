import copula_sandbox as cs


# define copula model:
mvd1 = cs.archimedes(type='frank', dim = 2)
mvd1.set_margin(0, 'sigmoid')
mvd1.set_margin(1, 'sigmoid')

mvd1.F_para[0] = [15, 5]
mvd1.F_para[1] = [3, 13]
mvd1.C_para=8

# simulate from copula model
X = mvd1.generate_x(1000)

# fit magins and copula to training data
mvd1.fit(X)

# print parameter
print(mvd1.C_para)
print(mvd1.F_para[0])
print(mvd1.F_para[1], '\n')


mvd2 = cs.archimedes(type='frank', dim = 2)
mvd2.set_margin(0, 'sigmoid')
mvd2.set_margin(1, 'uniform')

# # simulate from copula model
mvd2.fit(X)
U = mvd2.transform_u(X)

# ## visualize results
import matplotlib.pyplot as plt
plt.subplot(211)
plt.scatter(X.T[0],X.T[1],alpha=0.5)
plt.title('X')
plt.subplot(212)

plt.scatter(U.T[0],U.T[1],alpha=0.5)
plt.title('U')
plt.savefig('/tmp/tmp.pdf')
