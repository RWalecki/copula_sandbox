import Copula_Sandbox as CS


# define copula model:
mvd1 = CS.Archimedes(type='frank', dim = 2, para=10)
mvd1.set_margin(0, 'uniform',para=[3,4])
mvd1.set_margin(1, 'sigmoid',para=[1,2])
X = mvd1.generate_x(300)
mvd1.fit(X)
mvd1.plot_model(X)


'''
# fit magins and copula to training data
X = mvd1.generate_x(1000)
U = mvd1.transform_u(X)

# visualize results 
import matplotlib.pyplot as plt
plt.subplot(211)
plt.scatter(X.T[0],X.T[1],alpha=0.5)
plt.title('X')
plt.subplot(212)
plt.scatter(U.T[0],U.T[1],alpha=0.5)
plt.title('U')
plt.savefig('/tmp/tmp.pdf')
'''
