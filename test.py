import Multivariate_Sampling as ms


# define copula model:
mvd1 = ms.MVD(type='frank', dim = 3, para=5)
mvd1.add_margin(0, 'sigmoid',para=[3,4])
mvd1.add_margin(1, 'sigmoid',para=[1,2])
mvd1.plot_model()


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
