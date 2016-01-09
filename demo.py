import Copula_Sandbox as CS
import cPickle,gzip

# load training data:
[X,info] = cPickle.load(gzip.open('./tests/test_samples.pklz','rb'))
print info,'\n'

# define copula model:
mvd1 = CS.Archimedes(type='frank', dim = 2)
mvd1.set_margin(0, 'sigmoid')
mvd1.set_margin(1, 'sigmoid')


# fit magins and copula to training data
mvd1.fit(X)

# print parameter
print mvd1.C_para
print mvd1.F_para[0]
print mvd1.F_para[1], '\n'

# sample from copula model
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
