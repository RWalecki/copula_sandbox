import Multivariate_Sampling as ms
import cPickle,gzip

# generate samples from a frank coupula
# with a uniform and sigmoid margin
'''
mvd1 = ms.MVD(type='frank', dim = 2, para=-7, verbose=1)
mvd1.add_margin(0, 'sigmoid', para=[3,4])
mvd1.add_margin(1, 'uniform', para=[-10,3])
X = mvd1.generate_x(5000)
'''
[X,info] = cPickle.load(gzip.open('test_samples.pklz','rb'))
import numpy as np
print X.shape
print info
mvd1 = ms.MVD(type='frank', para=5, dim = 2, verbose = 1)
mvd1.add_margin(0, 'sigmoid')
mvd1.add_margin(1, 'uniform')
mvd1.fit(X)
mvd1.transform_u(X)
#X = mvd1.generate_x(5000)
#mvd1.fit(X)
'''

# visualize results 
mvd2.visual_model('/tmp/tmp.pdf',samples = X)
import matplotlib.pyplot as plt
plt.scatter(X.T[0],X.T[1],c='k',alpha=0.5)
plt.savefig('/tmp/tmp.pdf')
'''
