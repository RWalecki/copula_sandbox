import sympy as sy
import numpy as np
from scipy.optimize import minimize

def fit_copula(C, para, U):

    #  if there are no parameter to optimize,
    #  return empty array
    if len(para)==2:return []

    # compute empirical copula
    X,Y,N = [],[],20

    for i0 in np.linspace(0,1,N):
        for i1 in np.linspace(0,1,N):
            idx = U[:,0]<=i0
            idx = U[idx,1]<=i1
            X.append([i0,i1])
            Y.append(np.sum(idx)/float(U.shape[0]))

    X = np.array(X)
    Y = np.array(Y)

    # loss function
    y = sy.symbols('y')
    loss = (C-y)**2
    loss_F = sy.lambdify([y]+para, loss, 'numpy')

    def obj(p,u0,u1,y):
        return loss_F(y,u0,u1,*p).mean()

    # gradient functions
    loss_J = []
    for p in para[2:]:
        jac = sy.diff(loss,p)
        loss_J.append(sy.lambdify([y]+para, jac,'numpy'))

    def jac(p,u0,u1,y):
        res = []
        for i in loss_J:
            res.append(i(y,u0,u1,*p).mean())
        return  np.array(res)

    x1 = minimize(
            fun = obj,
            x0 = np.ones(len(para)-2),
            args=(X[:,0],X[:,1],Y),
            jac=jac,
            method='CG',
            ).x

    return x1


def fit_margin(F, para, X):

    # compute empirical margin
    X = np.sort(X)
    Y = np.arange(len(X))/float(len(X))

    # loss function
    u = sy.symbols('u')
    loss = (F-u)**2
    loss_F = sy.lambdify([u]+para, loss, 'numpy')

    def obj(p,x,y):
        return loss_F(y,x,*p).mean()
    
    # gradient functions
    loss_J = []
    for p in para[1:]:
        jac = sy.diff(loss,p)
        loss_J.append(sy.lambdify([u]+para, jac,'numpy'))

    def jac(p,x,y):
        res = []
        for i in loss_J:
            res.append(i(y,x,*p).mean())
        return  np.array(res)

    x1 = minimize(
            fun = obj,
            x0 = np.ones(len(para)-1),
            args=(X,Y),
            jac=jac,
            method='CG',
            ).x

    return x1
