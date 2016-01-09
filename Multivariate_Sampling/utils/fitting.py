import sympy as sy
import numpy as np
from scipy.optimize import minimize

def fit_copula(C, U, d, samples):
    '''
    '''
    # compute copula probability density function 
    P = C
    for u in U:P = sy.diff(P,u)


    # define loss funciton
    # negative log likelihood of copula parameter: loss(d) = -log(P(U|d))
    _loss = -sy.log(P)
    loss = sy.lambdify([d]+U,_loss,'numpy')

    # gradient of loss function
    _grad = sy.diff(_loss,d)
    grad = sy.lambdify([d]+U,_grad,'numpy')

    def obj(x,u_samp):
        args = []
        for tmp in u_samp.T:args.append(tmp)
        return loss(x,*args).mean()

    def jac(x,u_samp):
        args = []
        for tmp in u_samp.T:args.append(tmp)
        return grad(x,*args).mean()

    x1 = minimize(
            fun = obj,
            x0 = 10,
            args=samples,
            jac=jac,
            method='CG',
            ).x

    return x1

def fit_margin(F, x, P, samples, domain='cdf'):
    '''
    '''
    # minimize mean square error in cdf domain
    if domain=='cdf':

        # compute empirical margin
        X = np.sort(samples)
        Y = np.arange(len(samples))/float(len(samples))

        # define loss function as mean square error in cdf domain 
        y = sy.symbols('y')
        _loss_f = (F-y)**2
        loss_f = sy.lambdify([x]+[y]+P, _loss_f, 'numpy')

        def obj(p,x,y):
            return loss_f(x,y,*p).mean()

        # compute gradients to each parameter
        loss_g = []
        for p in P:
            tmp = sy.diff(_loss_f,p)
            loss_g.append(sy.lambdify([x]+[y]+P, tmp,'numpy'))

        def grad(p,x,y):
            res = []
            for i in loss_g:
                res.append(i(x,y,*p).mean())
            return  np.array(res)

    # maximize log_probability (TODO)
    if domain=='pdf':pass

    x1 = minimize(
            fun = obj,
            x0 = [samples.mean(),samples.std()],
            args=(X,Y),
            jac=grad,
            method='CG',
            ).x

    return x1
