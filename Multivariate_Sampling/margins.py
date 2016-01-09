import sympy as sy

def uniform(n='_0'):
    '''
    '''
    x, x0, s = sy.symbols('x'+n+',x0'+n+',s'+n)
    X = (x-x0)/s
    F = X
    return F, x, [x0, s]

def sigmoid(n='_0'):
    '''
    '''
    x, x0, s = sy.symbols('x'+n+',x0'+n+',s'+n)
    X = (x-x0)/s
    F = 1/(1+sy.exp(-X))
    return F, x, [x0, s]

def load(type,n):
    m = {
            'uniform':uniform,
            'sigmoid':sigmoid,
            }
    return m[type]('_'+str(n))
